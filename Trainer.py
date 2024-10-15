import torch
from torch_geometric.data import Data, Batch
from torch.optim import optimizer as optimizer_
from torch_geometric.utils import accuracy
from torch_geometric.nn import DataParallel
from torch.nn.utils import clip_grad_norm_
import time
from utils import construt_nosimilarity_graph,perturb_graph,update_teacher_params
import torch.nn.functional as F
import scipy.io as sio

class JointTrainer(object):
    r'''Joint trainer'''
    def __init__(self, models: list):
        super().__init__()
        self.models = models

    def train_kl(self, superpixel_features_tensor, fullGraph: Data, optimizer, device, monitor = None, is_l1=False, is_clip=False):

        extNet_student = self.models[0]
        extNet_teacher = self.models[1]
        mlpNet = self.models[3]

        extNet_student.train()
        extNet_teacher.train()
        mlpNet.eval()

        extNet_student.to(device)
        extNet_teacher.to(device)
        mlpNet.to(device)

        superpixel_features_tensor.to(device)


        fullGraph.x = superpixel_features_tensor
        fullGraph = fullGraph.to(device)


        # 构建扰动图 两个
        x1, edge_index1 = perturb_graph(fullGraph.x, fullGraph.edge_index, 0.10, 0.10)
        x2, edge_index2 = perturb_graph(fullGraph.x, fullGraph.edge_index, 0.05, 0.05)

        h1 = extNet_student(x1, edge_index1)
        h2 = extNet_teacher(x2, edge_index2)

        logits1 = mlpNet(h1)
        logits2 = mlpNet(h2)

        log_pred1 = F.log_softmax(logits1, dim=1)
        log_pred2 = F.softmax(logits2, dim=1)

        kl_loss = F.kl_div(log_pred1, log_pred2, reduction='batchmean')   # 原文给的一致性损失的系数 0.02
        kl_loss = 0.02 * kl_loss
        optimizer.zero_grad()
        kl_loss.backward()
        optimizer.step()

        update_teacher_params(extNet_teacher,extNet_student,0.90)

        return kl_loss.item()


    def train_t_f(self, superpixel_features_tensor, fullGraph: Data, optimizer, device, monitor = None, is_l1=False, is_clip=False):

        extNet_student = self.models[0]
        extNet_teacher = self.models[1]
        mlpNet = self.models[3]

        extNet_student.train()
        extNet_teacher.train()
        mlpNet.eval()

        extNet_student.to(device)
        extNet_teacher.to(device)
        mlpNet.to(device)

        superpixel_features_tensor.to(device)

        fullGraph.x = superpixel_features_tensor
        fullGraph = fullGraph.to(device)


        x1, edge_index1 = perturb_graph(fullGraph.x, fullGraph.edge_index, 0.10, 0.10)
        x2, edge_index2 = perturb_graph(fullGraph.x, fullGraph.edge_index, 0.05, 0.05)

        emb1 = extNet_student(x1, edge_index1)
        emb2 = extNet_teacher(x2, edge_index2)

        norm1 = (emb1 - emb1.mean(dim=1, keepdim=True)) / emb1.std(1, keepdim=True)
        norm2 = (emb2 - emb2.mean(dim=1, keepdim=True)) / emb2.std(1, keepdim=True)
        corr1 = torch.mul(norm1, norm1)
        corr2 = torch.mul(norm2, norm2)
        corr = torch.mm(norm1, norm2.t())

        loss_s = F.mse_loss(corr1, corr2)
        I = torch.eye(corr.shape[0]).to(device)
        loss_f = F.mse_loss(corr, I)
        loss_s_f = 0.01 * loss_s + 0.003 * loss_f


        optimizer.zero_grad()
        loss_s_f.backward()
        optimizer.step()


        update_teacher_params(extNet_teacher, extNet_student, 0.90)

        return loss_s_f.item()



    def train_ce(self, superpixel_features_tensor, fullGraph: Data, optimizer, criterion, device, monitor = None, is_l1=False, is_clip=False):

        extNet_student = self.models[0]
        extNet_teacher = self.models[1]
        mlpNet = self.models[2]

        extNet_student.train()
        extNet_teacher.train()
        mlpNet.train()

        extNet_student.to(device)
        extNet_teacher.to(device)
        mlpNet.to(device)
        criterion.to(device)

        superpixel_features_tensor.to(device)


        # External graph features
        fullGraph.x = superpixel_features_tensor
        fullGraph = fullGraph.to(device)

        x1, edge_index1 = perturb_graph(fullGraph.x, fullGraph.edge_index, 0.10, 0.10)

        h1 = extNet_student(x1, edge_index1)

        logits = mlpNet(fullGraph,h1)
        indices = torch.nonzero(fullGraph.tr_gt, as_tuple=True)

        y = fullGraph.tr_gt[indices].to(device) - 1
        node_number = fullGraph.seg[indices]
        pixel_logits = logits[node_number]
        loss = criterion(pixel_logits, y)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        update_teacher_params(extNet_teacher, extNet_student, 0.90)


        return loss.item()

    def evaluate(self, superpixel_features_tensor, fullGraph, criterion, device,r,epoch):

        extNet_student = self.models[0]
        extNet_teacher = self.models[1]
        mlpNet = self.models[2]

        extNet_student.eval()
        extNet_teacher.eval()
        mlpNet.eval()

        extNet_student.to(device)
        extNet_teacher.to(device)
        mlpNet.to(device)
        criterion.to(device)

        superpixel_features_tensor.to(device)

        with torch.no_grad():


            fullGraph.x = superpixel_features_tensor
            fullGraph = fullGraph.to(device)


            h = extNet_student(fullGraph.x, fullGraph.edge_index)

            logits = mlpNet(fullGraph, h)

            pred = torch.argmax(logits, dim=-1)
            indices = torch.nonzero(fullGraph.te_gt, as_tuple=True)

            y = fullGraph.te_gt[indices].to(device) - 1
            node_number = fullGraph.seg[indices]
            pixel_pred = pred[node_number]
            pixel_logits = logits[node_number]
            loss = criterion(pixel_logits, y)
        return loss.item(), accuracy(pixel_pred, y)

    # Getting prediction results
    def predict(self, superpixel_features_tensor, fullGraph, device: torch.device):

        extNet_student = self.models[0]
        extNet_teacher = self.models[1]
        mlpNet = self.models[2]

        extNet_student.eval()
        extNet_teacher.eval()
        mlpNet.eval()

        extNet_student.to(device)
        extNet_teacher.to(device)
        mlpNet.to(device)

        superpixel_features_tensor.to(device)

        with torch.no_grad():
            # Internal graph features

            # External graph features
            fullGraph.x = superpixel_features_tensor
            fullGraph = fullGraph.to(device)

            h = extNet_student(fullGraph.x, fullGraph.edge_index)

            logits = mlpNet(fullGraph, h)

        pred = torch.argmax(logits, dim=-1)


        return pred

    # Getting hidden features
    def getHiddenFeature(self, subGraph, fullGraph, device, gt = None, seg = None):
        intNet = DataParallel(self.models[0])
        extNet = self.models[1]
        intNet.eval()
        extNet.eval()
        intNet.to(device)
        extNet.to(device)
        with torch.no_grad():
            fe = intNet(subGraph.to_data_list())
            fullGraph.x = fe
            fullGraph = fullGraph.to(device)
            fe = extNet(fullGraph)
        if gt is not None and seg is not None:
            indices = torch.nonzero(gt, as_tuple=True)
            gt = gt[indices] - 1
            node_number = seg[indices].to(device)
            fe = fe[node_number]
            return fe.cpu(), gt
        else:
            return fe.cpu()

    def get_parameters(self):
        return self.models[0].parameters(), self.models[1].parameters()

    def save(self, paths):
        torch.save(self.models[0].cpu().state_dict(), paths[0])
        torch.save(self.models[1].cpu().state_dict(), paths[1])
        torch.save(self.models[2].cpu().state_dict(), paths[2])








