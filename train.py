'''Training'''
from scipy.io import loadmat
import numpy as np
import argparse
import configparser
import torch
from torch import nn
from skimage.segmentation import slic
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import scale, minmax_scale
import os
from PIL import Image
from utils import get_graph_list, split, get_edge_index
import math
from Model.module import SubGcnFeature, GraphNet1,GraphNet2,GraphNet3,GraphNet4, MLPNet,MLPNet_1,Conv1x1
from Trainer import JointTrainer
from Monitor import GradMonitor
# from visdom import Visdom
from tqdm import tqdm
import random
import time



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='TRAIN SUBGRAPH')
    parser.add_argument('--name', type=str, default='PaviaU',
                        help='DATASET NAME')
    parser.add_argument('--block', type=int, default=100,
                        help='BLOCK SIZE')
    parser.add_argument('--epoch', type=int, default=500,
                        help='ITERATION')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--comp', type=int, default=10,
                        help='COMPACTNESS')
    parser.add_argument('--batchsz', type=int, default=64,
                        help='BATCH SIZE')
    parser.add_argument('--run', type=int, default=10,
                        help='EXPERIMENT AMOUNT')
    parser.add_argument('--spc', type=int, default=5,
                        help='SAMPLE per CLASS')
    parser.add_argument('--hsz', type=int, default=256,
                        help='HIDDEN SIZE')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='LEARNING RATE')
    parser.add_argument('--wd', type=float, default=0.,
                        help='WEIGHT DECAY')
    arg = parser.parse_args()
    config = configparser.ConfigParser()
    config.read('dataInfo.ini')
    # viz = Visdom(port=8097)

    # Data processing
    # Reading hyperspectral image
    data_path = 'data/{0}/{0}.mat'.format(arg.name)
    m = loadmat(data_path)
    data = m[config.get(arg.name, 'data_key')]
    gt_path = 'data/{0}/{0}_gt.mat'.format(arg.name)
    m = loadmat(gt_path)
    gt = m[config.get(arg.name, 'gt_key')]
    # Normalizing data
    h, w, c = data.shape
    data = data.reshape((h * w, c))
    data = data.astype(float)
    if arg.name == 'Xiongan':
        minmax_scale(data, copy=False)
    data_normalization = scale(data).reshape((h, w, c))

    # Superpixel segmentation
    seg_root = 'data/rgb'
    seg_path = os.path.join(seg_root, '{}_seg_{}.npy'.format(arg.name, arg.block))
    if os.path.exists(seg_path):
        seg = np.load(seg_path)
    else:
        rgb_path = os.path.join(seg_root, '{}_rgb.jpg'.format(arg.name))
        img = Image.open(rgb_path)
        img_array = np.array(img)
        # The number of superpixel
        n_superpixel = int(math.ceil((h * w) / arg.block))
        seg = slic(img_array, n_superpixel, arg.comp)
        seg = seg - 1
        print(np.max(seg))
        # Saving
        np.save(seg_path, seg)


    num_superpixels = np.max(seg) + 1
    feature_dim = c

    superpixel_features = np.zeros((num_superpixels, feature_dim))

    for i in range(num_superpixels):

        mask = (seg == i)

        summed_features = np.sum(data_normalization[mask], axis=0)

        averaged_features = summed_features / np.sum(mask)

        superpixel_features[i] = averaged_features


    superpixel_features_tensor = torch.tensor(superpixel_features, dtype=torch.float32).to('cuda')

    full_edge_index_path = 'data/{}/{}_edge_index.npy'.format(arg.name, arg.block)
    if os.path.exists(full_edge_index_path):
        edge_index = np.load(full_edge_index_path)
    else:
        edge_index, _ = get_edge_index(seg)
        np.save(full_edge_index_path, edge_index if isinstance(edge_index, np.ndarray) else edge_index.cpu().numpy())
    fullGraph = Data(None,
                     edge_index=torch.from_numpy(edge_index) if isinstance(edge_index, np.ndarray) else edge_index,
                     seg=torch.from_numpy(seg) if isinstance(seg, np.ndarray) else seg)

    alltime = 0
    for r in range(arg.run):

        print('*'*5 + 'Run {}'.format(r) + '*'*5)
        # Reading the training data set and testing data set
        m = loadmat('trainTestSplit/{}/sample{}_run{}.mat'.format(arg.name, arg.spc, r))
        tr_gt, te_gt = m['train_gt'], m['test_gt']
        tr_gt_torch, te_gt_torch = torch.from_numpy(tr_gt).long(), torch.from_numpy(te_gt).long()
        fullGraph.tr_gt, fullGraph.te_gt = tr_gt_torch, te_gt_torch



        student_net = GraphNet3(c, 256,c)
        teacher_net = GraphNet3(c, 256,c)

        mlp = MLPNet(c, c, config.getint(arg.name, 'nc'))
        mlp1 = MLPNet_1(c, c, config.getint(arg.name, 'nc'))


        optimizer_gcn2 = torch.optim.Adam([{'params': student_net.parameters()}],
                                          weight_decay=arg.wd)


        optimizer_all = torch.optim.Adam([
                                        {'params': student_net.parameters()},
                                        {'params': mlp.parameters()}
        ],
            weight_decay=arg.wd)

        criterion = nn.CrossEntropyLoss()
        trainer = JointTrainer([student_net,teacher_net,mlp,mlp1])
        monitor = GradMonitor()


        device = torch.device('cuda:{}'.format(arg.gpu)) if arg.gpu != -1 else torch.device('cpu')
        max_acc = 0
        save_root = 'models/{}/{}/{}_overall_skip_2_SGConv_l1_clip'.format(arg.name, arg.spc, arg.block)
        pbar = tqdm(range(arg.epoch))
        # Training
        for epoch in pbar:

            start = time.time()

            pbar.set_description_str('Epoch: {}'.format(epoch))
            kl_loss = trainer.train_kl(superpixel_features_tensor, fullGraph, optimizer_gcn2, device, monitor.clear(),is_l1=True, is_clip=True)

            n_e_loss = trainer.train_t_f(superpixel_features_tensor, fullGraph, optimizer_gcn2, device, monitor.clear(), is_l1=True, is_clip=True)

            ce_loss = trainer.train_ce(superpixel_features_tensor, fullGraph, optimizer_all, criterion, device, monitor.clear(), is_l1=True, is_clip=True)

            te_loss, acc = trainer.evaluate(superpixel_features_tensor, fullGraph, criterion, device,r,epoch)

            pbar.set_postfix_str('KL loss: {} n_e_loss: {} train_ce_loss: {}  test loss:{} acc:{}'.format(kl_loss, n_e_loss,ce_loss,te_loss, acc))

            temp = time.time() - start
            alltime = alltime + temp

            if acc > max_acc:
                max_acc = acc
                if not os.path.exists(save_root):
                    os.makedirs(save_root)
                trainer.save([
                              os.path.join(save_root, 'studentNet_best_{}_{}.pkl'.format(arg.spc, r)),
                              os.path.join(save_root, 'teacherNet_best_{}_{}.pkl'.format(arg.spc, r)),
                              os.path.join(save_root, 'mlpNet_best_{}_{}.pkl'.format(arg.spc, r))])
    print("total time:",alltime)
    print('*'*5 + 'FINISH' + '*'*5)

