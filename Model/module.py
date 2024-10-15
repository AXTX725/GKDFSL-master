from torch_geometric import nn as gnn
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, BatchNorm


# Internal graph convolution
class SubGcn(nn.Module):
    def __init__(self, c_in, hidden_size, nc):
        super().__init__()
        self.gcn = gnn.SGConv(c_in, hidden_size, K=3)
        # 1
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(hidden_size // 2, nc)
        )

    def forward(self, graph):
        h = F.relu(self.gcn(graph.x, graph.edge_index))
        h_avg = gnn.global_mean_pool(h, graph.batch)
        logits = self.classifier(h_avg)
        return logits


# Internal graph convolution feature module
class SubGcnFeature(nn.Module):
    def __init__(self, c_in, hidden_size):
        super().__init__()
        self.gcn = gnn.SGConv(c_in, hidden_size, K=3)

    def forward(self, graph):
        h = F.relu(self.gcn(graph.x, graph.edge_index))
        h_avg = gnn.global_mean_pool(h, graph.batch)
        return h_avg


# External graph convolution
# class GraphNet(nn.Module):
#     def __init__(self, c_in, hidden_size, nc):
#         super().__init__()
#         self.bn_0 = gnn.BatchNorm(c_in)
#         self.gcn_1 = gnn.GCNConv(c_in, hidden_size)
#         self.bn_1 = gnn.BatchNorm(hidden_size)
#         self.gcn_2 = gnn.GraphConv(hidden_size, hidden_size)
#         self.bn_2 = gnn.BatchNorm(hidden_size)
#         # self.gcn_3 = gnn.GraphConv(hidden_size, hidden_size)
#         # self.bn_3 = gnn.BatchNorm(hidden_size)
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.ReLU(),
#             # nn.Dropout(),
#             nn.Linear(hidden_size // 2, nc)
#         )
#
#     def forward(self, graph):
#         # x_normalization = graph.x
#         # h = F.relu(self.gcn_1(x_normalization, graph.edge_index))
#         # h = F.relu(self.gcn_2(h, graph.edge_index))
#         x_normalization = self.bn_0(graph.x)
#         h = self.bn_1(F.relu(self.gcn_1(x_normalization, graph.edge_index)))
#         h = self.bn_2(F.relu(self.gcn_2(h, graph.edge_index)))
#         # h = self.bn_3(F.relu(self.gcn_3(h, graph.edge_index)))
#         # h = F.relu(self.gcn_2(h, graph.edge_index))
#         logits = self.classifier(h + x_normalization)
#         # logits = self.classifier(h)
#         return logits

#    1层GCN
class GraphNet1(nn.Module):
    def __init__(self, c_in, hidden_size1):
        super().__init__()
        self.bn_0 = gnn.BatchNorm(c_in)
        self.gcn_1 = gnn.GCNConv(c_in, hidden_size1)
        self.bn_1 = gnn.BatchNorm(hidden_size1)


    def forward(self, x,edge_index):

        x_normalization = self.bn_0(x)

        h = self.bn_1(F.relu(self.gcn_1(x_normalization, edge_index)))

        # h = F.relu(self.gcn_2(h, graph.edge_index))
        # logits = self.classifier(h)
        return h

#    2层GCN
class GraphNet2(nn.Module):
    def __init__(self, c_in, hidden_size1,hidden_size2):
        super().__init__()
        self.bn_0 = gnn.BatchNorm(c_in)
        self.gcn_1 = gnn.GCNConv(c_in, hidden_size1)
        self.bn_1 = gnn.BatchNorm(hidden_size1)
        self.gcn_2 = gnn.GraphConv(hidden_size1, hidden_size2)
        self.bn_2 = gnn.BatchNorm(hidden_size2)


    def forward(self, x,edge_index):
        x_normalization = self.bn_0(x)
        h = self.bn_1(F.relu(self.gcn_1(x_normalization, edge_index)))
        h = self.bn_2(F.relu(self.gcn_2(h, edge_index)))
        return h


#  外部图卷积 只返回特征  GCN    3层GCN
class GraphNet3(nn.Module):
    def __init__(self, c_in, hidden_size1,hidden_size2):
        super().__init__()
        self.bn_0 = gnn.BatchNorm(c_in)
        self.gcn_1 = gnn.GCNConv(c_in, hidden_size1)
        self.bn_1 = gnn.BatchNorm(hidden_size1)
        self.gcn_2 = gnn.GraphConv(hidden_size1, hidden_size1)
        self.bn_2 = gnn.BatchNorm(hidden_size1)
        self.gcn_3 = gnn.GraphConv(hidden_size1, hidden_size2)
        self.bn_3 = gnn.BatchNorm(hidden_size2)

    def forward(self, x,edge_index):
        # x_normalization = graph.x
        # h = F.relu(self.gcn_1(x_normalization, graph.edge_index))
        # h = F.relu(self.gcn_2(h, graph.edge_index))
        x_normalization = self.bn_0(x)

        h = self.bn_1(F.relu(self.gcn_1(x_normalization, edge_index)))
        h = self.bn_2(F.relu(self.gcn_2(h, edge_index)))
        h = self.bn_3(F.relu(self.gcn_3(h, edge_index)))
        # h = F.relu(self.gcn_2(h, graph.edge_index))
        # logits = self.classifier(h)
        return h


#  外部图卷积 只返回特征  GCN    4层GCN
class GraphNet4(nn.Module):
    def __init__(self, c_in, hidden_size1,hidden_size2):
        super().__init__()
        self.bn_0 = gnn.BatchNorm(c_in)
        self.gcn_1 = gnn.GCNConv(c_in, hidden_size1)
        self.bn_1 = gnn.BatchNorm(hidden_size1)
        self.gcn_2 = gnn.GraphConv(hidden_size1, hidden_size1)
        self.bn_2 = gnn.BatchNorm(hidden_size1)
        self.gcn_3 = gnn.GraphConv(hidden_size1, hidden_size1)
        self.bn_3 = gnn.BatchNorm(hidden_size1)
        self.gcn_4 = gnn.GraphConv(hidden_size1, hidden_size2)
        self.bn_4 = gnn.BatchNorm(hidden_size2)

    def forward(self, x,edge_index):
        # x_normalization = graph.x
        # h = F.relu(self.gcn_1(x_normalization, graph.edge_index))
        # h = F.relu(self.gcn_2(h, graph.edge_index))
        x_normalization = self.bn_0(x)

        h = self.bn_1(F.relu(self.gcn_1(x_normalization, edge_index)))
        h = self.bn_2(F.relu(self.gcn_2(h, edge_index)))
        h = self.bn_3(F.relu(self.gcn_3(h, edge_index)))
        h = self.bn_4(F.relu(self.gcn_4(h, edge_index)))

        return h

# External graph convolution feature module
class GraphNetFeature(nn.Module):
    def __init__(self, c_in, hidden_size):
        super().__init__()
        self.bn_0 = gnn.BatchNorm(c_in)
        self.gcn_1 = gnn.GCNConv(c_in, hidden_size)
        self.bn_1 = gnn.BatchNorm(hidden_size)
        self.gcn_2 = gnn.GCNConv(hidden_size, hidden_size)
        self.bn_2 = gnn.BatchNorm(hidden_size)

    def forward(self, graph):
        x_normalization = self.bn_0(graph.x)
        # x_normalization = graph.x
        h = self.bn_1(F.relu(self.gcn_1(x_normalization, graph.edge_index)))
        h = self.bn_2(F.relu(self.gcn_2(h, graph.edge_index)))
        return x_normalization + h



class MLPNet(nn.Module):    # 用于最后的交叉熵损失
    def __init__(self,c_in, hidden_size, nc):
        super().__init__()
        self.bn_0 = gnn.BatchNorm(c_in)
        self.classifier1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(hidden_size // 2, nc)
        )

    def forward(self, graph,h):
        x_normalization = self.bn_0(graph.x)
        logits = self.classifier1(h + x_normalization)
        return logits


class MLPNet_1(nn.Module):      # 用于中间的损失   不涉及反向传播
    def __init__(self,c_in, hidden_size, nc):
        super().__init__()
        self.bn_0 = gnn.BatchNorm(c_in)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(hidden_size // 2, nc)
        )

    def forward(self, h):
        x_normalization = self.bn_0(h)
        logits = self.classifier(x_normalization)
        return logits


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1D 卷积层

    def forward(self, x):
        # 将输入张量形状调整为 (batch_size, in_channels, features)
        x = x.unsqueeze(0)  # 假设 batch size 为 1，可以添加一个维度  添加第一个维度
        x = x.permute(0, 2, 1)  # 调整维度顺序，将特征维度放在最后
        x = x.unsqueeze(3)     # 添加最后一个维度 (1,C,H,W)
        return self.conv(x).squeeze(0).squeeze(2).permute(1, 0)  # 还原形状并返回结果
