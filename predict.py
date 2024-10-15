'''Predicting'''
from scipy.io import loadmat, savemat
import numpy as np
import argparse
import configparser
import torch
from torch import nn
from torch_geometric.data import Data, Batch
from skimage.segmentation import slic, mark_boundaries
from sklearn.preprocessing import scale
import os
from PIL import Image
from utils import get_graph_list, get_edge_index
import math
from Model.module import SubGcnFeature,GraphNet1,GraphNet2,GraphNet3,GraphNet4,MLPNet,Conv1x1
from Trainer import JointTrainer
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRAIN THE OVERALL')
    parser.add_argument('--name', type=str, default='PaviaU',
                        help='DATASET NAME')
    parser.add_argument('--block', type=int, default=100,
                        help='BLOCK SIZE')
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



    # Constructing full graphs
    full_edge_index_path = 'data/{}/{}_edge_index.npy'.format(arg.name, arg.block)
    if os.path.exists(full_edge_index_path):
        edge_index = np.load(full_edge_index_path)
    else:
        edge_index, _ = get_edge_index(seg)
        np.save(full_edge_index_path,
                edge_index if isinstance(edge_index, np.ndarray) else edge_index.cpu().numpy())
    fullGraph = Data(None,
                    edge_index=torch.from_numpy(edge_index) if isinstance(edge_index, np.ndarray) else edge_index,
                    seg=torch.from_numpy(seg) if isinstance(seg, np.ndarray) else seg)


    student_net = GraphNet3(c, 256, c)
    teacher_net = GraphNet3(c, 256, c)


    mlp = MLPNet(c, c, config.getint(arg.name, 'nc'))  # 用于最后的交叉熵损失



    device = torch.device('cuda:{}'.format(arg.gpu)) if arg.gpu != -1 else torch.device('cpu')

    alltime = 0

    for r in range(arg.run):

        start = time.time()

        student_net.load_state_dict(
                torch.load(f"models/{arg.name}/{arg.spc}/{arg.block}_overall_skip_2_SGConv_l1_clip/studentNet_best_{arg.spc}_{r}.pkl"))
        teacher_net.load_state_dict(
            torch.load(f"models/{arg.name}/{arg.spc}/{arg.block}_overall_skip_2_SGConv_l1_clip/teacherNet_best_{arg.spc}_{r}.pkl"))

        mlp.load_state_dict(
            torch.load(f"models/{arg.name}/{arg.spc}/{arg.block}_overall_skip_2_SGConv_l1_clip/mlpNet_best_{arg.spc}_{r}.pkl"))

        trainer = JointTrainer([student_net,teacher_net,mlp])
        # predicting
        preds = trainer.predict(superpixel_features_tensor, fullGraph, device)
        seg_torch = torch.from_numpy(seg)
        map = preds[seg_torch]

        temp = time.time() - start
        alltime = alltime + temp

        save_root = 'prediction/{}/{}/{}_overall_skip_2_SGConv_l1_clip'.format(arg.name, arg.spc, arg.block)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        save_path = os.path.join(save_root, '{}.mat'.format(r))
        savemat(save_path, {'pred': map.cpu().numpy()})
    print("total time:",alltime)
    print('*'*5 + 'FINISH' + '*'*5)

