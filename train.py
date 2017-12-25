#coding=utf-8
import torch
from torch.autograd import Variable
from torch.backends import cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pprint

from data_loader import KFDataset
from models import KFSGNet

config = dict()
config['lr'] = 0.000001
config['momentum'] = 0.9
config['weight_decay'] = 1e-4
config['epoch_num'] = 400
config['batch_size'] = 72
config['sigma'] = 5.
config['debug_vis'] = False         # 是否可视化heatmaps
config['fname'] = 'data/test.csv'
# config['fname'] = 'data/training.csv'
# config['is_test'] = False
config['is_test'] = True
config['save_freq'] = 10
config['checkout'] = 'data/weight/kd_epoch_909_model.ckpt'
config['start_epoch'] = 850
config['eval_freq'] = 5
config['debug'] = False
config['lookup'] = 'data/IdLookupTable.csv'
config['featurename2id'] = {
    'left_eye_center_x':0,
    'left_eye_center_y':1,
    'right_eye_center_x':2,
    'right_eye_center_y':3,
    'left_eye_inner_corner_x':4,
    'left_eye_inner_corner_y':5,
    'left_eye_outer_corner_x':6,
    'left_eye_outer_corner_y':7,
    'right_eye_inner_corner_x':8,
    'right_eye_inner_corner_y':9,
    'right_eye_outer_corner_x':10,
    'right_eye_outer_corner_y':11,
    'left_eyebrow_inner_end_x':12,
    'left_eyebrow_inner_end_y':13,
    'left_eyebrow_outer_end_x':14,
    'left_eyebrow_outer_end_y':15,
    'right_eyebrow_inner_end_x':16,
    'right_eyebrow_inner_end_y':17,
    'right_eyebrow_outer_end_x':18,
    'right_eyebrow_outer_end_y':19,
    'nose_tip_x':20,
    'nose_tip_y':21,
    'mouth_left_corner_x':22,
    'mouth_left_corner_y':23,
    'mouth_right_corner_x':24,
    'mouth_right_corner_y':25,
    'mouth_center_top_lip_x':26,
    'mouth_center_top_lip_y':27,
    'mouth_center_bottom_lip_x':28,
    'mouth_center_bottom_lip_y':29,
}

def get_peak_points(heatmaps):
    """

    :param heatmaps: numpy array (N,15,96,96)
    :return:numpy array (N,15,2)
    """
    N,C,H,W = heatmaps.shape
    all_peak_points = []
    for i in range(N):
        peak_points = []
        for j in range(C):
            yy,xx = np.where(heatmaps[i,j] == heatmaps[i,j].max())
            y = yy[0]
            x = xx[0]
            peak_points.append([x,y])
        all_peak_points.append(peak_points)
    all_peak_points = np.array(all_peak_points)
    return all_peak_points

def get_mse(pred_points,gts,indices_valid=None):
    """

    :param pred_points: numpy (N,15,2)
    :param gts: numpy (N,15,2)
    :return:
    """
    pred_points = pred_points[indices_valid[0],indices_valid[1],:]
    gts = gts[indices_valid[0],indices_valid[1],:]
    pred_points = Variable(torch.from_numpy(pred_points).float(),requires_grad=False)
    gts = Variable(torch.from_numpy(gts).float(),requires_grad=False)
    criterion = nn.MSELoss()
    loss = criterion(pred_points,gts)
    return loss

def calculate_mask(heatmaps_target):
    """

    :param heatmaps_target: Variable (N,15,96,96)
    :return: Variable (N,15,96,96)
    """
    N,C,_,_ = heatmaps_targets.size()
    N_idx = []
    C_idx = []
    for n in range(N):
        for c in range(C):
            max_v = heatmaps_targets[n,c,:,:].max().data[0]
            if max_v != 0.0:
                N_idx.append(n)
                C_idx.append(c)
    mask = Variable(torch.zeros(heatmaps_targets.size()))
    mask[N_idx,C_idx,:,:] = 1.
    mask = mask.float().cuda()
    return mask,[N_idx,C_idx]

if __name__ == '__main__':
    pprint.pprint(config)
    torch.manual_seed(0)
    cudnn.benchmark = True
    net = KFSGNet()
    net.float().cuda()
    net.train()
    criterion = nn.MSELoss()
    # optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=config['momentum'] , weight_decay=config['weight_decay'])
    optimizer = optim.Adam(net.parameters(),lr=config['lr'])
    trainDataset = KFDataset(config)
    trainDataset.load()
    trainDataLoader = DataLoader(trainDataset,config['batch_size'],True)
    sample_num = len(trainDataset)

    if (config['checkout'] != ''):
        net.load_state_dict(torch.load(config['checkout']))

    for epoch in range(config['start_epoch'],config['epoch_num']+config['start_epoch']):
        running_loss = 0.0
        for i, (inputs, heatmaps_targets, gts) in enumerate(trainDataLoader):
            inputs = Variable(inputs).cuda()
            heatmaps_targets = Variable(heatmaps_targets).cuda()
            mask,indices_valid = calculate_mask(heatmaps_targets)

            optimizer.zero_grad()
            outputs = net(inputs)
            outputs = outputs * mask
            heatmaps_targets = heatmaps_targets * mask
            loss = criterion(outputs, heatmaps_targets)
            loss.backward()
            optimizer.step()

            # 统计最大值与最小值
            v_max = torch.max(outputs)
            v_min = torch.min(outputs)

            # 评估
            all_peak_points = get_peak_points(heatmaps_targets.cpu().data.numpy())
            loss_coor = get_mse(all_peak_points, gts.numpy(),indices_valid)

            print('[ Epoch {:005d} -> {:005d} / {} ] loss : {:15} loss_coor : {:15} max : {:10} min : {}'.format(
                epoch, i * config['batch_size'],
                sample_num, loss.data[0],loss_coor.data[0],v_max.data[0],v_min.data[0]))



        if (epoch+1) % config['save_freq'] == 0 or epoch == config['epoch_num'] - 1:
            torch.save(net.state_dict(),'kd_epoch_{}_model.ckpt'.format(epoch))

