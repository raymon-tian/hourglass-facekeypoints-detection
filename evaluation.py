#coding=utf-8

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt

from data_loader import KFDataset
from models import KFSGNet
from train import config,get_peak_points,get_mse


def demo(img,heatmaps):
    """

    :param img: (96,96)
    :param heatmaps: ()
    :return:
    """
    # img = img.reshape(96, 96)
    # axis.imshow(img, cmap='gray')
    # axis.scatter(y[:, 0], y[:, 1], marker='x', s=10)
    pass

def evaluate():
    # 加载模型
    net = KFSGNet()
    net.float().cuda()
    net.eval()
    if (config['checkout'] != ''):
        net.load_state_dict(torch.load(config['checkout']))

    dataset = KFDataset(config)
    dataset.load()
    dataLoader = DataLoader(dataset,1)
    for i,(images,_,gts) in enumerate(dataLoader):
        images = Variable(images).float().cuda()

        pred_heatmaps = net.forward(images)
        demo_img = images[0].cpu().data.numpy()[0]
        demo_img = (demo_img * 255.).astype(np.uint8)
        demo_heatmaps = pred_heatmaps[0].cpu().data.numpy()[np.newaxis,...]
        demo_pred_poins = get_peak_points(demo_heatmaps)[0] # (15,2)
        plt.imshow(demo_img,cmap='gray')
        plt.scatter(demo_pred_poins[:,0],demo_pred_poins[:,1])
        plt.show()

        # loss = get_mse(demo_pred_poins[np.newaxis,...],gts)

if __name__ == '__main__':
    evaluate()