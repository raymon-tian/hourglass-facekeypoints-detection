#coding=utf-8

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import KFDataset
from models import KFSGNet
from train import config,get_peak_points


def test():
    # 加载模型
    net = KFSGNet()
    net.float().cuda()
    net.eval()
    if (config['checkout'] != ''):
        net.load_state_dict(torch.load(config['checkout']))

    dataset = KFDataset(config)
    dataset.load()
    dataLoader = DataLoader(dataset,32)
    all_result = []
    lookup_df = pd.read_csv(config['lookup'])
    num = len(dataset)
    for i,(images,ids) in enumerate(dataLoader):
        print('{} / {}'.format(i,num))
        images = Variable(images).float().cuda()
        ids = ids.numpy()
        pred_heatmaps = net.forward(images)

        """
        可视化预测结果
        demo_img = images[0].cpu().data.numpy()[0]
        demo_img = (demo_img * 255.).astype(np.uint8)
        demo_heatmaps = pred_heatmaps[0].cpu().data.numpy()[np.newaxis,...]
        demo_pred_poins = get_peak_points(demo_heatmaps)[0] # (15,2)
        plt.imshow(demo_img,cmap='gray')
        plt.scatter(demo_pred_poins[:,0],demo_pred_poins[:,1])
        plt.show()
        """

        pred_points = get_peak_points(pred_heatmaps.cpu().data.numpy()) #(N,15,2)
        pred_points = pred_points.reshape((pred_points.shape[0],-1)) #(N,30)

        # 筛选出要查询的features
        for idx,img_id in enumerate(ids):
            result_img = lookup_df[lookup_df['ImageId'] == img_id]
            # 映射feature names to ids
            fea_names = result_img['FeatureName'].as_matrix()
            fea_ids = [config['featurename2id'][name] for name in fea_names]
            pred_values = pred_points[idx][fea_ids]
            result_img['Location'] = pred_values
            all_result.append(result_img)


        # loss = get_mse(demo_pred_poins[np.newaxis,...],gts)
    result_df = pd.concat(all_result)
    result_df = result_df.drop(columns=['ImageId','FeatureName'])
    result_df.to_csv('data/result_909.csv',index=False)

if __name__ == '__main__':
    test()