#coding=utf-8

import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import copy

import matplotlib.pyplot as plt

# from train import config


def plot_sample(x, y, axis):
    """

    :param x: (9216,)
    :param y: (15,2)
    :param axis:
    :return:
    """
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[:,0], y[:,1], marker='x', s=10)

def plot_demo(X,y):
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X[i], y[i], ax)

    plt.show()


class KFDataset(Dataset):
    def __init__(self,config,X=None,gts=None):
        """

        :param X: (N,96*96)
        :param gts: (N,15,2)
        """
        self.__X = X
        self.__gts = gts
        self.__sigma = config['sigma']
        self.__debug_vis = config['debug_vis']
        self.__fname = config['fname']
        self.__is_test = config['is_test']
        # self.__ftrain = config['ftrain']
        # self.load(self.__ftrain)

    def load(self,cols=None):
        """

        :param fname:
        :param test:
        :param cols:
        :return: X (N,96*96) Y (N,15,2)
        """
        test = self.__is_test
        fname = self.__fname
        df = pd.read_csv(fname)

        # The Image column has pixel values separated by space; convert
        # the values to numpy arrays:
        df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

        if cols:  # get a subset of columns
            df = df[list(cols) + ['Image']]

        # print(df.count())  # prints the number of values for each column
        if test == False:
            # df = df.dropna()  # drop all rows that have missing values in them
            pass
        # 选出存在缺失值的iamge
        # df = df[df.isnull().any(axis=1)]
        # print(df.count())  # prints the number of values for each column
        df_np = df.as_matrix()

        X = df_np[:,-1]
        x_list = X.tolist()
        x_list = [item.tolist() for item in x_list]
        X = np.array(x_list)
        # X = X.astype(np.float32)
        # X = X / 255.  # scale pixel values to [0, 1]

        if not test:  # only FTRAIN has any target columns
            gts = df_np[:, :-1]
            gts = gts.reshape((gts.shape[0], -1, 2))
            gts = gts.astype(np.float32)
        else:
            gts = df['ImageId'].as_matrix()

        self.__X = X
        self.__gts = gts

        return X, gts

    def __len__(self):
        return len(self.__X)

    def __getitem__(self, item):
        H,W = 96,96
        x = self.__X[item]
        gt = self.__gts[item]

        if self.__is_test:
            x = x.reshape((1, 96, 96)).astype(np.float32)
            x = x / 255.
            return x,gt #返回图像以及其id


        heatmaps = self._putGaussianMaps(gt,H,W,1,self.__sigma)

        if self.__debug_vis == True:
            for i in range(heatmaps.shape[0]):
                img = copy.deepcopy(x).astype(np.uint8).reshape((H,W))
                self.visualize_heatmap_target(img,copy.deepcopy(heatmaps[i]),1)

        x = x.reshape((1,96,96)).astype(np.float32)
        x = x / 255.
        heatmaps = heatmaps.astype(np.float32)
        return x,heatmaps,gt

    def _putGaussianMap(self, center, visible_flag, crop_size_y, crop_size_x, stride, sigma):
        """
        根据一个中心点,生成一个heatmap
        :param center:
        :return:
        """
        grid_y = crop_size_y / stride
        grid_x = crop_size_x / stride
        if visible_flag == False:
            return np.zeros((grid_y,grid_x))
        start = stride / 2.0 - 0.5
        y_range = [i for i in range(grid_y)]
        x_range = [i for i in range(grid_x)]
        xx, yy = np.meshgrid(x_range, y_range)
        xx = xx * stride + start
        yy = yy * stride + start
        d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        exponent = d2 / 2.0 / sigma / sigma
        heatmap = np.exp(-exponent)
        return heatmap

    def _putGaussianMaps(self,keypoints,crop_size_y, crop_size_x, stride, sigma):
        """

        :param keypoints: (15,2)
        :param crop_size_y: int
        :param crop_size_x: int
        :param stride: int
        :param sigma: float
        :return:
        """
        all_keypoints = keypoints
        point_num = all_keypoints.shape[0]
        heatmaps_this_img = []
        for k in range(point_num):
            flag = ~np.isnan(all_keypoints[k,0])
            heatmap = self._putGaussianMap(all_keypoints[k],flag,crop_size_y,crop_size_x,stride,sigma)
            heatmap = heatmap[np.newaxis,...]
            heatmaps_this_img.append(heatmap)
        heatmaps_this_img = np.concatenate(heatmaps_this_img,axis=0) # (num_joint,crop_size_y/stride,crop_size_x/stride)
        return heatmaps_this_img

    def visualize_heatmap_target(self,oriImg,heatmap,stride):

        plt.imshow(oriImg)
        plt.imshow(heatmap, alpha=.5)
        plt.show()

if __name__ == '__main__':
    from train import config
    dataset = KFDataset(config)
    dataset.load()
    dataLoader = DataLoader(dataset=dataset,batch_size=64,shuffle=False)
    for i, (x, y ,gt) in enumerate(dataLoader):
        print(x.size())
        print(y.size())
        print(gt.size())
