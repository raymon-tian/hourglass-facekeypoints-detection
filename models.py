#coding=utf-8
import torch
import torch.nn as nn
from torch.nn import Upsample
from torch.autograd import Variable

class HourGlass(nn.Module):
    """不改变特征图的高宽"""
    def __init__(self,n=4,f=128):
        """
        :param n: hourglass模块的层级数目
        :param f: hourglass模块中的特征图数量
        :return:
        """
        super(HourGlass,self).__init__()
        self._n = n
        self._f = f
        self._init_layers(self._n,self._f)

    def _init_layers(self,n,f):
        # 上分支
        setattr(self,'res'+str(n)+'_1',Residual(f,f))
        # 下分支
        setattr(self,'pool'+str(n)+'_1',nn.MaxPool2d(2,2))
        setattr(self,'res'+str(n)+'_2',Residual(f,f))
        if n > 1:
            self._init_layers(n-1,f)
        else:
            self.res_center = Residual(f,f)
        setattr(self,'res'+str(n)+'_3',Residual(f,f))
        setattr(self,'unsample'+str(n),Upsample(scale_factor=2))


    def _forward(self,x,n,f):
        # 上分支
        up1 = x
        up1 = eval('self.res'+str(n)+'_1')(up1)
        # 下分支
        low1 = eval('self.pool'+str(n)+'_1')(x)
        low1 = eval('self.res'+str(n)+'_2')(low1)
        if n > 1:
            low2 = self._forward(low1,n-1,f)
        else:
            low2 = self.res_center(low1)
        low3 = low2
        low3 = eval('self.'+'res'+str(n)+'_3')(low3)
        up2 = eval('self.'+'unsample'+str(n)).forward(low3)

        return up1+up2

    def forward(self,x):
        return self._forward(x,self._n,self._f)

class Residual(nn.Module):
    """
    残差模块，并不改变特征图的宽高
    """
    def __init__(self,ins,outs):
        super(Residual,self).__init__()
        # 卷积模块
        self.convBlock = nn.Sequential(
            nn.BatchNorm2d(ins),
            nn.ReLU(inplace=True),
            nn.Conv2d(ins,outs/2,1),
            nn.BatchNorm2d(outs/2),
            nn.ReLU(inplace=True),
            nn.Conv2d(outs/2,outs/2,3,1,1),
            nn.BatchNorm2d(outs/2),
            nn.ReLU(inplace=True),
            nn.Conv2d(outs/2,outs,1)
        )
        # 跳层
        if ins != outs:
            self.skipConv = nn.Conv2d(ins,outs,1)
        self.ins = ins
        self.outs = outs
    def forward(self,x):
        residual = x
        x = self.convBlock(x)
        if self.ins != self.outs:
            residual = self.skipConv(residual)
        x += residual
        return x

class Lin(nn.Module):
    def __init__(self,numIn=128,numout=15):
        super(Lin,self).__init__()
        self.conv = nn.Conv2d(numIn,numout,1)
        self.bn = nn.BatchNorm2d(numout)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))


class KFSGNet(nn.Module):

    def __init__(self):
        super(KFSGNet,self).__init__()
        self.__conv1 = nn.Conv2d(1,64,1)
        self.__relu1 = nn.ReLU(inplace=True)
        self.__conv2 = nn.Conv2d(64,128,1)
        self.__relu2 = nn.ReLU(inplace=True)
        self.__hg = HourGlass()
        self.__lin = Lin()
    def forward(self,x):
        x = self.__relu1(self.__conv1(x))
        x = self.__relu2(self.__conv2(x))
        x = self.__hg(x)
        x = self.__lin(x)
        return x


from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch.optim as optim

class tempDataset(Dataset):
    def __init__(self):
        self.X = np.random.randn(100,1,96,96)
        self.Y = np.random.randn(100,30,96,96)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, item):
        # 这里返回的时候不要设置batch_size
        return self.X[item],self.Y[item]

if __name__ == '__main__':
    from torch.nn import MSELoss
    critical = MSELoss()

    dataset = tempDataset()
    dataLoader = DataLoader(dataset=dataset,batch_size=64)
    shg = KFSGNet().cuda()
    optimizer = optim.SGD(shg.parameters(), lr=0.001, momentum=0.9,weight_decay=1e-4)

    for e in range(200):
        for i,(x,y) in enumerate(dataLoader):
            x = Variable(x,requires_grad=True).float().cuda()
            y = Variable(y).float().cuda()
            y_pred = shg.forward(x)
            loss = critical(y_pred[0],y[0])
            print('loss : {}'.format(loss.data))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()