#coding=utf-8
""" pytorch实现：stacked hourglass network architecture"""

import torch
import torch.nn as nn
from torch.nn import UpsamplingNearest2d,Upsample
from torch.autograd import Variable

class StackedHourGlass(nn.Module):
    def __init__(self,nFeats=256,nStack=8,nJoints=18):
        """
        输入： 256^2
        """
        super(StackedHourGlass,self).__init__()
        self._nFeats = nFeats
        self._nStack = nStack
        self._nJoints = nJoints
        self.conv1 = nn.Conv2d(3,64,7,2,3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.res1 = Residual(64,128)
        self.pool1 = nn.MaxPool2d(2,2)
        self.res2 = Residual(128,128)
        self.res3 = Residual(128,self._nFeats)           #cmd:option('-nFeats',            256, 'Number of features in the hourglass')
        self._init_stacked_hourglass()

    def _init_stacked_hourglass(self):
        for i in range(self._nStack):
            setattr(self,'hg'+str(i),HourGlass(4,self._nFeats))
            setattr(self,'hg'+str(i)+'_res1',Residual(self._nFeats,self._nFeats))
            setattr(self,'hg'+str(i)+'_lin1',Lin(self._nFeats,self._nFeats))
            setattr(self,'hg'+str(i)+'_conv_pred',nn.Conv2d(self._nFeats,self._nJoints,1))
            if i < self._nStack - 1:
                setattr(self,'hg'+str(i)+'_conv1',nn.Conv2d(self._nFeats,self._nFeats,1))
                setattr(self,'hg'+str(i)+'_conv2',nn.Conv2d(self._nJoints,self._nFeats,1))

    def forward(self,x):
        # 初始图像处理
        x = self.relu1(self.bn1(self.conv1(x))) #(n,64,128,128)
        x = self.res1(x)                        #(n,128,128,128)
        x = self.pool1(x)                       #(n,128,64,64)
        x = self.res2(x)                        #(n,128,64,64)
        x = self.res3(x)                        #(n,256,64,64)

        out = []
        inter = x

        for i in range(self._nStack):                      #cmd:option('-nStack',              8, 'Number of hourglasses to stack')
            hg = eval('self.hg'+str(i))(inter)
            # Residual layers at output resolution
            ll = hg
            ll = eval('self.hg'+str(i)+'_res1')(ll)
            # Linear layer to produce first set of predictions
            ll = eval('self.hg'+str(i)+'_lin1')(ll)
            # Predicted heatmaps
            tmpOut = eval('self.hg'+str(i)+'_conv_pred')(ll)
            out.append(tmpOut)
            # Add predictions back
            if i < self._nStack - 1:
                ll_ = eval('self.hg'+str(i)+'_conv1')(ll)
                tmpOut_ = eval('self.hg'+str(i)+'_conv2')(tmpOut)
                inter = inter + ll_ + tmpOut_
        return out

class HourGlass(nn.Module):
    """不改变特征图的高宽"""
    def __init__(self,n=4,f=256):
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
        # setattr(self,'SUSN'+str(n),UpsamplingNearest2d(scale_factor=2))
        setattr(self,'SUSN'+str(n),Upsample(scale_factor=2))


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
        up2 = eval('self.'+'SUSN'+str(n)).forward(low3)

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
    def __init__(self,numIn,numout):
        super(Lin,self).__init__()
        self.conv = nn.Conv2d(numIn,numout,1)
        self.bn = nn.BatchNorm2d(numout)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))


from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch.optim as optim

class tempDataset(Dataset):
    def __init__(self):
        self.X = np.random.randn(100,3,512,512)
        self.Y = np.random.randn(100,18,128,128)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, item):
        # 这里返回的时候不要设置batch_size
        return self.X[item],self.Y[item]

if __name__ == '__main__':
    from torch.nn import MSELoss
    critical = MSELoss()

    dataset = tempDataset()
    dataLoader = DataLoader(dataset=dataset)
    shg = StackedHourGlass()
    optimizer = optim.SGD(shg.parameters(), lr=0.01, momentum=0.9,weight_decay=1e-4)
    for i,(x,y) in enumerate(dataLoader):
        x = Variable(x,requires_grad=True).float()
        y = Variable(y).float()
        y_pred = shg.forward(x)
        loss = critical(y_pred[0],y[0])
        print('loss : {}'.format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()