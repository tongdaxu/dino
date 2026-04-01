from dictdot import dictdot
import torch
import torch.nn as nn
import torch.nn.functional as F
import INN
import INN.INNAbstract as INNAbstract
from INN.CouplingModels.conv import CouplingConv
from INN.CouplingModels.NICEModel.conv import Conv2dNICE


class PixelUnshuffle2d(INNAbstract.PixelShuffleModule):
    def __init__(self, r):
        super(PixelUnshuffle2d, self).__init__()
        self.r = r
        self.shuffle = nn.PixelShuffle(r)
        self.unshuffle = nn.PixelUnshuffle(r)
    
    def PixelShuffle(self, x):
        return self.unshuffle(x)
    
    def PixelUnshuffle(self, x):
        return self.shuffle(x)

class PIXELVAE(nn.Module):
    def __init__(self, use_flow=True, *args, **kwargs):
        super().__init__()

        self.flow = INN.Sequential(
            Conv2dNICE(3, 3), # 16
            Conv2dNICE(3, 3), # 16
            INN.PixelShuffle2d(2), # 8
            Conv2dNICE(12, 3),
            Conv2dNICE(12, 3),
            INN.PixelShuffle2d(2), # 4
            Conv2dNICE(48, 3),
            Conv2dNICE(48, 3),
            INN.PixelShuffle2d(2), # 2
            Conv2dNICE(192, 3),
            Conv2dNICE(192, 3),
            INN.PixelShuffle2d(2), # 1
            Conv2dNICE(768, 3, w=1),
            Conv2dNICE(768, 3, w=1),
            INN.PixelShuffle2d(2), # 1
        )
        self.flow.computing_p(True)
        self.flow_lam = 1

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3072, 1000)

    def encode(self, x, *args, **kwargs):            
        z = self.flow(x)[0]
        # z = F.pixel_unshuffle(x, 2)
        return z

    def decode(self, z, *args, **kwargs):
        # x = F.pixel_shuffle(z, 2)
        z = self.flow.inverse(z)
        return dictdot(dict(sample=x))

    def forward(self, x):
        z = self.encode(x)
        z = self.avgpool(z)
        z = torch.flatten(z, 1)
        z = self.fc(z)
        return z