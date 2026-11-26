"""Review-only learned rank16 P2 bridge; training efficacy is untested."""
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F

class LowRankP2Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce=nn.Sequential(OrderedDict([
            ('conv',nn.Conv2d(64,16,1,bias=False)),
            ('bn',nn.BatchNorm2d(16)),('act',nn.ReLU(inplace=False))]))
        self.mix=nn.Conv2d(64,16,1,bias=False)
        self.restore=nn.Conv2d(16,256,1,bias=False)
        nn.init.orthogonal_(self.mix.weight.view(16,64))
        nn.init.zeros_(self.restore.weight)

    def forward(self,p2,p3):
        detail=self.reduce(p2);h,w=detail.shape[-2:]
        if h%2 or w%2:detail=F.pad(detail,(0,w%2,0,h%2))
        detail=F.pixel_unshuffle(detail,2)
        if detail.shape[-2:]!=p3.shape[-2:]:raise ValueError('P2/P3 spatial mismatch')
        return p3+self.restore(self.mix(detail))

    def effective_weight(self):
        return self.restore.weight.view(256,16)@self.mix.weight.view(16,64)
