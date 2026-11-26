import torch
from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
from engine.backbone.hgnetv2 import HGNetv2

def conv_bn_relu(cin, cout, kernel=1, stride=1, groups=1, dilation=1):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(cin, cout, kernel, stride,
                           padding=dilation * (kernel // 2), dilation=dilation,
                           groups=groups, bias=False)),
        ('bn', nn.BatchNorm2d(cout)),
        ('act', nn.ReLU(inplace=False)),
    ]))

class P2ToP3Residual(nn.Module):
    """P2 1x1 compression -> 2x spatial reorder -> zero-initialized P3 residual."""
    def __init__(self, p2_channels=64, p3_channels=256, width=16):
        super().__init__()
        self.reduce = conv_bn_relu(p2_channels, width)
        self.restore = nn.Conv2d(width * 4, p3_channels, 1, bias=False)
        nn.init.zeros_(self.restore.weight)

    def forward(self, p2, p3):
        detail = self.reduce(p2)
        h, w = detail.shape[-2:]
        if h % 2 or w % 2:
            detail = F.pad(detail, (0, w % 2, 0, h % 2))
        detail = F.pixel_unshuffle(detail, 2)
        if detail.shape[-2:] != p3.shape[-2:]:
            raise ValueError('P2 spatial reorder and P3 sizes differ')
        return p3 + self.restore(detail)

class P2SkipBackbone(HGNetv2):
    def forward(self, x):
        x=self.stem(x);outs=[];p2=None
        for idx,stage in enumerate(self.stages):
            x=stage(x)
            if idx==0:p2=x
            if idx==1:x=self.p2_skip(p2,x)
            if idx in self.return_idx:outs.append(x)
        return outs
