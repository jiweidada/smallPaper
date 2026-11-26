"""Standard R50/FPN and RoIAlignV2 components using Torch/TorchVision only.

Parameter names follow the selected Detectron2/DiffusionDet checkpoint. No claim
of a new backbone or pooling operator. Native stride-in-1x1=False is preserved.
"""
import math
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from torchvision.ops.misc import FrozenBatchNorm2d

class ConvNorm(nn.Conv2d):
    def __init__(self,cin,cout,k,stride=1,padding=0):
        super().__init__(cin,cout,k,stride=stride,padding=padding,bias=False)
        self.norm=FrozenBatchNorm2d(cout,eps=1e-5)
    def forward(self,x):return self.norm(super().forward(x))

class Stem(nn.Module):
    def __init__(self):
        super().__init__();self.conv1=ConvNorm(3,64,7,2,3)
    def forward(self,x):return F.max_pool2d(F.relu_(self.conv1(x)),3,2,1)

class Bottleneck(nn.Module):
    def __init__(self,cin,mid,stride):
        super().__init__();cout=mid*4
        self.conv1=ConvNorm(cin,mid,1)
        self.conv2=ConvNorm(mid,mid,3,stride,1)
        self.conv3=ConvNorm(mid,cout,1)
        self.shortcut=ConvNorm(cin,cout,1,stride) if cin!=cout or stride!=1 else None
    def forward(self,x):
        y=self.conv3(F.relu_(self.conv2(F.relu_(self.conv1(x)))))
        return F.relu_(y+(x if self.shortcut is None else self.shortcut(x)))

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__();self.stem=Stem();cin=64
        for stage,(mid,count) in enumerate([(64,3),(128,4),(256,6),(512,3)],2):
            blocks=[]
            for j in range(count):
                blocks.append(Bottleneck(cin,mid,2 if stage>2 and j==0 else 1));cin=4*mid
            self.add_module('res'+str(stage),nn.Sequential(*blocks))
    def forward(self,x):
        x=self.stem(x);out=OrderedDict()
        for stage in range(2,6):
            x=getattr(self,'res'+str(stage))(x);out['res'+str(stage)]=x
        return out

class FPN(nn.Module):
    def __init__(self):
        super().__init__();self.bottom_up=ResNet50()
        for level,cin in zip(range(2,6),[256,512,1024,2048]):
            self.add_module('fpn_lateral'+str(level),nn.Conv2d(cin,256,1))
            self.add_module('fpn_output'+str(level),nn.Conv2d(256,256,3,padding=1))
    def forward(self,x):
        bottom=self.bottom_up(x);last=None;out={}
        for level in range(5,1,-1):
            lateral=getattr(self,'fpn_lateral'+str(level))(bottom['res'+str(level)])
            last=lateral if last is None else lateral+F.interpolate(last,scale_factor=2.,mode='nearest')
            out['p'+str(level)]=getattr(self,'fpn_output'+str(level))(last)
        return OrderedDict((k,out[k]) for k in ['p2','p3','p4','p5'])

class Boxes:
    def __init__(self,tensor):
        assert tensor.ndim==2 and tensor.shape[-1]==4
        self.tensor=tensor
    def __len__(self):return self.tensor.shape[0]
    def area(self):
        wh=self.tensor[:,2:]-self.tensor[:,:2]
        return wh[:,0]*wh[:,1]

class ROIPooler(nn.Module):
    def __init__(self,output_size,scales,sampling_ratio,pooler_type,
                 canonical_box_size=224,canonical_level=4):
        super().__init__()
        assert pooler_type=='ROIAlignV2'
        self.output_size=(output_size,output_size) if isinstance(output_size,int) else tuple(output_size)
        self.scales=tuple(scales);self.sampling_ratio=sampling_ratio
        self.min_level=int(-math.log2(self.scales[0]));self.max_level=int(-math.log2(self.scales[-1]))
        assert self.max_level-self.min_level+1==len(self.scales)
        self.canonical_box_size=canonical_box_size;self.canonical_level=canonical_level
    def forward(self,features,box_lists):
        assert len(features)==len(self.scales) and len(box_lists)==features[0].shape[0]
        boxes=torch.cat([b.tensor for b in box_lists],dim=0)
        batch=torch.cat([torch.full((len(b),1),i,dtype=boxes.dtype,device=boxes.device) for i,b in enumerate(box_lists)])
        rois=torch.cat((batch,boxes),dim=1)
        size=torch.sqrt(torch.cat([b.area() for b in box_lists]))
        levels=torch.floor(self.canonical_level+torch.log2(size/self.canonical_box_size+1e-8))
        levels=levels.clamp(self.min_level,self.max_level).long()-self.min_level
        output=features[0].new_zeros((len(boxes),features[0].shape[1],*self.output_size))
        for index,(feature,scale) in enumerate(zip(features,self.scales)):
            selected=torch.where(levels==index)[0]
            if len(selected):
                value=roi_align(feature,rois[selected],self.output_size,spatial_scale=scale,
                                sampling_ratio=self.sampling_ratio,aligned=True)
                output.index_copy_(0,selected,value)
        return output
