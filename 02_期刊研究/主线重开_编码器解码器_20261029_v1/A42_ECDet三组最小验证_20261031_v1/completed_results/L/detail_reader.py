from __future__ import annotations
import torch
from torch import nn
from torchvision.ops import roi_align

class StemDetailReader(nn.Module):
    def __init__(self,channels=48,hidden_dim=192):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(channels*9,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self,feature,normalized_boxes):
        b,q,_=normalized_boxes.shape
        h,w=feature.shape[-2:]
        boxes=normalized_boxes.detach().float()
        xy0=boxes[...,:2]-boxes[...,2:]/2
        xy1=boxes[...,:2]+boxes[...,2:]/2
        xyxy=torch.cat([xy0,xy1],dim=-1).clamp(0,1)
        xyxy=xyxy*xyxy.new_tensor([w,h,w,h])
        bi=torch.arange(b,device=feature.device).view(b,1,1).expand(b,q,1)
        rois=torch.cat([bi,xyxy],dim=-1).reshape(-1,5)
        # ROIAlign has no BF16 kernel; only this spatial read is FP32.
        with torch.autocast(device_type=feature.device.type,enabled=False):
            pooled=roi_align(feature.float(),rois.float(),output_size=(3,3),spatial_scale=1.0,sampling_ratio=2,aligned=True)
        return self.mlp(pooled.flatten(1)).reshape(b,q,-1)
