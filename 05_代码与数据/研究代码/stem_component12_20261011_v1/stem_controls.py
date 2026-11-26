"""One changed operation per arm; all original parameters retained."""
import torch
from torch.nn import functional as F
from original_stem import EnhancedStemBlock

ARMS=('detail_lowres','highfreq_rgb','attention_spatial_mean','early_final_downsample')

class ControlledStemBlock(EnhancedStemBlock):
    def set_control(self,control):
        assert control in ARMS or control=='full'
        self.control=control
        if control=='early_final_downsample':
            self.fusion[1].conv.stride=(2,2)
            self.final_downsample.conv.stride=(1,1)

    def forward(self,x):
        if self.control=='detail_lowres':
            detail=self.detail_branch(F.avg_pool2d(x,2,2))
        else:
            detail=F.avg_pool2d(self.detail_branch(x),2,2)
        main=self.main_branch(x)
        source=x if self.control=='highfreq_rgb' else x-F.avg_pool2d(x,3,1,1)
        high=F.interpolate(self.high_freq_enhance(source),size=main.shape[-2:],mode='bilinear',align_corners=False)
        cat=torch.cat([main,detail,high],1)
        att=self.spatial_attention(cat)
        if self.control=='attention_spatial_mean':
            att=att.mean(dim=(-2,-1),keepdim=True)
        return self.final_downsample(self.fusion(cat*att+cat))
