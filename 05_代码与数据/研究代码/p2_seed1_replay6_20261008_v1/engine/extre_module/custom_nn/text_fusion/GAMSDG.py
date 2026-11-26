'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/自研模块-GAMSDG.drawio 
engine/extre_module/module_images/自研模块-GAMSDG.md     
'''

import warnings    
warnings.filterwarnings('ignore')  
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
from calflops import calculate_flops
import torch 
import torch.nn as nn   
import torch.nn.functional as F     
     
from engine.extre_module.ultralytics_nn.conv import Conv, DWConv
   
  
class GuideAdaptiveMultiScaleProj(nn.Module):  
    """由 guide 的全局语义向量动态生成 3x3/5x5/7x7 深度卷积分支的融合权重，
    实现"文本引导的感受野选择"，并对融合结果做 guide 条件下的通道重标定。
    """
  
    def __init__(self, c1, c2, gc, hidden=64):
        super().__init__()
        self.branch3 = DWConv(c1, c1, 3, act=True)    
        self.branch5 = DWConv(c1, c1, 5, act=True)
        self.branch7 = DWConv(c1, c1, 7, act=True)
 
        self.scale_gate = nn.Sequential(
            nn.Linear(gc, hidden),    
            nn.ReLU(inplace=True),   
            nn.Linear(hidden, 3),
        ) 
        self.channel_gate = nn.Sequential(   
            nn.Linear(gc, hidden), 
            nn.ReLU(inplace=True),
            nn.Linear(hidden, c2), 
        )
        self.merge = Conv(c1, c2, 1, act=False)  
  
    def forward(self, x, guide_pool): 
        b0 = self.branch3(x)
        b1 = self.branch5(x) 
        b2 = self.branch7(x)   
        stacked = torch.stack((b0, b1, b2), dim=1)  # (bs, 3, c1, h, w)
   
        scale_w = F.softmax(self.scale_gate(guide_pool), dim=-1)  # (bs, 3)
        bs = scale_w.shape[0]
        fused = (stacked * scale_w.view(bs, 3, 1, 1, 1)).sum(dim=1)  # (bs, c1, h, w)
        fused = self.merge(fused)

        channel_w = torch.sigmoid(self.channel_gate(guide_pool)).view(bs, -1, 1, 1)    
        return fused * channel_w     


class GAMSDG(nn.Module):
    """Guide-Adaptive Multi-Scale Dynamic Gating (GAMSDG)."""

    def __init__(self, c1: int, c2: int, nh: int = 1, gc: int = 256, scale: bool = False):
        super().__init__()
        self.nh = nh  
        self.hc = c2 // nh
        ec = self.nh * self.hc
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec) 
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj = GuideAdaptiveMultiScaleProj(c1, c2, gc)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0
    
    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:    
        bs, _, h, w = x.shape
        guide_pool = guide.mean(dim=1)     

        gl = self.gl(guide)  
        gl = gl.view(bs, gl.shape[1], self.nh, self.hc)   
        embed = self.ec(x) if self.ec is not None else x     
        embed = embed.view(bs, self.nh, self.hc, h, w)   
  
        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, gl) 
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc ** 0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale  
 
        x = self.proj(x, guide_pool)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2) 
        return x.view(bs, -1, h, w)     
   
  
if __name__ == '__main__':  
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"   
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width, n_prompt, gc, nh = 1, 64, 20, 20, 20, 256, 8
    inputs = torch.randn((batch_size, channel, height, width)).to(device)  
    guide = torch.randn((batch_size, n_prompt, gc)).to(device)
    
    module = GAMSDG(channel, channel, nh=nh, gc=gc).to(device)
    
    outputs = module(inputs, guide) 
    print(GREEN + f'inputs.size:{inputs.size()} guide.size:{guide.size()} outputs.size:{outputs.size()}' + RESET)
 
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,    
                                     args=[inputs, guide],
                                     output_as_string=True,  
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)   
