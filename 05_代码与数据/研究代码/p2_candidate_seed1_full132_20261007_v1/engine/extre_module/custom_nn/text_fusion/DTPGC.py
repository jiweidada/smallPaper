'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/自研模块-DTPGC.drawio
engine/extre_module/module_images/自研模块-DTPGC.md
'''
    
import warnings
warnings.filterwarnings('ignore')
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')   
from calflops import calculate_flops
import torch
import torch.nn as nn
import torch.nn.functional as F
  
from engine.extre_module.ultralytics_nn.conv import DSConv 
   

class DynamicPosBias(nn.Module):  
    """轻量位置偏置：由归一化空间坐标预测每个 head 的偏置，弥补纯语义门控的空间无关性。"""

    def __init__(self, nh, hidden=32):  
        super().__init__() 
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden),  
            nn.ReLU(inplace=True), 
            nn.Linear(hidden, nh),     
        )     

    def forward(self, h, w, device, dtype): 
        ys = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, w, device=device, dtype=dtype)  
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij') 
        coords = torch.stack((grid_y, grid_x), dim=-1)  # (h, w, 2)
        bias = self.mlp(coords)  # (h, w, nh)
        return bias.permute(2, 0, 1)  # (nh, h, w)  
   

class DTPGC(nn.Module):     
    """Dynamic Top-K Prompt Gating with Positional Calibration (DTPGC)."""    

    def __init__(self, c1: int, c2: int, nh: int = 1, gc: int = 256, scale: bool = False, topk_ratio: float = 0.5):  
        super().__init__()     
        self.nh = nh     
        self.hc = c2 // nh
        ec = self.nh * self.hc    
        self.topk_ratio = topk_ratio
  
        self.ec = DSConv(c1, ec, k=1, act=False) if c1 != ec else None    
        self.gl = nn.Linear(gc, ec) 
        self.bias = nn.Parameter(torch.zeros(nh))    
        self.proj_conv = DSConv(c1, c2, k=3, s=1, act=False) 
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0
        self.pos_bias = DynamicPosBias(nh)     
     
    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        bs, _, h, w = x.shape
        n_prompt = guide.shape[1] 
     
        guide = self.gl(guide)
        guide = guide.view(bs, n_prompt, self.nh, self.hc) 
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)
   
        aw_full = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)    
        aw_full = aw_full / (self.hc ** 0.5)  
   
        k = max(1, min(n_prompt, int(round(n_prompt * self.topk_ratio))))
        topk_val, _ = aw_full.topk(k, dim=-1, largest=True, sorted=False)   
        topk_w = F.softmax(topk_val, dim=-1)
        aw = (topk_val * topk_w).sum(dim=-1)  # (bs, nh, h, w)     
    
        pos_bias = self.pos_bias(h, w, x.device, aw.dtype)  # (nh, h, w)  
        aw = aw + pos_bias.unsqueeze(0) + self.bias[None, :, None, None] 
        aw = aw.sigmoid() * self.scale
 
        x = self.proj_conv(x)     
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)
    
   
if __name__ == '__main__': 
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width, n_prompt, gc, nh = 1, 64, 20, 20, 20, 256, 8    
    inputs = torch.randn((batch_size, channel, height, width)).to(device)
    guide = torch.randn((batch_size, n_prompt, gc)).to(device)  

    module = DTPGC(channel, channel, nh=nh, gc=gc).to(device)   
   
    outputs = module(inputs, guide)
    print(GREEN + f'inputs.size:{inputs.size()} guide.size:{guide.size()} outputs.size:{outputs.size()}' + RESET)  

    print(ORANGE)     
    flops, macs, _ = calculate_flops(model=module,
                                     args=[inputs, guide],   
                                     output_as_string=True,
                                     output_precision=4,     
                                     print_detailed=True)
    print(RESET)    
