'''  
本文件由BiliBili：魔傀面具整理   
engine/extre_module/module_images/自研模块-SCPGA.drawio  
engine/extre_module/module_images/自研模块-SCPGA.md
'''    
 
import warnings
warnings.filterwarnings('ignore')
import os, sys   
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
from calflops import calculate_flops     
import torch   
import torch.nn as nn     
import torch.nn.functional as F

from engine.extre_module.ultralytics_nn.conv import Conv, DSConv    
    

class PromptSelfAttention(nn.Module):     
    """对 guide 中的多个 prompt 做一次轻量自注意力，建模 prompt 间语义关系。"""

    def __init__(self, gc, num_heads=4):   
        super().__init__()
        if gc % num_heads != 0:
            num_heads = 1
        self.num_heads = num_heads   
        self.head_dim = gc // num_heads   
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(gc, gc * 3)
        self.proj = nn.Linear(gc, gc)

    def forward(self, guide):  
        bs, n, c = guide.shape
        qkv = self.qkv(guide).reshape(bs, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   
        attn = (q @ k.transpose(-2, -1)) * self.scale    
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(bs, n, c) 
        return guide + self.proj(out)


class SCPGA(nn.Module):
    """Soft-Max Cooperative Prompt-Gated Attention (SCPGA)."""

    def __init__(self, c1: int, c2: int, nh: int = 1, gc: int = 256, scale: bool = False):  
        super().__init__() 
        self.nh = nh  
        self.hc = c2 // nh     
        ec = self.nh * self.hc 
  
        self.prompt_attn = PromptSelfAttention(gc, num_heads=min(4, gc))
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None 
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh)) 
        self.proj_conv = DSConv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0     

        # 软/硬聚合混合系数与可学习温度：训练早期偏向 soft 聚合以稳定梯度，
        # 随温度收缩逐渐逼近 hard-max 以保持原有稀疏选择性。
        self.mix_logit = nn.Parameter(torch.zeros(1))
        self.log_temp = nn.Parameter(torch.zeros(1))   

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        bs, _, h, w = x.shape

        guide = self.prompt_attn(guide)
        guide = self.gl(guide) 
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)   
  
        embed = self.ec(x) if self.ec is not None else x    
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw_full = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)

        hard_max = aw_full.max(dim=-1)[0]
        temp = F.softplus(self.log_temp) + 1e-3
        soft_agg = temp * torch.logsumexp(aw_full / temp, dim=-1)
        beta = torch.sigmoid(self.mix_logit)
        aw = beta * hard_max + (1 - beta) * soft_agg
    
        aw = aw / (self.hc ** 0.5) 
        aw = aw + self.bias[None, :, None, None]  
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
 
    module = SCPGA(channel, channel, nh=nh, gc=gc).to(device)  

    outputs = module(inputs, guide)   
    print(GREEN + f'inputs.size:{inputs.size()} guide.size:{guide.size()} outputs.size:{outputs.size()}' + RESET)   

    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module,  
                                     args=[inputs, guide],   
                                     output_as_string=True, 
                                     output_precision=4, 
                                     print_detailed=True) 
    print(RESET)
