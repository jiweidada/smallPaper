"""Borrowed LN/MLP self-only control; close parameter budget, not equal function capacity."""
import torch
from torch import nn

class SelfCapacityReadout(nn.Module):
    requires_pool=False
    def __init__(self,zero_output=True):
        super().__init__()
        self.norm=nn.LayerNorm(256)
        self.hidden=nn.Linear(256,80)
        self.act=nn.SiLU()
        self.output=nn.Linear(80,10)
        if zero_output:
            nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,hidden,dense,native,indices,mask):
        if hidden.ndim!=3 or hidden.shape[-1]!=256 or native.shape!=(*hidden.shape[:2],10):
            raise ValueError('Requires native hidden and ten logits.')
        # Pool arguments are ignored deliberately. The wrapper skips dense bbox/pool.
        delta=self.output(self.act(self.hidden(self.norm(hidden.detach())))).to(native.dtype)
        return native.detach()+delta,{'delta':delta}
