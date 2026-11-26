"""Pretrained native Stem plus the user's EnhancedStem, with zero initial residual."""
import torch
from torch import nn
from original_stem import EnhancedStemBlock

class WarmStartEnhancedStem(nn.Module):
    def __init__(self, native_stem, channels=(3,16,16), use_lab=True, seed=1):
        super().__init__()
        if tuple(channels)!=(3,16,16):
            raise ValueError('Only the bound HGNetv2-B0 Stem is supported')
        self.native = native_stem
        # Preserve the established CPU data/DN random-number stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.enhanced = EnhancedStemBlock(*channels,use_lab=use_lab,act_type='relu')
        self.residual_gain = nn.Parameter(torch.zeros(1,channels[-1],1,1))

    def forward(self,x):
        native = self.native(x)
        enhanced = self.enhanced(x)
        if native.shape!=enhanced.shape or native.dtype!=enhanced.dtype:
            raise ValueError('Native and enhanced feature interfaces differ')
        gain = self.residual_gain.tanh().to(dtype=native.dtype)
        return native + gain * enhanced

