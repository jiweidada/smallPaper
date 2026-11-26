"""
HGNetv2 with DRC and MobileMamba - Complete Implementation
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, re
from .common import FrozenBatchNorm2d
from ..core import register
from .hgnetv2 import HG_Block, HG_Stage, HGNetv2, ConvBNAct, LightConvBNAct

from ..extre_module.custom_nn.conv_module.DilatedReparamConv import DilatedReparamConv
from ..extre_module.custom_nn.mamba.MobileMamba.mobilemamba import MobileMambaModule
from functools import partial

# Constants for initialization
kaiming_normal_ = nn.init.kaiming_normal_
zeros_ = nn.init.zeros_
ones_ = nn.init.ones_

__all__ = ['HGNetv2_DRC_MobileMamba']


# ============== DRC Blocks ==============
class HG_Block_DRC(HG_Block):
    """HG Block with Dilated Reparam Conv"""
    def __init__(self, in_chs, mid_chs, out_chs, layer_num, kernel_size=3, 
                 residual=False, light_block=False, use_lab=False, agg='ese', 
                 drop_path=0, drc_k=11):
        super().__init__(in_chs, mid_chs, out_chs, layer_num, kernel_size, 
                         residual, light_block, use_lab, agg, drop_path)
        
        self.layers = nn.ModuleList()
        for i in range(layer_num):
            self.layers.append(
                DilatedReparamConv(
                    in_chs if i == 0 else mid_chs,
                    mid_chs,
                    kernel_size=drc_k
                )
            )


class HG_Stage_DRC(HG_Stage):
    """HG Stage with DRC blocks"""
    def __init__(self, in_chs, mid_chs, out_chs, block_num, layer_num, 
                 downsample=True, light_block=False, kernel_size=3, 
                 use_lab=False, agg='se', drop_path=0, drc_k=11):
        super().__init__(in_chs, mid_chs, out_chs, block_num, layer_num, 
                         downsample, light_block, kernel_size, use_lab, agg, drop_path)
        
        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block_DRC(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                    drc_k=drc_k
                )
            )
        self.blocks = nn.Sequential(*blocks_list)


# ============== MobileMamba Blocks ==============
class MobileMambaBlock(nn.Module):
    """Single MobileMamba block with residual connection"""
    def __init__(self, dim, global_ratio=0.5, local_ratio=0.25, 
                 kernels=3, ssm_ratio=1.5, forward_type="v052d", 
                 drop_path=0.0):
        super().__init__()
        self.mamba = MobileMambaModule(
            dim=dim,
            global_ratio=global_ratio,
            local_ratio=local_ratio,
            kernels=kernels,
            ssm_ratio=ssm_ratio,
            forward_type=forward_type
        )
        self.drop_path = nn.Identity() if drop_path == 0 else DropPath(drop_path)
        
    def forward(self, x):
        return x + self.drop_path(self.mamba(x))

class HG_Stage_MobileMamba(nn.Module):
    """HG Stage with pure MobileMamba blocks"""
    def __init__(self, in_chs, mid_chs, out_chs, block_num, layer_num,
                 downsample=True, light_block=False, kernel_size=3,
                 use_lab=False, agg='se', drop_path=0,
                 global_ratio=0.5, local_ratio=0.25, ssm_ratio=1.5, 
                 forward_type="v052d"):
        super().__init__()
        
        # Downsample layer
        if downsample:
            self.downsample = ConvBNAct(
                in_chs, in_chs, kernel_size=3, stride=2,
                groups=in_chs, use_act=False, use_lab=use_lab
            )
        else:
            self.downsample = nn.Identity()
            
        # Channel adapter
        self.channel_adapter = None
        if in_chs != out_chs:
            self.channel_adapter = ConvBNAct(
                in_chs, out_chs, kernel_size=1,
                stride=1, use_lab=use_lab
            )
            
        # Build MobileMamba blocks
        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                MobileMambaBlock(
                    dim=out_chs,
                    global_ratio=global_ratio,
                    local_ratio=local_ratio,
                    kernels=kernel_size,
                    ssm_ratio=ssm_ratio,
                    forward_type=forward_type,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path
                )
            )
        self.blocks = nn.Sequential(*blocks_list)
        
    def forward(self, x):
        x = self.downsample(x)
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        x = self.blocks(x)
        return x


# ============== Hybrid Stage ==============
class HG_Stage_Hybrid(nn.Module):
    """Hybrid stage combining DRC and MobileMamba blocks"""
    def __init__(self, in_chs, mid_chs, out_chs, block_num, layer_num,
                 downsample=True, light_block=False, kernel_size=3,
                 use_lab=False, agg='se', drop_path=0,
                 drc_k=11, mamba_ratio=0.5,
                 global_ratio=0.5, local_ratio=0.25, ssm_ratio=1.5,
                 forward_type="v052d"):
        super().__init__()
        
        # Downsample
        if downsample:
            self.downsample = ConvBNAct(
                in_chs, in_chs, kernel_size=3, stride=2,
                groups=in_chs, use_act=False, use_lab=use_lab
            )
        else:
            self.downsample = nn.Identity()
            
        # Channel adapter for first block if needed
        self.channel_adapter = None
        if in_chs != out_chs:
            self.channel_adapter = ConvBNAct(
                in_chs, out_chs, kernel_size=1, use_lab=use_lab
            )
            
        # Calculate number of each block type
        mamba_blocks = int(block_num * mamba_ratio)
        drc_blocks = block_num - mamba_blocks
        
        # Build mixed blocks
        blocks_list = []
        
        # First, add MobileMamba blocks
        for i in range(mamba_blocks):
            blocks_list.append(
                MobileMambaBlock(
                    dim=out_chs,
                    global_ratio=global_ratio,
                    local_ratio=local_ratio,
                    kernels=kernel_size,
                    ssm_ratio=ssm_ratio,
                    forward_type=forward_type,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path
                )
            )
            
        # Then, add DRC blocks
        for i in range(drc_blocks):
            idx = i + mamba_blocks
            blocks_list.append(
                HG_Block_DRC(
                    out_chs,  # All blocks use out_chs since we have channel adapter
                    mid_chs,
                    out_chs,
                    layer_num,
                    residual=True,  # Always use residual in hybrid stage
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    drop_path=drop_path[idx] if isinstance(drop_path, (list, tuple)) else drop_path,
                    drc_k=drc_k
                )
            )
            
        self.blocks = nn.Sequential(*blocks_list)
        
    def forward(self, x):
        x = self.downsample(x)
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        x = self.blocks(x)
        return x


# ============== DropPath ==============
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


# ============== Main Model ==============
@register()
class HGNetv2_DRC_MobileMamba(HGNetv2):
    """
    HGNetv2 with DRC and MobileMamba modules
    
    Args:
        stage_mode: Stage configuration mode
            - 'last': Only use MobileMamba in the last stage (default)
            - 'hybrid': Use hybrid stages with both DRC and MobileMamba
            - 'progressive': Progressively increase MobileMamba ratio
            - 'custom': Custom configuration via stage_types parameter
        mamba_stages: List of stage indices to use MobileMamba (for 'custom' mode)
        mamba_ratio: Ratio of MobileMamba blocks in hybrid stages
        global_ratio: Global modeling ratio for MobileMamba
        local_ratio: Local modeling ratio for MobileMamba
        ssm_ratio: SSM expansion ratio for MobileMamba
        forward_type: Forward type for MobileMamba
    """
    def __init__(self, name, use_lab=False, return_idx=..., 
                 freeze_stem_only=True, freeze_at=0, freeze_norm=True, 
                 pretrained=True, agg='se', local_model_dir='weight/hgnetv2/',
                 # Stage configuration
                 stage_mode='hybrid',  # 'last', 'hybrid', 'progressive', 'custom'
                 mamba_stages=[2, 3],  # For 'custom' mode
                 mamba_ratio=0.5,      # For 'hybrid' mode
                 # MobileMamba parameters
                 global_ratio=0.5, local_ratio=0.25, ssm_ratio=1.5,
                 forward_type="v052d",
                 # DRC parameters
                 drc_kernels=[5, 7, 9, 11],  # DRC kernel sizes for each stage
                 # Training parameters
                 drop_path_rate=0.1):
        
        super().__init__(name, use_lab, return_idx, freeze_stem_only, 
                        freeze_at, freeze_norm, pretrained, agg, local_model_dir)
        
        # Stage configurations for different model sizes
        stage_configs = {
            'B0': [
                # [in_ch, mid_ch, out_ch, blocks, downsample, light, k_size, layers]
                [16, 16, 64, 1, False, False, 3, 3],
                [64, 32, 256, 1, True, False, 3, 3],
                [256, 64, 512, 2, True, True, 5, 3],
                [512, 128, 1024, 1, True, True, 5, 3],
            ],
            'B1': [
                [24, 24, 96, 1, False, False, 3, 3],
                [96, 48, 384, 1, True, False, 3, 3],
                [384, 96, 768, 3, True, True, 5, 3],
                [768, 192, 1536, 1, True, True, 5, 3],
            ],
            'B2': [
                [32, 32, 128, 1, False, False, 3, 3],
                [128, 64, 512, 1, True, False, 3, 3],
                [512, 128, 1024, 3, True, True, 5, 3],
                [1024, 256, 2048, 1, True, True, 5, 3],
            ],
        }
        
        # Get stage config for current model
        cfg = stage_configs.get(name, stage_configs['B0'])
        
        # Calculate drop path rates
        depth = sum([c[3] for c in cfg])  # Total number of blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Determine stage types based on mode
        stage_types = self._determine_stage_types(stage_mode, mamba_stages)
        
        # Rebuild stages
        self.stages = nn.ModuleList()
        block_idx = 0
        
        for i, (stage_cfg, stage_type) in enumerate(zip(cfg, stage_types)):
            in_ch, mid_ch, out_ch, blocks, downsample, light, k_size, layers = stage_cfg
            drc_k = drc_kernels[i] if i < len(drc_kernels) else 11
            
            # Get drop path rates for this stage
            stage_dpr = dpr[block_idx:block_idx + blocks]
            block_idx += blocks
            
            if stage_type == 'drc':
                self.stages.append(
                    HG_Stage_DRC(
                        in_ch, mid_ch, out_ch, blocks, layers,
                        downsample, light, k_size, use_lab, agg,
                        drop_path=stage_dpr, drc_k=drc_k
                    )
                )
            elif stage_type == 'mamba':
                self.stages.append(
                    HG_Stage_MobileMamba(
                        in_ch, mid_ch, out_ch, blocks, layers,
                        downsample, light, k_size, use_lab, agg,
                        drop_path=stage_dpr,
                        global_ratio=global_ratio,
                        local_ratio=local_ratio,
                        ssm_ratio=ssm_ratio,
                        forward_type=forward_type
                    )
                )
            elif stage_type == 'hybrid':
                self.stages.append(
                    HG_Stage_Hybrid(
                        in_ch, mid_ch, out_ch, blocks, layers,
                        downsample, light, k_size, use_lab, agg,
                        drop_path=stage_dpr,
                        drc_k=drc_k,
                        mamba_ratio=mamba_ratio,
                        global_ratio=global_ratio,
                        local_ratio=local_ratio,
                        ssm_ratio=ssm_ratio,
                        forward_type=forward_type
                    )
                )
                
        # Initialize weights
        self._init_weights()
        
    def _determine_stage_types(self, mode, mamba_stages):
        """Determine stage types based on configuration mode"""
        if mode == 'last':
            return ['drc', 'drc', 'drc', 'mamba']
        elif mode == 'hybrid':
            return ['drc', 'hybrid', 'hybrid', 'mamba']
        elif mode == 'progressive':
            return ['drc', 'drc', 'hybrid', 'mamba']
        elif mode == 'custom':
            types = ['drc'] * 4
            for idx in mamba_stages:
                if 0 <= idx < 4:
                    types[idx] = 'mamba'
            return types
        else:
            raise ValueError(f"Unknown stage mode: {mode}")
            
    def _init_weights(self):
        """Initialize weights for new modules"""
        for m in self.modules():
            if isinstance(m, MobileMambaModule):
                # MobileMamba initialization
                if hasattr(m, 'out_proj') and m.out_proj is not None:
                    nn.init.zeros_(m.out_proj.weight)
                    if hasattr(m.out_proj, 'bias') and m.out_proj.bias is not None:
                        nn.init.zeros_(m.out_proj.bias)
                        
                # Initialize SSM parameters if available
                if hasattr(m, 'A_log'):
                    nn.init.uniform_(m.A_log, -3.0, -1.0)
                if hasattr(m, 'D'):
                    nn.init.ones_(m.D)
                    
            elif isinstance(m, DilatedReparamConv):
                # DRC initialization
                if hasattr(m, 'lk_origin'):
                    nn.init.kaiming_normal_(m.lk_origin.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'small_conv'):
                    nn.init.kaiming_normal_(m.small_conv.weight, mode='fan_out', nonlinearity='relu')
                    
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)