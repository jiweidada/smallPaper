"""
HGNetv2 with DRC and MobileMamba - Fixed Implementation
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

__all__ = ['HGNetv2_DRC_MobileMamba']


def debug_tensor_shape(tensor, name="Tensor"):
    """调试函数：打印张量形状 - 用于诊断MobileMamba输入格式"""
    print(f"{name} shape: {tensor.shape}")
    return tensor


# ============== Helper Modules ==============
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


class MobileMambaWrapper(nn.Module):
    """Wrapper for MobileMamba - processes 4D spatial tensors directly"""
    def __init__(self, dim, global_ratio=0.5, local_ratio=0.25, 
                 kernels=3, ssm_ratio=1.5, forward_type="v052d", debug=False):
        super().__init__()
        self.dim = dim
        self.debug = debug
        
        # MobileMamba直接处理 [B, C, H, W] 格式
        self.mamba = MobileMambaModule(
            dim=dim,
            global_ratio=global_ratio,
            local_ratio=local_ratio,
            kernels=kernels,
            ssm_ratio=ssm_ratio,
            forward_type=forward_type
        )
        
    def forward(self, x):
        # x shape: [B, C, H, W] - 直接的空间张量
        if self.debug:
            print(f"MobileMambaWrapper input shape: {x.shape}")
            print(f"Expected: [B, {self.dim}, H, W]")
        
        # MobileMamba直接处理空间张量
        out = self.mamba(x)
        
        if self.debug:
            print(f"MobileMambaWrapper output shape: {out.shape}")
            
        return out


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
    """Fixed MobileMamba block - processes spatial tensors directly"""
    def __init__(self, dim, global_ratio=0.5, local_ratio=0.25, 
                 kernels=3, ssm_ratio=1.5, forward_type="v052d", 
                 drop_path=0.0, use_residual=True, debug=False):
        super().__init__()
        self.use_residual = use_residual
        self.mamba_wrapper = MobileMambaWrapper(
            dim=dim,
            global_ratio=global_ratio,
            local_ratio=local_ratio,
            kernels=kernels,
            ssm_ratio=ssm_ratio,
            forward_type=forward_type,
            debug=debug
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
    def forward(self, x):
        # x shape: [B, C, H, W] - 直接处理空间张量
        out = self.mamba_wrapper(x)
        if self.use_residual:
            return x + self.drop_path(out)
        else:
            return self.drop_path(out)


class HG_Stage_MobileMamba(nn.Module):
    """HG Stage with pure MobileMamba blocks - processes spatial tensors directly"""
    def __init__(self, in_chs, mid_chs, out_chs, block_num, layer_num,
                 downsample=True, light_block=False, kernel_size=3,
                 use_lab=False, agg='se', drop_path=0,
                 global_ratio=0.5, local_ratio=0.25, ssm_ratio=1.5, 
                 forward_type="v052d", use_residual=True):
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
        
        # Build MobileMamba blocks - 直接处理空间张量
        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                MobileMambaBlock(
                    dim=out_chs,  # 通道数匹配
                    global_ratio=global_ratio,
                    local_ratio=local_ratio,
                    kernels=kernel_size,
                    ssm_ratio=ssm_ratio,
                    forward_type=forward_type,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                    use_residual=use_residual
                )
            )
        self.blocks = nn.Sequential(*blocks_list)
        
    def forward(self, x):
        # Downsample
        x = self.downsample(x)
        
        # Channel adaptation
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        
        # Process through MobileMamba blocks - 直接处理空间张量 [B, C, H, W]
        x = self.blocks(x)
        
        return x


# ============== Hybrid Stage ==============
class HG_Stage_Hybrid(nn.Module):
    """Optimized hybrid stage - processes spatial tensors directly"""
    def __init__(self, in_chs, mid_chs, out_chs, block_num, layer_num,
                 downsample=True, light_block=False, kernel_size=3,
                 use_lab=False, agg='se', drop_path=0,
                 drc_k=11, mamba_ratio=0.5,
                 global_ratio=0.5, local_ratio=0.25, ssm_ratio=1.5,
                 forward_type="v052d", fusion_strategy='sequential'):
        super().__init__()
        
        self.fusion_strategy = fusion_strategy
        
        # Downsample
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
                in_chs, out_chs, kernel_size=1, use_lab=use_lab
            )
        
        # Calculate number of each block type
        mamba_blocks_num = max(1, int(block_num * mamba_ratio))
        drc_blocks_num = block_num - mamba_blocks_num
        
        if fusion_strategy == 'sequential':
            # Sequential: DRC blocks -> MobileMamba blocks
            self.drc_blocks = self._build_drc_blocks(
                out_chs, mid_chs, drc_blocks_num, layer_num,
                light_block, kernel_size, use_lab, agg, 
                drop_path[:drc_blocks_num] if isinstance(drop_path, (list, tuple)) else [drop_path] * drc_blocks_num, 
                drc_k
            )
            
            self.mamba_blocks = self._build_mamba_blocks(
                out_chs, mamba_blocks_num, kernel_size,
                drop_path[drc_blocks_num:] if isinstance(drop_path, (list, tuple)) else [drop_path] * mamba_blocks_num, 
                global_ratio, local_ratio, ssm_ratio, forward_type
            )
            
        elif fusion_strategy == 'parallel':
            # Parallel: DRC and MobileMamba process in parallel and merge
            self.drc_branch = self._build_drc_blocks(
                out_chs, mid_chs, block_num, layer_num,
                light_block, kernel_size, use_lab, agg, drop_path, drc_k
            )
            
            self.mamba_branch = self._build_mamba_blocks(
                out_chs, block_num, kernel_size,
                drop_path, global_ratio, local_ratio,
                ssm_ratio, forward_type
            )
            
            # Merge layer
            self.merge = nn.Conv2d(out_chs * 2, out_chs, 1)
            
    def _build_drc_blocks(self, out_chs, mid_chs, num_blocks, layer_num,
                         light_block, kernel_size, use_lab, agg, drop_path, drc_k):
        if num_blocks == 0:
            return None
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                HG_Block_DRC(
                    out_chs, mid_chs, out_chs, layer_num,
                    residual=True, kernel_size=kernel_size,
                    light_block=light_block, use_lab=use_lab,
                    agg=agg, drop_path=drop_path[i] if i < len(drop_path) else 0,
                    drc_k=drc_k
                )
            )
        return nn.Sequential(*blocks)
    
    def _build_mamba_blocks(self, dim, num_blocks, kernel_size,
                           drop_path, global_ratio, local_ratio,
                           ssm_ratio, forward_type):
        if num_blocks == 0:
            return None
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                MobileMambaBlock(
                    dim=dim,
                    global_ratio=global_ratio,
                    local_ratio=local_ratio,
                    kernels=kernel_size,
                    ssm_ratio=ssm_ratio,
                    forward_type=forward_type,
                    drop_path=drop_path[i] if i < len(drop_path) else 0
                )
            )
        return nn.Sequential(*blocks)
        
    def forward(self, x):
        x = self.downsample(x)
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        
        if self.fusion_strategy == 'sequential':
            # Process through DRC blocks first
            if self.drc_blocks is not None:
                x = self.drc_blocks(x)
            
            # Then through MobileMamba blocks - 直接处理空间张量
            if self.mamba_blocks is not None:
                x = self.mamba_blocks(x)
                
        elif self.fusion_strategy == 'parallel':
            # Parallel processing
            x_drc = self.drc_branch(x) if self.drc_branch is not None else x
            
            # MobileMamba branch - 直接处理空间张量
            if self.mamba_branch is not None:
                x_mamba = self.mamba_branch(x)
            else:
                x_mamba = x
            
            # Merge
            x = self.merge(torch.cat([x_drc, x_mamba], dim=1))
        
        return x


# ============== Main Model ==============
@register()
class HGNetv2_DRC_MobileMamba(HGNetv2):
    """
    Fixed HGNetv2 with DRC and MobileMamba modules
    
    Key fixes:
    1. MobileMamba directly processes 4D spatial tensors [B, C, H, W]
    2. Removed unnecessary spatial-to-sequence conversions
    3. Fixed dimension compatibility issues
    4. Better error handling for edge cases
    
    Note: MobileMamba expects and processes spatial tensors directly,
    no sequence conversion needed.
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
                 forward_type="v052d", use_mamba_residual=True,
                 # DRC parameters
                 drc_kernels=[5, 7, 9, 11],  # DRC kernel sizes for each stage
                 # Fusion strategy
                 fusion_strategy='sequential',  # 'sequential', 'parallel'
                 # Training parameters
                 drop_path_rate=0.1):
        
        super().__init__(name, use_lab, return_idx, freeze_stem_only, 
                        freeze_at, freeze_norm, pretrained, agg, local_model_dir)
        
        # Stage configurations
        stage_configs = {
            'B0': [
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
        
        cfg = stage_configs.get(name, stage_configs['B0'])
        
        # Calculate drop path rates
        depth = sum([c[3] for c in cfg])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Determine stage types
        stage_types = self._determine_stage_types(stage_mode, mamba_stages)
        
        # Rebuild stages
        self.stages = nn.ModuleList()
        block_idx = 0
        
        for i, (stage_cfg, stage_type) in enumerate(zip(cfg, stage_types)):
            in_ch, mid_ch, out_ch, blocks, downsample, light, k_size, layers = stage_cfg
            drc_k = drc_kernels[i] if i < len(drc_kernels) else 11
            
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
                        forward_type=forward_type,
                        use_residual=use_mamba_residual
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
                        forward_type=forward_type,
                        fusion_strategy=fusion_strategy
                    )
                )
                
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
            if isinstance(m, MobileMambaWrapper):
                # MobileMamba specific initialization
                if hasattr(m.mamba, 'out_proj') and m.mamba.out_proj is not None:
                    nn.init.zeros_(m.mamba.out_proj.weight)
                    if hasattr(m.mamba.out_proj, 'bias') and m.mamba.out_proj.bias is not None:
                        nn.init.zeros_(m.mamba.out_proj.bias)
                        
                if hasattr(m.mamba, 'A_log'):
                    nn.init.uniform_(m.mamba.A_log, -3.0, -1.0)
                if hasattr(m.mamba, 'D'):
                    nn.init.ones_(m.mamba.D)
                    
            elif isinstance(m, DilatedReparamConv):
                # DRC initialization
                if hasattr(m, 'lk_origin'):
                    nn.init.kaiming_normal_(m.lk_origin.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'small_conv') and m.small_conv is not None:
                    nn.init.kaiming_normal_(m.small_conv.weight, mode='fan_out', nonlinearity='relu')
                    
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)