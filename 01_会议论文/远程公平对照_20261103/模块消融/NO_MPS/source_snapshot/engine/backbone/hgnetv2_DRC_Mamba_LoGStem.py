"""
reference  
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, re
from .common import FrozenBatchNorm2d
from ..core import register
from .hgnetv2 import HGNetv2

from ..extre_module.custom_nn.conv_module.DilatedReparamConv import DilatedReparamConv
from ..extre_module.custom_nn.mamba.ASSM import ASSM
from ..extre_module.custom_nn.stem.LoG import LoGStem
from ..extre_module.ultralytics_nn.block import C2f_Block
from functools import partial

# Import the existing stage classes
from .hgnetv2_DRC_Mamba import HG_Stage_DRC, HG_Stage_Mamba

__all__ = ['HGNetv2_DRC_Mamba_LoGStem']


@register()
class HGNetv2_DRC_Mamba_LoGStem(HGNetv2):
    def __init__(self, name, use_lab=False, return_idx=..., freeze_stem_only=True, freeze_at=0, freeze_norm=True,
                 pretrained=True, agg='se', local_model_dir='weight/hgnetv2/', pretrained_backbone_path=None):
        super().__init__(name, use_lab, return_idx, freeze_stem_only, freeze_at, freeze_norm, pretrained, agg,
                         local_model_dir)

        # 1. Replace stem with LoGStem
        stem_channels = self.arch_configs[name]['stem_channels']
        self.stem = LoGStem(stem_channels[0], stem_channels[2], use_multiscale=True)

        # 2. Build DRC + Mamba stages (same as HGNetv2_DRC_Mamba)
        stage_config = {
            # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num, drc_k, stage_type
            "stage1": [16, 16, 64, 1, False, False, 3, 3, 5, 'DRC'],
            "stage2": [64, 32, 256, 1, True, False, 3, 3, 7, 'DRC'],
            "stage3": [256, 64, 512, 2, True, True, 5, 3, 9, 'DRC'],
            "stage4": [512, 128, 1024, 1, True, True, 5, 3, 11, 'Mamba'],
        }

        # 3. Build stages
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num, drc_k, stage_type = \
            stage_config[k]

            if stage_type == 'DRC':
                self.stages.append(
                    HG_Stage_DRC(
                        in_channels,
                        mid_channels,
                        out_channels,
                        block_num,
                        layer_num,
                        downsample,
                        light_block,
                        kernel_size,
                        use_lab,
                        agg,
                        drc_k=drc_k))
            elif stage_type == 'Mamba':
                self.stages.append(
                    HG_Stage_Mamba(
                        in_channels,
                        mid_channels,
                        out_channels,
                        block_num,
                        layer_num,
                        downsample,
                        light_block,
                        kernel_size,
                        use_lab,
                        agg)
                )

        # 4. Load pretrained backbone weights if provided
        if pretrained_backbone_path and os.path.exists(pretrained_backbone_path):
            self.load_pretrained_backbone(pretrained_backbone_path)

    def load_pretrained_backbone(self, pretrained_path):
        """Load pretrained weights from DRC_Mamba model"""
        print(f"Loading pretrained backbone from: {pretrained_path}")

        try:
            # Load checkpoint
            if pretrained_path.endswith('.pth') or pretrained_path.endswith('.pt'):
                checkpoint = torch.load(pretrained_path, map_location='cpu')

                # Extract model state dict
                if 'model' in checkpoint:
                    pretrained_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    pretrained_dict = checkpoint['state_dict']
                else:
                    pretrained_dict = checkpoint
            else:
                raise ValueError(f"Unsupported checkpoint format: {pretrained_path}")

            # Filter backbone weights (exclude stem)
            model_dict = self.state_dict()
            backbone_dict = {}

            for k, v in pretrained_dict.items():
                # Only load stages weights, skip stem
                if k.startswith('stages.') and k in model_dict:
                    if model_dict[k].shape == v.shape:
                        backbone_dict[k] = v
                        print(f"✓ Loaded: {k}")
                    else:
                        print(f"✗ Shape mismatch for {k}: {model_dict[k].shape} vs {v.shape}")
                elif k.startswith('stem.'):
                    print(f"✗ Skipped stem layer: {k} (using LoGStem instead)")
                else:
                    print(f"✗ Not found in model: {k}")

            # Update model
            model_dict.update(backbone_dict)
            self.load_state_dict(model_dict, strict=False)

            print(f"Successfully loaded {len(backbone_dict)} layers from pretrained backbone")
            print(f"Stem layers will be trained from scratch with LoGStem")

        except Exception as e:
            print(f"Warning: Failed to load pretrained backbone: {e}")
            print("Training from scratch...")