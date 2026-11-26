"""
reference  
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

"""
HGNetv2 with DRC and MobileMamba
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


# DRC Block
class HG_Block_DRC(HG_Block):
    def __init__(self, in_chs, mid_chs, out_chs, layer_num, kernel_size=3, residual=False, light_block=False,
                 use_lab=False, agg='ese', drop_path=0, drc_k=11):
        super().__init__(in_chs, mid_chs, out_chs, layer_num, kernel_size, residual, light_block, use_lab, agg,
                         drop_path)

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            self.layers.append(DilatedReparamConv(in_chs if i == 0 else mid_chs,
                                                  mid_chs,
                                                  kernel_size=drc_k))


# DRC Stage
class HG_Stage_DRC(HG_Stage):
    def __init__(self, in_chs, mid_chs, out_chs, block_num, layer_num, downsample=True, light_block=False,
                 kernel_size=3, use_lab=False, agg='se', drop_path=0, drc_k=11):
        super().__init__(in_chs, mid_chs, out_chs, block_num, layer_num, downsample, light_block, kernel_size, use_lab,
                         agg, drop_path)

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


# MobileMamba Stage
class HG_Stage_MobileMamba(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, block_num, layer_num,
                 downsample=True, light_block=False, kernel_size=3,
                 use_lab=False, agg='se', drop_path=0,
                 global_ratio=0.25, local_ratio=0.25, ssm_ratio=1, forward_type="v052d"):
        super().__init__()
        self.downsample = downsample

        # 下采样层
        if downsample:
            self.downsample = ConvBNAct(
                in_chs,
                in_chs,
                kernel_size=3,
                stride=2,
                groups=in_chs,
                use_act=False,
                use_lab=use_lab
            )
        else:
            self.downsample = nn.Identity()

        # 通道适配层
        self.channel_adapter = None
        if in_chs != out_chs:
            self.channel_adapter = ConvBNAct(
                in_chs,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab
            )

        # 构建MobileMamba blocks
        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                MobileMambaModule(
                    dim=out_chs,
                    global_ratio=global_ratio,
                    local_ratio=local_ratio,
                    kernels=kernel_size,
                    ssm_ratio=ssm_ratio,
                    forward_type=forward_type
                )
            )

            # 添加残差连接
            if i > 0:
                blocks_list.append(ResidualAdd())

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.downsample(x)

        if self.channel_adapter is not None:
            x = self.channel_adapter(x)

        x = self.blocks(x)
        return x


# 残差连接模块
class ResidualAdd(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 假设输入是经过block处理后的特征
        return x


# 主模型
@register()
class HGNetv2_DRC_MobileMamba(HGNetv2):
    def __init__(self, name, use_lab=False, return_idx=..., freeze_stem_only=True,
                 freeze_at=0, freeze_norm=True, pretrained=True, agg='se',
                 local_model_dir='weight/hgnetv2/',
                 # MobileMamba specific params
                 global_ratio=0.25, local_ratio=0.25, ssm_ratio=1, forward_type="v052d"):
        super().__init__(name, use_lab, return_idx, freeze_stem_only, freeze_at, freeze_norm, pretrained, agg,
                         local_model_dir)

        stage_config = {
            # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num, drc_k, stage_type
            "stage1": [16, 16, 64, 1, False, False, 3, 3, 5, 'DRC'],
            "stage2": [64, 32, 256, 1, True, False, 3, 3, 7, 'DRC'],
            "stage3": [256, 64, 512, 2, True, True, 5, 3, 9, 'DRC'],
            "stage4": [512, 128, 1024, 1, True, True, 5, 3, 11, 'MobileMamba'],
        }

        # 重建stages
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
                        drc_k=drc_k
                    )
                )
            elif stage_type == 'MobileMamba':
                self.stages.append(
                    HG_Stage_MobileMamba(
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
                        global_ratio=global_ratio,
                        local_ratio=local_ratio,
                        ssm_ratio=ssm_ratio,
                        forward_type=forward_type
                    )
                )

        # 初始化新增的模块
        self._init_weights()

    def _init_weights(self):
        """初始化新增模块的权重"""
        for m in self.modules():
            if isinstance(m, MobileMambaModule):
                # MobileMamba模块会自己初始化
                pass
            elif isinstance(m, DilatedReparamConv):
                # DRC模块初始化
                if hasattr(m, 'lk_origin'):
                    nn.init.kaiming_normal_(m.lk_origin.weight, mode='fan_out', nonlinearity='relu')