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
from .hgnetv2 import HG_Block, HG_Stage, HGNetv2

from ..extre_module.custom_nn.conv_module.DilatedReparamConv import DilatedReparamConv
from ..extre_module.custom_nn.module.FCM import FCM

# Constants for initialization
kaiming_normal_ = nn.init.kaiming_normal_
zeros_ = nn.init.zeros_
ones_ = nn.init.ones_

__all__ = ['HGNetv2_DRC_FCM']


class HG_Block_FCM(nn.Module):
    """FCM增强的HG_Block - 通过组合方式在标准HG_Block后面加载FCM"""

    def __init__(self, in_chs, mid_chs, out_chs, layer_num, kernel_size=3, residual=False, light_block=False,
                 use_lab=False, agg='ese', drop_path=0):
        super().__init__()

        # 创建标准的HG_Block
        self.hg_block = HG_Block(in_chs, mid_chs, out_chs, layer_num, kernel_size, residual, light_block, use_lab, agg,
                                 drop_path)

        # 在输出后添加FCM增强
        self.fcm_enhancement = FCM(out_chs, out_chs)

        # 保持与原有HG_Block相同的属性，以便其他组件访问
        self.residual = residual

    def forward(self, x):
        """
        前向传播：先通过标准HG_Block，再通过FCM增强
        """
        # 通过标准HG_Block
        x = self.hg_block(x)

        # 通过FCM增强
        x = self.fcm_enhancement(x)

        return x


class HG_Block_DRC(HG_Block):
    """DRC增强的HG_Block"""

    def __init__(self, in_chs, mid_chs, out_chs, layer_num, kernel_size=3, residual=False, light_block=False,
                 use_lab=False, agg='ese', drop_path=0, drc_k=11):
        super().__init__(in_chs, mid_chs, out_chs, layer_num, kernel_size, residual, light_block, use_lab, agg,
                         drop_path)

        # 用DRC替换标准卷积层
        self.layers = nn.ModuleList()
        for i in range(layer_num):
            self.layers.append(DilatedReparamConv(in_chs if i == 0 else mid_chs,
                                                  mid_chs,
                                                  kernel_size=drc_k))


class HG_Stage_FCM(HG_Stage):
    """FCM增强的HG_Stage"""

    def __init__(self, in_chs, mid_chs, out_chs, block_num, layer_num, downsample=True, light_block=False,
                 kernel_size=3, use_lab=False, agg='se', drop_path=0):
        super().__init__(in_chs, mid_chs, out_chs, block_num, layer_num, downsample, light_block, kernel_size, use_lab,
                         agg, drop_path)

        # 构建FCM blocks
        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block_FCM(
                    in_chs if i == 0 else out_chs,  # 第一个block需要匹配输入通道
                    mid_chs,  # 中间层通道数
                    out_chs,  # 输出通道数
                    layer_num,  # 该block内的层数
                    residual=False if i == 0 else True,  # 只有第一个块不使用残差连接
                    kernel_size=kernel_size,  # 卷积核大小
                    light_block=light_block,  # 是否使用轻量级卷积块
                    use_lab=use_lab,  # 是否使用可学习的仿射变换模块
                    agg=agg,  # 特征聚合方式
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,  # DropPath比例
                )
            )

        # 将所有HG_Block组成一个顺序执行的模块
        self.blocks = nn.Sequential(*blocks_list)


class HG_Stage_DRC(HG_Stage):
    """DRC增强的HG_Stage"""

    def __init__(self, in_chs, mid_chs, out_chs, block_num, layer_num, downsample=True, light_block=False,
                 kernel_size=3, use_lab=False, agg='se', drop_path=0, drc_k=11):
        super().__init__(in_chs, mid_chs, out_chs, block_num, layer_num, downsample, light_block, kernel_size, use_lab,
                         agg, drop_path)

        # 构建DRC blocks
        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block_DRC(
                    in_chs if i == 0 else out_chs,  # 第一个block需要匹配输入通道
                    mid_chs,  # 中间层通道数
                    out_chs,  # 输出通道数
                    layer_num,  # 该block内的层数
                    residual=False if i == 0 else True,  # 只有第一个块不使用残差连接
                    kernel_size=kernel_size,  # 卷积核大小
                    light_block=light_block,  # 是否使用轻量级卷积块
                    use_lab=use_lab,  # 是否使用可学习的仿射变换模块
                    agg=agg,  # 特征聚合方式
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,  # DropPath比例
                    drc_k=drc_k
                )
            )

        # 将所有HG_Block组成一个顺序执行的模块
        self.blocks = nn.Sequential(*blocks_list)


@register()
class HGNetv2_DRC_FCM(HGNetv2):
    """
    HGNetv2 with DRC and FCM enhancements
    简洁的配置风格，直接在stage_config中定义所有参数
    """

    def __init__(self, name, use_lab=False, return_idx=..., freeze_stem_only=True, freeze_at=0, freeze_norm=True,
                 pretrained=True, agg='se', local_model_dir='weight/hgnetv2/'):
        super().__init__(name, use_lab, return_idx, freeze_stem_only, freeze_at, freeze_norm, pretrained, agg,
                         local_model_dir)

        # 🚀 简洁的stage配置 - 直接定义每个stage的所有参数
        # 格式: [in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num, extra_param, stage_type]
        stage_config = {
            # in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num, extra_param, stage_type
            "stage1": [16, 16, 64, 1, False, False, 3, 3, 0, 'default'],
            "stage2": [64, 32, 256, 1, True, False, 3, 3, 0, 'default'],
            "stage3": [256, 64, 512, 2, True, True, 5, 3, 9, 'default'],
            "stage4": [512, 64, 1024, 1, True, True, 5, 3, 0, 'FCM'],
        }

        # 重建stages
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num, extra_param, stage_type = \
                stage_config[k]

            if stage_type == 'FCM':
                self.stages.append(
                    HG_Stage_FCM(
                        in_channels,
                        mid_channels,
                        out_channels,
                        block_num,
                        layer_num,
                        downsample,
                        light_block,
                        kernel_size,
                        use_lab,
                        agg
                    )
                )
            elif stage_type == 'DRC':
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
                        drc_k=extra_param
                    )
                )
            else:  # default
                self.stages.append(
                    HG_Stage(
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
                        drop_path=0
                    )
                )

