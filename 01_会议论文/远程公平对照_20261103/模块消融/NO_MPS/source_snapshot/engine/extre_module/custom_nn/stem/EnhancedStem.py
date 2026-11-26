import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re
from typing import Optional, List, Tuple

from engine import get_activation
from engine.backbone.hgnetv2 import ConvBNAct


class EnhancedStemBlock(nn.Module):
    """
    增强型Stem Block，专门针对小目标检测优化
    主要改进：
    1. 渐进式下采样，保留更多细节
    2. 添加细节保留分支
    3. 引入空间注意力机制
    """

    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False, act_type='relu'):
        super().__init__()

        # 第一阶段：温和的下采样（stride=1），保留原始分辨率
        self.detail_branch = nn.Sequential(
            ConvBNAct(in_chs, mid_chs // 2, kernel_size=3, stride=1,
                      use_lab=use_lab, act_type=act_type),
            ConvBNAct(mid_chs // 2, mid_chs // 2, kernel_size=3, stride=1,
                      use_lab=use_lab, act_type=act_type),
        )

        # 主干分支：原始的下采样路径
        self.main_branch = nn.Sequential(
            ConvBNAct(in_chs, mid_chs, kernel_size=3, stride=2,
                      use_lab=use_lab, act_type=act_type),
            ConvBNAct(mid_chs, mid_chs, kernel_size=3, stride=1,
                      use_lab=use_lab, act_type=act_type),
        )

        # 细节增强模块（捕获高频信息）
        self.high_freq_enhance = nn.Sequential(
            nn.Conv2d(in_chs, mid_chs // 4, kernel_size=1),
            nn.BatchNorm2d(mid_chs // 4),
            get_activation(act_type),
            nn.Conv2d(mid_chs // 4, mid_chs // 4, kernel_size=3, padding=1, groups=mid_chs // 4),
            nn.BatchNorm2d(mid_chs // 4),
        )

        # 空间注意力（增强小目标区域）
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(mid_chs + mid_chs // 2 + mid_chs // 4, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 特征融合
        self.fusion = ConvBNAct(
            mid_chs + mid_chs // 2 + mid_chs // 4,
            out_chs,
            kernel_size=1,
            use_lab=use_lab,
            act_type=act_type
        )

        # 最终下采样
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # 保留原始输入用于高频增强
        identity = x

        # 细节分支（保持高分辨率）
        detail = self.detail_branch(x)
        detail = F.avg_pool2d(detail, kernel_size=2, stride=2)  # 温和下采样

        # 主干分支
        main = self.main_branch(x)

        # 高频增强（捕获细节）
        high_freq = x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high_freq = self.high_freq_enhance(high_freq)
        high_freq = F.interpolate(high_freq, scale_factor=0.5, mode='bilinear')

        # 特征拼接
        concat_feat = torch.cat([main, detail, high_freq], dim=1)

        # 空间注意力加权
        spatial_att = self.spatial_attention(concat_feat)
        weighted_feat = concat_feat * spatial_att

        # 特征融合
        out = self.fusion(weighted_feat)

        # 最终下采样到目标尺寸
        out = self.downsample(out)

        return out