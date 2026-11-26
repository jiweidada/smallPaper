"""
Enhanced HGNetv2 with VisDrone-Optimized Stem Blocks for Small Object Detection
Optimized for VisDrone2019 Dataset

Copyright (c) 2024 The Enhanced D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 从已核对的 FULL_FRESH 源码摘录：engine/backbone/hgnetv2_MSCA_EnhancedStem.py
# 原文件 SHA-256：8d229219cce4ff8213e38b9f4fc3e6d72f227061b9ecca6a7a322e15d30195bc


# 原文件第 30–49 行
def get_activation(act_type):
    """
    根据激活函数类型返回对应的激活函数
    """
    if act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'silu' or act_type == 'swish':
        return nn.SiLU()
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'leaky_relu':
        return nn.LeakyReLU(0.1)
    elif act_type == 'hardswish':
        return nn.Hardswish()
    elif act_type == 'mish':
        return nn.Mish()
    elif act_type == 'elu':
        return nn.ELU()
    else:
        return nn.ReLU()


# 原文件第 52–63 行
class LearnableAffineBlock(nn.Module):
    """
    可学习的仿射变换模块
    """

    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x):
        return self.scale * x + self.bias


# 原文件第 66–112 行
class ConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            stride=1,
            groups=1,
            padding='',
            use_act=True,
            use_lab=False,
            act_type='relu'
    ):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab

        if padding == 'same':
            self.conv = nn.Sequential(
                nn.ZeroPad2d([0, 1, 0, 1]),
                nn.Conv2d(in_chs, out_chs, kernel_size, stride, groups=groups, bias=False)
            )
        else:
            self.conv = nn.Conv2d(
                in_chs, out_chs, kernel_size, stride,
                padding=(kernel_size - 1) // 2,
                groups=groups, bias=False
            )

        self.bn = nn.BatchNorm2d(out_chs)

        if self.use_act:
            self.act = get_activation(act_type)
        else:
            self.act = nn.Identity()

        if self.use_act and self.use_lab:
            self.lab = LearnableAffineBlock()
        else:
            self.lab = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x


# 原文件第 336–430 行
class EnhancedStemBlock(nn.Module):
    """
    增强型Stem Block，专门针对小目标检测优化

    主要改进：
    1. 渐进式下采样，保留更多细节
    2. 添加细节保留分支
    3. 引入空间注意力机制
    4. 高频信息增强
    """

    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False, act_type='relu'):
        super().__init__()

        # ========== 细节保留分支 ==========
        self.detail_branch = nn.Sequential(
            ConvBNAct(in_chs, mid_chs // 2, kernel_size=3, stride=1,
                      use_lab=use_lab, act_type=act_type),
            ConvBNAct(mid_chs // 2, mid_chs // 2, kernel_size=3, stride=1,
                      use_lab=use_lab, act_type=act_type),
        )

        # ========== 主干分支 ==========
        self.main_branch = nn.Sequential(
            ConvBNAct(in_chs, mid_chs, kernel_size=3, stride=2,
                      use_lab=use_lab, act_type=act_type),
            ConvBNAct(mid_chs, mid_chs, kernel_size=3, stride=1,
                      use_lab=use_lab, act_type=act_type),
        )

        # ========== 高频增强模块 ==========
        self.high_freq_enhance = nn.Sequential(
            nn.Conv2d(in_chs, mid_chs // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_chs // 4),
            get_activation(act_type),
            nn.Conv2d(mid_chs // 4, mid_chs // 4, kernel_size=3, padding=1,
                      groups=mid_chs // 4, bias=False),
            nn.BatchNorm2d(mid_chs // 4),
            get_activation(act_type),
        )

        # ========== 空间注意力模块 ==========
        total_channels = mid_chs + mid_chs // 2 + mid_chs // 4
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(total_channels, mid_chs // 8, kernel_size=1),
            nn.BatchNorm2d(mid_chs // 8),
            get_activation(act_type),
            nn.Conv2d(mid_chs // 8, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # ========== 特征融合模块 ==========
        self.fusion = nn.Sequential(
            ConvBNAct(total_channels, out_chs * 2, kernel_size=1,
                      use_lab=use_lab, act_type=act_type),
            ConvBNAct(out_chs * 2, out_chs, kernel_size=3, stride=1,
                      use_lab=use_lab, act_type=act_type)
        )

        # ========== 最终下采样 ==========
        self.final_downsample = ConvBNAct(
            out_chs, out_chs, kernel_size=3, stride=2,
            use_lab=use_lab, act_type=act_type
        )

    def forward(self, x):
        """前向传播"""
        # 1. 细节分支
        detail = self.detail_branch(x)
        detail = F.avg_pool2d(detail, kernel_size=2, stride=2)

        # 2. 主干分支
        main = self.main_branch(x)

        # 3. 高频增强
        blurred = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high_freq = x - blurred
        high_freq = self.high_freq_enhance(high_freq)
        high_freq = F.interpolate(high_freq, size=(main.shape[2], main.shape[3]),
                                  mode='bilinear', align_corners=False)

        # 4. 特征拼接
        concat_feat = torch.cat([main, detail, high_freq], dim=1)

        # 5. 空间注意力加权
        spatial_att = self.spatial_attention(concat_feat)
        weighted_feat = concat_feat * spatial_att + concat_feat

        # 6. 特征融合
        out = self.fusion(weighted_feat)

        # 7. 最终下采样
        out = self.final_downsample(out)

        return out
