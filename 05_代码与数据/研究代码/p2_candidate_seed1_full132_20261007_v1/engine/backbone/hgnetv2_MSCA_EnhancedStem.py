"""
Enhanced HGNetv2 with VisDrone-Optimized Stem Blocks for Small Object Detection
Optimized for VisDrone2019 Dataset

Copyright (c) 2024 The Enhanced D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, re
# from .common import FrozenBatchNorm2d
from ..core import register
from ..extre_module.custom_nn.attention.ema import EMA
from ..extre_module.custom_nn.attention.simam import SimAM
from ..extre_module.custom_nn.attention.lsk import LSKBlock
from ..extre_module.custom_nn.attention.DeformableLKA import DeformableLKA
from ..extre_module.custom_nn.attention.MSCA import MSCA, MSCA_Dilated, MSCA_Lite, MSCA_Enhanced
from ..misc.dist_utils import Multiprocess_sync, is_dist_available_and_initialized
import logging

# Constants for initialization
kaiming_normal_ = nn.init.kaiming_normal_
zeros_ = nn.init.zeros_
ones_ = nn.init.ones_

__all__ = ['HGNetv2_MSCA_EnhancedStem']


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


class LightConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            groups=1,
            use_lab=False,
            act_type='relu'
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_chs, out_chs, kernel_size=1,
            use_act=False, use_lab=use_lab, act_type=act_type
        )
        self.conv2 = ConvBNAct(
            out_chs, out_chs, kernel_size=kernel_size,
            groups=out_chs, use_act=True, use_lab=use_lab, act_type=act_type
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# ========== 🔥 新增：VisDrone优化的轻量级多尺度Stem Block ==========
class VDLightweightMultiScaleStem(nn.Module):
    """
    VisDrone优化的轻量级多尺度Stem Block

    设计理念：
    1. 渐进式下采样：1x -> 2x -> 4x (保护小目标)
    2. 多尺度并行处理但共享权重（减少参数）
    3. 自适应特征保护机制
    4. 计算量仅增加约20-30%
    """

    def __init__(self, in_chs=3, out_chs=16, act_type='silu'):
        super().__init__()

        # ========== Stage 1: 多尺度特征提取器（1x下采样） ==========
        # 使用不同感受野捕获不同尺度的目标
        self.multi_scale_extract = nn.ModuleList([
            # Branch 1: 小目标专用（3x3卷积）
            nn.Sequential(
                nn.Conv2d(in_chs, out_chs//4, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_chs//4),
                get_activation(act_type)
            ),

            # Branch 2: 中等目标（5x5卷积，用两个3x3模拟）
            nn.Sequential(
                nn.Conv2d(in_chs, out_chs//4, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_chs//4),
                get_activation(act_type),
                nn.Conv2d(out_chs//4, out_chs//4, 3, 1, 1, groups=out_chs//4, bias=False),  # DW
                nn.BatchNorm2d(out_chs//4),
            ),

            # Branch 3: 大目标（7x7卷积，stride=1保持分辨率）
            nn.Sequential(
                nn.Conv2d(in_chs, out_chs//4, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_chs//4),
                get_activation(act_type),
                nn.Conv2d(out_chs//4, out_chs//4, 3, 1,
                         dilation=2, padding=2, groups=out_chs//4, bias=False),  # 空洞卷积
                nn.BatchNorm2d(out_chs//4),
            ),

            # Branch 4: 全局上下文（stride=1的下采样分支）
            nn.Sequential(
                nn.AvgPool2d(3, 1, 1),  # 平滑处理
                nn.Conv2d(in_chs, out_chs//4, 1, bias=False),
                nn.BatchNorm2d(out_chs//4),
                get_activation(act_type)
            )
        ])

        # ========== Stage 2: 特征融合与第一次下采样（2x） ==========
        # 轻量级特征融合
        self.fusion_conv1 = nn.Sequential(
            nn.Conv2d(out_chs, out_chs, 1, bias=False),  # 1x1 融合
            nn.BatchNorm2d(out_chs),
            get_activation(act_type)
        )

        # 小目标保护分支（不下采样）
        self.protect_branch = nn.Sequential(
            nn.Conv2d(out_chs, out_chs//2, 1, bias=False),
            nn.BatchNorm2d(out_chs//2),
            get_activation(act_type),
            nn.Conv2d(out_chs//2, out_chs//2, 3, 1, 1, groups=out_chs//2, bias=False),
            nn.BatchNorm2d(out_chs//2),
        )

        # 主下采样分支（2x下采样）
        self.downsample_2x = nn.Sequential(
            nn.Conv2d(out_chs, out_chs, 3, 2, 1, groups=out_chs, bias=False),  # DW
            nn.BatchNorm2d(out_chs),
            nn.Conv2d(out_chs, out_chs, 1, bias=False),  # PW
            nn.BatchNorm2d(out_chs),
            get_activation(act_type)
        )

        # ========== Stage 3: 自适应特征增强 ==========
        # 简单但有效的注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_chs + out_chs//2, out_chs//4, 1),
            get_activation(act_type),
            nn.Conv2d(out_chs//4, out_chs + out_chs//2, 1),
            nn.Sigmoid()
        )

        # ========== Stage 4: 最终下采样（4x） ==========
        total_chs = out_chs + out_chs//2
        self.final_downsample = nn.Sequential(
            # 深度可分离卷积下采样
            nn.Conv2d(total_chs, total_chs, 3, 2, 1, groups=total_chs, bias=False),
            nn.BatchNorm2d(total_chs),
            nn.Conv2d(total_chs, out_chs, 1, bias=False),
            nn.BatchNorm2d(out_chs),
            get_activation(act_type)
        )

        # ========== 可选：残差连接 ==========
        self.residual_downsample = nn.Sequential(
            nn.AvgPool2d(4, 4),
            nn.Conv2d(in_chs, out_chs, 1, bias=False),
            nn.BatchNorm2d(out_chs)
        )

    def forward(self, x):
        # Stage 1: 多尺度特征提取
        multi_scale_feats = []
        for branch in self.multi_scale_extract:
            multi_scale_feats.append(branch(x))

        # 拼接多尺度特征
        ms_feat = torch.cat(multi_scale_feats, dim=1)  # [B, out_chs, H, W]

        # Stage 2: 特征融合
        fused = self.fusion_conv1(ms_feat)

        # 分离路径：保护分支 + 下采样分支
        protected = self.protect_branch(fused)  # [B, out_chs//2, H, W]
        downsampled = self.downsample_2x(fused)  # [B, out_chs, H/2, W/2]

        # 将保护分支下采样到相同分辨率
        protected_down = F.avg_pool2d(protected, 2, 2)  # [B, out_chs//2, H/2, W/2]

        # Stage 3: 特征拼接与增强
        combined = torch.cat([downsampled, protected_down], dim=1)  # [B, 1.5*out_chs, H/2, W/2]

        # 通道注意力增强
        channel_att = self.channel_attention(combined)
        enhanced = combined * channel_att

        # Stage 4: 最终下采样
        output = self.final_downsample(enhanced)  # [B, out_chs, H/4, W/4]

        # 添加残差连接（可选）
        residual = self.residual_downsample(x)
        output = output + 0.1 * residual  # 小权重残差

        return output


class VDUltraLightStem(nn.Module):
    """
    超轻量版本：专注于小目标保护，计算量最小化
    """

    def __init__(self, in_chs=3, out_chs=16, act_type='silu'):
        super().__init__()

        # 简化版：只有两个尺度
        # Path 1: 细节路径（保护小目标）
        self.detail_path = nn.Sequential(
            nn.Conv2d(in_chs, out_chs//2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_chs//2),
            get_activation(act_type),
        )

        # Path 2: 语义路径（快速下采样）
        self.semantic_path = nn.Sequential(
            nn.Conv2d(in_chs, out_chs//2, 3, 2, 1, bias=False),  # 直接2x下采样
            nn.BatchNorm2d(out_chs//2),
            get_activation(act_type),
            nn.Conv2d(out_chs//2, out_chs//2, 3, 1, 1, groups=out_chs//2, bias=False),
            nn.BatchNorm2d(out_chs//2),
        )

        # 特征对齐和融合
        self.align_and_fuse = nn.Sequential(
            nn.Conv2d(out_chs, out_chs, 1, bias=False),
            nn.BatchNorm2d(out_chs),
            get_activation(act_type),
            nn.Conv2d(out_chs, out_chs, 3, 2, 1, groups=out_chs, bias=False),  # DW下采样
            nn.BatchNorm2d(out_chs),
            nn.Conv2d(out_chs, out_chs, 1, bias=False),  # PW
            nn.BatchNorm2d(out_chs),
            get_activation(act_type),
        )

    def forward(self, x):
        # 两路处理
        detail = self.detail_path(x)  # [B, C/2, H, W]
        semantic = self.semantic_path(x)  # [B, C/2, H/2, W/2]

        # 对齐分辨率
        detail_down = F.max_pool2d(detail, 2, 2)  # 使用max_pool保护小目标特征

        # 融合
        combined = torch.cat([detail_down, semantic], dim=1)
        output = self.align_and_fuse(combined)

        return output


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


# ========== 原始Stem Block ==========
class StemBlock(nn.Module):
    """
    原始的HGNetv2 Stem Block
    """

    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False, act_type='relu'):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_chs, mid_chs, kernel_size=3, stride=2,
            use_lab=use_lab, act_type=act_type
        )
        self.stem2a = ConvBNAct(
            mid_chs, mid_chs // 2, kernel_size=2, stride=1,
            padding='same', use_lab=use_lab, act_type=act_type
        )
        self.stem2b = ConvBNAct(
            mid_chs // 2, mid_chs, kernel_size=2, stride=1,
            padding='same', use_lab=use_lab, act_type=act_type
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)
        self.stem3 = ConvBNAct(
            mid_chs * 2, mid_chs, kernel_size=3, stride=2,
            use_lab=use_lab, act_type=act_type
        )
        self.stem4 = ConvBNAct(
            mid_chs, out_chs, kernel_size=1, stride=1,
            use_lab=use_lab, act_type=act_type
        )

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class EseModule(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.conv = nn.Conv2d(chs, chs, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.mul(identity, x)


class HG_Block(nn.Module):
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            layer_num,
            kernel_size=3,
            residual=False,
            light_block=False,
            use_lab=False,
            agg='ese',
            drop_path=0.,
            act_type='relu',
    ):
        super().__init__()
        self.residual = residual

        # 构建卷积层序列
        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(
                    LightConvBNAct(
                        in_chs if i == 0 else mid_chs, mid_chs,
                        kernel_size=kernel_size, use_lab=use_lab, act_type=act_type
                    )
                )
            else:
                self.layers.append(
                    ConvBNAct(
                        in_chs if i == 0 else mid_chs, mid_chs,
                        kernel_size=kernel_size, stride=1,
                        use_lab=use_lab, act_type=act_type
                    )
                )

        # 计算聚合层的输入通道数
        total_chs = in_chs + layer_num * mid_chs

        # 特征聚合方式
        if agg == 'se':
            aggregation_squeeze_conv = ConvBNAct(
                total_chs, out_chs // 2, kernel_size=1, stride=1,
                use_lab=use_lab, act_type=act_type
            )
            aggregation_excitation_conv = ConvBNAct(
                out_chs // 2, out_chs, kernel_size=1, stride=1,
                use_lab=use_lab, act_type=act_type
            )
            self.aggregation = nn.Sequential(
                aggregation_squeeze_conv,
                aggregation_excitation_conv,
            )
        elif agg == 'ese':
            aggregation_conv = ConvBNAct(
                total_chs, out_chs, kernel_size=1, stride=1,
                use_lab=use_lab, act_type=act_type
            )
            att = EseModule(out_chs)
            self.aggregation = nn.Sequential(aggregation_conv, att)
        elif agg == 'ema':
            aggregation_conv = ConvBNAct(
                total_chs, out_chs, kernel_size=1, stride=1,
                use_lab=use_lab, act_type=act_type
            )
            att = EMA(out_chs)
            self.aggregation = nn.Sequential(aggregation_conv, att)
        elif agg == 'simam':
            aggregation_conv = ConvBNAct(
                total_chs, out_chs, kernel_size=1, stride=1,
                use_lab=use_lab, act_type=act_type
            )
            att = SimAM()
            self.aggregation = nn.Sequential(aggregation_conv, att)
        elif agg == 'lsk':
            aggregation_conv = ConvBNAct(
                total_chs, out_chs, kernel_size=1, stride=1,
                use_lab=use_lab, act_type=act_type
            )
            att = LSKBlock(out_chs)
            self.aggregation = nn.Sequential(aggregation_conv, att)
        elif agg == 'deformable_lka':
            aggregation_conv = ConvBNAct(
                total_chs, out_chs, kernel_size=1, stride=1,
                use_lab=use_lab, act_type=act_type
            )
            att = DeformableLKA(out_chs)
            self.aggregation = nn.Sequential(aggregation_conv, att)
        elif agg == 'msca':
            aggregation_conv = ConvBNAct(
                total_chs, out_chs, kernel_size=1, stride=1,
                use_lab=use_lab, act_type=act_type
            )
            att = MSCA(out_chs)
            self.aggregation = nn.Sequential(aggregation_conv, att)
        elif agg == 'msca_dilated':
            aggregation_conv = ConvBNAct(
                total_chs, out_chs, kernel_size=1, stride=1,
                use_lab=use_lab, act_type=act_type
            )
            att = MSCA_Dilated(out_chs)
            self.aggregation = nn.Sequential(aggregation_conv, att)
        elif agg == 'msca_lite':
            aggregation_conv = ConvBNAct(
                total_chs, out_chs, kernel_size=1, stride=1,
                use_lab=use_lab, act_type=act_type
            )
            att = MSCA_Lite(out_chs)
            self.aggregation = nn.Sequential(aggregation_conv, att)
        elif agg == 'msca_enhanced':
            aggregation_conv = ConvBNAct(
                total_chs, out_chs, kernel_size=1, stride=1,
                use_lab=use_lab, act_type=act_type
            )
            att = MSCA_Enhanced(out_chs)
            self.aggregation = nn.Sequential(aggregation_conv, att)
        else:
            raise Exception(f"param agg '{agg}' Illegal.")

        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        identity = x
        output = [x]

        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        x = self.aggregation(x)

        if self.residual:
            x = self.drop_path(x) + identity

        return x


class HG_Stage(nn.Module):
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            block_num,
            layer_num,
            downsample=True,
            light_block=False,
            kernel_size=3,
            use_lab=False,
            agg='se',
            drop_path=0.,
            act_type='relu',
    ):
        super().__init__()
        self.downsample = downsample

        if downsample:
            self.downsample = ConvBNAct(
                in_chs, in_chs, kernel_size=3, stride=2,
                groups=in_chs, use_act=False, use_lab=use_lab, act_type=act_type
            )
        else:
            self.downsample = nn.Identity()

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block(
                    in_chs if i == 0 else out_chs,
                    mid_chs, out_chs, layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                    act_type=act_type,
                )
            )

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


@register()
class HGNetv2_MSCA_EnhancedStem(nn.Module):
    """
    Enhanced HGNetV2 with MSCA and VisDrone-Optimized Stem Blocks

    New Features:
    1. Three Stem Block options: 'standard', 'enhanced', 'vd_lms', 'vd_ultralight'
    2. Multi-Scale Context-Aware Attention (MSCA)
    3. Flexible activation functions
    4. VisDrone dataset optimizations
    """

    arch_configs = {
        'B0': {
            'stem_channels': [3, 16, 16],
            'stage_config': {
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
                "stage4": [512, 128, 1024, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'B1': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                "stage1": [32, 32, 64, 1, False, False, 3, 3],
                "stage2": [64, 48, 256, 1, True, False, 3, 3],
                "stage3": [256, 96, 512, 2, True, True, 5, 3],
                "stage4": [512, 192, 1024, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth'
        },
        'B2': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                "stage1": [32, 32, 96, 1, False, False, 3, 4],
                "stage2": [96, 64, 384, 1, True, False, 3, 4],
                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                "stage4": [768, 256, 1536, 1, True, True, 5, 4],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth'
        },
        'B3': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                "stage1": [32, 32, 128, 1, False, False, 3, 5],
                "stage2": [128, 64, 512, 1, True, False, 3, 5],
                "stage3": [512, 128, 1024, 3, True, True, 5, 5],
                "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B3_stage1.pth'
        },
        'B4': {
            'stem_channels': [3, 32, 48],
            'stage_config': {
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth'
        },
        'B5': {
            'stem_channels': [3, 32, 64],
            'stage_config': {
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B5_stage1.pth'
        },
        'B6': {
            'stem_channels': [3, 48, 96],
            'stage_config': {
                "stage1": [96, 96, 192, 2, False, False, 3, 6],
                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                "stage3": [512, 384, 1024, 6, True, True, 5, 6],
                "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B6_stage1.pth'
        },
    }

    def __init__(self,
                 name,
                 use_lab=False,
                 return_idx=[1, 2, 3],
                 freeze_stem_only=True,
                 freeze_at=0,
                 freeze_norm=True,
                 pretrained=True,
                 agg='se',
                 hierarchical_attention=False,
                 act_type='relu',
                 stem_type='standard',  # 🔥 新增参数：Stem类型选择
                 local_model_dir='weight/hgnetv2/'):
        super().__init__()

        self.use_lab = use_lab
        self.return_idx = return_idx
        self.act_type = act_type
        self.stem_type = stem_type  # 保存配置

        stem_channels = self.arch_configs[name]['stem_channels']
        stage_config = self.arch_configs[name]['stage_config']
        download_url = self.arch_configs[name]['url']

        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]

        # ========== 🔥 核心修改：多种Stem Block类型选择 ==========
        if stem_type == 'vd_lms':
            print("🔥 Using VisDrone-Optimized Lightweight Multi-Scale Stem (VD-LMS)")
            self.stem = VDLightweightMultiScaleStem(
                in_chs=stem_channels[0],
                out_chs=stem_channels[2],
                act_type=act_type
            )
        elif stem_type == 'vd_ultralight':
            print("🔥 Using VisDrone-Optimized Ultra-Light Stem (VD-UltraLight)")
            self.stem = VDUltraLightStem(
                in_chs=stem_channels[0],
                out_chs=stem_channels[2],
                act_type=act_type
            )
        elif stem_type == 'enhanced':
            print("🔥 Using Enhanced Stem Block for small object detection")
            self.stem = EnhancedStemBlock(
                in_chs=stem_channels[0],
                mid_chs=stem_channels[1],
                out_chs=stem_channels[2],
                use_lab=use_lab,
                act_type=act_type
            )
        elif stem_type == 'standard':
            print("Using Standard HGNetv2 Stem Block")
            self.stem = StemBlock(
                in_chs=stem_channels[0],
                mid_chs=stem_channels[1],
                out_chs=stem_channels[2],
                use_lab=use_lab,
                act_type=act_type
            )
        else:
            raise ValueError(f"Unknown stem_type: {stem_type}. "
                           f"Supported: 'standard', 'enhanced', 'vd_lms', 'vd_ultralight'")

        # 处理分层注意力策略
        if hierarchical_attention:
            attention_list = ['simam', 'simam', 'lsk', 'lsk']
            print(f"Using hierarchical attention strategy: {attention_list}")
        elif isinstance(agg, list):
            attention_list = agg
            assert len(attention_list) == 4, "attention list length must be 4"
            print(f"Using custom attention list: {attention_list}")
        else:
            attention_list = [agg] * 4
            print(f"Using uniform attention: {agg}")

        # 构建stages
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = \
                stage_config[k]
            self.stages.append(
                HG_Stage(
                    in_channels, mid_channels, out_channels,
                    block_num, layer_num, downsample,
                    light_block, kernel_size, use_lab,
                    attention_list[i], act_type=act_type
                )
            )

        # 冻结参数
        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

        # 加载预训练权重
        if pretrained:
            self._load_pretrained_weights(name, download_url, local_model_dir)

    def _load_pretrained_weights(self, name, download_url, local_model_dir):
        """加载预训练权重"""
        RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"

        try:
            model_path = local_model_dir + 'PPHGNetV2_' + name + '_stage1.pth'

            if os.path.exists(model_path):
                state = torch.load(model_path, map_location='cpu')
                print(f"Loaded stage1 {name} HGNetV2 from local file.")
            else:
                if is_dist_available_and_initialized() and torch.distributed.get_rank() == 0:
                    print(GREEN + "Downloading pretrained model..." + RESET)
                    state = torch.hub.load_state_dict_from_url(
                        download_url, map_location='cpu', model_dir=local_model_dir
                    )
                    Multiprocess_sync()
                else:
                    Multiprocess_sync()
                    state = torch.load(model_path, map_location='cpu')
                print(f"Loaded stage1 {name} HGNetV2 from URL.")

            # 处理非标准Stem的权重
            if self.stem_type in ['vd_lms', 'vd_ultralight', 'enhanced']:
                print(RED + f"Using {self.stem_type} Stem - adapting pretrained weights..." + RESET)
                filtered_state = {}
                for k, v in state.items():
                    if not k.startswith('stem.'):
                        filtered_state[k] = v
                    else:
                        print(f"  Skipping: {k} ({self.stem_type} Stem will be trained from scratch)")

                self.load_state_dict(filtered_state, strict=False)
                print(GREEN + f"✅ {self.stem_type} Stem Block initialized with random weights" + RESET)
            else:
                print(RED + f'Loading Pretrained State Dict Keys: {list(state.keys())[:5]}...' + RESET)
                self.load_state_dict(state, strict=False)

        except Exception as e:
            if (is_dist_available_and_initialized() and torch.distributed.get_rank() == 0) or \
                    (not is_dist_available_and_initialized()):
                print(f"Error: {str(e)}")
                logging.error(RED + "Failed to load pretrained model" + RESET)

            if self.stem_type == 'standard':
                exit()
            else:
                print(GREEN + "Continuing with random initialization..." + RESET)

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            # m = FrozenBatchNorm2d(m.num_features)  # 注释掉避免导入错误
            pass
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs

    def get_model_info(self):
        """获取模型信息用于调试"""
        info = {
            'stem_type': self.stem_type,
            'act_type': self.act_type,
            'use_lab': self.use_lab,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

        # 统计各部分参数
        stem_params = sum(p.numel() for p in self.stem.parameters())
        stage_params = [sum(p.numel() for p in stage.parameters()) for stage in self.stages]

        info['stem_params'] = stem_params
        info['stage_params'] = stage_params

        print("\n" + "=" * 50)
        print("Model Information:")
        print("=" * 50)
        for k, v in info.items():
            print(f"  {k}: {v}")
        print("=" * 50 + "\n")

        return info