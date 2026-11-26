"""
DEIM: DETR with Improved Matching for Fast Convergence 
Copyright (c) 2024 The DEIM Authors. All Rights Reserved. 
---------------------------------------------------------------------------------   
Modified from D-FINE (https://github.com/Peterande/D-FINE/)    
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
""" 
 
import copy
from collections import OrderedDict
   
import torch   
import torch.nn as nn 
import torch.nn.functional as F  
  
from .utils import get_activation

from ..core import register
from .hybrid_encoder import HybridEncoder
from engine.extre_module.custom_nn.transformer.DilatedGCSA import DilatedGCSA

__all__ = ['HybridEncoder_AdaDilatedGCSA']

@register()
class HybridEncoder_AdaDilatedGCSA(HybridEncoder):
    def __init__(self, in_channels=..., feat_strides=..., hidden_dim=256, nhead=8,
                 dim_feedforward=1024, dropout=0, enc_act='gelu', use_encoder_idx=...,
                 num_encoder_layers=1, pe_temperature=10000, expansion=1, depth_mult=1,
                 act='silu', eval_spatial_size=None, version='dfine',
                 window_size_list=None, overlap_ratio_list=None):
        super().__init__(in_channels, feat_strides, hidden_dim, nhead, dim_feedforward,
                        dropout, enc_act, use_encoder_idx, num_encoder_layers,
                        pe_temperature, expansion, depth_mult, act, eval_spatial_size, version)

        # 如果没有提供window_size_list，则根据feat_strides自动选择
        if window_size_list is None:
            # 根据特征步长选择合适的window_size
            # feat_stride越大，特征图越小，使用更小的window_size
            window_size_map = {
                8: 10,   # 对于小的stride，使用大的window
                16: 10,   # 中等stride
                32: 10,   # 大的stride，使用小的window
                64: 10,
                128: 5
            }
            window_size_list = []
            for idx in use_encoder_idx:
                stride = feat_strides[idx]
                # 如果stride不在map中，选择最接近的
                if stride in window_size_map:
                    ws = window_size_map[stride]
                else:
                    # 选择最接近的window_size
                    sorted_strides = sorted(window_size_map.keys())
                    closest_stride = min(sorted_strides, key=lambda x: abs(x - stride))
                    ws = window_size_map[closest_stride]
                window_size_list.append(ws)

        # 如果没有提供overlap_ratio_list，则根据feat_strides自动选择
        if overlap_ratio_list is None:
            # 浅层使用更大的overlap，深层减少overlap
            overlap_ratio_map = {
                8: 0.7,   # 浅层更多重叠
                16: 0.6,
                32: 0.6,  # 深层减少重叠
                64: 0.6,
                128: 0.5
            }
            overlap_ratio_list = []
            for idx in use_encoder_idx:
                stride = feat_strides[idx]
                # 如果stride不在map中，选择最接近的
                if stride in overlap_ratio_map:
                    overlap = overlap_ratio_map[stride]
                else:
                    # 选择最接近的overlap_ratio
                    sorted_strides = sorted(overlap_ratio_map.keys())
                    closest_stride = min(sorted_strides, key=lambda x: abs(x - stride))
                    overlap = overlap_ratio_map[closest_stride]
                overlap_ratio_list.append(overlap)

        # 打印配置信息，便于调试
        # print(f"[HybridEncoder_DilatedGCSA] Configuration:")
        # print(f"  use_encoder_idx: {use_encoder_idx}")
        # print(f"  feat_strides for encoder: {[feat_strides[idx] for idx in use_encoder_idx]}")
        # print(f"  window_size_list: {window_size_list}")
        # print(f"  overlap_ratio_list: {overlap_ratio_list}")

        self.encoder = nn.ModuleList([
            DilatedGCSA(
                hidden_dim,
                num_heads=nhead,
                window_size=window_size_list[i],
                overlap_ratio=overlap_ratio_list[i]
            ) for i in range(len(use_encoder_idx))
        ])

    def forward(self, feats):
        """
        前向传播函数
        Args:
            feats (list[torch.Tensor]): 输入特征图列表，形状为 [B, C, H, W]，长度需与 in_channels 一致
        Returns:
            list[torch.Tensor]: 融合后的多尺度特征图列表
        """
        # 检查输入特征图数量是否与预期一致
        assert len(feats) == len(self.in_channels)

        # 输入投影：将所有特征图投影到 hidden_dim 通道
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # Transformer 编码器：对指定层进行特征增强
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                proj_feats[enc_ind] = self.encoder[i](proj_feats[enc_ind])

        # FPN 融合：自顶向下融合高层特征到低层特征
        inner_outs = [proj_feats[-1]]  # 从最顶层特征图开始
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]  # 当前高层特征
            feat_low = proj_feats[idx - 1]  # 当前低层特征
            # 横向卷积处理高层特征
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            # 上采样高层特征
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            # 拼接上采样后的高层特征和低层特征，并通过 FPN 块处理
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)  # 将融合结果插入列表开头

        # PAN 融合：自底向上融合低层特征到高层特征
        outs = [inner_outs[0]]  # 从最底层特征图开始
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]  # 当前低层特征
            feat_height = inner_outs[idx + 1]  # 当前高层特征
            # 下采样低层特征
            downsample_feat = self.downsample_convs[idx](feat_low)
            # 拼接下采样后的低层特征和高层特征，并通过 PAN 块处理
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)  # 将融合结果添加到列表

        # 返回融合后的多尺度特征图
        return outs