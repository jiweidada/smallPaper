'''
Multi-Scale Context-Aware Attention (MSCA) for VisDrone2019
Designed specifically for drone-based object detection with small objects and varying scales
Fixed version with proper spatial dimension handling
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# 从已核对的 FULL_FRESH 源码摘录：engine/extre_module/custom_nn/attention/MSCA.py
# 原文件 SHA-256：04a3ebb82954c060843493a27a582694d9f056e7badc0629ffe0718674a6e888


# 原文件第 157–196 行
class MSCA_Lite(nn.Module):
    """
    Lightweight version of MSCA with fewer parameters
    Better for real-time drone applications
    """
    def __init__(self, channels, e_lambda=1e-4):
        super(MSCA_Lite, self).__init__()
        self.e_lambda = e_lambda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # Multi-scale context using pooling (parameter-free)
        # Scale 1: Original
        scale1 = x

        # Scale 2: 2x2 pooling and upsample
        scale2 = F.avg_pool2d(x, kernel_size=2, stride=2)
        scale2 = F.interpolate(scale2, size=(h, w), mode='bilinear', align_corners=False)

        # Scale 3: 4x4 pooling and upsample
        scale3 = F.avg_pool2d(x, kernel_size=4, stride=4)
        scale3 = F.interpolate(scale3, size=(h, w), mode='bilinear', align_corners=False)

        # Aggregate multi-scale information
        ms_fusion = (scale1 + scale2 + scale3) / 3.0

        # Enhanced SimAM-style attention with multi-scale context
        mu = ms_fusion.mean(dim=[2, 3], keepdim=True)

        # Calculate attention weights
        diff_squared = (ms_fusion - mu).pow(2)
        sigma_squared = diff_squared.mean(dim=[2, 3], keepdim=True)

        # Attention calculation
        attention = diff_squared / (4 * (sigma_squared + self.e_lambda)) + 0.5
        attention = self.sigmoid(attention)

        return x * attention
