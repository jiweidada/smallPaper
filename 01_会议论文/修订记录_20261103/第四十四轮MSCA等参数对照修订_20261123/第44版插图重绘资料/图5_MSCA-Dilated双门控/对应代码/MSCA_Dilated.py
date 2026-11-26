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


# 原文件第 89–154 行
class MSCA_Dilated(nn.Module):
    """
    MSCA using dilated convolutions instead of different kernel sizes
    More efficient and avoids spatial dimension issues
    """
    def __init__(self, channels, dilations=[1, 2, 4], reduction=8, e_lambda=1e-4):
        super(MSCA_Dilated, self).__init__()
        self.channels = channels
        self.dilations = dilations
        self.e_lambda = e_lambda

        # Multi-scale dilated convolutions
        self.multi_scale_convs = nn.ModuleList()
        for dilation in dilations:
            self.multi_scale_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels//reduction, kernel_size=1),
                    nn.BatchNorm2d(channels//reduction),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels//reduction, channels, kernel_size=3,
                             stride=1, padding=dilation, dilation=dilation,
                             groups=channels//reduction),
                    nn.BatchNorm2d(channels),
                )
            )

        # Channel attention branch
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Context aggregation
        self.context_conv = nn.Conv2d(channels * len(dilations), channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # Multi-scale feature extraction using dilated convolutions
        multi_scale_features = []
        for conv in self.multi_scale_convs:
            multi_scale_features.append(conv(x))

        # Concatenate multi-scale features (same spatial dimensions guaranteed)
        ms_concat = torch.cat(multi_scale_features, dim=1)

        # Context aggregation
        context = self.context_conv(ms_concat)

        # Enhanced spatial attention
        mu = context.mean(dim=[2, 3], keepdim=True)
        variance = ((context - mu).pow(2)).mean(dim=[2, 3], keepdim=True)
        spatial_att = (context - mu).pow(2) / (4 * (variance + self.e_lambda)) + 0.5
        spatial_att = self.sigmoid(spatial_att)

        # Channel attention
        channel_att = self.channel_att(x)

        # Combine attentions
        attention = spatial_att * channel_att

        return x * attention
