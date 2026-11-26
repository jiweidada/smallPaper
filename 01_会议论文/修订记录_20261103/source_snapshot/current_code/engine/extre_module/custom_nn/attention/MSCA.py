'''
Multi-Scale Context-Aware Attention (MSCA) for VisDrone2019
Designed specifically for drone-based object detection with small objects and varying scales
Fixed version with proper spatial dimension handling
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class MSCA(nn.Module):
    """
    Multi-Scale Context-Aware Attention
    Combines multi-scale feature aggregation with context-aware spatial attention
    """
    def __init__(self, channels, scales=[1, 3, 5], reduction=8, e_lambda=1e-4):
        super(MSCA, self).__init__()
        self.channels = channels
        self.scales = scales
        self.e_lambda = e_lambda

        # Multi-scale convolutions for capturing different receptive fields
        self.multi_scale_convs = nn.ModuleList()
        for scale in scales:
            if scale == 1:
                self.multi_scale_convs.append(nn.Identity())
            else:
                # Use odd kernel sizes for proper padding
                self.multi_scale_convs.append(
                    nn.Sequential(
                        nn.Conv2d(channels, channels//reduction, kernel_size=1),
                        nn.BatchNorm2d(channels//reduction),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(channels//reduction, channels, kernel_size=scale,
                                 stride=1, padding=scale//2, groups=channels//reduction),
                        nn.BatchNorm2d(channels),
                    )
                )

        # Channel attention branch (lightweight)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Context aggregation
        self.context_conv = nn.Conv2d(channels * len(scales), channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # Multi-scale feature extraction
        multi_scale_features = []
        for conv in self.multi_scale_convs:
            feature = conv(x)
            # Ensure all features have the same spatial dimensions
            if feature.shape[2:] != x.shape[2:]:
                feature = F.interpolate(feature, size=(h, w), mode='bilinear', align_corners=False)
            multi_scale_features.append(feature)

        # Concatenate multi-scale features
        ms_concat = torch.cat(multi_scale_features, dim=1)

        # Context aggregation
        context = self.context_conv(ms_concat)

        # Spatial attention (inspired by SimAM but enhanced)
        # Calculate mean and variance
        mu = context.mean(dim=[2, 3], keepdim=True)
        variance = ((context - mu).pow(2)).mean(dim=[2, 3], keepdim=True)

        # Enhanced spatial attention with context
        spatial_att = (context - mu).pow(2) / (4 * (variance + self.e_lambda)) + 0.5
        spatial_att = self.sigmoid(spatial_att)

        # Channel attention
        channel_att = self.channel_att(x)

        # Combine spatial and channel attention
        attention = spatial_att * channel_att

        return x * attention


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


class MSCA_Enhanced(nn.Module):
    """
    Enhanced version with frequency domain processing
    Best for highest accuracy but more computationally intensive
    """
    def __init__(self, channels, reduction=8, e_lambda=1e-4):
        super(MSCA_Enhanced, self).__init__()
        self.channels = channels
        self.e_lambda = e_lambda

        # Spatial branch (multi-scale)
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(channels, channels//reduction, kernel_size=1),
            nn.BatchNorm2d(channels//reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Channel branch
        self.channel_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Frequency branch for capturing fine details (important for small objects)
        self.freq_branch = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.size()

        # Spatial attention
        spatial_att = self.spatial_branch(x)

        # Channel attention
        channel_att = self.channel_branch(x)

        # Frequency domain processing (DCT-like)
        # Simple approximation using convolutions
        freq_features = self.freq_branch(x)

        # High-frequency enhancement for small objects
        high_freq = x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high_freq_att = torch.sigmoid(high_freq.abs().mean(dim=1, keepdim=True))

        # Combine all attention mechanisms
        combined_att = spatial_att * channel_att * (1 + 0.5 * high_freq_att)

        return x * combined_att + freq_features * 0.1


# Usage example
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size, channel, height, width = 1, 16, 32, 32
    inputs = torch.randn((batch_size, channel, height, width)).to(device)

    # Test all variants
    print("Testing MSCA variants for VisDrone2019...")

    # Full version (fixed)
    msca = MSCA(channel).to(device)
    outputs = msca(inputs)
    print(f'MSCA Full - inputs: {inputs.size()} outputs: {outputs.size()}')
    print(f'Parameters: {sum(p.numel() for p in msca.parameters())}')

    # Dilated version (recommended for accuracy)
    msca_dilated = MSCA_Dilated(channel).to(device)
    outputs_dilated = msca_dilated(inputs)
    print(f'\nMSCA Dilated - inputs: {inputs.size()} outputs: {outputs_dilated.size()}')
    print(f'Parameters: {sum(p.numel() for p in msca_dilated.parameters())}')

    # Lite version (recommended for real-time)
    msca_lite = MSCA_Lite(channel).to(device)
    outputs_lite = msca_lite(inputs)
    print(f'\nMSCA Lite - inputs: {inputs.size()} outputs: {outputs_lite.size()}')
    print(f'Parameters: {sum(p.numel() for p in msca_lite.parameters())} (parameter-free!)')

    # Enhanced version
    msca_enhanced = MSCA_Enhanced(channel).to(device)
    outputs_enhanced = msca_enhanced(inputs)
    print(f'\nMSCA Enhanced - inputs: {inputs.size()} outputs: {outputs_enhanced.size()}')
    print(f'Parameters: {sum(p.numel() for p in msca_enhanced.parameters())}')

    print("\nAll tests passed successfully!")