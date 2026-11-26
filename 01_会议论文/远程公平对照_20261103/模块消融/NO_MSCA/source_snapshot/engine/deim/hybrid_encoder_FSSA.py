import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core import register
from .hybrid_encoder import HybridEncoder
from engine.extre_module.custom_nn.transformer.FSSA import FSSA

__all__ = ['HybridEncoder_FSSA']


@register()
class HybridEncoder_FSSA(HybridEncoder):
    """使用FSSA(频域+空域自注意力)替代传统Transformer的混合编码器"""

    def __init__(self, in_channels=..., feat_strides=..., hidden_dim=256,
                 nhead=8, dim_feedforward=1024, dropout=0, enc_act='gelu',
                 use_encoder_idx=..., num_encoder_layers=1, pe_temperature=10000,
                 expansion=1, depth_mult=1, act='silu', eval_spatial_size=None,
                 version='dfine'):
        super().__init__(in_channels, feat_strides, hidden_dim, nhead,
                         dim_feedforward, dropout, enc_act, use_encoder_idx,
                         num_encoder_layers, pe_temperature, expansion,
                         depth_mult, act, eval_spatial_size, version)

        # 使用FSSA替代原有的encoder - 修正参数
        self.encoder = nn.ModuleList([
            FSSA(
                dim=hidden_dim,
                num_heads=nhead,
                bias=True,  # 对应原来的qkv_bias
            ) for _ in range(len(use_encoder_idx))
        ])

        # 添加dropout层（因为FSSA本身不包含dropout）
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(len(use_encoder_idx))
        ])

        # 添加特征归一化层
        self.pre_norm = nn.ModuleList([
            nn.BatchNorm2d(hidden_dim) for _ in range(len(use_encoder_idx))
        ])

        # 添加后处理层
        self.post_process = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(len(use_encoder_idx))
        ])

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)

        # 输入投影
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # FSSA编码器处理
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                # 保存原始特征用于残差连接
                residual = proj_feats[enc_ind]

                # 预归一化
                x = self.pre_norm[i](proj_feats[enc_ind])

                # FSSA处理 - 直接传入2D特征图
                x = self.encoder[i](x)

                # 应用dropout
                x = self.dropout_layers[i](x)

                # 后处理
                x = self.post_process[i](x)

                # 残差连接
                proj_feats[enc_ind] = residual + x

        # FPN融合
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        # PAN融合
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        return outs