"""SOEP-inspired budgeted neck fusion; NOT a reproduction of full SOEP.

The unchanged HybridEncoder FPN/PAN forward is reproduced below with one
explicit insertion after input projection. Original DEIM/RT-DETR copyright
and repository license continue to apply to that encoder forward.
"""
import torch
from torch import nn
from torch.nn import functional as F
from ..core import register
from ..backbone.hgnetv2_light import conv_bn_relu
from .hybrid_encoder import HybridEncoder


class SOEPLiteFusion(nn.Module):
    def __init__(self, p2_channels=64, hidden_dim=256, width=16):
        super().__init__()
        spatial_channels = 4*p2_channels
        self.spatial = conv_bn_relu(spatial_channels, spatial_channels, 3, groups=spatial_channels)
        self.compress = conv_bn_relu(spatial_channels, width)
        self.mix = conv_bn_relu(hidden_dim+width, width)
        self.restore = nn.Conv2d(width, hidden_dim, 1, bias=False)
        nn.init.zeros_(self.restore.weight)

    def forward(self, p2, projected_p3):
        height, width = p2.shape[-2:]
        if height % 2 or width % 2:
            p2 = F.pad(p2, (0, width % 2, 0, height % 2))
        detail = self.compress(self.spatial(F.pixel_unshuffle(p2, 2)))
        if detail.shape[-2:] != projected_p3.shape[-2:]:
            raise ValueError('SOEP-lite P2 and projected P3 spatial sizes differ')
        mixed = self.mix(torch.cat([projected_p3, detail], dim=1))
        return projected_p3 + self.restore(mixed)


@register()
class HybridEncoder_SOEP_Lite(HybridEncoder):
    def __init__(self, in_channels=[256, 512, 1024], feat_strides=[8, 16, 32],
                 hidden_dim=256, nhead=8, dim_feedforward=1024, dropout=0.,
                 enc_act='gelu', use_encoder_idx=[2], num_encoder_layers=1,
                 pe_temperature=10000, expansion=.5, depth_mult=.34, act='silu',
                 eval_spatial_size=None, version='dfine', p2_channels=64, bridge_width=16):
        super().__init__(in_channels=in_channels, feat_strides=feat_strides,
                         hidden_dim=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                         dropout=dropout, enc_act=enc_act, use_encoder_idx=use_encoder_idx,
                         num_encoder_layers=num_encoder_layers, pe_temperature=pe_temperature,
                         expansion=expansion, depth_mult=depth_mult, act=act,
                         eval_spatial_size=eval_spatial_size, version=version)
        self.p2_fusion = SOEPLiteFusion(p2_channels, hidden_dim, bridge_width)

    def forward(self, feats):
        if len(feats) != len(self.in_channels)+1:
            raise ValueError('SOEP-lite requires backbone outputs P2/P3/P4/P5')
        p2, pyramid = feats[0], feats[1:]
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(pyramid)]
        proj_feats[0] = self.p2_fusion(p2, proj_feats[0])
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                height, width = proj_feats[enc_ind].shape[2:]
                flattened = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    position = self.build_2d_sincos_position_embedding(width, height, self.hidden_dim,
                                                                      self.pe_temperature).to(flattened.device)
                else:
                    position = getattr(self, f'pos_embed{enc_ind}').to(flattened.device)
                memory = self.encoder[i](flattened, pos_embed=position)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, height, width).contiguous()
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels)-1, 0, -1):
            high = self.lateral_convs[len(self.in_channels)-1-idx](inner_outs[0])
            low = proj_feats[idx-1]
            inner_outs[0] = high
            upsampled = F.interpolate(high, scale_factor=2., mode='nearest')
            fused = self.fpn_blocks[len(self.in_channels)-1-idx](torch.cat([upsampled, low], dim=1))
            inner_outs.insert(0, fused)
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels)-1):
            downsampled = self.downsample_convs[idx](outs[-1])
            outs.append(self.pan_blocks[idx](torch.cat([downsampled, inner_outs[idx+1]], dim=1)))
        return outs
