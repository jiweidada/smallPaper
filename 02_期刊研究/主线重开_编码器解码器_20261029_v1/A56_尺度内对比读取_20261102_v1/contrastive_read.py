"""A56 exploratory decoder residual; not a first differential-attention claim.

M: two globally normalized sample maps.
Q: match the negative map's mass to the positive map separately at each level.
The original zero-padded native read is always retained. The new residual uses
border sampling in both arms so that its levelwise offset property also holds
for sampling positions beyond an image edge. This is a property of a fixed-query
residual, not a background-purity assumption or full-detector invariance.
"""
import types
import torch
from torch import nn
from torch.nn import functional as F

NEW_SEED = 20261112


class ContrastiveRead(nn.Module):
    def __init__(self, dim, heads, points, arm):
        super().__init__()
        assert arm in ('M', 'Q')
        self.arm = arm
        self.heads = heads
        self.points = tuple(points)
        self.negative_logits = nn.Linear(dim, heads * sum(points))
        nn.init.normal_(self.negative_logits.weight, std=.02)
        nn.init.zeros_(self.negative_logits.bias)
        self.gain = nn.Parameter(torch.zeros(heads))

    def signed_weights(self, positive_logits, negative_logits):
        a = positive_logits.float().softmax(-1)
        if self.arm == 'M':
            b = negative_logits.float().softmax(-1)
        else:
            aa = a.split(self.points, -1)
            zz = negative_logits.float().split(self.points, -1)
            b = torch.cat([x.sum(-1, keepdim=True) * z.softmax(-1)
                           for x, z in zip(aa, zz)], -1)
        return a - b

    def residual(self, query, native_attention, reference_points, values, shapes):
        assert native_attention.method == 'default'
        assert reference_points.shape[-1] == 4
        bs, nq, _ = query.shape
        heads = self.heads
        positive = native_attention.attention_weights(query).reshape(bs, nq, heads, -1)
        negative = self.negative_logits(query).reshape(bs, nq, heads, -1)
        offsets = native_attention.sampling_offsets(query).reshape(bs, nq, heads, -1, 2)
        scale = native_attention.num_points_scale.to(query.dtype).unsqueeze(-1)
        locations = (reference_points[:, :, None, :, :2] + offsets * scale *
                     reference_points[:, :, None, :, 2:] * native_attention.offset_scale)
        # Native output still uses the unmodified author's core and zero padding.
        with torch.autocast(device_type=query.device.type, enabled=False):
            signed = self.signed_weights(positive, negative)
            weights = signed.permute(0, 2, 1, 3).reshape(bs * heads, 1, nq, -1)
            grids = (2 * locations.float() - 1).permute(0, 2, 1, 3, 4).flatten(0, 1)
            sampled = []
            for val, (h, w), grid in zip(values, shapes, grids.split(self.points, -2)):
                channels = val.shape[-2]
                val = val.float().reshape(bs * heads, channels, h, w)
                sampled.append(F.grid_sample(val, grid, mode='bilinear',
                                             padding_mode='border', align_corners=False))
            r = (torch.cat(sampled, -1) * weights).sum(-1)
            channels = r.shape[1]
            r = r.reshape(bs, heads, channels, nq)
            r = r * self.gain.float().tanh()[None, :, None, None]
            return r.reshape(bs, heads * channels, nq).permute(0, 2, 1)


def install(model, arm):
    assert arm in ('M', 'Q')
    installed = []
    for index, layer in enumerate(model.decoder.decoder.layers):
        attn = layer.cross_attn
        assert not hasattr(attn, 'a56_contrast')
        # No alteration of the native RNG stream or any native tensor.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(NEW_SEED + index)
            branch = ContrastiveRead(attn.embed_dim, attn.num_heads,
                                    attn.num_points_list, arm)
        attn.add_module('a56_contrast', branch)
        original_forward = attn.forward

        def make_forward(original):
            def forward(this, query, reference_points, value, value_spatial_shapes):
                native = original(query, reference_points, value, value_spatial_shapes)
                residual = this.a56_contrast.residual(query, this, reference_points,
                                                     value, value_spatial_shapes)
                return native + residual.to(native.dtype)
            return forward

        attn.forward = types.MethodType(make_forward(original_forward), attn)
        installed.append(branch)
    assert len(installed) == 4
    return installed


def diagnostics(branches):
    return {'gains': [b.gain.detach().float().tanh().cpu().tolist() for b in branches],
            'negative_weight_norms': [float(b.negative_logits.weight.detach().float().norm())
                                      for b in branches]}
