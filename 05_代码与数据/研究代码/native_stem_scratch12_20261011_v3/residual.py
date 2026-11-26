"""Selected local geometry residual; zero output, no global RNG consumption."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def selected_patches(feature, batch_indices, local_ids):
    """Gather only selected 3x3 patches, row-major, with zero exterior padding."""
    if feature.ndim != 4 or batch_indices.shape != local_ids.shape:
        raise ValueError('Expected BCHW and matching index vectors')
    b, c, h, w = feature.shape
    if local_ids.ndim != 1 or local_ids.numel() and (
        local_ids.min() < 0 or local_ids.max() >= h*w or
        batch_indices.min() < 0 or batch_indices.max() >= b):
        raise ValueError('Selected patch index out of range')
    offset = torch.arange(-1, 2, device=local_ids.device)
    dy, dx = torch.meshgrid(offset, offset, indexing='ij')
    yy = local_ids[:, None] // w + dy.reshape(1, 9)
    xx = local_ids[:, None] % w + dx.reshape(1, 9)
    inside = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
    flat = feature.flatten(2).transpose(1, 2)
    ids = yy.clamp(0, h-1)*w + xx.clamp(0, w-1)
    values = flat[batch_indices[:, None], ids]
    return values.masked_fill(~inside[..., None], 0)


class LocalGeometryResidual(nn.Module):
    def __init__(self, channels=256, hidden=64, initialization_seed=0):
        super().__init__()
        self.channels = channels
        # Initialize fresh tensors before registering parameters. No checkpoint
        # mutation and no consumption of the native initialization RNG stream.
        generator = torch.Generator(device='cpu').manual_seed(initialization_seed)
        bound = 1/math.sqrt(2*channels)
        self.kernel = nn.Parameter(torch.full((9, channels), 1/9))
        self.mix_weight = nn.Parameter(torch.empty(hidden, 2*channels).uniform_(
            -bound, bound, generator=generator))
        self.mix_bias = nn.Parameter(torch.empty(hidden).uniform_(
            -bound, bound, generator=generator))
        self.output_weight = nn.Parameter(torch.zeros(4, hidden))
        self.output_bias = nn.Parameter(torch.zeros(4))

    def forward(self, semantic, feature, ids, valid_mask):
        if semantic.shape[:2] != ids.shape or semantic.shape[-1] != self.channels:
            raise ValueError('Selected semantic/query dimensions differ')
        if feature.shape[:2] != (ids.shape[0], self.channels):
            raise ValueError('Source feature batch/channel dimensions differ')
        p3_count = feature.shape[-2] * feature.shape[-1]
        bi, qi = torch.where(ids < p3_count)
        local_ids = ids[bi, qi]
        patches = selected_patches(feature, bi, local_ids)
        local = (patches * self.kernel[None]).sum(1)
        local = F.gelu(F.layer_norm(local, (self.channels,)))
        hidden = F.gelu(F.linear(torch.cat((semantic[bi, qi], local), -1),
                                 self.mix_weight, self.mix_bias))
        selected = F.linear(hidden, self.output_weight, self.output_bias)
        valid = valid_mask.expand(ids.shape[0], -1, -1)[bi, local_ids]
        selected = selected.masked_fill(~valid, 0)
        # Retain the computed delta dtype under autocast. Unselected P4/P5
        # entries remain exact zero; native anchor arithmetic is unchanged.
        delta = torch.zeros((*ids.shape, 4), device=semantic.device, dtype=selected.dtype)
        delta[bi, qi] = selected
        return delta
