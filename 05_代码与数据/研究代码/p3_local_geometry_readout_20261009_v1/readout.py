"""Box-relative P3 readout. This is a new hypothesis, not fine_residual v5."""
import torch
from torch import nn
from torch.nn import functional as F


def sample_locations(z0, arm):
    if arm not in ('center', 'spatial'):
        raise ValueError(arm)
    box = z0.detach().sigmoid()
    axis = box.new_tensor([-0.75, 0., 0.75])
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    offsets = torch.stack((xx, yy), -1).reshape(9, 2)
    if arm == 'center':
        offsets = torch.zeros_like(offsets)
    return box[..., None, :2] + box[..., None, 2:] * offsets


def sample_nine(feature, locations):
    # [B,Q,9,2] normalized xy -> [B,Q,9,C], preserving row-major order.
    if locations.shape[1] == 0:
        return feature.new_empty((feature.shape[0], 0, 9, feature.shape[1]))
    values = F.grid_sample(feature, 2 * locations - 1, mode='bilinear',
                           padding_mode='border', align_corners=False)
    return values.permute(0, 2, 3, 1)


@torch.no_grad()
def sampling_diagnostics(z0, arm, height, width):
    """Optional observer; never changes samples or runs in the normal forward.

    Border clamping begins at the outer pixel centers for align_corners=False.
    Duplicate counts refer to clamped coordinates, not feature independence.
    """
    xy = sample_locations(z0, arm)
    low = xy.new_tensor([.5/width, .5/height])
    high = 1-low
    clamped = torch.maximum(low, torch.minimum(high, xy))
    same = (clamped[..., :, None, :] == clamped[..., None, :, :]).all(-1)
    prior = torch.ones(9,9,device=xy.device,dtype=torch.bool).tril(-1)
    return {
        'outside_normalized_image_points': ((xy < 0) | (xy > 1)).any(-1).sum(-1),
        'border_clamped_points': (clamped != xy).any(-1).sum(-1),
        'duplicate_after_clamp_points': (same & prior).any(-1).sum(-1),
    }


class P3LocalReadout(nn.Module):
    def __init__(self, arm, initialization_seed=0):
        super().__init__()
        if arm not in ('center', 'spatial'):
            raise ValueError(arm)
        self.arm = arm
        # CPU-only fork: no CUDA state initialization and no change to the common
        # model/data RNG stream. Only newly created parameters are initialized.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(initialization_seed)
            self.projection = nn.Conv2d(256, 32, 1, bias=True, device='cpu')
            self.hidden = nn.Linear(544, 128, device='cpu')
            self.output = nn.Linear(128, 4, device='cpu')
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, semantic, p3, z0):
        if p3.ndim != 4 or p3.shape[:2] != (semantic.shape[0], 256):
            raise ValueError('Expected projected P3 [B,256,H,W]')
        if semantic.shape != (*z0.shape[:2], 256) or z0.shape[-1] != 4:
            raise ValueError('Expected selected semantic [B,Q,256] and logits [B,Q,4]')
        local = sample_nine(self.projection(p3), sample_locations(z0, self.arm))
        joined = torch.cat((semantic, local.flatten(2)), dim=-1)
        return self.output(F.silu(self.hidden(joined)))
