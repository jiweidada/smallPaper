"""Phase-mechanism controls for the C P2-to-P3 bridge (2026-10-12).

Separate from the locked C file (hgnetv2_light.py) and the existing low-pass
control (hgnetv2_light_control.py); both are left unchanged.

Training controls (new arms):
- single_phase: keep only one of the four 2x2 sub-cell positions, so P3 gets
  P2 content at stride 8 with no phase diversity. Capacity-matched to the
  existing low-pass control (reduce 64->16, expand 16->60, restore 60->256).

Inference-only ablations on a trained C checkpoint (no retraining; state_dict
keys are identical to HGNetv2_Light with use_p2_skip=True):
- none:    unchanged C
- mean:    replace the four phases by their mean (low-pass at test time)
- shuffle: fixed permutation of the four phases (same content, wrong position)
- zero:    remove the bridge residual entirely
"""
import torch
from torch import nn
from torch.nn import functional as F
from ..core import register
from .hgnetv2_light import HGNetv2_Light, P2ToP3Residual, conv_bn_relu

PHASE_EVAL_MODES = ('none', 'mean', 'shuffle', 'zero')
# Cyclic shift of (00, 01, 10, 11): every phase lands on a different position.
SHUFFLE_ORDER = (1, 2, 3, 0)


def _pad_even(x):
    h, w = x.shape[-2:]
    if h % 2 or w % 2:
        x = F.pad(x, (0, w % 2, 0, h % 2))
    return x


class P2ToP3PhaseAblation(P2ToP3Residual):
    """C bridge with a test-time phase ablation switch."""
    def __init__(self, p2_channels=64, p3_channels=256, width=16, phase_eval='none'):
        super().__init__(p2_channels, p3_channels, width)
        if phase_eval not in PHASE_EVAL_MODES:
            raise ValueError(f'unknown phase_eval={phase_eval}')
        self.width = width
        self.phase_eval = phase_eval

    def forward(self, p2, p3):
        if self.training and self.phase_eval != 'none':
            raise RuntimeError('phase_eval ablations are inference-only')
        if self.phase_eval == 'zero':
            return p3
        detail = F.pixel_unshuffle(_pad_even(self.reduce(p2)), 2)
        if detail.shape[-2:] != p3.shape[-2:]:
            raise ValueError('P2 spatial reorder and P3 sizes differ')
        if self.phase_eval != 'none':
            # pixel_unshuffle layout: channel = c * 4 + (2 * i + j)
            b, _, h, w = detail.shape
            detail = detail.view(b, self.width, 4, h, w)
            if self.phase_eval == 'mean':
                detail = detail.mean(dim=2, keepdim=True).expand(-1, -1, 4, -1, -1)
            else:
                detail = detail[:, :, list(SHUFFLE_ORDER)]
            detail = detail.reshape(b, self.width * 4, h, w)
        return p3 + self.restore(detail)


class P2ToP3SinglePhaseResidual(nn.Module):
    """Capacity-matched bridge that keeps only the top-left 2x2 sub-cell."""
    def __init__(self, p2_channels=64, p3_channels=256, width=16, expanded=60):
        super().__init__()
        self.reduce = conv_bn_relu(p2_channels, width)
        self.expand = nn.Conv2d(width, expanded, 1, bias=False)
        self.restore = nn.Conv2d(expanded, p3_channels, 1, bias=False)
        nn.init.zeros_(self.restore.weight)

    def forward(self, p2, p3):
        detail = _pad_even(self.reduce(p2))[:, :, 0::2, 0::2]
        if detail.shape[-2:] != p3.shape[-2:]:
            raise ValueError('P2 single phase and P3 sizes differ')
        return p3 + self.restore(self.expand(detail))


def _phase_mean(detail, width):
    """Replace the four 2x2 phases of each channel by their mean (layout c*4+phase)."""
    b, _, h, w = detail.shape
    d = detail.view(b, width, 4, h, w)
    return d.mean(dim=2, keepdim=True).expand(-1, -1, 4, -1, -1).reshape(b, width * 4, h, w)


class P2ToP3PhaseMeanResidual(P2ToP3Residual):
    """C-mean (trainable): C's exact modules and init; phases averaged before restore.

    Same parameters/state_dict/construction order as C; only access to
    within-cell differences is removed. Separate class so the inference-only
    guard of P2ToP3PhaseAblation stays intact.
    """
    def __init__(self, p2_channels=64, p3_channels=256, width=16):
        super().__init__(p2_channels, p3_channels, width)
        self.width = width

    def forward(self, p2, p3):
        detail = F.pixel_unshuffle(_pad_even(self.reduce(p2)), 2)
        if detail.shape[-2:] != p3.shape[-2:]:
            raise ValueError('P2 spatial reorder and P3 sizes differ')
        return p3 + self.restore(_phase_mean(detail, self.width))


class P2ToP3LowRankResidual(nn.Module):
    """rank16 bridge, same modules/init as code/p2_lowrank12_20261011_v3/lowrank_branch.py.

    phase_mean=True gives rank16-mean: identical modules and init, phases
    averaged after pixel_unshuffle and before mix.
    """
    def __init__(self, p2_channels=64, p3_channels=256, width=16, rank=16, phase_mean=False):
        super().__init__()
        self.width = width
        self.phase_mean = phase_mean
        self.reduce = conv_bn_relu(p2_channels, width)
        self.mix = nn.Conv2d(width * 4, rank, 1, bias=False)
        self.restore = nn.Conv2d(rank, p3_channels, 1, bias=False)
        nn.init.orthogonal_(self.mix.weight.view(rank, width * 4))
        nn.init.zeros_(self.restore.weight)

    def forward(self, p2, p3):
        detail = F.pixel_unshuffle(_pad_even(self.reduce(p2)), 2)
        if detail.shape[-2:] != p3.shape[-2:]:
            raise ValueError('P2/P3 spatial mismatch')
        if self.phase_mean:
            detail = _phase_mean(detail, self.width)
        return p3 + self.restore(self.mix(detail))


@register()
class HGNetv2_Light_Phase(HGNetv2_Light):
    """HGNetv2_Light with p2_skip_mode in
    {'c', 'single_phase', 'c_mean', 'rank16', 'rank16_mean'}.

    p2_phase_eval only applies to mode 'c' and only at inference.
    """
    def __init__(self, name, use_lab=False, return_idx=[1, 2, 3],
                 freeze_stem_only=True, freeze_at=-1, freeze_norm=False,
                 pretrained=False, agg='se', local_model_dir='weight/hgnetv2/',
                 use_detail=False, use_context=False, detail_width=8,
                 context_width=64, context_dilations=[1, 2, 3],
                 use_p2_skip=True, p2_skip_width=16, p2_skip_mode='c',
                 p2_expanded=60, p2_phase_eval='none'):
        valid_modes = ('c', 'single_phase', 'c_mean', 'rank16', 'rank16_mean')
        if p2_skip_mode not in valid_modes:
            raise ValueError(f'unknown p2_skip_mode={p2_skip_mode}')
        if p2_phase_eval not in PHASE_EVAL_MODES:
            raise ValueError(f'unknown p2_phase_eval={p2_phase_eval}')
        if p2_phase_eval != 'none' and (p2_skip_mode != 'c' or not use_p2_skip):
            raise ValueError('p2_phase_eval requires an enabled p2_skip_mode=c bridge')
        super().__init__(name=name, use_lab=use_lab, return_idx=return_idx,
                         freeze_stem_only=freeze_stem_only, freeze_at=freeze_at,
                         freeze_norm=freeze_norm, pretrained=pretrained, agg=agg,
                         local_model_dir=local_model_dir, use_detail=use_detail,
                         use_context=use_context, detail_width=detail_width,
                         context_width=context_width,
                         context_dilations=context_dilations,
                         use_p2_skip=False, p2_skip_width=p2_skip_width)
        c2, c3 = self._out_channels[0], self._out_channels[1]
        if not use_p2_skip:
            self.p2_skip = None
        elif p2_skip_mode == 'c':
            self.p2_skip = P2ToP3PhaseAblation(c2, c3, p2_skip_width, p2_phase_eval)
        elif p2_phase_eval != 'none':
            raise ValueError('p2_phase_eval requires p2_skip_mode=c')
        elif p2_skip_mode == 'single_phase':
            self.p2_skip = P2ToP3SinglePhaseResidual(c2, c3, p2_skip_width, p2_expanded)
        elif p2_skip_mode == 'c_mean':
            self.p2_skip = P2ToP3PhaseMeanResidual(c2, c3, p2_skip_width)
        elif p2_skip_mode in ('rank16', 'rank16_mean'):
            self.p2_skip = P2ToP3LowRankResidual(c2, c3, p2_skip_width, 16,
                                                 phase_mean=p2_skip_mode == 'rank16_mean')
        else:
            raise ValueError(f'unknown p2_skip_mode={p2_skip_mode}')
