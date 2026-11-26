"""Input-dependent P3 subband attention; Haar versus spatial-phase control.

Inspired by frequency-conditioned gating, not a reproduction of WPFormer WCA,
FDAM, or WTConv. Four bands share all gate parameters. Spatial phases refer to
the four positions of a 2x2 sampling block, not Fourier phase.
"""
import torch
from torch import nn
from torch.nn import functional as F


class P3BandAttention(nn.Module):
    identity_zero_parameters = ('local.weight', 'local.bias',
                                'global_expand.weight', 'global_expand.bias')

    def __init__(self, arm, channels=256, reduction=8, local_kernel=3):
        super().__init__()
        if arm not in ('wavelet', 'spatial'):
            raise ValueError(arm)
        if channels != 256 or reduction != 8 or local_kernel != 3:
            raise ValueError('Locked module contract is C256/reduction8/kernel3')
        self.arm = arm
        self.channels = channels
        # CPU-only initialization: restores the native CPU stream and never
        # calls torch.manual_seed (which would also change CUDA RNG streams).
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            self.local = nn.Conv2d(channels, channels, 3, padding=1,
                                   groups=channels, bias=True, device='cpu')
            self.global_reduce = nn.Conv2d(channels, channels // reduction, 1, device='cpu')
            self.global_expand = nn.Conv2d(channels // reduction, channels, 1, device='cpu')
            nn.init.zeros_(self.local.weight)
            nn.init.zeros_(self.local.bias)
            nn.init.zeros_(self.global_expand.weight)
            nn.init.zeros_(self.global_expand.bias)

    def decompose(self, x):
        """BCHW -> B,4,C,ceil(H/2),ceil(W/2); right/bottom zero padding."""
        if x.ndim != 4 or x.shape[1] != self.channels or min(x.shape[-2:]) < 1:
            raise ValueError('Expected nonempty B256HW')
        h, w = x.shape[-2:]
        x = F.pad(x.float(), (0, w % 2, 0, h % 2))
        b, c, hp, wp = x.shape
        phases = F.pixel_unshuffle(x, 2).reshape(b, c, 4, hp // 2, wp // 2)
        a, b0, c0, d = phases.unbind(2)
        if self.arm == 'spatial':
            bands = phases.permute(0, 2, 1, 3, 4).contiguous()
        else:
            bands = torch.stack(((a+b0+c0+d)*.5, (-a+b0-c0+d)*.5,
                                 (-a-b0+c0+d)*.5, (a-b0-c0+d)*.5), dim=1)
        return bands, (h, w)

    def reconstruct(self, bands, original_hw):
        """Inverse of the fixed orthonormal basis, then crop padding."""
        if bands.ndim != 5 or bands.shape[1:3] != (4, self.channels):
            raise ValueError('Expected B4C h w bands')
        bands = bands.float()
        b, _, c, h, w = bands.shape
        if self.arm == 'wavelet':
            ll, lh, hl, hh = bands.unbind(1)
            phases = torch.stack(((ll-lh-hl+hh)*.5, (ll+lh-hl-hh)*.5,
                                  (ll-lh+hl-hh)*.5, (ll+lh+hl+hh)*.5), dim=2)
        else:
            phases = bands.permute(0, 2, 1, 3, 4).contiguous()
        x = F.pixel_shuffle(phases.reshape(b, c*4, h, w), 2)
        oh, ow = original_hw
        if not (2*h-1 <= oh <= 2*h and 2*w-1 <= ow <= 2*w):
            raise ValueError('Invalid inverse crop')
        return x[:, :, :oh, :ow]

    @staticmethod
    def _conv(x, layer):
        # Explicit FP32 also supports an externally half-cast module without
        # introducing FP16 gate convolution or changing master-parameter policy.
        return F.conv2d(x, layer.weight.float(), layer.bias.float(),
                        padding=layer.padding, groups=layer.groups)

    def forward(self, x):
        if not x.is_floating_point():
            raise ValueError('Expected floating-point P3')
        with torch.autocast(device_type=x.device.type, enabled=False):
            bands, hw = self.decompose(x)
            b, _, c, h, w = bands.shape
            z = bands.reshape(b*4, c, h, w)
            local = self._conv(z, self.local)
            pooled = z.mean(dim=(-2, -1), keepdim=True)
            glob = self._conv(F.relu(self._conv(pooled, self.global_reduce)), self.global_expand)
            gate = 2 * torch.sigmoid(local + glob) - 1
            residual = self.reconstruct((z * gate).reshape_as(bands), hw)
        # Invert only the residual, not X itself: initial zero logits give
        # exactly X, without DWT/IDWT round-off on the native identity path.
        return x + residual.to(dtype=x.dtype)
