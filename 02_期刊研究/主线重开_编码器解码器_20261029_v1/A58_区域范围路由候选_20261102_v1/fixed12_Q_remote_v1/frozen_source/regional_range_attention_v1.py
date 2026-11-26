"""Fixed-budget regional attention; interval routing is a known primitive.

M uses an exact Quest-style key interval score for the mean query.
Q bounds every fine query/key pair in the two spatial regions. That bound
is not a guarantee of attention mass or detection quality.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

COUNTS = {'module_forwards': 0, 'M_route_calls': 0, 'Q_route_calls': 0}


def regional_stats(tokens, valid):
    """tokens[B,H,R,T,D], valid[R,T]; padding never enters extrema/means."""
    mask = valid[None, None, :, :, None]
    counts = valid.sum(-1).clamp_min(1)[None, None, :, None]
    values = tokens.detach().float() if tokens.dtype != torch.float64 else tokens.detach()
    mean = torch.where(mask, values, 0).sum(-2) / counts
    low = torch.where(mask, values, float('inf')).amin(-2)
    high = torch.where(mask, values, -float('inf')).amax(-2)
    nonempty = valid.any(-1)[None, None, :, None]
    return mean, torch.where(nonempty, low, 0), torch.where(nonempty, high, 0)


@torch.no_grad()
def mean_query_key_interval(q_mean, k_low, k_high):
    COUNTS['M_route_calls'] += 1
    # The score is for a point query and a key box. No clamp-to-zero on K.
    with torch.autocast(device_type=q_mean.device.type, enabled=False):
        return (q_mean.clamp_min(0) @ k_high.transpose(-2, -1)
                + q_mean.clamp_max(0) @ k_low.transpose(-2, -1)) / math.sqrt(q_mean.shape[-1])


@torch.no_grad()
def two_sided_interval(q_low, q_high, k_low, k_high, chunk=32):
    COUNTS['Q_route_calls'] += 1
    pieces = []
    with torch.autocast(device_type=q_low.device.type, enabled=False):
        kl, kh = k_low.unsqueeze(-3), k_high.unsqueeze(-3)
        for first in range(0, q_low.shape[-2], chunk):
            ql = q_low[..., first:first + chunk, :].unsqueeze(-2)
            qh = q_high[..., first:first + chunk, :].unsqueeze(-2)
            a = torch.maximum(ql * kl, ql * kh)
            b = torch.maximum(qh * kl, qh * kh)
            pieces.append(torch.maximum(a, b).sum(-1))
    return torch.cat(pieces, dim=-2) / math.sqrt(q_low.shape[-1])


def pack_regions(x, heads, side):
    b, c, h, w = x.shape
    assert c % heads == 0 and h > 0 and w > 0
    nh, nw = (h + side - 1) // side, (w + side - 1) // side
    padded = F.pad(x, (0, nw * side - w, 0, nh * side - h))
    tokens = padded.reshape(b, heads, c // heads, nh, side, nw, side)
    tokens = tokens.permute(0, 1, 3, 5, 4, 6, 2).reshape(b, heads, nh * nw, side * side, c // heads)
    valid = torch.ones((h, w), device=x.device, dtype=torch.bool)
    valid = F.pad(valid, (0, nw * side - w, 0, nh * side - h), value=False)
    valid = valid.reshape(nh, side, nw, side).permute(0, 2, 1, 3).reshape(nh * nw, side * side)
    return tokens, valid, (h, w, nh, nw)


def unpack_regions(tokens, layout, side):
    h, w, nh, nw = layout
    b, heads, _, _, d = tokens.shape
    x = tokens.reshape(b, heads, nh, nw, side, side, d)
    return x.permute(0, 1, 6, 2, 4, 3, 5).reshape(b, heads * d, nh * side, nw * side)[:, :, :h, :w].contiguous()


def attend_regions(q, k, v, valid, indices):
    b, heads, regions, count, dim = q.shape
    bi = torch.arange(b, device=q.device)[:, None, None, None]
    hi = torch.arange(heads, device=q.device)[None, :, None, None]
    selected_k = k[bi, hi, indices].reshape(b, heads, regions, -1, dim)
    selected_v = v[bi, hi, indices].reshape(b, heads, regions, -1, dim)
    selected_valid = valid[indices].reshape(b, heads, regions, -1)
    # Each spatial region has at least one valid image token, including borders.
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=q.device.type, enabled=False):
        logits = q.to(dtype) @ selected_k.to(dtype).transpose(-2, -1) / math.sqrt(dim)
        logits = logits.masked_fill(~selected_valid.unsqueeze(-2), -float('inf'))
        output = logits.softmax(-1) @ selected_v.to(dtype)
    return output.to(v.dtype) * valid[None, None, :, :, None]


class RegionalRangeContext(nn.Module):
    def __init__(self, channels=192, heads=3, side=4, topk=4, mode='B', chunk=32):
        super().__init__()
        assert mode in ['B', 'M', 'Q'] and channels % heads == 0
        self.channels, self.heads, self.side, self.topk = channels, heads, side, topk
        self.mode, self.chunk = mode, chunk
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        COUNTS['module_forwards'] += 1
        if self.mode == 'B':
            return x
        z = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        q0, k0, v0 = self.qkv(z).chunk(3, 1)
        q, valid, layout = pack_regions(q0, self.heads, self.side)
        k, _, _ = pack_regions(k0, self.heads, self.side)
        v, _, _ = pack_regions(v0, self.heads, self.side)
        with torch.no_grad():
            qm, ql, qh = regional_stats(q, valid)
            _, kl, kh = regional_stats(k, valid)
            scores = mean_query_key_interval(qm, kl, kh) if self.mode == 'M' else two_sided_interval(ql, qh, kl, kh, self.chunk)
            scores = scores.masked_fill(~valid.any(-1)[None, None, None, :], -float('inf'))
            indices = scores.topk(min(self.topk, scores.shape[-1]), dim=-1, sorted=False).indices
        selected = attend_regions(q, k, v, valid, indices)
        context = self.proj(unpack_regions(selected, layout, self.side))
        return x + self.gate.to(x.dtype) * context.to(x.dtype)


def install(model, mode):
    encoder = model.encoder
    assert not hasattr(encoder, 'a58_context') and encoder.hidden_dim == 192
    # Module construction must not change the pre-forward native RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20261102)
        encoder.add_module('a58_context', RegionalRangeContext(mode=mode))
    original = encoder.forward

    def wrapped(features):
        outs = original(features)
        assert isinstance(outs, list) and len(outs) == 3 and outs[0].shape[1] == 192
        return [encoder.a58_context(outs[0]), *outs[1:]]

    encoder.forward = wrapped
    return {'mode': mode, 'site': 'encoder native highest-resolution output after fusion',
            'extra_parameters': sum(p.numel() for p in encoder.a58_context.parameters()),
            'native_encoder_source_modified': False}
