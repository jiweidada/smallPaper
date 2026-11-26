"""Query-conditioned crop-context prototype. Geometry is not semantic confidence."""
import math
import torch
from torch import nn


def area(box):
    return (box[..., 2:] - box[..., :2]).clamp_min(0).prod(-1)


def intersection(a, b):
    return torch.cat((torch.maximum(a[..., :2], b[..., :2]),
                      torch.minimum(a[..., 2:], b[..., 2:])), -1)


def grid_boxes(hw, device):
    h, w = hw
    y, x = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                          torch.arange(w, device=device, dtype=torch.float32), indexing='ij')
    return torch.stack((x / w, y / h, (x + 1) / w, (y + 1) / h), -1).reshape(-1, 4)


def position_encoding(xy, dim):
    if dim % 4:
        raise ValueError('Position encoding requires a multiple of four channels')
    freq = 10000. ** (-torch.arange(dim // 4, device=xy.device, dtype=torch.float32) / (dim // 4))
    angle = xy.float()[..., None] * (2 * math.pi) * freq
    return torch.cat((angle.sin(), angle.cos()), -1).flatten(-2)


@torch.no_grad()
def support_geometry(ref_boxes, crop_norm, cells, mode, tile_size=640,
                     scale=4., min_support=256., max_support=640.):
    """All boxes are original-normalized xyxy except local reference cxcywh.

    Original W/H are recovered ONLY from the fixed native-640 crop contract:
    crop_span=(640/W,640/H). No resized-crop recipe is supported.
    Cell overlap describes feature locations, not their actual receptive fields.
    """
    ref, crop = ref_boxes.detach().float(), crop_norm.detach().float()
    if ref.ndim != 3 or ref.shape[-1] != 4 or crop.shape != (ref.shape[0], 4):
        raise ValueError('Expected references [B,Q,4] and crop_norm [B,4]')
    span = crop[:, 2:] - crop[:, :2]
    if not torch.isfinite(crop).all() or not (span > 0).all():
        raise ValueError('Invalid crop geometry')
    if not ((crop[:, :2] >= 0) & (crop[:, :2] < 1)).all():
        raise ValueError('Only crops anchored within the original image are supported')
    if not torch.isfinite(ref).all() or not (ref[..., 2:] >= 0).all():
        raise ValueError('Nonfinite or negative reference box')
    center = crop[:, None, :2] + ref[..., :2] * span[:, None]
    original_wh = tile_size / span
    ref_wh_pixels = ref[..., 2:] * tile_size
    support_wh = (scale * ref_wh_pixels).clamp(min_support, max_support)
    radius = support_wh / original_wh[:, None] / 2
    region = torch.cat((center - radius, center + radius), -1).clamp(0, 1)
    visible = crop.clamp(0, 1)[:, None]
    inside = intersection(region, visible)
    region_area = area(region)
    gap_area = (region_area - area(inside)).clamp_min(0)
    # A reference located in right/bottom padding must not acquire real context.
    valid = ((ref[..., :2] >= 0) & (ref[..., :2] < 1) &
             (center >= 0) & (center < 1)).all(-1) & (region_area > 0)
    rho = (gap_area / region_area.clamp_min(1e-12)).clamp(0, 1) * valid
    cell = cells[None, None]
    cell_area = area(cells)[None, None]
    overlap_region = area(intersection(region[:, :, None], cell))
    if mode in ('gap', 'gap_unit'):
        overlap_visible = area(intersection(inside[:, :, None], cell))
        overlap = (overlap_region - overlap_visible).clamp_min(0)
        gate = rho if mode == 'gap' else (gap_area > 0).float() * valid
    elif mode == 'support':
        # Same query window and gate, but do not subtract the visible crop.
        overlap, gate = overlap_region, rho
    elif mode == 'global':
        overlap, gate = cell_area.expand(ref.shape[0], ref.shape[1], -1), valid.float()
    elif mode == 'local':
        # Local-memory control: mapped local feature cells, valid original pixels only.
        overlap = area(intersection(visible[:, :, None], cell))
        gate = valid.float()
    else:
        raise ValueError(mode)
    fraction = (overlap / cell_area.clamp_min(1e-12)).clamp(0, 1) * valid[..., None]
    return {'fraction': fraction, 'gate': gate, 'rho': rho, 'valid': valid,
            'center': center, 'region': region, 'gap_area': gap_area,
            'original_wh': original_wh}


class QueryContext(nn.Module):
    def __init__(self, dim=256, attention_dim=128, heads=4, scale=4.,
                 min_support=256., max_support=640.):
        super().__init__()
        if attention_dim % heads:
            raise ValueError('Attention width must be divisible by heads')
        self.dim, self.width, self.heads = dim, attention_dim, heads
        self.geometry_kwargs = dict(scale=scale, min_support=min_support, max_support=max_support)
        self.query_norm, self.memory_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.query = nn.Linear(dim, attention_dim, bias=False)
        self.key = nn.Linear(dim, attention_dim, bias=False)
        self.value = nn.Linear(dim, attention_dim, bias=False)
        self.output = nn.Linear(attention_dim, dim, bias=False)
        nn.init.zeros_(self.output.weight)
        self.trace_enabled = False
        self.last_trace = None

    def prepare_memory(self, feature, crop_norm=None):
        """Project once per original at eval; retain autograd in paired training."""
        b, c, h, w = feature.shape
        if c != self.dim:
            raise ValueError('Expected the encoded P4 channel width')
        cells = grid_boxes((h, w), feature.device)
        cells = cells[None].expand(b, -1, -1)
        if crop_norm is not None:
            crop = crop_norm.detach().float()
            cells = crop[:, None, :2].repeat(1, 1, 2) + cells * (crop[:, None, 2:] - crop[:, None, :2]).repeat(1, 1, 2)
        centers = (cells[..., :2] + cells[..., 2:]) / 2
        tokens = self.memory_norm(feature.flatten(2).transpose(1, 2))
        k = self.key(tokens + position_encoding(centers, c).to(tokens.dtype))
        v = self.value(tokens)
        d = self.width // self.heads
        return {'key': k.view(b, h*w, self.heads, d).transpose(1, 2),
                'value': v.view(b, h*w, self.heads, d).transpose(1, 2), 'cells': cells}

    def forward(self, query, references, crop_norm, memory, mode='gap'):
        b, n, _ = query.shape
        if memory['key'].shape[0] != b:
            raise ValueError('Context batch must match paired/cached local batch')
        # Different mapped local grids per batch are handled explicitly. Global
        # grids coincide; using the same path keeps geometry comparable.
        with torch.autocast(device_type=query.device.type, enabled=False):
            rows = [support_geometry(references[i:i+1], crop_norm[i:i+1], memory['cells'][i],
                                     mode, **self.geometry_kwargs) for i in range(b)]
            g = {key: torch.cat([row[key] for row in rows], 0) for key in rows[0]}
        q = self.query(self.query_norm(query) + position_encoding(g['center'], self.dim).to(query.dtype))
        q = q.view(b, n, self.heads, self.width // self.heads).transpose(1, 2)
        # Geometry/log-softmax/matmul accumulation stay FP32 under AMP. A fixed
        # zero-key/zero-value option prevents all-masked softmax rows and permits
        # rejection of irrelevant context. No new classifier or confidence head.
        with torch.autocast(device_type=query.device.type, enabled=False):
            logits = q.float() @ memory['key'].float().transpose(-1, -2) / math.sqrt(q.shape[-1])
            fraction = g['fraction'][:, None]
            prior = fraction.clamp_min(1e-30).log().masked_fill(fraction == 0, -torch.inf)
            logits = logits + prior
            weights = torch.cat((logits, logits.new_zeros(b, self.heads, n, 1)), -1).softmax(-1)
            context = weights[..., :-1] @ memory['value'].float()
        context = context.transpose(1, 2).reshape(b, n, self.width).to(query.dtype)
        residual = self.output(context) * g['gate'][..., None].to(query.dtype)
        result = query + residual
        if self.trace_enabled:
            self.last_trace = {k: g[k].detach().cpu() for k in ('rho', 'gate', 'valid', 'center')}
            self.last_trace.update(nonzero_cells=(g['fraction'] > 0).sum(-1).detach().cpu(),
                                   null_weight=weights[..., -1].mean(1).detach().cpu(),
                                   residual_norm=residual.float().norm(dim=-1).detach().cpu())
        return result
