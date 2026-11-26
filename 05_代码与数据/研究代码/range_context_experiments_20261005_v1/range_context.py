"""Per-edge range feedback for local evidence reading, before detection.

The active controls have identical weights and read budgets. The adaptive
geometry retains the physical mean and interdecile span, not quantile skew.
No additional learned offsets, confidence correction or GT input are used.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

VARIANTS = {
    'independent': ('disabled', 'disabled'),
    'fixed_range_global_context': ('fixed', 'global'),
    'adaptive_range_global_context': ('adaptive', 'global'),
    'adaptive_range_local_context': ('adaptive', 'local'),
}


@torch.no_grad()
def edge_geometry(corners, initial, support, reg_scale, geometry='adaptive'):
    """Coordinates are normalized to the native crop, ordered L,T,R,B.

Sort *physical* coordinates before the CDF: L/T reverse the native support.
The mean box is always computed from the original full distribution, including
the fixed-range control. Quantile collapse is preserved and measurable.
"""
    if geometry not in ('fixed', 'adaptive'):
        raise ValueError(geometry)
    if initial.shape != (*corners.shape[:2], 4) or corners.shape[-1] != 4 * support.numel():
        raise ValueError('Expected corners[B,Q,4*K], initial[B,Q,4], support[K]')
    with torch.autocast(device_type=corners.device.type, enabled=False):
        p = corners.detach().float().reshape(*corners.shape[:2], 4, -1).softmax(-1)
        ref = initial.detach().float()
        project = support.detach().float().reshape(-1)
        scale = torch.as_tensor(reg_scale, device=p.device).detach().float().abs()
        centre = ref[..., [0, 1, 0, 1]]
        side = ref[..., [2, 3, 2, 3]]
        sign = p.new_tensor([-1, -1, 1, 1])
        coordinates = centre[..., None] + sign[:, None] * (.5 * scale + project) * (side[..., None] / scale)
        # Use the native FDR operation order for the mean. Averaging large
        # physical coordinates can lose precision on very narrow boxes.
        mean_distance = (p * project).sum(-1)
        mean = centre + sign * (.5 * scale + mean_distance) * (side / scale)
        order = coordinates.argsort(-1)
        ordered_coordinates = coordinates.gather(-1, order)
        cdf = p.gather(-1, order).cumsum(-1).contiguous()
        levels = p.new_tensor([.1, .5, .9]).expand(*p.shape[:-1], 3).contiguous()
        indices = torch.searchsorted(cdf, levels).clamp_max(project.numel() - 1)
        quantiles = ordered_coordinates.gather(-1, indices)
        adaptive_radius = (quantiles[..., 2] - quantiles[..., 0]) * .5
        radius = adaptive_radius if geometry == 'adaptive' else .25 * side
        selected = mean[..., None] + radius[..., None] * p.new_tensor([-1., 0., 1.])
        box = torch.cat(((mean[..., :2] + mean[..., 2:]) * .5,
                         mean[..., 2:] - mean[..., :2]), -1)
        points = box[..., None, None, :2].expand(*box.shape[:2], 4, 3, 2).clone()
        points[..., [0, 2], :, 0] = selected[..., [0, 2], :]
        points[..., [1, 3], :, 1] = selected[..., [1, 3], :]
        return points.flatten(2, 3), box, {
            'mean': mean, 'quantiles': quantiles, 'radius': radius,
            'adaptive_radius': adaptive_radius, 'selected': selected,
        }


def original_coordinates(points, crop):
    return crop[:, None, None, :2] + points * (crop[:, None, None, 2:] - crop[:, None, None, :2])


def valid_points(points):
    return torch.isfinite(points).all(-1) & ((points >= 0) & (points <= 1)).all(-1)


class RangeContext(nn.Module):
    def __init__(self, variant='adaptive_range_global_context', dim=256, width=128, heads=4):
        super().__init__()
        if variant not in VARIANTS or width % heads:
            raise ValueError((variant, width, heads))
        self.variant, self.width, self.heads = variant, width, heads
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, width, bias=False)
        self.context_kv = nn.Conv2d(dim, width * 2, 1, bias=False)
        self.local_kv = nn.Conv2d(dim, width * 2, 1, bias=False)
        self.position = nn.Linear(2, width, bias=False)
        self.context_to_query = nn.Linear(width, width, bias=False)
        self.restore = nn.Linear(width, dim, bias=False)
        nn.init.zeros_(self.restore.weight)
        if variant == 'independent':
            self.requires_grad_(False)

    def prepare_context(self, feature):
        return self.context_kv(feature)

    def prepare_local(self, feature):
        return self.local_kv(feature)

    def read(self, memory, points, valid, query, relative):
        b, q, n, _ = points.shape
        # Invalid NaN/Inf coordinates must not create NaN values multiplied by
        # zero attention. They use a finite dummy grid and explicit zero values.
        safe_points = torch.where(valid[..., None], points, torch.zeros_like(points))
        safe_relative = torch.where(valid[..., None], relative, torch.zeros_like(relative))
        sampled = F.grid_sample(memory.float(), safe_points.reshape(b, q * n, 1, 2) * 2 - 1,
                                align_corners=False, padding_mode='zeros')
        sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, q, n, 2 * self.width)
        sampled = torch.where(valid[..., None], sampled, torch.zeros_like(sampled))
        k, v = sampled.chunk(2, -1)
        k = k + self.position(safe_relative)
        depth = self.width // self.heads
        k = k.reshape(b, q, n, self.heads, depth)
        v = v.reshape(b, q, n, self.heads, depth)
        scores = (k * query.reshape(b, q, 1, self.heads, depth)).sum(-1) / math.sqrt(depth)
        scores = scores.masked_fill(~valid[..., None], float('-inf'))
        # One fixed zero null token per query: all invalid -> exactly zero.
        scores = torch.cat((scores, scores.new_zeros(b, q, 1, self.heads)), 2)
        weights = scores.softmax(2)[:, :, :n]
        return (v * weights[..., None]).sum(2).flatten(-2)

    def sampling_geometry(self, corners, initial, support, reg_scale, crop, global_context_hw):
        geometry, context_source = VARIANTS[self.variant]
        if geometry == 'disabled':
            raise RuntimeError('Independent variant does not execute the reader')
        points, box, stats = edge_geometry(corners, initial, support, reg_scale, geometry)
        crop = crop.detach().float()
        if crop.ndim != 2 or crop.shape != (points.shape[0], 4):
            raise ValueError('Expected crop[B,4] in normalized original xyxy')
        span = crop[:, 2:] - crop[:, :2]
        if not torch.isfinite(crop).all() or not (span > 0).all():
            raise ValueError('Nonfinite or inverted crop coordinates')
        h, w = global_context_hw
        # Same physical context positions in global/local controls. Local
        # control retains the global P4 cell radius rather than changing it.
        global_cell_in_crop = crop.new_tensor([1. / w, 1. / h]) / span
        radius = torch.maximum(box[..., 2:].abs() * .5, global_cell_in_crop[:, None])
        pattern = crop.new_tensor([[-1, -1], [1, -1], [-1, 1], [1, 1]])
        context_crop = box[..., None, :2] + radius[..., None, :] * pattern
        context_global = original_coordinates(context_crop, crop)
        if context_source == 'global':
            context_points = context_global
            context_valid = valid_points(context_global)
        else:
            context_points = context_crop
            context_valid = valid_points(context_crop) & valid_points(context_global)
        local_valid = valid_points(points) & valid_points(original_coordinates(points, crop))
        return {
            **stats, 'box': box, 'local_points': points, 'local_valid': local_valid,
            'context_crop': context_crop, 'context_global': context_global,
            'context_points': context_points, 'context_valid': context_valid,
            'context_relative': (context_crop - box[..., None, :2]) / radius[..., None, :].clamp_min(1e-8),
            'local_relative': (points - box[..., None, :2]) / initial.detach().float()[..., None, 2:].clamp_min(1e-8),
        }

    def forward(self, query, corners, initial, support, reg_scale, crop, context, local, global_context_hw):
        if self.variant == 'independent':
            return query
        dtype = query.dtype
        with torch.autocast(device_type=query.device.type, enabled=False):
            geometry = self.sampling_geometry(corners, initial, support, reg_scale, crop, global_context_hw)
            q = self.query(self.norm(query.float()))
            ctx = self.read(context, geometry['context_points'], geometry['context_valid'], q,
                            geometry['context_relative'])
            # Global evidence changes only local reading weights; there is no
            # context-only value residual and no output confidence correction.
            enhanced_q = q + self.context_to_query(ctx)
            detail = self.read(local, geometry['local_points'], geometry['local_valid'], enhanced_q,
                               geometry['local_relative'])
            return query + self.restore(detail).to(dtype)
