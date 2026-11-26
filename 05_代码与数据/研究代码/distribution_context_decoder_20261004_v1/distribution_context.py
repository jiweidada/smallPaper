"""Detached FDR geometry -> context-conditioned local feature read.

Four shared-across-head context positions and twelve local positions per query.
No targets, class scores, cross-query aggregation, or post-detection calibration.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def two_bin(prob, support):
    mean = (prob * support).sum(-1)
    j = (torch.searchsorted(support.contiguous(), mean.contiguous(), right=True)-1).clamp(0, len(support)-2)
    alpha = ((mean-support[j])/(support[j+1]-support[j])).clamp(0, 1)
    out = torch.zeros_like(prob)
    out.scatter_(-1, j[..., None], (1-alpha)[..., None])
    out.scatter_add_(-1, (j+1)[..., None], alpha[..., None])
    return out


@torch.no_grad()
def boundary_points(corners, initial, support, reg_scale, geometry):
    """Returns B,Q,12,2 normalized crop coordinates and FDR mean cxcywh."""
    with torch.autocast(device_type=corners.device.type, enabled=False):
        p = corners.detach().float().reshape(*corners.shape[:2], 4, -1).softmax(-1)
        support, initial = support.detach().float(), initial.detach().float()
        scale = torch.as_tensor(reg_scale, device=initial.device).detach().float()
        centre = initial[..., [0, 1, 0, 1]]
        size = initial[..., [2, 3, 2, 3]]
        signs = initial.new_tensor([-1, -1, 1, 1])
        edges = centre[..., None] + signs[:, None] * (.5*scale+support) * size[..., None]/scale
        means = (p*edges).sum(-1)
        box = torch.cat(((means[..., :2]+means[..., 2:])/2, means[..., 2:]-means[..., :2]), -1)
        if geometry == 'two_bin':
            p = two_bin(p, support)
        if geometry == 'fixed':
            selected = means[..., None] + size[..., None]*initial.new_tensor([-.25, 0., .25])
        else:
            order = edges.argsort(-1)
            cdf = p.gather(-1, order).cumsum(-1).contiguous()
            levels = p.new_tensor([.1, .5, .9]).expand(*cdf.shape[:-1], 3).contiguous()
            ix = torch.searchsorted(cdf, levels).clamp_max(p.shape[-1]-1)
            selected = edges.gather(-1, order.gather(-1, ix))
        points = box[..., None, None, :2].expand(-1, -1, 4, 3, -1).clone()
        points[..., [0, 2], :, 0] = selected[..., [0, 2], :]
        points[..., [1, 3], :, 1] = selected[..., [1, 3], :]
        return points.flatten(2, 3), box


def original_coordinates(points, crop):
    return crop[:, None, None, :2] + points*(crop[:, None, None, 2:]-crop[:, None, None, :2])


def valid_points(points):
    return torch.isfinite(points).all(-1) & ((points >= 0) & (points <= 1)).all(-1)


class DistributionContext(nn.Module):
    def __init__(self, arm, dim=256, width=128, heads=4):
        super().__init__()
        if arm not in ('A', 'B', 'C', 'D', 'E'):
            raise ValueError(arm)
        self.arm, self.width, self.heads = arm, width, heads
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, width, bias=False)
        self.context_kv = nn.Conv2d(dim, width*2, 1, bias=False)
        self.local_kv = nn.Conv2d(dim, width*2, 1, bias=False)
        self.position = nn.Linear(2, width, bias=False)
        # Identical learned offsets in every active arm; B has fixed mean-edge
        # bases, D/C full quantile bases, E equal-mean two-bin quantile bases.
        self.offset = nn.Linear(dim, (4+12)*2)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.context_to_query = nn.Linear(width, width, bias=False)
        self.restore = nn.Linear(width, dim, bias=False)
        nn.init.zeros_(self.restore.weight)
        if arm == 'A':
            self.requires_grad_(False)

    def prepare_context(self, feature):
        return self.context_kv(feature)

    def prepare_local(self, feature):
        return self.local_kv(feature)

    def read(self, memory, points, valid, query, relative):
        # FP32 interpolation and attention avoid half precision grid/softmax
        # issues. Conversion is differentiable for feature/projection weights.
        b, q, n, _ = points.shape
        sampled = F.grid_sample(memory.float(), points.reshape(b, q*n, 1, 2)*2-1,
                                align_corners=False, padding_mode='zeros')
        sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, q, n, 2*self.width)
        k, v = sampled.chunk(2, -1)
        k = k + self.position(relative)
        depth = self.width//self.heads
        k = k.reshape(b, q, n, self.heads, depth)
        v = v.reshape(b, q, n, self.heads, depth)
        scores = (k*query.reshape(b, q, 1, self.heads, depth)).sum(-1)/math.sqrt(depth)
        scores = scores.masked_fill(~valid[..., None], float('-inf'))
        # Fixed zero null token gives exactly zero if all sampled points invalid.
        scores = torch.cat((scores, scores.new_zeros(b, q, 1, self.heads)), 2)
        weights = scores.softmax(2)[:, :, :n]
        return (v*weights[..., None]).sum(2).flatten(-2)

    def forward(self, query, corners, initial, support, reg_scale, crop, context, local):
        if self.arm == 'A':
            return query
        geometry = {'B': 'fixed', 'E': 'two_bin'}.get(self.arm, 'full')
        base, box = boundary_points(corners, initial, support, reg_scale, geometry)
        dtype = query.dtype
        with torch.autocast(device_type=query.device.type, enabled=False):
            normed = self.norm(query.float())
            q = self.query(normed)
            offsets = self.offset(normed).reshape(*query.shape[:2], 16, 2).tanh()
            local_points = base + .25*initial.detach().float()[..., None, 2:]*offsets[:, :, 4:]
            # Context radius is at least one global/local P4 cell in crop
            # units and otherwise spans the current mean box. Identical bases
            # in C and D; C changes only which context image supplies evidence.
            crop = crop.detach().float()
            span = crop[:, 2:]-crop[:, :2]
            global_cell = 1./(40.*span[:, None, :])
            radius = torch.maximum(box[..., 2:].abs()*.5, global_cell)
            pattern = q.new_tensor([[-1, -1], [1, -1], [-1, 1], [1, 1]])
            context_crop = box[..., None, :2]+radius[..., None, :]*(pattern+.5*offsets[:, :, :4])
            context_global = original_coordinates(context_crop, crop)
            if self.arm == 'C':
                context_points = context_crop
                context_valid = valid_points(context_crop) & valid_points(context_global)
            else:
                context_points = context_global
                # Global context may lie outside crop, but never original image.
                context_valid = valid_points(context_global)
            ctx = self.read(context, context_points, context_valid, q,
                            (context_crop-box[..., None, :2])/radius[..., None, :].clamp_min(1e-5))
            enhanced_q = q+self.context_to_query(ctx)
            valid = valid_points(local_points) & valid_points(original_coordinates(local_points, crop))
            detail = self.read(local, local_points, valid, enhanced_q,
                               (local_points-box[..., None, :2])/initial.detach().float()[..., None, 2:].clamp_min(1e-5))
            return query + self.restore(detail).to(dtype)
