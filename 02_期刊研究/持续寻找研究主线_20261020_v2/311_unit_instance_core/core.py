"""Synthetic CPU prototype of 296/304/306; no loader, native model or optimizer.

The transport and S interfaces are the proposed contracts. This is not a trained
completion system: no M0 solver, GT scatter/loss adapter or native fusion lives here.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

CLASSES, DIM, HEADS, QUERIES, SIDE = 10, 128, 4, 4, 16
TOTAL_FLOOR = 1e-4


def candidate_inputs(boxes, classes, scores, valid):
    """15 values per real candidate; padding may contain arbitrary sentinels."""
    assert boxes.device.type == "cpu" and boxes.ndim == 3 and boxes.shape[-1] == 4
    assert classes.shape == scores.shape == valid.shape == boxes.shape[:2]
    assert valid.dtype == torch.bool and classes.dtype == torch.long
    assert bool(torch.isfinite(boxes[valid]).all())
    assert bool(torch.isfinite(scores[valid]).all())
    assert bool(((scores[valid] >= 0) & (scores[valid] <= 1)).all())
    assert bool(((classes[valid] >= 0) & (classes[valid] < CLASSES)).all())
    real_boxes = boxes[valid]
    assert bool((real_boxes[:, 2:] > real_boxes[:, :2]).all())
    safe_class = torch.where(valid, classes, torch.zeros_like(classes))
    default_box = boxes.new_tensor([0., 0., 1., 1.])
    safe_box = torch.where(valid[..., None], boxes, default_box)
    onehot = F.one_hot(safe_class, CLASSES).to(boxes.dtype) * valid[..., None]
    token = torch.cat((torch.where(valid[..., None], safe_box, 0.), onehot,
                       torch.where(valid, scores, 0.)[..., None]), dim=-1)
    counts = onehot.sum(dim=1)
    return token, safe_box, safe_class, counts


def independent_cap(mass, cost, null_cost, classes, valid, detach_mass=True):
    """mass [B,C,I], cost [B,I,J]; padding columns have zero total supply.

    Uses the mass-reference kernel m*exp(-C), not a uniform grid prior. Every
    valid column has supply one; excess row mass returns to that column's null.
    """
    assert mass.device.type == "cpu" and mass.ndim == 3
    b, c, i = mass.shape
    assert cost.ndim == 3 and cost.shape[:2] == (b, i)
    j = cost.shape[2]
    assert classes.shape == valid.shape == null_cost.shape == (b, j)
    assert valid.dtype == torch.bool and classes.dtype == torch.long
    assert bool(torch.isfinite(mass).all()) and bool((mass >= 0).all())
    assert bool(torch.isfinite(cost).all()) and bool(torch.isfinite(null_cost).all())
    assert bool(((classes[valid] >= 0) & (classes[valid] < c)).all())
    m = mass.detach() if detach_mass else mass
    if j == 0:
        return dict(p=cost, p0=null_cost, T=cost, z=null_cost,
                    u=torch.zeros_like(m), explained=torch.zeros_like(m),
                    scale=torch.ones_like(m), residual=m, mass=m)
    cls = torch.where(valid, classes, torch.zeros_like(classes))
    onehot = F.one_hot(cls, c).to(m.dtype) * valid[..., None]
    selected_m = m.gather(1, cls[..., None].expand(b, j, i)).transpose(1, 2)
    logm = torch.where(selected_m > 0,
                       selected_m.clamp_min(torch.finfo(m.dtype).tiny).log(),
                       torch.full_like(selected_m, -torch.inf))
    joined = torch.cat((logm - cost, -null_cost[:, None, :]), dim=1)
    probabilities = joined.softmax(dim=1) * valid[:, None, :]
    p, p0 = probabilities[:, :-1], probabilities[:, -1]
    u = torch.einsum("bij,bjc->bci", p, onehot)
    safe_u = torch.where(u > 0, u, torch.ones_like(u))
    scale = torch.minimum(torch.ones_like(m), m / safe_u)
    scale = torch.where(u > 0, scale, torch.ones_like(scale))
    selected_scale = scale.gather(1, cls[..., None].expand(b, j, i)).transpose(1, 2)
    plan = p * selected_scale
    null = p0 + (p - plan).sum(dim=1)
    explained = torch.einsum("bij,bjc->bci", plan, onehot)
    residual = (m - explained).clamp_min(0.)  # Roundoff guard, not a new threshold.
    return dict(p=p, p0=p0, T=plan, z=null, u=u, explained=explained,
                scale=scale, residual=residual, mass=m)


def density_inputs(field, total_floor=TOTAL_FLOOR):
    assert field.ndim == 3 and bool((field >= 0).all())
    return field.log1p().transpose(1, 2), (field.sum(1) + total_floor / field.shape[2]).log()


def grid_positions(dtype=torch.float32):
    axis = (torch.arange(SIDE, dtype=dtype) + .5) / SIDE
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((x, y), dim=-1).reshape(-1, 2)


def sine_position():
    xy = grid_positions() * (2 * math.pi)
    frequencies = 10000. ** (torch.arange(DIM // 4) / (DIM // 4))
    x, y = xy[:, 0:1] / frequencies, xy[:, 1:2] / frequencies
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1)


class RGBTrunk(nn.Module):
    # RGB convolution bias=False is an explicit prototype implementation convention;
    # 296 did not lock those biases. The mass head separately retains its prior bias.
    def __init__(self):
        super().__init__()
        blocks, channels = [], 3
        for out in (32, 64, DIM):
            blocks.extend([nn.Conv2d(channels, out, 3, 2, 1, bias=False),
                           nn.GroupNorm(8, out), nn.SiLU()])
            channels = out
        self.down = nn.Sequential(*blocks)
        self.residual = nn.Sequential(nn.Conv2d(DIM, DIM, 3, 1, 1, bias=False),
                                      nn.GroupNorm(8, DIM), nn.SiLU())

    def forward(self, rgb):
        assert rgb.device.type == "cpu" and rgb.shape[1:] == (3, 128, 128)
        f = self.down(rgb)
        return f + self.residual(f)


class Association(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(15, DIM), nn.SiLU(), nn.Linear(DIM, DIM))
        self.query = nn.Linear(DIM, DIM)
        self.key = nn.Linear(DIM, DIM)
        self.null = nn.Sequential(nn.Linear(2 * DIM, DIM), nn.SiLU(), nn.Linear(DIM, 1))
        self.register_buffer("xy", grid_positions(), persistent=False)

    def forward(self, f, tokens, boxes, classes, valid, mass):
        a = self.encoder(tokens)
        image = f.flatten(2).transpose(1, 2)
        content = -torch.einsum("bid,bjd->bij", self.key(image), self.query(a)) / math.sqrt(DIM)
        xy = self.xy.to(f.dtype)[None, :, None, :]
        outside = (boxes[:, None, :, :2] - xy).clamp_min(0.)
        outside = outside + (xy - boxes[:, None, :, 2:]).clamp_min(0.)
        wh = boxes[..., 2:] - boxes[..., :2]  # Validated positive; padding uses unit boxes.
        geometry = (outside / wh[:, None]).sum(dim=-1)
        global_f = image.mean(1)[:, None, :].expand(-1, a.shape[1], -1)
        null_cost = self.null(torch.cat((a, global_f), dim=-1)).squeeze(-1)
        plan = independent_cap(mass, content + geometry, null_cost, classes, valid)
        return a, plan


class ImageAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q, self.k, self.v, self.out = [nn.Linear(DIM, DIM) for _ in range(4)]
        self.mass_value = nn.Linear(CLASSES, DIM, bias=False)
        self.register_buffer("position", sine_position(), persistent=False)

    def forward(self, query, image, density_value, density_bias):
        b, q, _ = query.shape
        i, d = image.shape[1], DIM // HEADS
        qq = self.q(query).reshape(b, q, HEADS, d).transpose(1, 2)
        kk = self.k(image + self.position.to(image.dtype)).reshape(b, i, HEADS, d).transpose(1, 2)
        vv = (self.v(image) + self.mass_value(density_value)).reshape(b, i, HEADS, d).transpose(1, 2)
        logits = torch.matmul(qq, kk.transpose(-2, -1)) / math.sqrt(d)
        weights = (logits + density_bias[:, None, None, :]).softmax(dim=-1)
        value = torch.matmul(weights, vv).transpose(1, 2).reshape(b, q, DIM)
        return self.out(value)


class DecoderBlock(nn.Module):
    def __init__(self, direct_set=False):
        super().__init__()
        self.self_norm = nn.LayerNorm(DIM)
        self.self_attention = nn.MultiheadAttention(DIM, HEADS, dropout=0., batch_first=True)
        self.p_norm = nn.LayerNorm(DIM) if direct_set else None
        self.p_attention = nn.MultiheadAttention(DIM, HEADS, dropout=0., batch_first=True) if direct_set else None
        self.image_norm, self.ff_norm = nn.LayerNorm(DIM), nn.LayerNorm(DIM)
        self.image_attention = ImageAttention()
        self.ff = nn.Sequential(nn.Linear(DIM, 512), nn.GELU(), nn.Linear(512, DIM))

    def forward(self, q, image, dv, db, a, valid):
        normalized = self.self_norm(q)
        q = q + self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        if self.p_attention is not None and a.shape[1] > 0:
            active = valid.any(dim=1)
            # Empty rows never enter an all-masked softmax. index_copy preserves gradients.
            ids = active.nonzero(as_tuple=False).flatten()
            if ids.numel():
                normalized = self.p_norm(q.index_select(0, ids))
                subset = a.index_select(0, ids)
                update = self.p_attention(normalized, subset, subset,
                                          key_padding_mask=~valid.index_select(0, ids),
                                          need_weights=False)[0]
                q = q + torch.zeros_like(q).index_copy(0, ids, update)
        q = q + self.image_attention(self.image_norm(q), image, dv, db)
        return q + self.ff(self.ff_norm(q))


class CompletionHead(nn.Module):
    """N/S/R synthetic interface. Forward takes only RGB and predicted candidates."""
    def __init__(self, arm):
        super().__init__()
        assert arm in ("N", "S", "R")
        self.arm = arm
        self.trunk = RGBTrunk()
        self.mass_head = nn.Conv2d(DIM, CLASSES, 1)
        self.association = Association()
        self.queries = nn.Parameter(torch.empty(QUERIES, DIM))
        self.count_projection = nn.Linear(CLASSES, DIM, bias=False) if arm == "S" else None
        self.blocks = nn.ModuleList([DecoderBlock(arm == "S") for _ in range(2)])
        self.classifier = nn.Linear(DIM, CLASSES)
        self.box_head = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU(),
                                      nn.Linear(DIM, DIM), nn.ReLU(), nn.Linear(DIM, 4))
        refs = torch.tensor([[.25, .25, .25, .25], [.75, .25, .25, .25],
                             [.25, .75, .25, .25], [.75, .75, .25, .25]])
        self.register_buffer("reference_logits", torch.logit(refs))
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.zeros_(module.in_proj_bias)
        nn.init.normal_(self.queries, std=.02)
        nn.init.zeros_(self.mass_head.weight)
        nn.init.constant_(self.mass_head.bias, math.log(math.expm1(1. / (SIDE * SIDE))))
        nn.init.zeros_(self.association.query.weight)
        nn.init.zeros_(self.association.query.bias)
        nn.init.zeros_(self.association.null[-1].weight)
        nn.init.zeros_(self.association.null[-1].bias)
        nn.init.zeros_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, math.log(1e-4 / (1. - 1e-4)))
        nn.init.zeros_(self.box_head[-1].weight)
        nn.init.zeros_(self.box_head[-1].bias)
        if self.arm == "S":
            nn.init.zeros_(self.count_projection.weight)
            for block in self.blocks:
                nn.init.zeros_(block.p_attention.out_proj.weight)
                nn.init.zeros_(block.p_attention.out_proj.bias)

    def forward_features(self, f, boxes, classes, scores, valid):
        assert f.device.type == "cpu" and f.shape[1:] == (DIM, SIDE, SIDE)
        assert boxes.shape[0] == f.shape[0] and boxes.dtype == scores.dtype == f.dtype
        tokens, safe_boxes, safe_classes, counts = candidate_inputs(boxes, classes, scores, valid)
        mass = F.softplus(self.mass_head(f)).flatten(2)
        a, plan = self.association(f, tokens, safe_boxes, safe_classes, valid, mass)
        field = plan["residual"] if self.arm == "R" else mass.detach()
        dv, db = density_inputs(field)
        q = self.queries[None].expand(f.shape[0], -1, -1)
        if self.arm == "S":
            q = q + self.count_projection(counts.log1p())[:, None, :]
        image = f.flatten(2).transpose(1, 2)
        for block in self.blocks:
            q = block(q, image, dv, db, a, valid)
        return {"pred_boxes": (self.box_head(q) + self.reference_logits).sigmoid(),
                "pred_logits": self.classifier(q), "query_states": q,
                "features": f, "mass": mass, "plan": plan, "counts": counts,
                "tokens15": tokens, "density_field": field,
                "density_value": dv, "density_bias": db}

    def forward(self, rgb, boxes, classes, scores, valid):
        return self.forward_features(self.trunk(rgb), boxes, classes, scores, valid)


def copy_common(source, target):
    """Copy all common tensors; S-specific modules must be the only missing keys."""
    source_state, target_state = source.state_dict(), target.state_dict()
    shared = {k: v for k, v in source_state.items() if k in target_state}
    result = target.load_state_dict(shared, strict=False)
    def s_key(key):
        return key.startswith("count_projection.") or ".p_attention." in key or ".p_norm." in key
    assert not result.unexpected_keys
    assert all(s_key(k) for k in result.missing_keys)
    assert all(s_key(k) for k in source_state if k not in target_state)


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())
