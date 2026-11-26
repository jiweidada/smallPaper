"""A52 candidate: native marginal-shape features as decoder pairwise attention bias.

No new probabilistic/Fourier principle or calibrated joint-box posterior is claimed.
M: characteristic features of same-mean/variance Gaussian marginals.
Q: characteristic features of the actual native categorical edge marginals.
The legacy point-geometry features, trainable map, initialization and placement match.
"""
import hashlib
import inspect
import math
import textwrap
import types
import torch
from torch import nn
from torch.nn import functional as F

FREQUENCIES = 8
HIDDEN = 48
BIAS_SCALE = 0.1
NEW_INIT_SEED = 20261102


def physical_marginals(corners, reference, project, reg_scale):
    raw = corners.detach().float()
    ref = reference.detach().float()
    proj = project.detach().float()
    prob = raw.reshape(*raw.shape[:2], 4, len(proj)).softmax(-1)
    scale = abs(float(reg_scale))
    assert scale > 0 and ref.shape[-1] == 4
    centers = ref[..., [0, 1, 0, 1], None]
    widths = ref[..., [2, 3, 2, 3], None]
    sign = ref.new_tensor([-1., -1., 1., 1.])[None, None, :, None]
    supports = centers + sign * (0.5 * scale + proj[None, None, None, :]) * widths / scale
    return prob, supports


def characteristic_features(prob, supports, mode):
    assert mode in ("M", "Q") and prob.shape == supports.shape
    mean = (prob * supports).sum(-1)
    var = (prob * (supports - mean[..., None]).square()).sum(-1).clamp_min(0)
    cs, sn = [], []
    for k in range(FREQUENCIES):
        omega = 2. * math.pi * (2. ** k)
        if mode == "M":
            attenuation = torch.exp(-0.5 * omega * omega * var)
            cs.append(torch.cos(omega * mean) * attenuation)
            sn.append(torch.sin(omega * mean) * attenuation)
        else:
            phase = supports * omega
            cs.append((prob * phase.cos()).sum(-1))
            sn.append((prob * phase.sin()).sum(-1))
    return torch.stack(cs, -1), torch.stack(sn, -1)


def pair_features(prob, supports, boxes, mode):
    """Only descriptive marginal embeddings; not P(same object) or a joint-box model."""
    cs, sn = characteristic_features(prob, supports, mode)
    kernel = cs[:, :, None] * cs[:, None, :] + sn[:, :, None] * sn[:, None, :]
    kernel = kernel.flatten(-2)
    center = boxes[..., :2].detach().to(dtype=prob.dtype)
    wh = boxes[..., 2:].detach().to(dtype=prob.dtype).clamp_min(1e-4)
    delta = (center[:, :, None] - center[:, None, :]).abs()
    offset = torch.log1p(delta / wh[:, :, None])
    ratio = (wh[:, :, None] / wh[:, None, :]).log()
    return torch.cat((offset, ratio, kernel), -1)


class RelationBias(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.first = nn.Linear(4 + 4 * FREQUENCIES, HIDDEN)
        self.last = nn.Linear(HIDDEN, heads, bias=False)
        # Every attention row would cancel a common scalar bias; do not add it.
        nn.init.zeros_(self.last.weight)

    def forward(self, features):
        return BIAS_SCALE * self.last(F.silu(self.first(features)))


class DistributionRelation(nn.Module):
    def __init__(self, heads, layers, ordinary_queries, mode, reg_scale):
        super().__init__()
        assert mode in ("M", "Q") and layers == 4 and ordinary_queries == 300
        self.mode = mode
        self.heads = heads
        self.ordinary_queries = ordinary_queries
        self.reg_scale = float(reg_scale)
        self.maps = nn.ModuleList([RelationBias(heads) for _ in range(layers - 1)])

    def mask(self, index, context, original_mask, target):
        corners, refs, project, boxes = context
        n = self.ordinary_queries
        total = target.shape[1]
        assert corners.shape[1] == refs.shape[1] == boxes.shape[1] == total
        assert total >= n and index in (0, 1, 2)
        # DN can have a variable count. Geometry is computed ONLY for the final
        # 300 ordinary queries; the original DN hard mask remains authoritative.
        with torch.autocast(device_type=target.device.type, enabled=False):
            prob, supports = physical_marginals(corners[:, -n:], refs[:, -n:], project, self.reg_scale)
            features = pair_features(prob, supports, boxes[:, -n:], self.mode)
        bias = self.maps[index](features).permute(0, 3, 1, 2).float()
        pad = total - n
        if pad:
            bias = F.pad(bias, (pad, 0, pad, 0))
        if original_mask is not None:
            assert original_mask.ndim == 2 and tuple(original_mask.shape) == (total, total)
            if original_mask.dtype == torch.bool:
                bias = bias.masked_fill(original_mask[None, None], float("-inf"))
            else:
                bias = bias + original_mask[None, None].float()
        return bias.reshape(target.shape[0] * self.heads, total, total)


def install(model, arm):
    if arm == "B":
        return None
    assert arm in ("M", "Q")
    decoder = model.decoder.decoder
    assert decoder.num_layers == 4 and decoder.eval_idx == 3 and decoder.layer_scale == 1
    assert model.decoder.num_queries == 300
    assert not hasattr(decoder, "a52_relation")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(NEW_INIT_SEED)
        module = DistributionRelation(decoder.num_head, decoder.num_layers, model.decoder.num_queries,
                                      arm, decoder.reg_scale.detach().item())
    decoder.add_module("a52_relation", module)
    original = type(decoder).forward
    source = textwrap.dedent(inspect.getsource(original))
    replacements = [
        ("    output_detach = pred_corners_undetach = 0\n",
         "    output_detach = pred_corners_undetach = 0\n    a52_context = None\n"),
        ("        output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)\n",
         "        a52_mask = attn_mask\n"
         "        if i > 0:\n"
         "            assert a52_context is not None\n"
         "            a52_mask = self.a52_relation.mask(i - 1, a52_context, attn_mask, output)\n"
         "        output = layer(output, ref_points_input, value, spatial_shapes, a52_mask, query_pos_embed)\n"),
        ("        inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)\n",
         "        inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)\n"
         "        a52_context = (pred_corners.detach(), ref_points_initial.detach(), project.detach(), inter_ref_bbox.detach())\n"),
    ]
    for before, after in replacements:
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    namespace = dict(original.__globals__)
    exec(compile(source, "<A52 isolated decoder forward>", "exec"), namespace)
    decoder.forward = types.MethodType(namespace["forward"], decoder)
    module.patched_forward_sha256 = hashlib.sha256(source.encode()).hexdigest()
    return module

