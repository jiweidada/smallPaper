"""A49 experimental known-ignore supervision. Not a validated detection method.

Native MAL algebra is retained from the pinned EdgeCrafter implementation;
only fully unmatched ordinary-query classification weights can change.
"""
from contextlib import contextmanager
from types import MethodType
import torch
import torch.nn.functional as F


def disjoint_union(rectangles):
    """Partition clipped xyxy rectangles into non-overlapping x-slabs on CPU."""
    rects = [tuple(map(float, r)) for r in rectangles if r[2] > r[0] and r[3] > r[1]]
    xs = sorted({r[k] for r in rects for k in (0, 2)})
    result = []
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((r[1], r[3]) for r in rects if r[0] < right and r[2] > left)
        merged = []
        for lo, hi in intervals:
            if not merged or lo > merged[-1][1]:
                merged.append([lo, hi])
            else:
                merged[-1][1] = max(hi, merged[-1][1])
        result.extend([left, lo, right, hi] for lo, hi in merged)
    return result


@torch.no_grad()
def fractional_field(tiles, height, width):
    """Exact union-area fraction per feature-grid cell, not a semantic mask."""
    tiles = tiles.detach().float()
    if tiles.numel() == 0:
        return tiles.new_zeros((height, width))
    if tiles.ndim != 2 or tiles.shape[-1] != 4 or tiles.min() < 0 or tiles.max() > 1:
        raise ValueError('Expected normalized clipped disjoint xyxy tiles.')
    x0 = torch.arange(width, device=tiles.device, dtype=torch.float32) / width
    y0 = torch.arange(height, device=tiles.device, dtype=torch.float32) / height
    dx = (torch.minimum(tiles[:, 2, None], x0[None] + 1 / width) -
          torch.maximum(tiles[:, 0, None], x0[None])).clamp_min(0)
    dy = (torch.minimum(tiles[:, 3, None], y0[None] + 1 / height) -
          torch.maximum(tiles[:, 1, None], y0[None])).clamp_min(0)
    field = torch.einsum('rh,rw->hw', dy, dx) * (height * width)
    if not torch.isfinite(field).all() or field.min() < -1e-6 or field.max() > 1.0001:
        raise ValueError('Invalid union field; inputs must be disjoint clipped tiles.')
    return field.clamp(0, 1)


@torch.no_grad()
def box_ignore_fraction(boxes, tiles):
    """IoF of clipped predicted cxcywh boxes against the union of ignore tiles."""
    boxes = boxes.detach().float()
    xyxy = torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                      boxes[..., :2] + boxes[..., 2:] / 2), dim=-1).clamp(0, 1)
    tiles = tiles.detach().float()
    if tiles.numel() == 0:
        return boxes.new_zeros(boxes.shape[:-1])
    lo = torch.maximum(xyxy[:, None, :2], tiles[None, :, :2])
    hi = torch.minimum(xyxy[:, None, 2:], tiles[None, :, 2:])
    overlap = (hi - lo).clamp_min(0).prod(-1).sum(-1)
    area = (xyxy[:, 2:] - xyxy[:, :2]).clamp_min(0).prod(-1)
    fraction = torch.where(area > 0, overlap / area.clamp_min(1e-12), torch.zeros_like(area))
    if not torch.isfinite(fraction).all() or fraction.max() > 1.0001:
        raise ValueError('Invalid predicted-box union IoF.')
    return fraction.clamp(0, 1)


@torch.no_grad()
def read_ignore_mass(native_core, ignore_tiles, spatial_shapes, sampling_locations,
                     attention_weights, points_per_level, field_cache=None):
    """Replay the identical native sampling rule with one mask channel per head."""
    batch, _, heads = attention_weights.shape[:3]
    assert len(ignore_tiles) == batch
    with torch.autocast(device_type=sampling_locations.device.type, enabled=False):
        cache = {} if field_cache is None else field_cache
        values = []
        for height, width in spatial_shapes:
            height, width = int(height), int(width)
            key = (height, width)
            if key not in cache:
                cache[key] = torch.stack([fractional_field(t, height, width) for t in ignore_tiles])
            field = cache[key]
            values.append(field.reshape(batch, 1, 1, height * width).expand(batch, heads, 1, height * width))
        per_head = native_core(values, spatial_shapes, sampling_locations.detach().float(),
                               attention_weights.detach().float(), points_per_level)
        assert per_head.shape[-1] == heads
        mass = per_head.mean(-1)
    if not torch.isfinite(mass).all() or mass.min() < -1e-6 or mass.max() > 1.0001:
        raise ValueError('Invalid ignore read mass.')
    return mass.clamp(0, 1)


@contextmanager
def native_inference(model):
    """Restore literal native core callables, including on an EMA model copy."""
    changed = []
    for module in model.modules():
        if hasattr(module, '_a49_native_core'):
            changed.append((module, module.ms_deformable_attn_core))
            module.ms_deformable_attn_core = module._a49_native_core
    try:
        yield
    finally:
        for module, original in changed:
            module.ms_deformable_attn_core = original


class IgnorePolicy:
    def __init__(self, arm):
        assert arm in ('B', 'M', 'Q')
        self.arm = arm
        self.active_tiles = None
        self.masses = []
        self.mask_core_calls = 0
        self.native_loss = None
        self.loss_statistics = []
        self.fields = {}

    @contextmanager
    def training_step(self, targets):
        assert self.active_tiles is None, 'Nested/concurrent contexts are unsupported.'
        self.active_tiles = [t['ignore_tiles'] for t in targets]
        self.masses = []
        self.loss_statistics = []
        self.fields = {}
        try:
            yield self
        finally:
            self.active_tiles = None
            self.masses = []
            self.fields = {}

    def install_model(self, model):
        if self.arm != 'Q':
            return []
        installed = []
        for name, module in model.named_modules():
            if not name.endswith('.cross_attn') or not hasattr(module, 'ms_deformable_attn_core'):
                continue
            assert not hasattr(module, '_a49_native_core')
            native = module.ms_deformable_attn_core
            module._a49_native_core = native

            def capture(value, shapes, locations, weights, points, _native=native):
                if self.active_tiles is not None:
                    self.masses.append(read_ignore_mass(_native, self.active_tiles, shapes,
                                                        locations, weights, points, self.fields))
                    self.mask_core_calls += 1
                return _native(value, shapes, locations, weights, points)

            module.ms_deformable_attn_core = capture
            installed.append(name)
        assert installed, 'No native deformable core found; inspect host rather than continue silently.'
        return installed

    def annotate_outputs(self, outputs):
        if self.arm == 'B':
            return outputs
        ordinary = [outputs] + list(outputs.get('aux_outputs', []))
        ordinary += list(outputs.get('enc_aux_outputs', []))
        if 'pre_outputs' in outputs:
            ordinary.append(outputs['pre_outputs'])
        # Append metadata after native tensors; native criterion reads the first tensor's device.
        for prediction in ordinary:
            prediction['_a49_ordinary'] = True
        if self.arm == 'Q':
            decoder_outputs = list(outputs.get('aux_outputs', [])) + [outputs]
            assert len(decoder_outputs) == len(self.masses), (len(decoder_outputs), len(self.masses))
            start = int(outputs['dn_meta']['dn_num_split'][0]) if 'dn_meta' in outputs else 0
            for prediction, mass in zip(decoder_outputs, self.masses):
                count = prediction['pred_logits'].shape[1]
                assert mass.shape[1] == start + count
                prediction['_a49_read_mass'] = mass[:, start:].detach()
        return outputs

    def install_loss(self, criterion):
        assert self.native_loss is None, 'Do not install twice.'
        self.native_loss = criterion.loss_labels_mal
        if self.arm == 'B':
            return
        policy = self

        def modified(current, outputs, targets, indices, num_boxes, values=None):
            if not outputs.get('_a49_ordinary', False):
                return policy.native_loss(outputs, targets, indices, num_boxes, values=values)
            if not any(t['ignore_tiles'].numel() for t in targets):
                return policy.native_loss(outputs, targets, indices, num_boxes, values=values)
            assert 'pred_boxes' in outputs
            idx = current._get_src_permutation_idx(indices)
            if values is None:
                # Use the identical native box functions, not a separately reimplemented IoU.
                namespace = policy.native_loss.__func__.__globals__
                xyxy = namespace['box_cxcywh_to_xyxy']
                box_iou = namespace['box_iou']
                src_boxes = outputs['pred_boxes'][idx]
                target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
                ious = torch.diag(box_iou(xyxy(src_boxes), xyxy(target_boxes))[0]).detach()
            else:
                ious = values
            src_logits = outputs['pred_logits']
            target_classes_o = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)])
            target_classes = torch.full(src_logits.shape[:2], current.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o
            target = F.one_hot(target_classes, num_classes=current.num_classes + 1)[..., :-1]
            target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
            target_score_o[idx] = ious.to(target_score_o.dtype)
            target_score = (target_score_o.unsqueeze(-1) * target).pow(current.gamma)
            pred_score = F.sigmoid(src_logits).detach()
            if current.mal_alpha is not None:
                weight = current.mal_alpha * pred_score.pow(current.gamma) * (1 - target) + target
            else:
                weight = pred_score.pow(current.gamma) * (1 - target) + target
            with torch.no_grad():
                if policy.arm == 'Q' and '_a49_read_mass' in outputs:
                    coverage = outputs['_a49_read_mass'].detach()
                    rule = 'read_mass'
                else:
                    coverage = torch.stack([box_ignore_fraction(b, t['ignore_tiles'])
                                            for b, t in zip(outputs['pred_boxes'], targets)])
                    rule = 'box_union_IoF'
                keep = coverage < .5
                keep[idx] = True
                policy.loss_statistics.append({'rule': rule, 'query_rows': keep.numel(),
                                               'omitted_unmatched_rows': int((~keep).sum().item())})
            weight = weight * keep.to(weight.dtype).unsqueeze(-1)
            loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
            loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
            return {'loss_mal': loss}

        criterion.loss_labels_mal = MethodType(modified, criterion)
