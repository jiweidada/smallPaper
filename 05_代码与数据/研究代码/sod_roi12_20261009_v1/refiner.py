"""Isolated fixed-output ROI experiment using PyTorch and NumPy.

Coordinate contract: boxes are ORIGINAL-image continuous xyxy pixel edges;
images are unnormalised RGB float [3,H,W] in [0,1]; labels are 0-based and
query_ids come from the original flattened query x class TopK. Input detections
are never re-ranked, duplicated, or removed here. GT is NEVER a model input.

Training contract: epochs 1..8 stage='refiner', 9..12 stage='gate' (runner owns
the <=12 cap). Gate labels measure each SINGLE output replacement in the
unchanged original output set, rematching all outputs at .50:.05:.95:
mean_t(number of newly hit GT - number of lost GT), lambda_harm=1.
This is the all-output diagnostic protocol, not formal COCO maxDets100 or AP.
Invalid identity target=-1 is a rejection convention, not measured utility.
All targets are detached; jointly accepting several boxes need not equal
the sum of these single-replacement utilities.

Default losses (runner should record/freeze the actual values):
  geometry  (1-IoU):1, (1-GIoU):1, original-wh-normalised L1:1;
  old-.75 harm mean relu(IoU0-IoU1):2 (only cached old-success bindings);
  gate      SmoothL1(tanh(gain_logit), target_y), beta=1, unit weight;
            target_y>0, ==0, <0 groups have equal mean-loss weight by default.
Each geometry term is a mean over eligible bound items, except the harm term
which is a mean over their old-success subset. Gate includes ALL eligible
items, takes the mean within each target-sign group, then the equal mean of
present groups (empty groups omitted). This prevents background count alone
dominating gain learning. No extra step/distillation/class loss.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _boxes(boxes: Tensor) -> None:
    if boxes.ndim != 2 or boxes.shape[-1] != 4 or not boxes.is_floating_point():
        raise ValueError('boxes must be floating [N,4] original xyxy')
    if not bool(torch.isfinite(boxes).all()):
        raise ValueError('nonfinite boxes')


def pairwise_iou(a: Tensor, b: Tensor) -> Tensor:
    _boxes(a); _boxes(b)
    a, b = a.float(), b.float()
    inter = (torch.minimum(a[:, None, 2:], b[None, :, 2:]) -
             torch.maximum(a[:, None, :2], b[None, :, :2])).clamp_min(0).prod(-1)
    aa = (a[:, 2:] - a[:, :2]).clamp_min(0).prod(-1)
    bb = (b[:, 2:] - b[:, :2]).clamp_min(0).prod(-1)
    return inter / (aa[:, None] + bb[None] - inter).clamp_min(1e-8)


def aligned_iou_giou(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    """Elementwise metrics, avoiding an unnecessary NxN loss matrix."""
    if a.shape != b.shape:
        raise ValueError('aligned boxes must have the same shape')
    a, b = a.float(), b.float()
    inter = (torch.minimum(a[:, 2:], b[:, 2:]) -
             torch.maximum(a[:, :2], b[:, :2])).clamp_min(0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp_min(0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp_min(0).prod(-1)
    union = (area_a + area_b - inter).clamp_min(1e-8)
    enclosing = (torch.maximum(a[:, 2:], b[:, 2:]) -
                 torch.minimum(a[:, :2], b[:, :2])).clamp_min(0).prod(-1).clamp_min(1e-8)
    iou = inter / union
    return iou, iou - (enclosing - union) / enclosing


@dataclass
class ROIBatch:
    crops: Tensor             # [U,4,S,S], RGB + valid pixel mask
    det_to_roi: Tensor        # [N] long; -1 for ineligible detections
    box_roi: Tensor           # [N,4], original box in letterboxed ROI / S
    eligible: Tensor          # [N] bool; no GT in this decision
    roi_query_ids: Tensor     # [U] original query IDs
    roi_boxes: Tensor         # [U,4] original physical crop edges
    image_hw: Tuple[int, int]


@torch.no_grad()
def prepare_rois(image: Tensor, boxes0: Tensor, query_ids: Tensor, *,
                 arm: str = 'raw', input_hw: Tuple[int, int] = (640, 640),
                 degraded_image: Optional[Tensor] = None,
                 input_scale_xy: Optional[Tensor] = None,
                 short_side: float = 16., size: int = 64,
                 context: float = 2., min_extent: float = 32.) -> ROIBatch:
    """Identical continuous crop ranges/masks in raw and degraded arms.

    This minimal experiment supports independent x/y resize (no letterbox in
    the BASE input). If input_scale_xy is supplied it must agree with this
    input_hw/image_hw mapping: other affine input transforms fail explicitly.
    ROI letterboxing itself is supported and is identical in both arms.
    For exact native PIL-resize provenance, caller supplies degraded_image
    [3,input_H,input_W] float RGB[0,1], AFTER native PILResize/quantisation and
    BEFORE normalisation. It is sampled on the SAME physical grid as raw.
    If omitted, the fallback is torch bilinear+antialias full-image resize;
    this float fallback is NOT claimed pixel-identical to PIL quantisation.
    degraded_image is used only for arm='degraded'. Neither mode separately
    resizes cropped raw pixels to create the low-resolution control.
    """
    _boxes(boxes0)
    if image.ndim != 3 or image.shape[0] != 3 or not image.is_floating_point():
        raise ValueError('image must be float RGB [3,H,W] in [0,1]')
    if not bool(torch.isfinite(image).all()) or bool((image < 0).any()) or bool((image > 1).any()):
        raise ValueError('image must be unnormalised finite RGB in [0,1]')
    if arm not in ('raw', 'degraded'):
        raise ValueError('arm must be raw or degraded')
    n = len(boxes0)
    if query_ids.shape != (n,) or query_ids.dtype != torch.long:
        raise ValueError('query_ids must be int64 [N]')
    if boxes0.device != image.device or query_ids.device != image.device:
        raise ValueError('image, boxes and IDs must share device')
    if size <= 0 or context <= 0 or min_extent <= 0 or min(input_hw) <= 0:
        raise ValueError('invalid crop or input size')
    h, w = image.shape[-2:]
    boxes = boxes0.detach().float()
    expected_scale = boxes.new_tensor([input_hw[1]/w, input_hw[0]/h])
    if input_scale_xy is not None:
        provided = torch.as_tensor(input_scale_xy, device=boxes.device, dtype=boxes.dtype)
        if provided.shape != (2,) or not torch.allclose(provided, expected_scale, atol=1e-6, rtol=1e-5):
            raise ValueError('only the archived independent xy-resize is supported; do not silently use letterbox')
    wh = boxes[:, 2:] - boxes[:, :2]
    eligible = (wh > 0).all(-1) & ((wh * expected_scale).amin(-1) < short_side)
    det_to_roi = torch.full((n,), -1, dtype=torch.long, device=boxes.device)
    box_roi = torch.zeros_like(boxes)
    uq = torch.unique(query_ids[eligible], sorted=True)
    if not len(uq):
        return ROIBatch(image.new_empty((0, 4, size, size)), det_to_roi, box_roi,
                        eligible, uq, boxes.new_empty((0, 4)), (h, w))
    representatives = []
    for i, q in enumerate(uq):
        entries = torch.nonzero(query_ids == q, as_tuple=False).flatten()
        first = entries[0]
        if not torch.allclose(boxes[entries], boxes[first].expand(len(entries), 4), atol=1e-4, rtol=0):
            raise ValueError('same native query must have identical baseline boxes across classes')
        if not bool(eligible[entries].all()):
            raise ValueError('inconsistent eligibility within a native query')
        representatives.append(first)
        det_to_roi[entries] = i
    rep = torch.stack(representatives)
    b = boxes[rep]
    extent = ((b[:, 2:] - b[:, :2]) * context).clamp_min(min_extent)
    center = (b[:, :2] + b[:, 2:]) / 2
    start = center - extent / 2
    roi_boxes = torch.cat((start, start + extent), -1)
    scale = size / extent.amax(-1)
    pad = (size - extent * scale[:, None]) / 2
    yy, xx = torch.meshgrid(torch.arange(size, device=boxes.device, dtype=boxes.dtype) + .5,
                            torch.arange(size, device=boxes.device, dtype=boxes.dtype) + .5,
                            indexing='ij')
    output_xy = torch.stack((xx, yy), -1)[None]
    local_xy = (output_xy - pad[:, None, None]) / scale[:, None, None, None]
    original_xy = start[:, None, None] + local_xy
    inside_crop = ((local_xy >= 0) & (local_xy < extent[:, None, None])).all(-1)
    inside_image = ((original_xy >= 0) & (original_xy < boxes.new_tensor([w, h]))).all(-1)
    mask = (inside_crop & inside_image)[:, None].float()
    grid = 2 * original_xy / boxes.new_tensor([w, h]) - 1
    source = image.detach().float()[None]
    if arm == 'degraded':
        if degraded_image is None:
            source = F.interpolate(source, size=input_hw, mode='bilinear', align_corners=False, antialias=True)
        else:
            if degraded_image.shape != (3, *input_hw) or not degraded_image.is_floating_point():
                raise ValueError('degraded_image must be float [3,input_H,input_W] RGB before normalisation')
            if degraded_image.device != image.device:
                raise ValueError('degraded_image must share the original image device')
            if (not bool(torch.isfinite(degraded_image).all()) or bool((degraded_image < 0).any())
                    or bool((degraded_image > 1).any())):
                raise ValueError('degraded_image must be finite and in [0,1]')
            source = degraded_image.detach().float()[None]
    rgb = F.grid_sample(source.expand(len(uq), -1, -1, -1), grid,
                        mode='bilinear', padding_mode='border', align_corners=False) * mask
    valid = det_to_roi >= 0
    ix = det_to_roi[valid]
    box_roi[valid] = ((boxes[valid] - start[ix].repeat(1, 2)) * scale[ix, None]
                      + pad[ix].repeat(1, 2)) / size
    return ROIBatch(torch.cat((rgb, mask), 1), det_to_roi, box_roi,
                    eligible, uq, roi_boxes, (h, w))


class ROIRefiner(nn.Module):
    """One small ROI encoder, class-conditioned geometry head, one gain head.

    forward(rois, boxes0[N,4], labels[N], query_features[N,query_dim],
            stats[N,stats_dim]) -> dict boxes1[N,4], delta[N,4], gain[N].
    q features are indexed to DETECTION ITEMS by caller (duplicates permitted).
    Their values and ROI inputs are detached; GT never enters forward.
    """
    def __init__(self, num_classes: int = 10, query_dim: int = 256,
                 stats_dim: int = 0, max_edge_fraction: float = .2,
                 initialization_seed: int = 0):
        super().__init__()
        if not 0 < max_edge_fraction < .5:
            raise ValueError('edge fraction must be <.5 to preserve positive width/height')
        self.num_classes, self.query_dim, self.stats_dim = num_classes, query_dim, stats_dim
        self.max_edge_fraction = max_edge_fraction
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(initialization_seed)
            self.encoder = nn.Sequential(
                nn.Conv2d(4, 16, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.SiLU(),
                # Fixed 64px input yields 8x8 here; identical nonoverlapping
                # 4x4 averages avoid adaptive-pool nondeterministic CUDA backward.
                nn.AvgPool2d(4), nn.Flatten())
            joined = 128 + 4 + num_classes + query_dim + stats_dim
            self.geometry = nn.Sequential(nn.Linear(joined, 64), nn.SiLU(), nn.Linear(64, 4))
            self.gate = nn.Sequential(nn.Linear(joined + 4, 32), nn.SiLU(), nn.Linear(32, 1))
            nn.init.zeros_(self.geometry[-1].weight); nn.init.zeros_(self.geometry[-1].bias)
            nn.init.zeros_(self.gate[-1].weight); nn.init.zeros_(self.gate[-1].bias)
        self.set_stage('refiner')

    def set_stage(self, stage: str) -> None:
        """Call before constructing each phase optimizer. No hidden epoch logic."""
        if stage not in ('refiner', 'gate'):
            raise ValueError('stage must be refiner or gate')
        self.stage = stage
        for part in (self.encoder, self.geometry):
            for p in part.parameters(): p.requires_grad_(stage == 'refiner')
        for p in self.gate.parameters(): p.requires_grad_(stage == 'gate')

    def forward(self, rois: ROIBatch, boxes0: Tensor, labels: Tensor,
                query_features: Optional[Tensor] = None,
                stats: Optional[Tensor] = None) -> dict:
        _boxes(boxes0)
        n = len(boxes0)
        if labels.shape != (n,) or labels.dtype != torch.long:
            raise ValueError('labels must be 0-based int64 [N]')
        if n and (bool((labels < 0).any()) or bool((labels >= self.num_classes).any())):
            raise ValueError('labels outside configured 0-based class range')
        base = boxes0.detach().float()
        valid = rois.eligible & (rois.det_to_roi >= 0)
        out_boxes, deltas = base.clone(), torch.zeros_like(base)
        gains = base.new_full((n,), -1.)
        if not bool(valid.any()):
            # Keep a zero graph edge so an all-ineligible image can backward.
            anchor = next(p for p in self.parameters() if p.requires_grad).sum() * 0
            return {'boxes1': out_boxes + anchor, 'delta': deltas + anchor,
                    'gain': gains + anchor, 'eligible': valid}
        feature = self.encoder(rois.crops.detach().float())
        pieces = [feature[rois.det_to_roi[valid]], rois.box_roi[valid].detach().float(),
                  F.one_hot(labels[valid], self.num_classes).float()]
        for value, dim, name in ((query_features, self.query_dim, 'query_features'),
                                 (stats, self.stats_dim, 'stats')):
            if dim:
                if value is None or value.shape != (n, dim):
                    raise ValueError(f'{name} must be [{n},{dim}], do not fabricate missing features')
                pieces.append(value[valid].detach().float())
        joined = torch.cat(pieces, -1)
        delta = self.max_edge_fraction * torch.tanh(self.geometry(joined))
        wh = base[valid, 2:] - base[valid, :2]
        proposal = base[valid] + delta.float() * wh.repeat(1, 2)
        out_boxes[valid], deltas[valid] = proposal, delta.float()
        # Gate gradients cannot change ROI features or geometry, even if a
        # caller accidentally leaves a refiner parameter trainable.
        gains[valid] = torch.tanh(self.gate(torch.cat((joined.detach(), delta.detach()), -1))).flatten().float()
        return {'boxes1': out_boxes, 'delta': deltas, 'gain': gains, 'eligible': valid}


@dataclass
class FixedBindings:
    det_to_gt: Tensor          # [N] long, -1 unbound
    old_success: Tensor        # [N] bool, ORIGINAL .75 greedy bindings only


@torch.no_grad()
def fixed_bindings(boxes0: Tensor, labels: Tensor, scores: Tensor,
                   gt_boxes: Tensor, gt_labels: Tensor) -> FixedBindings:
    """First lock original .75 greedy TP bindings; then bind remaining at .5.

    Same-class, stable descending score; equal IoU chooses later GT index,
    matching the archived diagnostic tie convention. Includes ALL original
    detections, not only ROI-eligible items. GT order must remain fixed.
    """
    _boxes(boxes0); _boxes(gt_boxes)
    n, g = len(boxes0), len(gt_boxes)
    if labels.shape != (n,) or scores.shape != (n,) or gt_labels.shape != (g,):
        raise ValueError('bad detection/GT vector shape')
    binding = torch.full((n,), -1, device=boxes0.device, dtype=torch.long)
    old = torch.zeros(n, device=boxes0.device, dtype=torch.bool)
    free = torch.ones(g, device=boxes0.device, dtype=torch.bool)
    ious = pairwise_iou(boxes0, gt_boxes)
    order = torch.argsort(scores.detach(), descending=True, stable=True)
    for threshold in (.75, .5):
        for d in order.tolist():
            if int(binding[d]) >= 0: continue
            candidate = torch.nonzero(free & (gt_labels == labels[d]) &
                                      (ious[d] >= threshold), as_tuple=False).flatten().flip(0)
            if len(candidate):
                k = candidate[torch.argmax(ious[d, candidate])]
                binding[d], free[k] = k, False
                if threshold == .75: old[d] = True
    return FixedBindings(binding.detach(), old.detach())


def _single_replacement_counts(iou0: np.ndarray, iou1: np.ndarray,
                               labels: np.ndarray, gt_labels: np.ndarray,
                               scores: np.ndarray, thresholds: np.ndarray):
    """CPU batched greedy matcher, with no per-GT/scenario/threshold loop.

    Scenarios and thresholds are vectorised; the unavoidable greedy order
    loops over detections within each class. Scenario zero is always the
    unmodified BASE. Only one row changes in each other scenario. Each
    threshold has fresh free-GT state; ties select the later original GT.
    Classes/rows that can never match and identical-IoU proposals are exact
    no-ops and skipped. Peak temporary storage is O((K+1)*10*G_class).
    """
    n = len(labels)
    new = np.zeros((n, len(thresholds)), dtype=np.int64)
    lost = np.zeros_like(new)
    for cls in np.unique(labels):
        ds = np.flatnonzero(labels == cls)
        gs = np.flatnonzero(gt_labels == cls)
        if not len(gs):
            continue
        ds = ds[np.argsort(-scores[ds], kind='stable')]
        base = iou0[np.ix_(ds, gs)]
        proposal = iou1[np.ix_(ds, gs)]
        can_match = np.any((base >= thresholds[0]) | (proposal >= thresholds[0]), axis=1)
        changed = np.flatnonzero(np.any(base != proposal, axis=1) & can_match)
        if not len(changed):
            continue
        # Dropping never-reachable GT preserves original relative tie order.
        useful_gt = np.any((base >= thresholds[0]) | (proposal >= thresholds[0]), axis=0)
        base, proposal = base[:, useful_gt], proposal[:, useful_gt]
        scenarios = np.concatenate((np.array([-1], dtype=np.int64), changed))
        free = np.ones((len(scenarios), len(thresholds), base.shape[1]), dtype=bool)
        sr = np.arange(len(scenarios))[:, None]
        tr = np.arange(len(thresholds))[None, :]
        for d in np.flatnonzero(can_match):
            overlap = np.where((scenarios == d)[:, None], proposal[d], base[d])
            candidate = np.where(free & (overlap[:, None, :] >= thresholds[None, :, None]),
                                 overlap[:, None, :], np.float32(-1))
            # argmax of the reversed last axis gives the later-GT tie rule.
            k = base.shape[1] - 1 - np.argmax(candidate[:, :, ::-1], axis=-1)
            matched = candidate[sr, tr, k] >= thresholds[None, :]
            rr, tt = np.nonzero(matched)
            free[rr, tt, k[rr, tt]] = False
        hits = ~free
        new[ds[changed]] = np.count_nonzero(hits[1:] & ~hits[:1], axis=-1)
        lost[ds[changed]] = np.count_nonzero(hits[:1] & ~hits[1:], axis=-1)
    return new, lost


@torch.no_grad()
def utility_targets(boxes0: Tensor, boxes1: Tensor, labels: Tensor,
                    gt_boxes: Tensor, gt_labels: Tensor, bindings: FixedBindings,
                    eligible: Optional[Tensor] = None, identity_margin: float = 1e-6,
                    *, scores: Tensor) -> dict:
    """Detached SINGLE-item replacement/rematching utility, lambda_harm=1.

    All N final outputs participate in score-stable same-class greedy matching
    at each of 10 independent thresholds .50:.05:.95. For item d, only its
    box is replaced; scores/classes/order of every item stay fixed. A later
    GT index wins equal IoU. This uses all supplied trainable GT/all outputs,
    not COCO ignore/maxDets logic. Counts compare hit GT sets, not identities
    assigned to a particular detection. N=0 and no GT are valid zero results.

    measured_u[N] = mean(new_count_by_threshold - lost_count_by_threshold).
    It is computed even for unbound/background/ineligible items. The unchanged
    row gives exact zero. Counts are int64 [N,10]. A single changed row has
    net hit-count change in [-1,1]; assert this invariant, NEVER clip utility.

    identity_valid additionally requires eligibility, a fixed same-class GT,
    positive proposal, IoU>=.5 to that binding and no equal/better same-class
    neighbour within identity_margin. target_y = measured_u only when valid;
    otherwise -1 is a rejection convention, NOT a measured loss. The optional
    local_fixed_gt_delta (also named measured_delta for old diagnostic readers)
    is NOT the gate label and is NaN for an unbound item. All returns detach.
    Nonfinite boxes/scores fail explicitly, avoiding silent label corruption.
    """
    _boxes(boxes0); _boxes(boxes1); _boxes(gt_boxes)
    if boxes1.shape != boxes0.shape: raise ValueError('proposal shape mismatch')
    n, g = len(boxes0), len(gt_boxes)
    if labels.shape != (n,) or scores.shape != (n,) or gt_labels.shape != (g,):
        raise ValueError('bad detection/GT vector shape')
    if bindings.det_to_gt.shape != (n,) or bindings.old_success.shape != (n,):
        raise ValueError('bad binding shape')
    if eligible is not None and eligible.shape != (n,):
        raise ValueError('eligible must be [N]')
    if identity_margin < 0 or not bool(torch.isfinite(scores).all()):
        raise ValueError('invalid identity margin or nonfinite score')
    if bool((bindings.det_to_gt < -1).any()) or bool((bindings.det_to_gt >= g).any()):
        raise ValueError('binding outside GT range')
    devices = (boxes1, labels, scores, gt_boxes, gt_labels, bindings.det_to_gt,
               bindings.old_success) + (() if eligible is None else (eligible,))
    if any(x.device != boxes0.device for x in devices):
        raise ValueError('utility inputs must share one device')
    thresholds = torch.arange(50, 100, 5, device=boxes0.device, dtype=torch.float32) / 100
    iou0, iou1 = pairwise_iou(boxes0, gt_boxes), pairwise_iou(boxes1, gt_boxes)
    new_np, lost_np = _single_replacement_counts(
        iou0.cpu().numpy(), iou1.cpu().numpy(), labels.cpu().numpy(),
        gt_labels.cpu().numpy(), scores.detach().cpu().numpy(), thresholds.cpu().numpy())
    new = torch.from_numpy(new_np).to(boxes0.device)
    lost = torch.from_numpy(lost_np).to(boxes0.device)
    net = new - lost
    if bool((net.abs() > 1).any()):
        raise AssertionError('single-replacement net TP outside [-1,1]; do not clip')
    measured = net.float().mean(-1)
    target = boxes0.new_full((n,), -1., dtype=torch.float32)
    local = boxes0.new_full((n,), float('nan'), dtype=torch.float32)
    valid = torch.zeros(n, dtype=torch.bool, device=boxes0.device)
    b = bindings.det_to_gt
    ix = torch.nonzero(b >= 0, as_tuple=False).flatten()
    if len(ix):
        bound = b[ix]
        old_iou, new_iou = iou0[ix, bound], iou1[ix, bound]
        local[ix] = ((new_iou[:, None] >= thresholds).float().mean(-1) -
                     (old_iou[:, None] >= thresholds).float().mean(-1))
        other = (labels[ix, None] == gt_labels[None]).clone()
        other[torch.arange(len(ix), device=ix.device), bound] = False
        competitor = iou1[ix].masked_fill(~other, -1.).amax(-1)
        ok = (boxes1[ix, 2:] > boxes1[ix, :2]).all(-1)
        ok &= (labels[ix] == gt_labels[bound]) & (new_iou >= .5)
        ok &= competitor < (new_iou - identity_margin)
        if eligible is not None: ok &= eligible[ix]
        valid[ix] = ok
        target[ix[ok]] = measured[ix[ok]]
    return {'target_y': target.detach(), 'measured_u': measured.detach(),
            'identity_valid': valid.detach(), 'new_count_by_threshold': new.detach(),
            'lost_count_by_threshold': lost.detach(),
            'local_fixed_gt_delta': local.detach(), 'measured_delta': local.detach()}


def refinement_loss(boxes0: Tensor, boxes1: Tensor, gt_boxes: Tensor,
                    bindings: FixedBindings, eligible: Tensor, *,
                    giou_weight: float = 1., iou_weight: float = 1.,
                    l1_weight: float = 1., harm_weight: float = 2.) -> dict:
    """Refiner only: GT losses on fixed bound eligible items; old-.75 harm.

    L1 is normalised by ORIGINAL width/height. Unmatched/background items
    never receive an invented nearest-GT geometry target. Gate rejects them.
    """
    keep = eligible & (bindings.det_to_gt >= 0)
    zero = boxes1.sum() * 0
    if not bool(keep.any()):
        return {k: zero for k in ('loss', 'iou', 'giou', 'l1', 'old_success_harm')}
    b0, b1 = boxes0[keep].detach().float(), boxes1[keep].float()
    target = gt_boxes[bindings.det_to_gt[keep]].detach().float()
    iou1, giou1 = aligned_iou_giou(b1, target)
    iou0, _ = aligned_iou_giou(b0, target)
    normalizer = (b0[:, 2:] - b0[:, :2]).clamp_min(1e-3).repeat(1, 2)
    liou, lgiou = (1 - iou1).mean(), (1 - giou1).mean()
    ll1 = ((b1 - target).abs() / normalizer).mean()
    old = bindings.old_success[keep]
    harm = F.relu(iou0[old] - iou1[old]).mean() if bool(old.any()) else zero
    loss = iou_weight*liou + giou_weight*lgiou + l1_weight*ll1 + harm_weight*harm
    return {'loss': loss, 'iou': liou, 'giou': lgiou, 'l1': ll1, 'old_success_harm': harm}


def gain_loss(gain: Tensor, targets: dict, eligible: Tensor, *,
              balance_groups: bool = True) -> Tensor:
    """Unit-weight gate loss; beta=1. Default sign-balanced eligible targets.

    Each present positive/zero/negative target group contributes equally,
    independent of its item count. This is fixed identically in both arms,
    not selected after examining validation/test results. Empty input gives a
    differentiable zero. balance_groups=False is the explicit plain mean.
    """
    if not bool(eligible.any()): return gain.sum() * 0
    y = targets['target_y'][eligible].detach().float()
    losses = F.smooth_l1_loss(gain[eligible].float(), y, reduction='none', beta=1.)
    if not balance_groups: return losses.mean()
    grouped = [losses[mask].mean() for mask in (y > 0, y == 0, y < 0) if bool(mask.any())]
    return torch.stack(grouped).mean()


@torch.no_grad()
def apply_refinement(boxes0: Tensor, boxes1: Tensor, gain: Tensor, eligible: Tensor,
                     *, policy: str = 'gate', threshold: float = 0.,
                     clip_image_hw: Optional[Tuple[int, int]] = None) -> Tuple[Tensor, Tensor]:
    """Only replaces boxes. Caller keeps original q/class/score/order unchanged.

    Rejected/ineligible entries are bitwise originals even when clipping is
    requested. Default no extra clipping, preserving native postprocessing.
    ROIRefiner's tanh gain is bounded [-1,1]; strict gain>threshold with
    threshold=1 therefore guarantees all-reject for its outputs.
    """
    if policy not in ('gate', 'all', 'none'): raise ValueError('invalid policy')
    valid = torch.isfinite(boxes1).all(-1) & (boxes1[:, 2:] > boxes1[:, :2]).all(-1)
    # A zero-init/no-op proposal must remain an exact no-op even if explicit
    # clipping was requested and a native baseline box crosses the image edge.
    changed = (boxes1.to(boxes0.dtype) != boxes0).any(-1)
    accepted = eligible.bool() & valid & changed
    if policy == 'gate': accepted &= torch.isfinite(gain) & (gain > threshold)
    if policy == 'none': accepted = torch.zeros_like(accepted)
    proposal = boxes1.detach().to(boxes0.dtype)
    if clip_image_hw is not None:
        h, w = clip_image_hw
        proposal = proposal.clamp_min(0).minimum(proposal.new_tensor([w, h, w, h]))
        accepted &= (proposal[:, 2:] > proposal[:, :2]).all(-1)
    return torch.where(accepted[:, None], proposal, boxes0.detach()), accepted
