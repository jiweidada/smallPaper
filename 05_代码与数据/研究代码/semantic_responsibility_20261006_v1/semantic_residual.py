"""Isolated, untrained research prototype. No claim of novelty or effectiveness.

The original detector's losses use original outputs. Auxiliary CE updates only
this head; detached features/logits protect the native parameter path. Inference
takes features and logits only. It preserves a query's maximum score, not its
class, flattened query/class TopK, NMS decisions, or precision.
"""
import torch
from torch import nn
import torch.nn.functional as F


def retain_query_max(base_logits, residual):
    if base_logits.shape != residual.shape or base_logits.shape[-1] < 2:
        raise ValueError('Logit and residual shapes must match with >=2 classes.')
    if base_logits.dtype != residual.dtype:
        raise ValueError('Use a common input dtype explicitly.')
    if not torch.isfinite(base_logits).all() or not torch.isfinite(residual).all():
        raise ValueError('Nonfinite inputs are not supported.')
    dtype = base_logits.dtype
    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    z, r = base_logits.to(work_dtype), residual.to(work_dtype)
    maximum = z.amax(-1, keepdim=True)
    shifted = z + r
    # This spelling makes zero residual an exact identity on finite inputs.
    out = z + (r - (shifted.amax(-1, keepdim=True) - maximum))
    # Correct only roundoff at the winning coordinate; float16 range is bounded.
    out = out.clamp(min=torch.finfo(dtype).min).to(dtype)
    maximum = maximum.to(dtype)
    out = torch.minimum(out, maximum)
    out = out.scatter(-1, shifted.argmax(-1, keepdim=True), maximum)
    if not torch.isfinite(out).all():
        raise ValueError('Output is not representable in requested dtype.')
    return out


class SemanticResidual(nn.Module):
    def __init__(self, features=256, width=64, classes=10):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(features, width), nn.SiLU(), nn.Linear(width, classes))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def residual(self, query_features):
        return self.head(query_features.detach())

    def forward(self, query_features, native_logits):
        return retain_query_max(native_logits, self.residual(query_features).to(native_logits.dtype))

    def auxiliary_scores(self, query_features, native_logits):
        return native_logits.detach() + self.residual(query_features).to(native_logits.dtype)


@torch.no_grad()
def stable_geometry_labels(enc_boxes, pre_boxes, gt_boxes, gt_labels, visible_fraction,
                           assigned_gt, iou_threshold=.5, min_visible=.999):
    """Training only; boxes are normalized xyxy, matcher indices local to GT.

    Returns geometry GT indices and semantic labels, or -1 for excluded queries.
    Positive targets assigned to a different class are excluded. No predictions
    of classes enter geometry eligibility. Visibility must come from original GT,
    not be recomputed from an already-clipped area as if it were the original.
    """
    from torchvision.ops import box_iou
    nq, ng = len(enc_boxes), len(gt_boxes)
    if enc_boxes.shape != pre_boxes.shape or enc_boxes.shape != (nq, 4):
        raise ValueError('Expected paired query boxes.')
    if assigned_gt.shape != (nq,) or visible_fraction.shape != (ng,) or gt_labels.shape != (ng,):
        raise ValueError('Target metadata shape mismatch.')
    if ((assigned_gt < -1) | (assigned_gt >= ng)).any():
        raise ValueError('Invalid matcher target index.')
    if not torch.isfinite(visible_fraction).all() or ((visible_fraction < 0) | (visible_fraction > 1.00001)).any():
        raise ValueError('Invalid visibility fraction.')
    labels = torch.full((nq,), -1, dtype=torch.long, device=enc_boxes.device)
    identity = labels.clone()
    if ng == 0 or nq == 0:
        return identity, labels
    ie, ip = box_iou(enc_boxes, gt_boxes), box_iou(pre_boxes, gt_boxes)
    je, jp = ie.argmax(1), ip.argmax(1)
    eligible = ((ie >= iou_threshold).sum(1) == 1) & ((ip >= iou_threshold).sum(1) == 1)
    eligible &= (je == jp) & (visible_fraction[je] >= min_visible)
    assigned = assigned_gt >= 0
    conflict = assigned & (gt_labels[assigned_gt.clamp_min(0)] != gt_labels[je])
    eligible &= ~conflict
    identity[eligible] = je[eligible]
    labels[eligible] = gt_labels[je[eligible]]
    return identity, labels


def instance_balanced_semantic_loss(scores, labels, geometry_gt):
    """One crop at a time: average within GT, then across covered GT.

    No class balancing, no hard-error selection, and no duplicate-count gain.
    Crop overlap can still repeat an object across images; this is not globally
    instance-balanced over an epoch.
    """
    valid = labels >= 0
    if not torch.equal(valid, geometry_gt >= 0):
        raise ValueError('Label/instance eligibility mismatch.')
    if not valid.any():
        return scores.sum() * 0
    values = F.cross_entropy(scores[valid], labels[valid], reduction='none')
    _, inverse, counts = geometry_gt[valid].unique(return_inverse=True, return_counts=True)
    return (values / counts[inverse]).sum() / len(counts)
