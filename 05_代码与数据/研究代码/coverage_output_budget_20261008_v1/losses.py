"""Training objectives; no optimizer construction, fitting, or score rewriting."""
import torch
from torch.nn import functional as F
from matching import coverage_assignment


def budget_boundary(scores, positives, budget=300, margin=0.1):
    """Per-image hinge, keeping the selected negative boundary differentiable."""
    if scores.ndim != 1 or budget <= 0 or margin <= 0:
        raise ValueError('Require vector scores and positive budget/margin')
    scores = scores.float()
    if not torch.isfinite(scores).all():
        raise ValueError('Nonfinite ranking scores')
    positives = torch.as_tensor(positives, device=scores.device, dtype=torch.long)
    if positives.ndim != 1 or len(positives) > budget or len(positives.unique()) != len(positives):
        raise ValueError('Positive set must be unique and fit the budget')
    if len(positives) and (positives.min() < 0 or positives.max() >= len(scores)):
        raise ValueError('Positive index outside score vector')
    if not len(positives) or len(scores) <= budget:
        return scores.sum() * 0
    mask = torch.ones(len(scores), dtype=torch.bool, device=scores.device)
    mask[positives] = False
    competitors = scores[mask]
    kth = budget - len(positives) + 1
    # topk indices are discrete; the gathered value retains its gradient.
    tau = competitors.topk(kth).values[-1]
    return (margin + tau - scores[positives]).clamp_min(0).mean()


def image_mean(values):
    if not values:
        raise ValueError('Empty batch')
    count = values[0].new_tensor(float(len(values)))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(count)
        count /= torch.distributed.get_world_size()
    return torch.stack(values).sum() / count.clamp_min(1)


def pairwise_iou(boxes, targets):
    # Half precision is insufficient around the fixed geometry threshold.
    a, b = boxes.float(), targets.float()
    a = torch.cat((a[:, :2] - a[:, 2:] / 2, a[:, :2] + a[:, 2:] / 2), -1)
    b = torch.cat((b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2), -1)
    inter = (torch.minimum(a[:, None, 2:], b[None, :, 2:]) -
             torch.maximum(a[:, None, :2], b[None, :, :2])).clamp_min(0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp_min(0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp_min(0).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp_min(1e-12)


def admission_losses(aux, targets, mode, budget=300, margin=0.1,
                     weight=0.1, residual_weight=1e-4, threshold=0.5):
    if mode not in ('cover', 'quality'):
        raise ValueError(mode)
    per_image, regularizers, diagnostics = [], [], []
    for b, target in enumerate(targets):
        scores = aux['priority'][b]
        with torch.no_grad():
            overlap = pairwise_iou(aux['boxes'][b], target['boxes'])
            eligible = (overlap >= threshold) & aux['valid'][b, :, None]
            eligible &= aux['classes'][b, :, None] == target['labels'][None, :]
            qi, gi = torch.where(eligible)
            edges = list(zip(qi.cpu().tolist(), gi.cpu().tolist(), overlap[qi, gi].cpu().tolist()))
            if mode == 'cover':
                pairs = coverage_assignment(scores.detach().float().cpu().tolist(), edges, budget)
                positives = [q for q, _ in pairs]
            else:
                quality = torch.where(eligible, overlap, torch.zeros_like(overlap))
                quality = quality.max(-1).values if quality.shape[1] else scores.new_zeros(len(scores))
                pairs, positives = [], []
        if mode == 'cover':
            value = budget_boundary(scores, positives, budget, margin)
        else:
            value = F.binary_cross_entropy_with_logits(scores.float(), quality.float())
        per_image.append(value)
        regularizers.append(aux['residual'][b].float().square().mean())
        diagnostics.append({'GT': len(target['labels']), 'eligible_edges': len(edges),
                            'representatives': pairs, 'mode': mode})
    return {'loss_admission_' + mode: weight * image_mean(per_image),
            'loss_admission_residual': residual_weight * image_mean(regularizers)}, diagnostics


def output_budget_loss(outputs, targets, indices, budget=300, margin=0.1,
                       weight=0.1, threshold=0.5):
    logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
    values, diagnostics = [], []
    for b, (queries, gt_ids) in enumerate(indices):
        queries = queries.to(logits.device)
        gt_ids = gt_ids.to(logits.device)
        with torch.no_grad():
            overlap = pairwise_iou(boxes[b, queries], targets[b]['boxes'][gt_ids]).diag()
            keep = overlap >= threshold
            selected_queries = queries[keep]
            selected_classes = targets[b]['labels'][gt_ids[keep]]
            positive_entries = selected_queries * logits.shape[-1] + selected_classes
        values.append(budget_boundary(logits[b].flatten(), positive_entries, budget, margin))
        diagnostics.append({'native_matched': len(queries), 'qualified_positive_pairs': len(positive_entries),
                            'positive_entries': positive_entries.cpu().tolist()})
    return {'loss_output_budget': weight * image_mean(values)}, diagnostics
