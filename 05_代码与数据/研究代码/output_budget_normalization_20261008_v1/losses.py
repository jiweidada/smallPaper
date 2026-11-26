"""Final-output budget objective with two explicitly isolated normalizers.

No optimizer, checkpoint access, or parameter updates. Eligibility is detached;
the selected competing logit remains differentiable, exactly as in the old B.
"""
import math
import torch


def boundary_terms(scores, positives, budget=300, margin=0.1):
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
        return scores[positives] * 0, None
    mask = torch.ones(len(scores), dtype=torch.bool, device=scores.device)
    mask[positives] = False
    competitors = scores[mask]
    kth = budget - len(positives) + 1
    tau = competitors.topk(kth).values[-1]
    return (margin + tau - scores[positives]).clamp_min(0), tau


def paired_iou(boxes, targets):
    """Matched pairs only; no quadratic pairwise matrix is required."""
    if boxes.shape != targets.shape or boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError('Expected equally shaped matched cxcywh boxes')
    a, b = boxes.float(), targets.float()
    a = torch.cat((a[:, :2] - a[:, 2:] / 2, a[:, :2] + a[:, 2:] / 2), -1)
    b = torch.cat((b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2), -1)
    inter = (torch.minimum(a[:, 2:], b[:, 2:]) -
             torch.maximum(a[:, :2], b[:, :2])).clamp_min(0).prod(-1)
    aa = (a[:, 2:] - a[:, :2]).clamp_min(0).prod(-1)
    ab = (b[:, 2:] - b[:, :2]).clamp_min(0).prod(-1)
    return inter / (aa + ab - inter).clamp_min(1e-12)


def output_budget_loss(outputs, targets, indices, native_num_boxes,
                       normalization, budget=300, margin=0.1,
                       weight=0.1, threshold=0.5):
    if normalization not in ('image', 'gt'):
        raise ValueError('Unknown normalization')
    if not math.isfinite(float(native_num_boxes)) or native_num_boxes < 1:
        raise ValueError('Use the clamped native MAL normalizer')
    if weight < 0 or not 0 <= threshold <= 1:
        raise ValueError('Invalid weight or geometry threshold')
    logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
    batch = logits.shape[0]
    if batch == 0 or not len(targets) == len(indices) == batch:
        raise ValueError('Nonempty, consistent batch required')
    sums, means, diagnostics = [], [], []
    for b, (queries, gt_ids) in enumerate(indices):
        queries, gt_ids = queries.to(logits.device), gt_ids.to(logits.device)
        with torch.no_grad():
            overlap = paired_iou(boxes[b, queries], targets[b]['boxes'][gt_ids])
            keep = overlap >= threshold
            positive_entries = (queries[keep] * logits.shape[-1] +
                                targets[b]['labels'][gt_ids[keep]])
        terms, tau = boundary_terms(logits[b].flatten(), positive_entries, budget, margin)
        # Empty images contribute a differentiable zero to the IMAGE mean too.
        total = terms.sum()
        sums.append(total)
        means.append(terms.mean() if len(terms) else total)
        diagnostics.append({
            'native_matched': len(queries), 'qualified_positive_pairs': len(positive_entries),
            'active_hinges': int((terms.detach() > 0).sum().item()),
            'positive_entries': positive_entries.cpu().tolist(),
            'hinge_sum': float(total.detach()),
            'tau': None if tau is None else float(tau.detach()),
        })
    image_count = logits.new_tensor(float(batch), dtype=torch.float32)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(image_count)
        image_count /= torch.distributed.get_world_size()
    if normalization == 'image':
        value = torch.stack(means).sum() / image_count.clamp_min(1)
    else:
        # Captured from the same final MAL call; do not use GO-union cardinality.
        value = torch.stack(sums).sum() / native_num_boxes
    for d in diagnostics:
        m = d['qualified_positive_pairs']
        scale = (weight / (float(image_count) * m) if normalization == 'image' and m
                 else weight / native_num_boxes if m else 0.0)
        d.update(normalization=normalization, native_num_boxes=float(native_num_boxes),
                 active_positive_logit_gradient_magnitude=scale,
                 relative_to_mal=scale * native_num_boxes,
                 boundary_gradient_magnitude=scale * d['active_hinges'])
    return {'loss_output_budget': weight * value}, diagnostics
