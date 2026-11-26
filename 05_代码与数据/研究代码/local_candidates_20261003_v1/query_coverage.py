"""Fixed-budget, GT-free P3/space coverage selection and optional candidate trace."""
import torch
import bootstrap
from engine.deim.dfine_decoder import DFINETransformer


@torch.no_grad()
def coverage_indices(logits, anchors_unact, k=300, reserve=75, grid=4, max_logit_gap=2.):
    if not 0 <= reserve <= k or grid <= 0 or max_logit_gap < 0:
        raise ValueError('Invalid coverage parameters')
    scores = logits.max(-1).values
    if reserve == 0:
        # Exact native default including its tie behaviour.
        return torch.topk(scores, k, dim=-1).indices
    if anchors_unact.shape[0] == 1:
        anchors_unact = anchors_unact.expand(logits.shape[0], -1, -1)
    if anchors_unact.shape[:2] != scores.shape:
        raise ValueError('Anchor/token mismatch')
    valid = torch.isfinite(scores) & torch.isfinite(anchors_unact).all(-1)
    if (valid.sum(-1) < k).any():
        raise ValueError('Fewer than k valid candidates; do not silently fill invalid tokens')
    scores = scores.masked_fill(~valid, -torch.inf)
    anchors = anchors_unact.sigmoid()
    areas = anchors[..., 2:].prod(-1)
    # Smallest anchor group denotes P3, not predicted object size or GT size.
    smallest = areas.masked_fill(~valid, torch.inf).amin(-1, keepdim=True)
    p3 = valid & torch.isclose(areas, smallest, rtol=1e-4, atol=1e-8)
    cells = (anchors[..., :2] * grid).long().clamp(0, grid - 1)
    cells = cells[..., 1] * grid + cells[..., 0]
    ranked = torch.argsort(scores, dim=-1, descending=True, stable=True)
    result = []
    for b in range(scores.shape[0]):
        selected = ranked[b, :k - reserve]
        used = torch.zeros(scores.shape[1], dtype=torch.bool, device=scores.device)
        used[selected] = True
        eligible = p3[b] & ~used & (scores[b] >= scores[b].max() - max_logit_gap)
        pools = [ranked[b][eligible[ranked[b]] & (cells[b, ranked[b]] == c)] for c in range(grid * grid)]
        extra = []
        # Each round takes at most one per cell, ranking cell winners by score.
        # Empty/weak cells get no forced background query. Remaining slots backfill.
        for depth in range(reserve):
            candidates = [p[depth] for p in pools if len(p) > depth]
            if not candidates or len(extra) >= reserve:
                break
            candidates = torch.stack(candidates)
            candidates = candidates[torch.argsort(scores[b, candidates], descending=True, stable=True)]
            extra.extend(candidates[:reserve - len(extra)].unbind())
        if extra:
            covered = torch.stack(extra)
            used[covered] = True
            selected = torch.cat((selected, covered))
        remaining = ranked[b][~used[ranked[b]] & valid[b, ranked[b]]][:k - len(selected)]
        selected = torch.cat((selected, remaining))
        # Preserve score-descending presentation without changing the selected set.
        selected = selected[torch.argsort(scores[b, selected], descending=True, stable=True)]
        result.append(selected)
    return torch.stack(result)


class DFINECoverageTransformer(DFINETransformer):
    def __init__(self, decoder_kwargs, reserve=75, grid=4, max_logit_gap=2., trace=False):
        if decoder_kwargs.get('query_select_method', 'default') != 'default':
            raise ValueError('Coverage candidate only supports native default selection')
        if decoder_kwargs.get('mgsa_groups', 0) or decoder_kwargs.get('o2m_t', False):
            raise ValueError('Do not combine coverage with group supervision')
        if decoder_kwargs.get('num_queries', 300) != 300:
            raise ValueError('This candidate fixes the matching-query budget at 300')
        super().__init__(**decoder_kwargs)
        self.reserve, self.grid, self.max_logit_gap = reserve, grid, max_logit_gap
        self.trace_enabled, self.last_query_trace = trace, None

    def _select_topk(self, memory, outputs_logits, outputs_anchors_unact, topk):
        indices = coverage_indices(outputs_logits.detach(), outputs_anchors_unact.detach(),
                                   topk, self.reserve, self.grid, self.max_logit_gap)
        anchors = outputs_anchors_unact
        if anchors.shape[0] == 1:
            anchors = anchors.expand(memory.shape[0], -1, -1)
        if self.trace_enabled:
            with torch.no_grad():
                # Extra dense bbox-head computation is diagnostic only, excluded
                # from normal timing/capacity and never consulted by the selector.
                boxes = (self.enc_bbox_head(memory.detach()) + anchors).sigmoid()
                self.last_query_trace = {
                    'logits': outputs_logits.detach().cpu(), 'boxes': boxes.cpu(),
                    'anchors': anchors.sigmoid().detach().cpu(), 'selected_indices': indices.cpu(),
                    'valid': torch.isfinite(anchors).all(-1).cpu(),
                }
        def gather(value):
            return value.gather(1, indices[..., None].expand(-1, -1, value.shape[-1]))
        return gather(memory), gather(outputs_logits) if self.training else None, gather(anchors)

