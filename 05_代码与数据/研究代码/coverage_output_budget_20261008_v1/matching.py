"""Exact cardinality/score/IoU assignment. No detector or trainable parameters."""
import heapq
import math


def _dyadic_integers(values):
    pairs = [float(x).as_integer_ratio() for x in values]
    denominator = max((d for _, d in pairs), default=1)
    return [n * (denominator // d) for n, d in pairs]


def coverage_assignment(priorities, edges, budget):
    """Return (token, GT) pairs with cardinality <= budget.

    Edges are (token_id, gt_id, IoU). Objectives are lexicographic:
    maximum cardinality; maximum total token priority; maximum total IoU;
    minimum sum of ordered token/GT ranks. Remaining ties use fixed edge order.
    IEEE scores are converted to exact dyadic integers, not epsilon-weighted.
    """
    if budget < 0:
        raise ValueError('Negative budget')
    edges = sorted(edges)
    if not edges or budget == 0:
        return []
    if len({(i, j) for i, j, _ in edges}) != len(edges):
        raise ValueError('Duplicate graph edge')
    tokens = sorted({i for i, _, _ in edges})
    gts = sorted({j for _, j, _ in edges})
    if any(i < 0 or i >= len(priorities) for i in tokens) or any(j < 0 for j in gts):
        raise ValueError('Invalid graph ID')
    if any(not math.isfinite(float(priorities[i])) for i in tokens):
        raise ValueError('Nonfinite priority')
    if any(not math.isfinite(v) or not 0 <= v <= 1 for _, _, v in edges):
        raise ValueError('Invalid IoU')
    k = min(budget, len(tokens), len(gts))
    token_order = {v: i for i, v in enumerate(tokens)}
    gt_order = {v: i for i, v in enumerate(gts)}
    score_int = _dyadic_integers([priorities[i] for i in tokens])
    iou_int = _dyadic_integers([v for _, _, v in edges])
    max_rank = len(tokens) * (len(gts) + 1) + len(gts)
    iou_factor = k * max_rank + 1
    score_factor = k * max(iou_int) * iou_factor + k * max_rank + 1
    source = 0
    gt_offset = 1 + len(tokens)
    sink = gt_offset + len(gts)
    graph = [[] for _ in range(sink + 1)]

    def add(a, b, cost):
        forward = [b, len(graph[b]), 1, cost]
        backward = [a, len(graph[a]), 0, -cost]
        graph[a].append(forward)
        graph[b].append(backward)
        return forward

    potential = [0] * len(graph)
    for r, i in enumerate(tokens):
        cost = -score_int[r] * score_factor
        add(source, 1 + r, cost)
        potential[1 + r] = cost
    gt_min = {}
    references = []
    for e, (i, j, _) in enumerate(edges):
        ti, gj = token_order[i], gt_order[j]
        cost = -iou_int[e] * iou_factor + ti * (len(gts) + 1) + gj
        a, b = 1 + ti, gt_offset + gj
        ref = add(a, b, cost)
        references.append((i, j, ref))
        gt_min[b] = min(gt_min.get(b, potential[a] + cost), potential[a] + cost)
    for b, v in gt_min.items():
        potential[b] = v
        add(b, sink, 0)
    potential[sink] = min(gt_min.values())
    flow = 0
    while flow < k:
        distance = [None] * len(graph)
        previous = [None] * len(graph)
        distance[source] = 0
        queue = [(0, source)]
        while queue:
            dist, a = heapq.heappop(queue)
            if distance[a] != dist:
                continue
            for ei, (b, _, capacity, cost) in enumerate(graph[a]):
                if not capacity:
                    continue
                reduced = cost + potential[a] - potential[b]
                if reduced < 0:
                    raise RuntimeError('Invalid flow potential')
                nd = dist + reduced
                if distance[b] is None or nd < distance[b]:
                    distance[b] = nd
                    previous[b] = (a, ei)
                    heapq.heappush(queue, (nd, b))
        if distance[sink] is None:
            break
        for i, d in enumerate(distance):
            if d is not None:
                potential[i] += d
        b = sink
        while b != source:
            a, ei = previous[b]
            edge = graph[a][ei]
            edge[2] -= 1
            graph[b][edge[1]][2] += 1
            b = a
        flow += 1
    result = [(i, j) for i, j, edge in references if edge[2] == 0]
    assert len(result) == flow
    return sorted(result)
