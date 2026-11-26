"""191: exact dyadic lexicographic transport on fixed cached ownership budgets.

Written only; root reviews and executes. No detection model/checkpoint/gradient.
NetworkX integer min-cost flow plus an independently checked integer dual.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import ast
import hashlib
import importlib.util
import itertools
import logging
import pathlib
import sys
import time
import traceback
from fractions import Fraction
from typing import Dict

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent
PROTOCOL = ROOT / '191_支持域完整分配与强对照_事前协议.txt'
REUSE = ROOT / '183_C0_sameclass_support_bound.py'
REUSE_SHA = '800f840b20adbb6569633dc71dd552a64ee12a52dcde7e33544ed531c17fb5f4'
GO = ROOT / '188_C0_true_GO_support_cache'
GO_SHA = '8a100f920dff9c961cc13ac5814e338dd37fc0bf5f390cc453a607f546b0ced9'
COLLECTOR = ROOT / '187_C0_collect_true_GO_support.py'
COLLECTOR_SHA = '16bd61b9e2397ae9ed6cff29d85e5615e1b4da4569ee6f08929e3694a0b08c5a'
NX_SHA = 'df6b9eb686568feffb28f3b6389a6eb4fd05a53aec43fa89c67b42f17c4856bd'
if hashlib.sha256(REUSE.read_bytes()).hexdigest() != REUSE_SHA:
    raise RuntimeError('frozen 183 changed')
_spec = importlib.util.spec_from_file_location('_audit192_reuse183', REUSE)
if _spec is None or _spec.loader is None:
    raise RuntimeError('183 import unavailable')
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)  # Definitions only; no historical main is invoked.
A, H = B.A, B.H
np, torch, psutil = B.np, B.torch, B.psutil
require, sha, save, array = B.require, B.sha, B.save, B.array
import networkx as nx

ARMS = ('G0', 'G', 'final_IoU_face', 'ref_IoU_face', 'native_cost_face')
CONTROLS = ('G0', 'final_IoU_face', 'ref_IoU_face', 'native_cost_face')
DOMAINS = ('physical64', 'tight', 'expanded', 'native32')
METRICS = ('outside', 'v', 'final_IoU', 'ref_IoU', 'native_cost', 'owner_changes')
RSS_PEAK = 0
FLOW_STATS = []


def guard():
    global RSS_PEAK
    A.memory_guard()
    RSS_PEAK = max(RSS_PEAK, psutil.Process().memory_info().rss)


def rational(value):
    return Fraction(*float(value).as_integer_ratio())


def exact_sum(values):
    return sum((rational(x) for x in values), Fraction(0))


def fraction_report(value):
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator), 'float': float(value)}


def fraction_read(value):
    return Fraction(int(value['numerator']), int(value['denominator']))


def dyadic_level(matrix):
    matrix = np.asarray(matrix)
    require(matrix.ndim == 2 and matrix.size > 0 and np.isfinite(matrix).all(), 'finite objective matrix')
    ratios = [[float(value).as_integer_ratio() for value in row] for row in matrix]
    denominator = max(d for row in ratios for _, d in row)
    require(denominator & (denominator - 1) == 0, 'dyadic denominator')
    integers = [[n * (denominator // d) for n, d in row] for row in ratios]
    shifts = [min(row) for row in integers]
    shifted = [[value - shift for value in row] for row, shift in zip(integers, shifts)]
    bound = sum(max(row) for row in shifted)
    return shifted, {'denominator': str(denominator), 'row_shift_sum': str(sum(shifts)),
                     'total_nonnegative_upper_bound': str(bound)}


def pack_objectives(objectives):
    levels, proof = [], []
    shape = np.asarray(objectives[0][1]).shape
    for name, matrix in objectives:
        require(np.asarray(matrix).shape == shape, 'objective shapes')
        integer, meta = dyadic_level(matrix)
        levels.append(integer)
        proof.append(dict(meta, name=name))
    packed = [row[:] for row in levels[0]]
    for level, meta in zip(levels[1:], proof[1:]):
        base = int(meta['total_nonnegative_upper_bound']) + 1
        meta['packing_base'] = str(base)
        packed = [[old * base + new for old, new in zip(oldrow, newrow)] for oldrow, newrow in zip(packed, level)]
    return packed, proof, levels


def dual_certificate(cost, cols, capacities):
    """Exact transport dual, independent of network_simplex's private internals."""
    n, m = len(cost), len(cost[0])
    # All capacities are positive, so every GT has at least one assigned q.
    arcs = [[None] * m for _ in range(m)]
    for q, g in enumerate(cols):
        for h in range(m):
            value = cost[q][h] - cost[q][g]
            if arcs[g][h] is None or value < arcs[g][h]:
                arcs[g][h] = value
    require(all(value is not None for row in arcs for value in row), 'dual missing occupied GT')
    potential = [0] * m
    rounds = 0
    for rounds in range(1, m + 1):
        changed = False
        for g in range(m):
            for h in range(m):
                proposed = potential[g] + arcs[g][h]
                if proposed < potential[h]:
                    potential[h], changed = proposed, True
        if not changed:
            break
    require(not changed, 'negative residual cost cycle / no exact optimality certificate')
    row_dual = [cost[q][g] - potential[g] for q, g in enumerate(cols)]
    require(all(row_dual[q] + potential[g] <= cost[q][g] for q in range(n) for g in range(m)), 'dual edge inequality')
    primal = sum(cost[q][g] for q, g in enumerate(cols))
    dual = sum(row_dual) + sum(int(cap) * pot for cap, pot in zip(capacities, potential))
    require(primal == dual, 'integer primal-dual gap')
    return {'q_dual': [str(x) for x in row_dual], 'GT_dual': [str(x) for x in potential],
            'primal': str(primal), 'dual': str(dual), 'zero_exact_gap': True, 'Bellman_Ford_rounds': rounds}


def exact_flow(objectives, capacities, owner_cols):
    guard()
    started = time.monotonic()
    n, m = np.asarray(objectives[0][1]).shape
    caps = [int(x) for x in capacities]
    require(len(caps) == m and all(x > 0 for x in caps) and sum(caps) == n, 'transport budget')
    require(np.bincount(owner_cols, minlength=m).tolist() == caps, 'G0 budget')
    change = np.arange(m)[None, :] != np.asarray(owner_cols)[:, None]
    all_objectives = list(objectives) + [('owner_changes', change.astype(np.int64))]
    cost, proof, levels = pack_objectives(all_objectives)
    graph = nx.DiGraph()
    for q in range(n):
        graph.add_node(('q', q), demand=-1)
    for g, cap in enumerate(caps):
        graph.add_node(('g', g), demand=cap)
    for q in range(n):
        for g in range(m):
            graph.add_edge(('q', q), ('g', g), capacity=1, weight=cost[q][g])
    returned_cost, flow = nx.network_simplex(graph)
    require(type(returned_cost) is int, 'network simplex cost must be Python int')
    cols = []
    for q in range(n):
        row = flow[('q', q)]
        require(all(type(x) is int and x in (0, 1) for x in row.values()) and sum(row.values()) == 1, 'integral q flow')
        cols.append(next(g for g in range(m) if row[('g', g)] == 1))
    require(np.bincount(cols, minlength=m).tolist() == caps, 'GT multiplicity changed')
    require(returned_cost == sum(cost[q][g] for q, g in enumerate(cols)), 'packed cost mismatch')
    sums = []
    for (name, matrix), meta, integer in zip(all_objectives, proof, levels):
        shifted_sum = sum(integer[q][g] for q, g in enumerate(cols))
        numerator = shifted_sum + int(meta['row_shift_sum'])
        total = Fraction(numerator, int(meta['denominator']))
        require(total == exact_sum(np.asarray(matrix)[np.arange(n), cols]), 'objective decoding')
        sums.append(dict(meta, exact_sum=fraction_report(total), shifted_integer_sum=str(shifted_sum)))
    cert = dual_certificate(cost, cols, caps)
    report = {'objectives': sums, 'packed_cost': str(returned_cost), 'dual_certificate': cert,
              'maximum_edge_weight_bits': max(value.bit_length() for row in cost for value in row),
              'nodes': n + m, 'edges': n * m, 'seconds': time.monotonic() - started,
              'final_tie': 'minimum owner changes, then fixed NetworkX sorted insertion; not lexicographic ID optimum'}
    FLOW_STATS.append({key: report[key] for key in ('maximum_edge_weight_bits', 'nodes', 'edges', 'seconds')})
    guard()
    return np.asarray(cols, dtype=np.int64), report


def native_cost_reader(box_ops):
    """Only the source matcher class is instantiated, never a detection model."""
    observed = []
    def solver(cost):
        observed.append(cost.detach().cpu().clone() if torch.is_tensor(cost) else torch.as_tensor(cost).clone())
        return A.linear_sum_assignment(cost)
    namespace = {'__name__': '_audit192_native_matcher', 'torch': torch, 'nn': torch.nn,
                 'F': torch.nn.functional, 'np': np, 'os': os, 'Dict': Dict,
                 'linear_sum_assignment': solver, 'register': lambda: (lambda cls: cls),
                 'logger': logging.getLogger('audit192'), 'VIS_MATCH': False, 'point_sample': A.forbidden}
    for name in ('batch_dice_loss', 'batch_sigmoid_ce_loss', 'box_cxcywh_to_xyxy', 'generalized_box_iou',
                 'box_iou', 'generalized_box_inner_iou'):
        namespace[name] = getattr(box_ops, name)
    tree = ast.parse((A.NATIVE / 'matcher.py').read_text(encoding='utf-8-sig'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'HungarianMatcher')
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(A.NATIVE / 'matcher.py'), 'exec'), namespace)
    _, config = A.read_configuration()
    matcher = namespace['HungarianMatcher'](**config, use_focal_loss=True)
    def compute(boxes, logits, target):
        observed.clear()
        result = matcher({'pred_boxes': boxes[None], 'pred_logits': logits[None]}, [target], epoch=94)
        require(len(observed) == 1 and observed[0].dtype == torch.float32, 'one native FP32 cost capture')
        cost = observed[0].numpy()
        require(np.isfinite(cost).all(), 'native cost finite')
        return cost, pairs_array(result['indices'])
    return compute


def pairs_array(indices):
    require(len(indices) == 1 and len(indices[0]) == 2, 'one-image native index schema')
    q, g = [array(x, 'indices', integer=True) for x in indices[0]]
    require(q.ndim == g.ndim == 1 and q.shape == g.shape, 'q/GT indices shape')
    return np.stack((q, g), axis=1)


def synthetic_checks(native, compute_cost):
    checks = B.self_checks(native)
    cases = [
        ([('primary', np.array([[0., 1.], [1., 0.]])), ('huge_lower', np.array([[1e100, 0.], [0., 1e100]]))], [1, 1], [0, 1]),
        ([('tiny_primary', np.array([[2.**-80, 0.], [0., 0.]])), ('lower', np.array([[0., 1.], [1., 0.]]))], [1, 1], [0, 1]),
        ([('outside', np.array([[0., 1.], [1., 0.], [0., 1.]]))], [2, 1], [0, 0, 1]),
        ([('outside', np.array([[0., 0.], [1., 1.]]))], [1, 1], [0, 1]),
        ([('negative_IoU', np.array([[-.5, -.5], [-.5, -.5]], dtype=np.float32)), ('outside', np.array([[1, 0], [0, 1]]))], [1, 1], [0, 1]),
    ]
    chosen = []
    for objectives, caps, owner in cases:
        cols, proof = exact_flow(objectives, caps, owner)
        candidates = [p for p in itertools.product(range(len(caps)), repeat=sum(caps)) if np.bincount(p, minlength=len(caps)).tolist() == caps]
        all_obj = objectives + [('owner_changes', np.arange(len(caps))[None, :] != np.asarray(owner)[:, None])]
        def value(p):
            return tuple(exact_sum(np.asarray(mat)[np.arange(len(p)), p]) for _, mat in all_obj)
        require(value(cols) == min(value(p) for p in candidates), 'flow not brute-force lex optimal')
        require(proof['dual_certificate']['zero_exact_gap'], 'synthetic exact certificate')
        chosen.append(cols.tolist())
    require(chosen[0] == [0, 1] and chosen[1] == [1, 0] and chosen[4] == [1, 0], 'priority / IoU face')
    # Separate class transports cannot change q class, and may share a GT within class.
    budgets = [(np.array([1, 4, 8]), np.array([2, 7]), [2, 1], [0, 0, 1], 3),
               (np.array([2]), np.array([5]), [1], [0], 6)]
    rebuilt = []
    for qs, gs, caps, owner, label in budgets:
        cols, _ = exact_flow([('zero', np.zeros((len(qs), len(gs))))], caps, owner)
        rebuilt.extend((int(q), int(gs[g]), label) for q, g in zip(qs, cols))
    require(len({q for q, _, _ in rebuilt}) == 4 and sum(g == 2 for _, g, _ in rebuilt) == 2
            and all((q in (1, 4, 8)) == (c == 3) for q, _, c in rebuilt), 'same class/multiple GT budget')
    boxes = torch.tensor([[.5, .5, .2, .2], [.3, .3, .1, .1]], dtype=torch.float32)
    target = {'boxes': boxes.flip(0), 'labels': torch.tensor([0, 1])}
    cost, pairs = compute_cost(boxes, torch.zeros((2, 10)), target)
    require(cost.shape == (2, 2) and len(pairs) == 2, 'native cached matcher cost capture')
    checks.update(exact_dyadic_priority=True, exact_dual_and_bruteforce=True, GT_multiplicity_and_same_class=True,
                  non_simultaneous_takeover=True, IoU_optimal_face=True, native_cost_capture=True)
    return checks


def verify_collector(cache_sha):
    manifest = H.json.loads((GO / '05_output_sha256.json').read_text(encoding='utf-8-sig'))
    require(manifest.get('01_cache.pt') == cache_sha == GO_SHA, '188 cache SHA/manifest')
    for name in ('01_cache.pt', '02_summary.json', '04_execution.json', '00_inputs_sha256.json'):
        require(sha(GO / name) == manifest[name], '188 artifact hash: ' + name)
    summary = H.json.loads((GO / '02_summary.json').read_text(encoding='utf-8-sig'))
    execution = H.json.loads((GO / '04_execution.json').read_text(encoding='utf-8-sig'))
    require(summary['state'] == execution['state'] == 'complete', 'collector incomplete')
    require(execution['source_unchanged'] and all(value is True for value in execution['source_unchanged'].values()), 'collector sources changed')
    require(set(execution['source_hashes']) == set(execution['source_unchanged']), 'source provenance keys')
    for prefix in ('model_state', 'BN_state', 'criterion_state'):
        require(execution[prefix + '_before'] == execution[prefix + '_after'], 'collector state changed')
    require(execution['checkpoint_unchanged'] is True and execution['new_training'] is False
            and execution['optimizer_constructed'] is False and execution['heldout_reads'] is False, 'collector scope')
    counts = execution['counts']
    require(counts == {'model_forwards': 4, 'criterion_calls': 4, 'model_backwards': 0, 'model_vjps': 0,
                       'cache_autograd': 0, 'optimizer_steps': 0, 'eval_forwards': 0}, 'collector execution budget')
    require(execution['mode'] == {'model': 'eval', 'backbone': 'eval', 'encoder': 'eval', 'decoder': 'train',
        'all_BN': 'eval', 'torch_no_grad': True, 'mha_fastpath_enabled': False}, 'collector mode')
    require(summary['GT'] == 902 and summary['original_sub4'] == 453 and not summary['method_effect_evaluated'], 'collector population')
    source_matches = [digest for path, digest in execution['source_hashes'].items() if pathlib.Path(path).resolve() == COLLECTOR.resolve()]
    require(source_matches == [COLLECTOR_SHA], 'collector script identity')
    return summary, execution


def verify_GO_cache(cache, summary, execution):
    require(cache['schema'] == 'true-GO-train-ref-v1' and cache['checkpoint_sha256'] == A.CHECKPOINT_SHA
            and cache['weight_key'] == 'model_online' and cache['epoch'] == 94, '188 cache identity')
    require(cache['source_hashes'] == execution['source_hashes'] and cache['counts'] == execution['counts']
            and cache['mode'] == execution['mode'] and cache['heldout_reads'] is False, '188 cache provenance')
    for key, prefix in [('model_state_sha256', 'model_state'), ('BN_state_sha256', 'BN_state'), ('criterion_state_sha256', 'criterion_state')]:
        require(cache[key] == execution[prefix + '_before'], '188 cache state provenance')
    records = sorted(cache['records'], key=lambda x: int(x['image_index']))
    require([int(x['image_index']) for x in records] == list(range(4)), 'GO four image IDs')
    originals = {}
    for record, expected_n, expected_unique, summary_row in zip(records, (172, 290, 300, 70), (146, 265, 280, 58), summary['per_image']):
        index, match = int(record['image_index']), record['matching']
        pairs = pairs_array(match['indices_go'])
        gs = pairs[:, 1]
        require(len(pairs) == expected_n and len(set(gs.tolist())) == expected_unique and len(set(pairs[:, 0])) == expected_n, 'actual GO budgets')
        require(match['num_boxes_go'] == expected_n and match['num_boxes_ordinary'] == len(record['target']['boxes']), 'native denominators')
        require(np.array_equal(pairs, pairs_array(match['GO_native_call']['indices_go'])), 'native GO call mismatch')
        metadata = match['GO_summary']
        require(np.array_equal(array(metadata['GT_multiplicity'], 'm_g', integer=True), np.bincount(gs, minlength=len(record['target']['boxes'])))
                and metadata['query_unique'] is True and metadata['pairs'] == expected_n, 'GO multiplicity summary')
        require(record['reference_origin'] == "this true train forward output['ref_points'][0]; never borrowed from eval"
                and record['native_selection_gather_exact'] is True, 'native reference/selection source')
        outputs = record['native_outputs']
        for key, native_key in [('pred_boxes', 'pred_boxes'), ('pred_logits', 'pred_logits'), ('ref_points_initial', 'ref_points')]:
            require(torch.equal(record[key], outputs[native_key][0]), 'cached/native final field mismatch')
        require(torch.equal(record['ref_points_initial'], outputs['pre_outputs']['pred_boxes'][0]), 'pre/FDR ref mismatch')
        for stage in outputs['aux_outputs']:
            require(torch.equal(record['ref_points_initial'], stage['ref_points'][0]), 'FDR layer ref changed')
        require(record['stage_ref_sharing']['groups']['ordinary_FDR']['all_stage_values_exact'] is True, 'FDR sharing observer')
        for loss, expected in [('boxes', match['indices_go']), ('mal', match['indices_own_final'])]:
            calls = [x for x in record['loss_calls'] if x['stage'] == 'final' and x['loss'] == loss]
            require(len(calls) == 1 and np.array_equal(pairs_array(calls[0]['indices']), pairs_array(expected)), 'actual final loss ownership')
        require(H.jsonable(record['comparison_114']) == summary_row['comparison_114'] and index == int(summary_row['image_index']), 'recorded 114 comparison')
        originals[index] = pairs
    return originals


def encoding_rect(native, refs64, gt64, rm, up32, scale32):
    n, m = len(refs64), len(gt64)
    raw32, legacy = np.empty((n, m, 4), dtype=np.float32), np.empty((n, m, 4), dtype=np.float64)
    interpolation = 0.
    for start in range(0, n * m, B.BATCH):
        guard()
        ids = np.arange(start, min(start + B.BATCH, n * m))
        encoded = H.encode_native_fp32(native, torch.from_numpy(refs64[ids // m]), torch.from_numpy(gt64[ids % m]), rm, up32, scale32)
        raw32.reshape(-1, 4)[ids], legacy.reshape(-1, 4)[ids] = encoded['raw32'].numpy(), encoded['raw64'].numpy()
        interpolation = max(interpolation, encoded['audit']['interpolation_max_error'])
    return raw32, legacy, interpolation


def metrics_for(cols, graph, violation, iou, ref_iou, cost, owner):
    rows = np.arange(len(cols))
    return {'outside': Fraction(int((~graph[rows, cols]).sum())),
            'v': exact_sum(violation[rows, cols]), 'final_IoU': exact_sum(iou[rows, cols]),
            'ref_IoU': exact_sum(ref_iou[rows, cols]), 'native_cost': exact_sum(cost[rows, cols]),
            'owner_changes': Fraction(int((np.asarray(cols) != owner).sum()))}


def solve_domain(graph, violation, iou, ref_iou, cost, capacities, owner):
    outside = (~graph).astype(np.int64)
    specs = {'G': [('outside', outside), ('v', violation), ('native_cost', cost)],
             'final_IoU_face': [('negative_final_IoU', -iou), ('outside', outside)],
             'ref_IoU_face': [('negative_ref_IoU', -ref_iou), ('outside', outside)],
             'native_cost_face': [('native_cost', cost), ('outside', outside)]}
    cols, certs = {'G0': owner.copy()}, {'G0': None}
    for name, objectives in specs.items():
        cols[name], certs[name] = exact_flow(objectives, capacities, owner)
    stats = {name: metrics_for(value, graph, violation, iou, ref_iou, cost, owner) for name, value in cols.items()}
    for name in CONTROLS:
        require(tuple(stats['G'][key] for key in ('outside', 'v', 'native_cost')) <= tuple(stats[name][key] for key in ('outside', 'v', 'native_cost')), 'candidate lex optimum failed')
    return cols, certs, stats


def sum_metrics(groups, domain):
    return {arm: {key: sum((fraction_read(group['domains'][domain]['metrics'][arm][key]) for group in groups), Fraction(0))
                  for key in METRICS} for arm in ARMS}


def report_metrics(stats):
    return {arm: {key: fraction_report(value) for key, value in values.items()} for arm, values in stats.items()}


def necessary_gate(images, domain):
    comparisons = {}
    for control in CONTROLS:
        rows = []
        for image in images:
            stats = image['domains'][domain]['totals']
            candidate = tuple(fraction_read(stats['G'][key]) for key in ('outside', 'v'))
            baseline = tuple(fraction_read(stats[control][key]) for key in ('outside', 'v'))
            changed = image['domains'][domain]['owner_difference_G'][control]
            rows.append({'image_index': image['image_index'], 'strict_support_violation_lex_improvement': candidate < baseline,
                         'changed_owners': changed, 'qualifies': candidate < baseline and changed > 0})
        comparisons[control] = {'per_image': rows, 'has_actual_noncopy': any(row['qualifies'] for row in rows)}
    return {'necessary_gate': all(row['has_actual_noncopy'] for row in comparisons.values()), 'comparisons': comparisons}


def analyze(cache, originals, name, native, compute_cost, matrices, prior=None):
    rm, up, scale = int(cache['reg_max']), H.scalar(cache['up'], 'up'), abs(H.scalar(cache['reg_scale'], 'scale'))
    require(rm == 32 and up == .5 and scale > 0, 'FDR configuration')
    up32, scale32 = torch.tensor([up], dtype=torch.float32), torch.tensor([scale], dtype=torch.float32)
    nodes = native.weighting_function(rm, up32, scale32)
    a, b = float(nodes.min()), float(nodes.max())
    require(a == -scale and b == scale and bool((nodes[1:] > nodes[:-1]).all()), 'actual support endpoints')
    box = sys.modules[native.__package__ + '.box_ops']
    tolerance, images = B.NEAR_REL * max(1., scale), []
    records = sorted(cache['records'], key=lambda x: int(x['image_index']))
    require([int(x['image_index']) for x in records] == list(range(4)), 'four image IDs')
    for record in records:
        guard()
        index = int(record['image_index'])
        boxes, refs, logits = [torch.as_tensor(record[key]).detach().cpu() for key in ('pred_boxes', 'ref_points_initial', 'pred_logits')]
        target = {key: torch.as_tensor(record['target'][key]).detach().cpu() for key in ('boxes', 'labels')}
        gt, labels = target['boxes'], target['labels'].numpy()
        require(boxes.shape == refs.shape == (300, 4) and logits.shape == (300, 10)
                and all(value.dtype == torch.float32 for value in (boxes, refs, logits, gt)), 'FP32 output schema')
        require(gt.shape[1:] == (4,) and target['labels'].dtype == torch.int64
                and all(bool(torch.isfinite(value).all()) for value in (boxes, refs, logits, gt))
                and all(bool((value[:, 2:] > 0).all()) for value in (boxes, refs, gt)), 'finite valid boxes')
        small = array(record['original_sub4'], 'sub4').astype(bool)
        require(len(gt) == (146, 280, 418, 58)[index] and int(small.sum()) == (46, 121, 271, 15)[index], 'cohort')
        owner_pairs = originals[index]
        require(owner_pairs.ndim == 2 and owner_pairs.shape[1] == 2 and len(set(owner_pairs[:, 0])) == len(owner_pairs)
                and np.all((owner_pairs[:, 0] >= 0) & (owner_pairs[:, 0] < 300))
                and np.all((owner_pairs[:, 1] >= 0) & (owner_pairs[:, 1] < len(gt))), 'fixed q/GT ownership')
        owner_map = dict(owner_pairs.tolist())
        native_cost, rematched = compute_cost(boxes, logits, target)
        expected_own = originals[index] if name == 'ordinary' else pairs_array(record['matching']['indices_own_final'])
        require(np.array_equal(rematched, expected_own), 'native own matcher reproduction from same cache')
        if prior is not None:
            row = next(row for row in prior['images'] if int(row['image_index']) == index)
            require(record['input_sha256'] == row['input_sha256'] and str(record['event']) == row['event'], 'ordinary identity')
        groups, image_domains = [], {}
        all_iou = box.box_iou(box.box_cxcywh_to_xyxy(boxes), box.box_cxcywh_to_xyxy(gt))[0].numpy()
        all_ref_iou = box.box_iou(box.box_cxcywh_to_xyxy(refs), box.box_cxcywh_to_xyxy(gt))[0].numpy()
        for cls in sorted(set(int(labels[g]) for g in owner_map.values())):
            qs = np.array(sorted(q for q, g in owner_map.items() if labels[g] == cls), dtype=np.int64)
            gs, caps = np.unique([owner_map[q] for q in qs], return_counts=True)
            owner = np.searchsorted(gs, [owner_map[q] for q in qs])
            require(np.all(labels[gs] == cls) and caps.sum() == len(qs), 'same-class budget')
            refs64, targets64 = refs[qs].double().numpy(), gt[gs].double().numpy()
            raw = B.raw_physical64(refs64, targets64, scale)
            raw32, legacy64, interpolation = encoding_rect(native, refs64, targets64, rm, up32, scale32)
            margin = B.margin(raw, a, b)
            graphs = {'physical64': margin >= 0., 'tight': margin >= tolerance, 'expanded': margin >= -tolerance,
                      'native32': B.margin(raw32, a, b) >= 0.}
            violation = (np.maximum(a - raw, 0.) + np.maximum(raw - b, 0.)).sum(axis=-1) / scale
            iou, ref_iou, cost = [value[np.ix_(qs, gs)] for value in (all_iou, all_ref_iou, native_cost)]
            domains, solved = {}, {}
            for domain, graph in graphs.items():
                equal = next((old for old in solved if np.array_equal(graph, graphs[old])), None)
                if equal is None:
                    cols, certs, stats = solve_domain(graph, violation, iou, ref_iou, cost, caps, owner)
                else:
                    cols, certs, stats = solved[equal]
                solved[domain] = (cols, certs, stats)
                domains[domain] = {'metrics': report_metrics(stats), 'pairs': {arm: [[int(q), int(gs[c])] for q, c in zip(qs, col)] for arm, col in cols.items()},
                    'owner_difference_G': {arm: int((cols['G'] != cols[arm]).sum()) for arm in CONTROLS},
                    'GT_support': {arm: [{'GT': int(g), 'm_g': int(cap), 'supported_slots': int(graph[np.arange(len(qs)), col][col == j].sum()),
                                         'support_rate': float(graph[np.arange(len(qs)), col][col == j].sum() / cap), 'original_sub4': bool(small[g])}
                                        for j, (g, cap) in enumerate(zip(gs, caps))] for arm, col in cols.items()},
                    'certificate_reused_from_domain': equal, 'certificates': certs if equal is None else None}
            group = {'class_id': int(cls), 'queries': qs, 'GT_ids': gs, 'm_g': caps, 'domains': domains,
                'near_boundary_edges': int((np.minimum(np.abs(raw - a), np.abs(raw - b)).min(axis=-1) <= tolerance).sum()),
                'physical64_native32_edge_disagreements': int((graphs['physical64'] != graphs['native32']).sum()),
                'raw32_minus_physical64_max_abs': float(np.max(np.abs(raw32.astype(np.float64) - raw))),
                'legacy136_eps64_minus_physical64_max_abs': float(np.max(np.abs(legacy64 - raw))),
                'native_interpolation_max_error': interpolation,
                'G0_zero_IoU': {'pairs': int((iou[np.arange(len(qs)), owner] == 0.).sum()),
                    'support_inside': int(((iou[np.arange(len(qs)), owner] == 0.) & graphs['physical64'][np.arange(len(qs)), owner]).sum())}}
            if name == 'ordinary':
                require(np.all(caps == 1), 'ordinary capacity')
                rows, sci_cols = A.linear_sum_assignment(-iou.astype(np.float64))
                require(np.array_equal(rows, np.arange(len(qs))), 'SciPy rows')
                simple = metrics_for(sci_cols, graphs['physical64'], violation, iou, ref_iou, cost, owner)
                exact_I = solved['physical64'][2]['final_IoU_face']['final_IoU']
                require(exact_I >= simple['final_IoU'], 'exact final IoU below SciPy')
                max_support = len(qs) - int(solved['physical64'][2]['G']['outside'])
                require(max_support == B.completion(graphs['physical64'])[0], 'transport/max matching discrepancy')
                group['ordinary_184_replay'] = {'SciPy_metrics': {key: fraction_report(value) for key, value in simple.items()},
                    'SciPy_pairs': [[int(q), int(gs[g])] for q, g in zip(qs, sci_cols)],
                    'SciPy_primary_is_exact_optimal': exact_I == simple['final_IoU'], 'max_support': max_support,
                    'max_support_minus_SciPy': max_support - (len(qs) - int(simple['outside'])),
                    'max_support_minus_exact_IoU_face': int(solved['physical64'][2]['final_IoU_face']['outside'] - solved['physical64'][2]['G']['outside'])}
            groups.append(group)
            prefix = f'{name}_image{index}_class{cls}_'
            matrices.update({prefix + key: value for key, value in {'q_ids': qs, 'GT_ids': gs, 'capacities': caps, 'owner_cols': owner,
                'physical_inverse64': raw, 'native_inverse32': raw32, 'v64': violation, 'final_IoU32': iou, 'ref_IoU32': ref_iou,
                'native_cost32': cost, 'refs64': refs64, 'GT64': targets64, **graphs}.items()})
        for domain in DOMAINS:
            totals = sum_metrics(groups, domain)
            gt_stats = {arm: [row for group in groups for row in group['domains'][domain]['GT_support'][arm]] for arm in ARMS}
            image_domains[domain] = {'totals': report_metrics(totals), 'support_slots': {arm: len(owner_pairs) - int(totals[arm]['outside']) for arm in ARMS},
                'owner_difference_G': {arm: sum(group['domains'][domain]['owner_difference_G'][arm] for group in groups) for arm in CONTROLS},
                'unique_GT_with_any_support': {arm: sum(row['supported_slots'] > 0 for row in gt_stats[arm]) for arm in ARMS},
                'unique_GT_with_all_slots_supported': {arm: sum(row['supported_slots'] == row['m_g'] for row in gt_stats[arm]) for arm in ARMS}}
            for arm in ARMS:
                pairs = [pair for group in groups for pair in group['domains'][domain]['pairs'][arm]]
                mapping = dict(pairs)
                require(len(pairs) == len(mapping) == len(owner_pairs) and set(mapping) == set(owner_map)
                        and sorted(mapping.values()) == sorted(owner_map.values())
                        and all(labels[g] == labels[owner_map[q]] for q, g in mapping.items()), 'global class/query/GT multiplicity conservation')
        image = {'image_index': index, 'event': str(record['event']), 'input_sha256': record['input_sha256'], 'GT': len(gt),
                 'original_sub4': int(small.sum()), 'slots': len(owner_pairs), 'unique_GT': len(set(owner_pairs[:, 1])),
                 'original_pairs': owner_pairs, 'GT_not_in_fixed_budget': sorted(set(range(len(gt))) - set(owner_pairs[:, 1])),
                 'groups': groups, 'domains': image_domains}
        if name == 'true_GO':
            image['comparison_114_preserved'] = record['comparison_114']
            image['MAL_own_denominator_unchanged'] = record['matching']['num_boxes_ordinary']
            image['GO_geometry_denominator_unchanged'] = record['matching']['num_boxes_go']
        images.append(image)
        print(name, 'image', index, 'completed exact assignments', flush=True)
    result = {'population': name, 'images': images, 'configuration': {'reg_max': rm, 'up': up, 'scale': scale, 'endpoints': [a, b], 'near_tolerance': tolerance},
              'scope': 'Fixed cached assignment premise only; optimized support/violation gains are not detection or training effects.'}
    result['domains'] = {}
    for domain in DOMAINS:
        totals = {arm: {key: sum((fraction_read(image['domains'][domain]['totals'][arm][key]) for image in images), Fraction(0)) for key in METRICS} for arm in ARMS}
        result['domains'][domain] = {'totals': report_metrics(totals), 'necessary_screen': necessary_gate(images, domain)}
    if name == 'ordinary':
        old_gains = [sum(g['ordinary_184_replay']['max_support_minus_SciPy'] for g in image['groups']) for image in images]
        face_gains = [sum(g['ordinary_184_replay']['max_support_minus_exact_IoU_face'] for g in image['groups']) for image in images]
        require(old_gains == [1, 6, 3, 0] and sum(image['slots'] for image in images) == 784, '184 fixed replay changed')
        result['ordinary_184_unchanged'] = {'original_gain': old_gains, 'original_gate': sum(old_gains) > 0 and sum(x > 0 for x in old_gains) >= 2,
            'gain_after_exact_IoU_face': face_gains, 'sum_original_10_removed_by_face': sum(old_gains) - sum(face_gains),
            'not_a_true_GO_conclusion': True}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--go-cache-sha256', default=GO_SHA)
    parser.add_argument('--out', type=pathlib.Path, default=ROOT / '193_C0_support_assignment_exact')
    args = parser.parse_args()
    args.out = args.out.resolve()
    require(args.go_cache_sha256 == GO_SHA, 'frozen188 expected hash')
    require(args.out.parent == ROOT and args.out.name.startswith('193_') and not args.out.exists(), 'new local193 output only')
    args.out.mkdir(exist_ok=False)
    nx_source = pathlib.Path(nx.__file__).parent / 'algorithms/flow/networksimplex.py'
    files = {'script': pathlib.Path(__file__), 'protocol': PROTOCOL, 'reuse183': REUSE, 'reuse159': B.REUSE,
             'helper136': A.HELPER, 'recipe': A.RECIPE, 'matcher': A.NATIVE / 'matcher.py', 'criterion': A.NATIVE / 'deim_criterion.py',
             'box_ops': A.NATIVE / 'box_ops.py', 'dfine_utils': A.NATIVE / 'dfine_utils.py', 'network_simplex': nx_source,
             'collector187': COLLECTOR, 'ordinary_cache127': B.CACHE, 'own160': B.OWN,
             'GO_cache188': GO / '01_cache.pt', 'GO_summary': GO / '02_summary.json',
             'GO_execution': GO / '04_execution.json', 'GO_manifest': GO / '05_output_sha256.json', 'GO_inputs': GO / '00_inputs_sha256.json'}
    started, before, resources, checks, loaded = time.monotonic(), {}, None, None, False
    stage, result = 'resources', {'state': 'error', 'primary': {'necessary_gate': False}}
    try:
        resources = H.resource_limits()
        require(resources['priority'] == 'below_normal' and torch.version.cuda is None and nx.__version__ == '3.3'
                and A.scipy.__version__ == '1.13.1', 'CPU/package contract')
        content_keys = {'ordinary_cache127', 'own160', 'GO_cache188', 'GO_summary', 'GO_execution', 'GO_manifest', 'GO_inputs'}
        before = {key: sha(path) for key, path in files.items() if key not in content_keys}
        expected = {'reuse183': REUSE_SHA, 'reuse159': B.REUSE_SHA, 'helper136': A.HELPER_SHA, 'network_simplex': NX_SHA,
                    'collector187': COLLECTOR_SHA, **{key: A.EXPECTED[key] for key in ('recipe', 'matcher', 'criterion', 'box_ops', 'dfine_utils')}}
        require(all(before[key] == value for key, value in expected.items()), 'dependency hashes')
        native = H.load_native_math()
        compute_cost = native_cost_reader(sys.modules[native.__package__ + '.box_ops'])
        stage = 'synthetic_before_any_cache_or_result_content'
        checks = synthetic_checks(native, compute_cost)
        before.update({key: sha(files[key]) for key in content_keys})
        require(before['ordinary_cache127'] == A.CACHE_SHA and before['own160'] == B.OWN_SHA
                and before['GO_cache188'] == GO_SHA, 'input hashes')
        stage, loaded = 'verify188_provenance', True
        summary, execution = verify_collector(args.go_cache_sha256)
        old = H.json.loads(B.OWN.read_text(encoding='utf-8-sig'))
        require(old['state'] == 'complete' and old['sources_unchanged'] is True
                and old['sha256_before']['cache'] == A.CACHE_SHA and old['sha256_before']['script'] == B.REUSE_SHA, '160 provenance')
        matrices = {}
        stage = 'ordinary_exact_face'
        ordinary_cache = torch.load(B.CACHE, map_location='cpu', weights_only=False)
        require(ordinary_cache['checkpoint_sha256'] == A.CHECKPOINT_SHA and ordinary_cache['weight_key'] == 'model_online', 'ordinary C0 identity')
        own = {int(row['image_index']): np.asarray(row['own_Hungarian_pairs'], dtype=np.int64) for row in old['images']}
        ordinary = analyze(ordinary_cache, own, 'ordinary', native, compute_cost, matrices, old)
        del ordinary_cache, old
        stage = 'true_GO_exact_assignment'
        go_cache = torch.load(GO / '01_cache.pt', map_location='cpu', weights_only=False)
        originals = verify_GO_cache(go_cache, summary, execution)
        go = analyze(go_cache, originals, 'true_GO', native, compute_cost, matrices)
        del go_cache
        primary = dict(go['domains']['physical64']['necessary_screen'])
        sensitive = any(go['domains'][domain]['necessary_screen']['necessary_gate'] != primary['necessary_gate'] for domain in ('tight', 'expanded'))
        primary.update(boundary_sensitive=sensitive, training_authorized=False,
            decision='precision_review_no_method_conclusion' if sensitive else ('noncopy_premise_only_may_discuss_supervision_test' if primary['necessary_gate'] else 'stop_exact_fixed_budget_GO_assignment_candidate'),
            optimization_advantage_is_not_supervision_evidence=True)
        stage = 'save'
        save(args.out / '01_ordinary_assignments_and_certificates.json', ordinary)
        save(args.out / '02_true_GO_assignments_and_certificates.json', go)
        np.savez_compressed(args.out / '03_pairwise_graphs.npz', **matrices)
        result = {'state': 'complete', 'primary': primary, 'ordinary_184_unchanged': ordinary['ordinary_184_unchanged'],
            'ordinary_domains': ordinary['domains'], 'true_GO_domains': go['domains'],
            'GO_identity': {'cache_sha256': GO_SHA, 'collector_state': execution['state'], 'collector_counts': execution['counts'],
                'parity_with114_exact': summary['parity_with_114_all_saved_train_fields_exact'], 'all_comparisons_use188_internal_baseline': True},
            'artifact_sha256': {name: sha(args.out / name) for name in ('01_ordinary_assignments_and_certificates.json', '02_true_GO_assignments_and_certificates.json', '03_pairwise_graphs.npz')}}
        stage = 'complete'
    except Exception as exc:
        result = {'state': 'error', 'primary': {'necessary_gate': False}, 'error': str(exc), 'traceback': traceback.format_exc(),
                  'scope': 'Implementation/provenance/certificate failure, not a method negative.'}
    finally:
        after = {key: sha(files[key]) for key in before if files[key].exists()}
        unchanged = bool(before and before == after)
        if result['state'] == 'complete':
            unchanged = unchanged and set(before) == set(files)
        if not unchanged:
            result['state'] = 'error'
            result['primary']['necessary_gate'] = False
            result['hash_error'] = 'source/input missing or changed'
        result.update(source_paths={key: str(path.resolve()) for key, path in files.items()}, sha256_before=before, sha256_after=after,
            sources_unchanged=unchanged, resources=resources, synthetic_self_checks=checks, cache_content_loaded=loaded, stage=stage,
            model_forwards=0, model_backwards=0, optimizer_steps=0, model_vjps=0, new_training=False,
            networkx_version=nx.__version__, scipy_version=A.scipy.__version__, flow_statistics=FLOW_STATS,
            RSS_observed_peak_bytes=RSS_PEAK, OS_peak_working_set_bytes=getattr(psutil.Process().memory_info(), 'peak_wset', None),
            free_RAM_GiB_at_end=psutil.virtual_memory().available / 2**30,
            seconds=time.monotonic() - started)
        save(args.out / '00_results.json', result)
        print(H.json.dumps(H.jsonable({'state': result['state'], 'primary': result['primary'], 'seconds': result['seconds']}), ensure_ascii=False), flush=True)
    if result['state'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
