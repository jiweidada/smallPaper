"""182: same-class, fixed-positive-slot support upper bound; cache only.

Written for root review, not executed by the author. No model/gradient/optimizer.
The feasible graph uses the closed real-arithmetic decoder support. Its matching
number is exact for the computed graph, not a learnability or AP result.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import hashlib
import importlib.util
import pathlib
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent
PROTOCOL = ROOT / '182_同类所有权支持域_必要界事前协议.txt'
REUSE = ROOT / '159_C0_exchange_label_audit.py'
REUSE_SHA = 'ffd80cee9fe685fe6ebc69f3e4575ca856a82e2b58a5d0a466439198135f1d64'
OWN = ROOT / '160_C0_exchange_label_audit/00_results.json'
OWN_SHA = '750d13d7dc3376ae34e3f2f05a4a0635a16172a96c106f827a68e525e4f4f39c'
CACHE = ROOT / '127_C0_attention_template_registration/01_cache.pt'
if hashlib.sha256(REUSE.read_bytes()).hexdigest() != REUSE_SHA:
    raise RuntimeError('frozen 159 changed')
_spec = importlib.util.spec_from_file_location('_audit183_reuse159', REUSE)
if _spec is None or _spec.loader is None:
    raise RuntimeError('159 import unavailable')
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)  # Definitions only. Neither 159.main nor 136.main runs.
H, np, torch, psutil = A.H, A.np, A.torch, A.psutil
require, sha, save, array = H.require, H.sha, H.save, H.array
BATCH, NEAR_REL = 512, 1e-10


def raw_physical64(refs, targets, scale):
    """Pairwise inverse of distance2bbox, WITHOUT encoder-only denominator eps."""
    require(refs.ndim == targets.ndim == 2 and refs.shape[1] == targets.shape[1] == 4, 'box shapes')
    require(refs.dtype == targets.dtype == np.float64 and scale > 0., 'physical FP64 contract')
    require(np.isfinite(refs).all() and np.isfinite(targets).all()
            and (refs[:, 2:] > 0.).all() and (targets[:, 2:] > 0.).all(), 'nonpositive/nonfinite physical box')
    corners = H.xyxy(targets)
    denominator = refs[:, None, 2:] / scale
    left_top = (refs[:, None, :2] - corners[None, :, :2]) / denominator - scale / 2.
    right_bottom = (corners[None, :, 2:] - refs[:, None, :2]) / denominator - scale / 2.
    return np.concatenate((left_top, right_bottom), axis=-1)


def margin(raw, a, b):
    return np.minimum(raw - a, b - raw).min(axis=-1)


def completion(edges):
    """A maximum feasible matching plus a deterministic infeasible completion."""
    require(edges.ndim == 2 and edges.shape[0] == edges.shape[1], 'square budget graph')
    n = len(edges)
    pairs = H.match_capacity_one(edges)
    k = len(pairs)
    require(k == A.matching_count(edges), 'SciPy/Kuhn cardinality mismatch')
    used_rows, used_cols = {q for q, _ in pairs}, {g for _, g in pairs}
    rem_rows = sorted(set(range(n)) - used_rows)
    rem_cols = sorted(set(range(n)) - used_cols)
    require(not edges[np.ix_(rem_rows, rem_cols)].any(), 'maximum matching has an addable feasible edge')
    full = sorted(pairs + list(zip(rem_rows, rem_cols)))
    rows = np.array([q for q, _ in full], dtype=np.int64)
    cols = np.array([g for _, g in full], dtype=np.int64)
    require(np.array_equal(rows, np.arange(n)) and len(set(cols.tolist())) == n, 'completion budget')
    require(int(edges[rows, cols].sum()) == k, 'completion adds a feasible pair')
    return k, cols


def assignment_report(cols, edges, native_edges, overlaps, qs, gs):
    rows = np.arange(len(qs))
    require(len(cols) == len(qs) and len(set(cols.tolist())) == len(gs), 'assignment is not a full permutation')
    return {'support64': int(edges[rows, cols].sum()), 'native_encoder_support32': int(native_edges[rows, cols].sum()),
            'current_IoU32_sum': float(overlaps[rows, cols].astype(np.float64).sum()),
            'current_IoU32_zero_pairs': int((overlaps[rows, cols] == 0.).sum()),
            'pairs_q_gt': [[int(q), int(gs[c])] for q, c in zip(qs, cols)]}


def self_checks(native):
    checks = {'native_FP32_encoding': H.native_encoding_self_checks(native)}
    graphs = [np.array([[1, 1], [0, 0]], dtype=bool),
              np.array([[1, 1, 0], [0, 0, 0], [0, 0, 1]], dtype=bool),
              np.zeros((2, 2), dtype=bool), np.ones((3, 3), dtype=bool)]
    sizes = [completion(graph)[0] for graph in graphs]
    require(sizes == [1, 2, 0, 3] and graphs[0].any(axis=0).sum() == 2, 'individual reachability is not simultaneous')
    checks['SciPy_Kuhn_and_infeasible_completion'] = True
    # Two class blocks preserve q/GT sets and every original positive class slot.
    owner = np.array([[8, 6], [2, 7], [5, 1]], dtype=np.int64)
    gt_class = {6: 1, 7: 1, 1: 4}
    rebuilt = []
    for c in (1, 4):
        group = owner[[gt_class[g] == c for _, g in owner]]
        qs, gs = np.sort(group[:, 0]), np.sort(group[:, 1])
        _, cols = completion(np.ones((len(qs), len(qs)), dtype=bool))
        rebuilt.extend((int(q), int(gs[col])) for q, col in zip(qs, cols))
    old_class = {int(q): gt_class[int(g)] for q, g in owner}
    require({q for q, _ in rebuilt} == set(owner[:, 0]) and {g for _, g in rebuilt} == set(owner[:, 1])
            and all(old_class[q] == gt_class[g] for q, g in rebuilt), 'synthetic class/slot budget')
    overlaps = np.array([[.9, .8], [.7, 0.]], dtype=np.float32)
    rows, cols = A.linear_sum_assignment(-overlaps.astype(np.float64))
    require(cols.tolist() == [1, 0] and rows.tolist() == [0, 1], 'IoU Hungarian control')
    checks['same_positive_slots_and_IoU_control'] = True
    ref = torch.tensor([[.5, .5, .125, .25]], dtype=torch.float64)
    offsets = torch.tensor([[0., 0., 0., 0.], [4., 4., 4., 4.], [-4., 0., 4., 0.],
                            [4., 0., -4., 0.], [4.001, 0., 4., 0.]], dtype=torch.float64)
    targets = native.distance2bbox(ref.repeat(5, 1), offsets, torch.tensor([4.], dtype=torch.float64))
    raw = raw_physical64(ref.numpy(), targets.numpy(), 4.)[0]
    require(np.max(np.abs(raw - offsets.numpy())) < 2e-12, 'native decode inverse')
    require((margin(raw, -4., 4.) >= 0.).tolist() == [True, True, True, True, False], 'physical endpoint/outside cases')
    encoded = H.encode_native_fp32(native, ref.repeat(5, 1), targets, 32,
                                    torch.tensor([.5]), torch.tensor([4.]))
    require((margin(encoded['raw32'].numpy(), -4., 4.) >= 0.).tolist() == [True, True, True, True, False], 'native32 endpoint/outside cases')
    invalid = native.distance2bbox(ref, torch.full((1, 4), -4., dtype=torch.float64), torch.tensor([4.], dtype=torch.float64))
    rejected = False
    try:
        raw_physical64(ref.numpy(), invalid.numpy(), 4.)
    except RuntimeError:
        rejected = True
    require(rejected, 'nonpositive width must be rejected')
    checks['native_decode_closed_boundary_and_invalid_width'] = True
    return checks


def native_encoding_matrix(native, refs64, gt64, reg_max, up32, scale32):
    n = len(refs64)
    raw32 = np.empty((n, n, 4), dtype=np.float32)
    legacy64 = np.empty((n, n, 4), dtype=np.float64)
    interpolation_max = 0.
    for start in range(0, n * n, BATCH):
        A.memory_guard()
        ids = np.arange(start, min(start + BATCH, n * n))
        encoded = H.encode_native_fp32(native, torch.from_numpy(refs64[ids // n]), torch.from_numpy(gt64[ids % n]),
                                        reg_max, up32, scale32)
        raw32.reshape(-1, 4)[ids] = encoded['raw32'].numpy()
        legacy64.reshape(-1, 4)[ids] = encoded['raw64'].numpy()
        interpolation_max = max(interpolation_max, encoded['audit']['interpolation_max_error'])
    return raw32, legacy64, interpolation_max


def analyze(cache, prior, native):
    require(prior['state'] == 'complete' and prior['sources_unchanged'] is True
            and prior['sha256_before']['cache'] == A.CACHE_SHA and prior['sha256_before']['script'] == REUSE_SHA, '160 provenance')
    require(cache['checkpoint_sha256'] == A.CHECKPOINT_SHA and cache['weight_key'] == 'model_online', 'C0 cache identity')
    require(prior['native_MAL']['positive_slots'] == 784 and prior['native_MAL']['zero_IoU_positive_slots'] == 251
            and prior['native_MAL']['denominator'] == 902., 'known own denominator/scope')
    rm, up, scale = int(cache['reg_max']), H.scalar(cache['up'], 'up'), abs(H.scalar(cache['reg_scale'], 'scale'))
    require(rm == 32 and up == .5 and scale > 0., 'native support configuration')
    up32, scale32 = torch.tensor([up], dtype=torch.float32), torch.tensor([scale], dtype=torch.float32)
    nodes = native.weighting_function(rm, up32, scale32)
    a, b = float(nodes.min()), float(nodes.max())
    require(a == -scale and b == scale and bool((nodes[1:] > nodes[:-1]).all()), 'actual node endpoints')
    box_ops = sys.modules[native.__package__ + '.box_ops']
    records = sorted(cache['records'], key=lambda item: int(item['image_index']))
    old_images = {int(item['image_index']): item for item in prior['images']}
    require(len(records) == 4 and {int(r['image_index']) for r in records} == set(old_images), 'four identical images')
    images, original_pairs, matrices = [], [], {'native_nodes32': nodes.numpy()}
    total_gt = total_small = total_n = total_zero = 0
    tol = NEAR_REL * max(1., scale)
    for item in records:
        A.memory_guard()
        index, old = int(item['image_index']), old_images[int(item['image_index'])]
        require(item['input_sha256'] == old['input_sha256'] and str(item['event']) == old['event'], 'event/input identity')
        tensors = [torch.as_tensor(item[key]).detach().cpu() for key in ('pred_boxes', 'ref_points_initial')]
        boxes, refs = tensors
        gt = torch.as_tensor(item['target']['boxes']).detach().cpu()
        labels = array(item['target']['labels'], 'GT labels', (len(gt),), integer=True)
        require(boxes.shape == refs.shape == (300, 4) and boxes.dtype == refs.dtype == gt.dtype == torch.float32
                and np.all((labels >= 0) & (labels < 10)), 'FP32 schema')
        require(bool(torch.isfinite(boxes).all() and (boxes[:, 2:] > 0.).all()), 'invalid current prediction boxes')
        require(gt.ndim == 2 and gt.shape[1] == 4 and bool(torch.isfinite(refs).all() and torch.isfinite(gt).all())
                and bool((refs[:, 2:] > 0.).all() and (gt[:, 2:] > 0.).all()), 'invalid reference/GT boxes')
        small = torch.as_tensor(item['original_sub4']).detach().cpu()
        require(small.dtype == torch.bool and small.shape == (len(gt),) and len(gt) == old['GT'], 'GT/sub4 schema')
        own = np.asarray(old['own_Hungarian_pairs'], dtype=np.int64)
        n = len(own)
        require(own.shape == (min(300, len(gt)), 2) and len(set(own[:, 0])) == n and len(set(own[:, 1])) == n
                and np.all((own[:, 0] >= 0) & (own[:, 0] < 300))
                and np.all((own[:, 1] >= 0) & (own[:, 1] < len(gt))), 'original own mapping')
        gt_by_q = dict(own.tolist())
        class_by_q = {q: int(labels[g]) for q, g in own}
        groups, witness_pairs, simple_pairs = [], [], []
        for c in sorted(set(class_by_q.values())):
            qs = np.array(sorted(q for q in gt_by_q if class_by_q[q] == c), dtype=np.int64)
            gs = np.array(sorted(gt_by_q[q] for q in qs), dtype=np.int64)
            require(np.all(labels[gs] == c), 'class block')
            orig_cols = np.searchsorted(gs, [gt_by_q[q] for q in qs])
            refs64, gt64 = refs[qs].double().numpy(), gt[gs].double().numpy()
            raw64 = raw_physical64(refs64, gt64, scale)
            raw32, old64, interpolation = native_encoding_matrix(native, refs64, gt64, rm, up32, scale32)
            margins = margin(raw64, a, b)
            support, inside32 = margins >= 0., margin(raw32, a, b) >= 0.
            near = np.minimum(np.abs(raw64 - a), np.abs(raw64 - b)).min(axis=-1) <= tol
            iou32 = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(boxes[qs]), box_ops.box_cxcywh_to_xyxy(gt[gs]))[0].numpy()
            require(np.isfinite(iou32).all() and (iou32 >= 0.).all() and (iou32 <= 1. + 1e-6).all(), 'native IoU values')
            rows, simple_cols = A.linear_sum_assignment(-iou32.astype(np.float64))
            require(np.array_equal(rows, np.arange(len(qs))), 'simple full rows')
            k, witness_cols = completion(support)
            reports = {name: assignment_report(cols, support, inside32, iou32, qs, gs)
                       for name, cols in [('original', orig_cols), ('max_IoU_simple', simple_cols), ('max_support_witness', witness_cols)]}
            require(reports['max_IoU_simple']['current_IoU32_sum'] + 1e-12 >= reports['original']['current_IoU32_sum'], 'IoU simple not optimal to original')
            require(k >= max(reports['original']['support64'], reports['max_IoU_simple']['support64']), 'upper bound below feasible assignment')
            sensitivity = {}
            for name, graph in [('tight', margins >= tol), ('expanded', margins >= -tol), ('native32', inside32)]:
                sensitivity[name] = {'max_support': completion(graph)[0], 'simple_support': int(graph[rows, simple_cols].sum()),
                                     'original_support': int(graph[rows, orig_cols].sum())}
            groups.append({'class_id': c, 'n': len(qs), 'edges64': int(support.sum()),
                           'individually_reachable_GT': int(support.any(axis=0).sum()), 'max_support': k,
                           'assignments': reports, 'sensitivity': sensitivity, 'near_boundary_pairs': int(near.sum()),
                           'physical64_vs_native32_edge_disagreements': int((support != inside32).sum()),
                           'native32_raw_minus_physical64_max_abs': float(np.max(np.abs(raw32.astype(np.float64) - raw64))),
                           'legacy136_eps64_minus_exact_inverse_max_abs': float(np.max(np.abs(old64 - raw64))),
                           'native_interpolation_max_error': interpolation})
            simple_pairs.extend(reports['max_IoU_simple']['pairs_q_gt'])
            witness_pairs.extend(reports['max_support_witness']['pairs_q_gt'])
            for row, (q, col) in enumerate(zip(qs, orig_cols)):
                g, current = int(gs[col]), float(iou32[row, col])
                original_pairs.append({'image_index': index, 'query': int(q), 'gt': g, 'class_id': c,
                    'current_IoU32': current, 'original_sub4': bool(small[g]), 'support64': bool(support[row, col]),
                    'native_encoder_support32': bool(inside32[row, col]), 'near_boundary': bool(near[row, col]),
                    'ref_points_initial_FP32_as_FP64': refs64[row], 'GT_FP32_as_FP64': gt64[col],
                    'raw_physical64': raw64[row, col], 'raw_encoder32': raw32[row, col]})
                total_zero += current == 0.
            prefix = 'image' + str(index) + '_class' + str(c) + '_'
            matrices.update({prefix + key: value for key, value in {'q_ids': qs, 'gt_ids': gs, 'support64': support,
                'support32': inside32, 'margin64': margins, 'near': near, 'IoU32': iou32, 'refs64': refs64, 'GT64': gt64}.items()})
        for pairs in (simple_pairs, witness_pairs):
            require({q for q, _ in pairs} == set(own[:, 0]) and {g for _, g in pairs} == set(own[:, 1])
                    and len(pairs) == n and all(class_by_q[q] == labels[g] for q, g in pairs), 'real class/positive-slot budget')
        totals = {name: sum(group['assignments'][name]['support64'] for group in groups)
                  for name in ('original', 'max_IoU_simple', 'max_support_witness')}
        images.append({'image_index': index, 'event': old['event'], 'input_sha256': old['input_sha256'], 'GT': len(gt),
                       'own_slots': n, 'unmatched_GT_ids_unchanged': sorted(set(range(len(gt))) - set(own[:, 1])),
                       'groups': groups, 'totals_support64': totals,
                       'max_support_minus_simple': totals['max_support_witness'] - totals['max_IoU_simple'],
                       'current_IoU32_sum': {name: sum(group['assignments'][name]['current_IoU32_sum'] for group in groups) for name in totals}})
        total_gt, total_small, total_n = total_gt + len(gt), total_small + int(small.sum()), total_n + n
    require((total_gt, total_small, total_n, total_zero) == (902, 453, 784, 251), 'frozen 160 population/IoU-zero mismatch')
    gains = [image['max_support_minus_simple'] for image in images]
    gate = sum(gains) > 0 and sum(gain > 0 for gain in gains) >= 2
    alternatives = {}
    for name in ('tight', 'expanded', 'native32'):
        diffs = [sum(g['sensitivity'][name]['max_support'] - g['sensitivity'][name]['simple_support'] for g in image['groups']) for image in images]
        alternatives[name] = {'per_image_gain': diffs, 'gate': bool(sum(diffs) > 0 and sum(d > 0 for d in diffs) >= 2)}
    sensitive = any(alternatives[name]['gate'] != gate for name in ('tight', 'expanded'))
    zero = [pair for pair in original_pairs if pair['current_IoU32'] == 0.]
    zero_stats = {'total': len(zero), 'physical64_inside': sum(p['support64'] for p in zero),
                  'physical64_outside': sum(not p['support64'] for p in zero),
                  'native32_inside': sum(p['native_encoder_support32'] for p in zero),
                  'native32_outside': sum(not p['native_encoder_support32'] for p in zero), 'near_boundary': sum(p['near_boundary'] for p in zero)}
    decision = ('boundary_precision_review_no_method_conclusion' if sensitive else
                ('ordinary_own_premise_passes_only_register_actual_candidate_next' if gate else 'stop_exact_ordinary_own_fixed_slot_prototype_only'))
    return {'state': 'complete', 'images': images, 'GT': 902, 'own_positive_slots': 784, 'unchanged_MAL_denominator': 902,
            'unmatched_GT_excluded': 118, 'zero_IoU_own_scope': zero_stats,
            'configuration': {'reg_max': rm, 'up': up, 'reg_scale_abs': scale, 'endpoints': [a, b], 'near_tolerance': tol},
            'primary': {'necessary_gate': bool(gate), 'per_image_max_support_minus_simple': gains, 'total_extra': sum(gains),
                        'positive_images': sum(gain > 0 for gain in gains), 'boundary_sensitive': sensitive, 'decision': decision,
                        'training_authorized': False, 'candidate_IoU_sum_non_decrease_not_established': True,
                        'cannot_conclude_true_GO_multiplicity_budget': True},
            'precision_sensitivity': alternatives, 'scope': 'Closed-support necessary upper bound, original ordinary-own budget only; not full GO, AP, learnability or novelty.'}, original_pairs, matrices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=pathlib.Path, default=CACHE)
    parser.add_argument('--own', type=pathlib.Path, default=OWN)
    parser.add_argument('--out', type=pathlib.Path, default=ROOT / '184_C0_sameclass_support_bound')
    args = parser.parse_args()
    args.cache, args.own, args.out = args.cache.resolve(), args.own.resolve(), args.out.resolve()
    require(args.cache == CACHE.resolve() and args.own == OWN.resolve(), 'fixed input paths')
    require(args.out.parent == ROOT and args.out.name.startswith('184_') and not args.out.exists(), 'new local 184_* output required')
    args.out.mkdir(exist_ok=False)
    files = {'script': pathlib.Path(__file__), 'protocol': PROTOCOL, 'reuse159': REUSE, 'helper136': A.HELPER,
             'box_ops': A.NATIVE / 'box_ops.py', 'dfine_utils': A.NATIVE / 'dfine_utils.py', 'cache': args.cache, 'own160': args.own}
    started, before, checks, resources = time.monotonic(), {}, None, None
    stage, loaded = 'resources', False
    result = {'state': 'error', 'primary': {'necessary_gate': False}}
    try:
        resources = H.resource_limits()
        require(resources['priority'] == 'below_normal' and torch.version.cuda is None and A.scipy.__version__ == '1.13.1', 'CPU/runtime contract')
        before = {key: sha(path) for key, path in files.items() if key not in ('cache', 'own160')}
        require(before['reuse159'] == REUSE_SHA and before['helper136'] == A.HELPER_SHA
                and before['box_ops'] == A.EXPECTED['box_ops'] and before['dfine_utils'] == A.EXPECTED['dfine_utils'], 'dependency hashes')
        native = H.load_native_math()
        stage = 'synthetic_before_cache_and_own_content'
        checks = self_checks(native)
        before.update({key: sha(files[key]) for key in ('cache', 'own160')})
        require(before['cache'] == A.CACHE_SHA and before['own160'] == OWN_SHA, 'fixed cache/160 hashes')
        stage, loaded = 'cached_bound', True
        cache = torch.load(args.cache, map_location='cpu', weights_only=False)
        prior = H.json.loads(args.own.read_text(encoding='utf-8-sig'))
        result, pairs, matrices = analyze(cache, prior, native)
        del cache, prior
        stage = 'save'
        save(args.out / '01_original_own_pairs.json', {'pairs': pairs})
        np.savez_compressed(args.out / '02_pairwise_graphs.npz', **matrices)
        result['artifact_sha256'] = {name: sha(args.out / name) for name in ('01_original_own_pairs.json', '02_pairwise_graphs.npz')}
        stage = 'complete'
    except Exception as exc:
        result = {'state': 'error', 'primary': {'necessary_gate': False}, 'error': str(exc), 'traceback': traceback.format_exc(),
                  'scope': 'Implementation/configuration failclosed, not a method negative.'}
    finally:
        after = {key: sha(files[key]) for key in before if files[key].exists()}
        unchanged = bool(before and before == after)
        if result['state'] == 'complete':
            unchanged = unchanged and set(before) == set(files)
        if not unchanged:
            result['state'] = 'error'
            result['primary']['necessary_gate'] = False
            result['hash_error'] = 'source missing or changed'
        result.update({'source_paths': {key: str(path.resolve()) for key, path in files.items()}, 'sha256_before': before,
                       'sha256_after': after, 'sources_unchanged': unchanged, 'stage': stage, 'cache_content_loaded': loaded,
                       'synthetic_self_checks': checks, 'resources': resources, 'scipy_version': A.scipy.__version__,
                       'model_forwards': 0, 'model_backwards': 0, 'optimizer_steps': 0, 'new_training': False,
                       'seconds': time.monotonic() - started})
        save(args.out / '00_results.json', result)
        print(H.json.dumps(H.jsonable({'state': result['state'], 'primary': result['primary'], 'seconds': result['seconds']}), ensure_ascii=False))
    if result['state'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
