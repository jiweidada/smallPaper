"""145 preregistered support-family audit. Cache only; root reviews before running.

Exact real-arithmetic arrangement-vertex oracle; FP64 implementation limitations
are recorded in 145. No model, training, backward, or PMF compatibility analysis.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
import argparse
import hashlib
import importlib.util
import pathlib
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent
PROTOCOL = ROOT / '145_C0_conditional_unfold_可达族事前协议.txt'
HELPER = ROOT / '136_C0_pmf_translation_prior_fp32.py'
HELPER_SHA = '10bc190cce57688cd8c04b31190fc356244cc2afd6df5b6c74f8bed4eca6ab26'
CACHE_SHA = 'f2fb2120ccd670f846a0a70651a3fabbf9bcfb636c94d43d33f8dc34eceac79c'
CHECKPOINT_SHA = '9d6c0dd7a9cd6cb56465c4eab9416c0e89547c7e45df936e559fa12b921f275e'
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
    raise RuntimeError('frozen helper source changed; review required')
_spec = importlib.util.spec_from_file_location('_audit146_frozen_helpers', HELPER)
if _spec is None or _spec.loader is None:
    raise RuntimeError('helper import unavailable')
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)  # Defines helpers only; its guarded main is not called.
np, torch, psutil = H.np, H.torch, H.psutil
require, sha, save, array, scalar = H.require, H.sha, H.save, H.array, H.scalar
ARMS = ('original', 'conditional', 'rectangle', 'range2')
TOL, DET_TOL, NEST_TOL = 1e-10, 1e-14, 1e-8
BATCH = 64


def halfplanes(name):
    if name in ('original', 'range2'):
        lo_l, hi_l, lo_h, hi_h = ((-1.5, .5, -.5, 1.5) if name == 'original'
                                  else (-2.5, 1.5, -1.5, 2.5))
        aa = [[-1., 0.], [1., 0.], [0., -1.], [0., 1.], [1., -1.]]
        bb = [-lo_l, hi_l, -lo_h, hi_h, 0.]
    else:
        require(name in ('rectangle', 'conditional'), 'unknown family')
        aa = [[1., 1.], [-1., -1.], [1., -1.], [-1., 1.]]
        bb = [2., 2., 0., 3.]
        if name == 'conditional':
            aa += [[-4.5, 5.5], [-5.5, 4.5]]
            bb += [15., 15.]
    return np.array(aa, dtype=np.float64), np.array(bb, dtype=np.float64)


def inside(points, aa, bb):
    return np.all(np.asarray(points) @ aa.T <= bb + TOL, axis=-1)


def polygon_vertices(aa, bb):
    result = []
    for i in range(len(bb)):
        for j in range(i + 1, len(bb)):
            mat = aa[[i, j]]
            if abs(np.linalg.det(mat)) <= DET_TOL:
                continue
            point = np.linalg.solve(mat, bb[[i, j]])
            if inside(point, aa, bb) and not any(np.max(np.abs(point - old)) <= 1e-12 for old in result):
                result.append(point)
    require(len(result) >= 3, 'nonpolygon domain')
    return np.array(result)


DOMAINS = {}
for _arm in ARMS:
    _aa, _bb = halfplanes(_arm)
    DOMAINS[_arm] = (_aa, _bb, polygon_vertices(_aa, _bb))


def axis_candidates(name, a, b):
    """All vertices of the domain/GT-line arrangement, batched over GT pairs."""
    aa, bb, base = DOMAINS[name]
    require(a.ndim == b.ndim == 1 and a.shape == b.shape and np.all(b > a), 'invalid GT interval')
    count = len(a)
    pieces = [np.broadcast_to(base, (count, len(base), 2))]
    # Each line fixes L or H to a GT endpoint. Intersect with every domain edge.
    for fixed_axis in (0, 1):
        free_axis = 1 - fixed_axis
        for value in (a, b):
            for row, rhs in zip(aa, bb):
                if abs(row[free_axis]) <= DET_TOL:
                    continue
                point = np.empty((count, 2), dtype=np.float64)
                point[:, fixed_axis] = value
                point[:, free_axis] = (rhs - row[fixed_axis] * value) / row[free_axis]
                pieces.append(point[:, None, :])
    # Nonparallel intersections among the four GT lines.
    for left in (a, b):
        for right in (a, b):
            pieces.append(np.stack((left, right), axis=-1)[:, None, :])
    points = np.concatenate(pieces, axis=1)
    valid = inside(points, aa, bb)
    width = points[..., 1] - points[..., 0]
    valid &= width >= -TOL
    width = np.maximum(width, 0.)
    overlap = np.maximum(np.minimum(points[..., 1], b[:, None])
                         - np.maximum(points[..., 0], a[:, None]), 0.)
    require(np.all(valid.any(axis=1)), 'empty arrangement candidate set')
    return width, overlap, valid


def joint_normalized(name, a, b):
    """a,b are pair-count x 2 GT endpoints relative to reference cx/cy,w/h."""
    require(a.shape == b.shape and a.ndim == 2 and a.shape[1] == 2 and np.all(b > a), 'GT box invalid')
    wx, ix, vx = axis_candidates(name, a[:, 0], b[:, 0])
    wy, iy, vy = axis_candidates(name, a[:, 1], b[:, 1])
    area = wx[:, :, None] * wy[:, None, :]
    intersection = ix[:, :, None] * iy[:, None, :]
    gt_area = np.prod(b - a, axis=1)[:, None, None]
    # Invalid points are still finite but can imply negative unions. They never
    # participate in division or maximization.
    valid = vx[:, :, None] & vy[:, None, :]
    union = area + gt_area - intersection
    require(np.all(union[valid] > 0.), 'nonpositive legal IoU union')
    values = np.zeros_like(union)
    np.divide(intersection, union, out=values, where=valid)
    require(np.all(values >= -TOL) and np.all(values <= 1. + TOL), 'IoU outside [0,1]')
    return np.clip(values.max(axis=(1, 2)), 0., 1.)


def slice_interval(name, k):
    """Exact polygon intersection with H=L+k, expressed as an L interval."""
    require(np.all(np.isfinite(k)) and np.all(k > 0.), 'fixed current dimension must be positive')
    aa, bb, _ = DOMAINS[name]
    low = np.full(k.shape, -np.inf)
    high = np.full(k.shape, np.inf)
    for row, rhs in zip(aa, bb):
        coef = row.sum()
        bound = rhs - row[1] * k
        if abs(coef) <= DET_TOL:
            require(np.all(bound >= -TOL), 'fixed dimension has no legal slice')
        elif coef > 0.:
            high = np.minimum(high, bound / coef)
        else:
            low = np.maximum(low, bound / coef)
    require(np.all(np.isfinite(low)) and np.all(np.isfinite(high)) and np.all(high >= low),
            'invalid fixed-size slice; no clipping to a different size')
    return low, high


def fixed_normalized(name, a, b, k):
    low, high = slice_interval(name, k)
    center = np.clip((a + b) / 2., low + k / 2., high + k / 2.)
    overlap = np.maximum(np.minimum(center + k / 2., b) - np.maximum(center - k / 2., a), 0.)
    inter = np.prod(overlap, axis=1)
    union = np.prod(k, axis=1) + np.prod(b - a, axis=1) - inter
    require(np.all(union > 0.), 'fixed-size union invalid')
    value = inter / union
    require(np.all((value >= -TOL) & (value <= 1. + TOL)), 'fixed IoU range')
    return np.clip(value, 0., 1.)


def validate_boxes(refs, boxes, gt):
    for name, values in (('references', refs), ('current boxes', boxes), ('GT', gt)):
        require(values.ndim == 2 and values.shape[1] == 4 and np.all(np.isfinite(values))
                and np.all(values[:, 2:] > 0.), name + ' must have finite positive dimensions')


def oracle_matrices(refs, boxes, gt):
    validate_boxes(refs, boxes, gt)
    shape = (len(refs), len(gt))
    result = {mode: {arm: np.zeros(shape, dtype=np.float64) for arm in ARMS}
              for mode in ('joint', 'fixed_current_wh')}
    gt_lo, gt_hi = gt[:, :2] - gt[:, 2:] / 2., gt[:, :2] + gt[:, 2:] / 2.
    a = ((gt_lo[None] - refs[:, None, :2]) / refs[:, None, 2:]).reshape(-1, 2)
    b = ((gt_hi[None] - refs[:, None, :2]) / refs[:, None, 2:]).reshape(-1, 2)
    current_k = boxes[:, 2:] / refs[:, 2:]
    k = np.broadcast_to(current_k[:, None, :], shape + (2,)).reshape(-1, 2)
    for arm in ARMS:
        _, _, vertices = DOMAINS[arm]
        # Necessary and sufficient for potential positive overlap on each axis;
        # zero-overlap pairs cannot contribute to any positive IoU threshold.
        possible = np.all((a < vertices[:, 1].max()) & (b > vertices[:, 0].min()), axis=1)
        ids = np.flatnonzero(possible)
        # Validate all current widths before pruning far-away GTs.
        slice_interval(arm, current_k)
        joint_flat = result['joint'][arm].reshape(-1)
        fixed_flat = result['fixed_current_wh'][arm].reshape(-1)
        for start in range(0, len(ids), BATCH):
            require(psutil.virtual_memory().available / 2**30 > 8, 'RAM guard')
            take = ids[start:start + BATCH]
            joint_flat[take] = joint_normalized(arm, a[take], b[take])
            fixed_flat[take] = fixed_normalized(arm, a[take], b[take], k[take])
        require(np.all(joint_flat + NEST_TOL >= fixed_flat), 'joint oracle below fixed-size oracle')
    for mode in result:
        for smaller, larger in zip(ARMS[:-1], ARMS[1:]):
            require(np.all(result[mode][smaller] <= result[mode][larger] + NEST_TOL), 'domain nesting failed')
    return result


def solver_self_checks():
    checks = {}
    for smaller, larger in zip(ARMS[:-1], ARMS[1:]):
        aa, bb, _ = DOMAINS[larger]
        require(np.all(inside(DOMAINS[smaller][2], aa, bb)), 'synthetic vertex inclusion')
    checks['all_domain_inclusions_by_vertices'] = True
    # (u,k) strict witnesses: conditional\original, rectangle\conditional, range2\rectangle.
    for u, k, yes, no in ((.9, .2, 'conditional', 'original'),
                         (1., 3., 'rectangle', 'conditional'), (0., 4., 'range2', 'rectangle')):
        point = np.array([u - k / 2., u + k / 2.])
        require(inside(point, *DOMAINS[yes][:2]) and not inside(point, *DOMAINS[no][:2]), 'strict inclusion witness')
    checks['strict_inclusion_witnesses'] = True
    unit_a, unit_b = np.array([[-.5, -.5]]), np.array([[.5, .5]])
    for arm in ARMS:
        require(abs(joint_normalized(arm, unit_a, unit_b)[0] - 1.) < 1e-12, 'exact reachable GT')
        require(not inside(np.array([1., -1.]), *DOMAINS[arm][:2]), 'reversed endpoints accepted')
        require(inside(np.zeros(2), *DOMAINS[arm][:2]), 'zero-size closure missing')
    checks['exact_GT_and_endpoint_validity'] = True
    # Zero-area boxes have zero intersection, hence IoU zero with positive GT.
    require(float(H.iou(np.array([0., 0., 0., 1.]), np.array([0., 0., 1., 1.]))) == 0., 'zero-area IoU')
    valid_box = np.array([[0., 0., 1., 1.]])
    for source in range(3):
        for bad in (0., -1.):
            values = [valid_box.copy() for _ in range(3)]
            values[source][0, 2] = bad
            try:
                validate_boxes(*values)
            except RuntimeError:
                pass
            else:
                raise RuntimeError('nonpositive physical box accepted')
    checks['nonpositive_physical_dimensions_rejected'] = True
    r = 4.
    tiny_k = 2. ** -20
    s = r * (tiny_k - 1.) / 2.
    require(abs((r - abs(s)) / r - (.5 + tiny_k / 2.)) < 1e-14, 'tiny legal cap')
    require(1. + 2. * (-r) / r < 0., 's=-R must be invalid negative width')
    checks['tiny_cap_tends_R_over_2'] = True
    # Direct conditional forward/inverse, including capped-q and wide endpoint.
    for left in (-r, -.5 * r, 0., .5 * r, r):
        for right in (-r, -.5 * r, 0., .5 * r, r):
            s, t = (left + right) / 2., (right - left) / 2.
            k = 1. + 2. * s / r
            if k <= 0.:
                continue
            q = max(1. - abs(s) / r, .1)
            u = t / (r * q)
            point = np.array([u - k / 2., u + k / 2.])
            require(inside(point, *DOMAINS['conditional'][:2]), 'conditional forward outside polygon')
            recovered = np.array([s - q * r * u, s + q * r * u])
            require(np.max(np.abs(recovered - [left, right])) < 1e-12, 'inverse units or algebra')
    for lo, hi in DOMAINS['conditional'][2]:
        k, u = hi - lo, (lo + hi) / 2.
        s = r * (k - 1.) / 2.
        q = max(1. - abs(s) / r, .1)
        require(max(abs(s - q * r * u), abs(s + q * r * u)) <= r + 1e-12,
                'conditional vertex lacks physical inverse')
    checks['conditional_formula_and_inverse'] = True
    two_a, two_b = -np.ones((1, 2)), np.ones((1, 2))
    require(abs(joint_normalized('original', two_a, two_b)[0] - 1.) < 1e-12, 'joint width/height oracle')
    require(abs(fixed_normalized('original', two_a, two_b, np.ones((1, 2)))[0] - .25) < 1e-12,
            'joint and fixed-size counterexample')
    checks['both_axis_resize_vs_fixed_1_vs_quarter'] = True
    big_a, big_b = -5. * np.ones((1, 2)), 5. * np.ones((1, 2))
    for arm in ARMS:
        expected = .25 if arm == 'range2' else .09
        require(abs(joint_normalized(arm, big_a, big_b)[0] - expected) < 1e-12, 'enclosing GT area ratio')
    checks['enclosing_GT_known_area_ratio'] = True
    # Deterministic asymmetric cases; fixed-size projection is checked against
    # all breakpoints of its 1D interval, not an approximate dense grid.
    a = np.array([[-.8, .3], [1.1, -.7], [-2.3, -1.8], [.7, .8]])
    b = a + np.array([[.2, .9], [.7, 1.3], [2.2, .4], [.1, .2]])
    k = np.array([[.4, .6], [1., 1.4], [.7, .2], [2., 2.]])
    previous = None
    for arm in ARMS:
        joint = joint_normalized(arm, a, b)
        fixed = fixed_normalized(arm, a, b, k)
        require(np.all(joint + 1e-12 >= fixed), 'synthetic joint/fixed nesting')
        if previous is not None:
            require(np.all(joint + 1e-12 >= previous), 'synthetic family nesting')
        previous = joint
        lo, hi = slice_interval(arm, k)
        candidates = np.stack((lo, hi, np.clip(a, lo, hi), np.clip(b, lo, hi),
                               np.clip(a - k, lo, hi), np.clip(b - k, lo, hi)), axis=1)
        overlap = np.maximum(np.minimum(candidates + k[:, None], b[:, None])
                             - np.maximum(candidates, a[:, None]), 0.)
        max_overlap = overlap.max(axis=1)
        inter = np.prod(max_overlap, axis=1)
        brute = inter / (np.prod(k, axis=1) + np.prod(b - a, axis=1) - inter)
        require(np.max(np.abs(brute - fixed)) < 1e-12, 'fixed projection vs interval-breakpoint oracle')
    checks['fixed_slice_independent_breakpoints'] = True
    edges = np.array([[True, True], [True, False]], dtype=bool)
    require(len(H.match_capacity_one(edges)) == 2, 'augmenting path / capacity-one check')
    checks['capacity_one_matching'] = True
    return checks


def metric_report(matrix, class_edges, small):
    report = {}
    for group, mask in (('all', np.ones(len(small), dtype=bool)), ('sub4', small)):
        ids = np.flatnonzero(mask)
        for kind in ('geometry', 'trueclass'):
            permit = class_edges[:, mask] if kind == 'trueclass' else np.ones((300, len(ids)), dtype=bool)
            values = matrix[:, mask]
            for suffix, threshold in (('05', .5), ('075', .75)):
                key = group + '_' + kind + suffix
                edges = (values >= threshold) & permit
                pairs = H.match_capacity_one(edges)
                near = (np.abs(values - threshold) <= TOL) & permit
                lo_count = hi_count = len(pairs)
                if near.any():
                    lo_count = len(H.match_capacity_one((values >= threshold + TOL) & permit))
                    hi_count = len(H.match_capacity_one((values >= threshold - TOL) & permit))
                report[key] = {'matched': len(pairs), 'edges': int(edges.sum()),
                               'pairs_query_gt': [[q, int(ids[g])] for q, g in pairs],
                               'near_threshold_edges': int(near.sum()),
                               'count_threshold_plus_tol': lo_count, 'count_threshold_minus_tol': hi_count}
    return report


def analyze(cache):
    require(isinstance(cache, dict) and cache['checkpoint_sha256'] == CHECKPOINT_SHA
            and cache['weight_key'] == 'model_online', 'cache identity')
    require(scalar(cache['reg_max'], 'reg_max') == 32., 'reg_max changed')
    require(scalar(cache['up'], 'up') == .5, 'domain assumes up=.5')
    r = abs(scalar(cache['reg_scale'], 'reg_scale'))
    require(r > 0., 'reg_scale must be nonzero')
    records = cache['records']
    require(isinstance(records, list) and len(records) == 4, 'four-image panel required')
    require(len({int(item['image_index']) for item in records}) == 4, 'duplicate image index')
    records = sorted(records, key=lambda item: int(item['image_index']))
    images, matrices, totals = [], {}, {}
    gt_total = sub4_total = 0
    for record in records:
        index = int(record['image_index'])
        refs = array(record['ref_points_initial'], 'ref_points_initial', (300, 4))
        boxes = array(record['pred_boxes'], 'pred_boxes', (300, 4))
        array(record['pred_logits'], 'pred_logits', (300, 10))
        require(torch.as_tensor(record['pred_logits']).dtype == torch.float32, 'native topk requires FP32 logits')
        gt = array(record['target']['boxes'], 'target.boxes')
        labels = array(record['target']['labels'], 'target.labels', (len(gt),), integer=True)
        require(np.all((labels >= 0) & (labels < 10)), 'invalid class')
        raw_small = torch.as_tensor(record['original_sub4']).detach().cpu()
        require(raw_small.dtype == torch.bool and tuple(raw_small.shape) == (len(gt),), 'original_sub4 schema')
        small = raw_small.numpy().copy()
        ihash = record['input_sha256']
        require(isinstance(ihash, str) and len(ihash) == 64
                and all(c in '0123456789abcdef' for c in ihash.lower()), 'input SHA format')
        validate_boxes(refs, boxes, gt)
        gt_total += len(gt)
        sub4_total += int(small.sum())
        allow, flat_top300 = H.allowed_classes(record['pred_logits'])
        class_edges = allow[:, labels]
        native = H.iou(boxes[:, None, :], gt[None, :, :])
        oracle = oracle_matrices(refs, boxes, gt)
        reports = {'native_predictions': metric_report(native, class_edges, small)}
        for mode in oracle:
            reports[mode] = {arm: metric_report(oracle[mode][arm], class_edges, small) for arm in ARMS}
        # Native FP32 decoded coordinates need not lie exactly on the FP64
        # support boundary. Report this numerical residue without moving boxes.
        lower = (boxes[:, :2] - boxes[:, 2:] / 2. - refs[:, :2]) / refs[:, 2:]
        upper = (boxes[:, :2] + boxes[:, 2:] / 2. - refs[:, :2]) / refs[:, 2:]
        aa, bb, _ = DOMAINS['original']
        violation = np.stack((lower, upper), axis=-1) @ aa.T - bb
        diagnostics = {'native_original_domain_max_positive_residual': float(max(0., violation.max())),
                       'native_minus_original_fixed_IoU_max_positive_residual':
                           float(max(0., (native - oracle['fixed_current_wh']['original']).max()))}
        image = {'image_index': index, 'event': str(record['event']), 'input_sha256': ihash,
                 'GT': len(gt), 'sub4': int(small.sum()), 'native_flat_top300': flat_top300,
                 'metrics': reports, 'numerical_diagnostics': diagnostics}
        images.append(image)
        prefix = 'image' + str(index) + '_'
        matrices[prefix + 'native'] = native
        matrices[prefix + 'class_edges'] = class_edges
        matrices[prefix + 'original_sub4'] = small
        for mode in oracle:
            for arm in ARMS:
                matrices[prefix + mode + '_' + arm] = oracle[mode][arm]
        for mode, mode_report in reports.items():
            arms = {'native_predictions': mode_report} if mode == 'native_predictions' else mode_report
            for arm, metrics in arms.items():
                destination = totals.setdefault(mode, {}).setdefault(arm, {})
                for metric, detail in metrics.items():
                    row = destination.setdefault(metric, {key: 0 for key in
                            ('matched', 'edges', 'near_threshold_edges', 'count_threshold_plus_tol', 'count_threshold_minus_tol')})
                    for key in row:
                        row[key] += detail[key]
    require(gt_total == 902 and sub4_total == 453, 'panel counts changed')
    native = totals['native_predictions']['native_predictions']
    expected = {'sub4_trueclass05': 51, 'sub4_trueclass075': 10, 'sub4_geometry05': 66, 'sub4_geometry075': 15}
    require(all(native[key]['matched'] == value for key, value in expected.items()), 'C0 native reproduction mismatch')
    primary_key = 'sub4_trueclass05'
    gains = [item['metrics']['joint']['conditional'][primary_key]['matched']
             - item['metrics']['joint']['original'][primary_key]['matched'] for item in images]
    net = sum(gains)
    positives = sum(value > 0 for value in gains)
    require(net == totals['joint']['conditional'][primary_key]['matched']
            - totals['joint']['original'][primary_key]['matched'], 'primary aggregation mismatch')
    return {'state': 'complete', 'checkpoint_sha256': CHECKPOINT_SHA, 'weight_key': 'model_online',
            'reg_max': 32, 'up': .5, 'reg_scale_abs_R': r, 'GT': gt_total, 'sub4': sub4_total,
            'images': images, 'totals': totals,
            'primary': {'metric': primary_key, 'comparison': 'joint conditional minus joint original family',
                        'per_image_net': gains, 'net_gain': net, 'positive_images': positives,
                        'minimum_gain': 10, 'minimum_positive_images': 2, 'gate': bool(net >= 10 and positives >= 2)},
            'scope': 'C0 development panel support oracle only; no AP, training permission, learnability or novelty claim.'}, matrices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=pathlib.Path, default=ROOT / '127_C0_attention_template_registration/01_cache.pt')
    parser.add_argument('--out', type=pathlib.Path, default=ROOT / '147_C0_conditional_unfold_reachable')
    args = parser.parse_args()
    args.out.mkdir(parents=False, exist_ok=False)
    started = time.monotonic()
    files = {'script': pathlib.Path(__file__), 'protocol': PROTOCOL, 'helper': HELPER, 'cache': args.cache}
    before, resources, checks = {}, None, None
    result = {'state': 'error', 'primary': {'gate': False}}
    cache_loaded = False
    stage = 'resources'
    try:
        resources = H.resource_limits()
        require(resources['priority'] == 'below_normal', 'BelowNormal priority required')
        before = {key: sha(path) for key, path in files.items() if key != 'cache'}
        require(before['helper'] == HELPER_SHA, 'helper hash changed')
        stage = 'synthetic_checks_before_cache'
        checks = solver_self_checks()
        stage = 'cache_identity_and_load'
        before['cache'] = sha(args.cache)
        require(before['cache'] == CACHE_SHA, 'unexpected cache bytes')
        cache = torch.load(args.cache, map_location='cpu', weights_only=False)
        cache_loaded = True
        stage = 'cached_oracle_analysis'
        result, matrices = analyze(cache)
        stage = 'save_pairwise_oracles'
        matrix_path = args.out / '01_pairwise_oracles.npz'
        np.savez_compressed(matrix_path, **matrices)
        result['pairwise_oracles_path'] = str(matrix_path.resolve())
        result['pairwise_oracles_sha256'] = sha(matrix_path)
        stage = 'complete'
    except Exception as exc:
        result = {'state': 'error', 'primary': {'gate': False}, 'error': str(exc),
                  'traceback': traceback.format_exc(), 'scope': 'Failclosed; no method-utility inference.'}
    finally:
        after = {key: sha(files[key]) for key in before if files[key].exists()}
        unchanged = bool(before and before == after)
        if result['state'] == 'complete':
            unchanged = unchanged and set(before) == set(files)
        if not unchanged:
            result['state'] = 'error'
            result.setdefault('primary', {})['gate'] = False
            result['hash_error'] = 'Source missing or changed during audit'
        result.update({'source_paths': {key: str(path.resolve()) for key, path in files.items()},
                       'sha256_before': before, 'sha256_after': after, 'sources_unchanged': unchanged,
                       'stage': stage, 'cache_content_loaded': cache_loaded, 'resources': resources,
                       'synthetic_self_checks': checks, 'model_forwards': 0, 'model_backwards': 0,
                       'optimizer_steps': 0, 'new_training': False, 'batch_pairs': BATCH,
                       'floating_point_tolerances': {'support': TOL, 'nonparallel': DET_TOL, 'nesting': NEST_TOL},
                       'seconds': time.monotonic() - started})
        save(args.out / '00_results.json', result)
        print(H.json.dumps(H.jsonable({'state': result['state'], 'primary': result['primary'],
                                     'output': str(args.out / '00_results.json'), 'seconds': result['seconds']}),
                           ensure_ascii=False, allow_nan=False))
    if result['state'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
