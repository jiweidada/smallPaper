"""170 preregistered single-edge secondary-peak necessary upper bound.

Write-only delivery to root; do not import this file to run model code.
No model/checkpoint/optimizer construction, autograd, forward or remote access.
Synthetic tests precede even hashing/opening the cache. Never overwrite output.
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
HELPER = ROOT / '136_C0_pmf_translation_prior_fp32.py'
PROTOCOL = ROOT / '170_单边次峰多假设_必要上界事前协议.txt'
SOURCE = ROOT / '168_双阴性回交_首答后调用失败_完整原始.json'
CACHE = ROOT / '127_C0_attention_template_registration/01_cache.pt'
DEFAULT_OUT = ROOT / '172_C0_secondary_peak_upperbound'
CHECKPOINT_SHA = '9d6c0dd7a9cd6cb56465c4eab9416c0e89547c7e45df936e559fa12b921f275e'
CACHE_SHA = 'f2fb2120ccd670f846a0a70651a3fabbf9bcfb636c94d43d33f8dc34eceac79c'
HELPER_SHA = '10bc190cce57688cd8c04b31190fc356244cc2afd6df5b6c74f8bed4eca6ab26'
SOURCE_SHA = '739e21f065edfb1a8760599eb20115be5e4a7e287f5cc5e2b6bfd82f9adf5a26'
STATE_SHA = 'c10a2b8b8f7244c6bdb4d1a3261a85af829d303b18469ceefb25acc5fdd5e60b'
EDGES = ('l', 't', 'r', 'b')
EXPECTED = {'all_geo05': 370, 'sub4_geo05': 66, 'all_geo075': 184, 'sub4_geo075': 15,
            'all_class05': 324, 'sub4_class05': 51, 'all_class075': 162, 'sub4_class075': 10}


def file_sha(path):
    result = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


if file_sha(HELPER) != HELPER_SHA:
    raise RuntimeError('136 helper changed; audit before execution')
_spec = importlib.util.spec_from_file_location('_secondary171_helpers', HELPER)
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)
np, torch = h.np, h.torch
require = h.require


def local_peaks(prob):
    """One representative (lowest index) per strict maximal equal-value run."""
    prob = np.asarray(prob, dtype=np.float64)
    require(prob.ndim == 1 and len(prob) >= 3 and np.isfinite(prob).all(), 'invalid PMF')
    require((prob >= 0).all() and abs(float(prob.sum()) - 1.) < 2e-6, 'PMF normalization')
    peaks, start = [], 0
    while start < len(prob):
        end = start
        while end + 1 < len(prob) and prob[end + 1] == prob[start]:
            end += 1
        left_ok = start == 0 or prob[start] > prob[start - 1]
        right_ok = end + 1 == len(prob) or prob[end] > prob[end + 1]
        if left_ok and right_ok:
            peaks.append({'index': start, 'end': end, 'mass': float(prob[start])})
        start = end + 1
    peaks.sort(key=lambda item: (-item['mass'], item['index']))
    for a in range(len(peaks)):
        for b in range(a):
            lo, hi = sorted((peaks[a], peaks[b]), key=lambda item: item['index'])
            require(hi['index'] - lo['end'] >= 2, 'peaks are adjacent')
    return peaks


def legal_boxes(boxes):
    boxes = np.asarray(boxes)
    return np.isfinite(boxes).all(-1) & (boxes[..., 2] > 0) & (boxes[..., 3] > 0)


def expanded_order(main_scores, native_order, ratios):
    """Synthetic budget proof: main first on ties, old main order preserved."""
    main_scores = np.asarray(main_scores, dtype=np.float64)
    require(sorted(native_order) == list(range(len(main_scores))), 'not full main order')
    position = {q: rank for rank, q in enumerate(native_order)}
    entries = [(float(s), 0, position[q], -1, q) for q, s in enumerate(main_scores)]
    for q, ratio in ratios:
        require(0 <= ratio <= 1, 'ratio outside [0,1]')
        entries.append((float(main_scores[q] * ratio), 1, position[q], len(entries), q))
    return sorted(entries, key=lambda a: (-a[0], a[1], a[2], a[3]))


def synthetic_tests(native):
    require([p['index'] for p in local_peaks([.3, .1, .2, .1, .3])] == [0, 4, 2], 'endpoint/tie test')
    plateau = local_peaks([.05, .3, .3, .05, .2, .1])
    require(plateau[0] == {'index': 1, 'end': 2, 'mass': .3}, 'plateau representative')
    require(plateau[1]['index'] == 4, 'plateau secondary')
    require(len(local_peaks(np.ones(33) / 33.)) == 1, 'flat PMF has only one plateau')
    monotone = np.arange(1., 34.)
    require(local_peaks(monotone / monotone.sum())[0]['index'] == 32, 'one-sided endpoint')
    require(plateau[1]['mass'] / plateau[0]['mass'] < 1, 'ratio test')
    equal = local_peaks([.3, .1, .2, .1, .3])
    require(equal[1]['mass'] / equal[0]['mass'] == 1, 'equal peaks must remain legal')
    # Plateau mass is not accumulated: .06*3 > .1 but its peak mass is .06.
    low = np.array([.02, .06, .06, .06, .02, .78])
    require(local_peaks(low)[1]['mass'] < .1, 'raw node mass, not plateau mass')
    counts = 0
    for scores, order, k in [([.9, .8, .7, .6], [0, 1, 2, 3], 2),
                             ([.9, .5, .5, .5], [0, 2, 1, 3], 2)]:
        ratios = [(q, r) for q in range(4) for r in (0., .5, 1.)]
        chosen = expanded_order(scores, order, ratios)[:k]
        require(all(item[4] in set(order[:k]) for item in chosen), 'eligibility expanded')
        counts += 1
    # K=1, main0 originally selected. At equality an arbitrary secondary-first
    # tie rule can incorrectly admit outside main1's secondary instead.
    tied = expanded_order([.5, .5], [0, 1], [(1, 1.)])
    require(tied[0][4] == 0, 'main-first rule must preserve cutoff eligibility')
    arbitrary = sorted(tied, key=lambda item: (-item[0], -item[1], -item[2]))
    require(arbitrary[0][1] == 1 and arbitrary[0][4] == 1, 'tie counterexample failed')
    capacity = len(h.match_capacity_one(np.ones((1, 2), dtype=bool)))
    duplicate_capacity = len(h.match_capacity_one(np.ones((2, 2), dtype=bool)))
    require(capacity == 1 and duplicate_capacity == 2, 'must preserve query capacity')
    scale = torch.tensor([4.], dtype=torch.float32)
    points = torch.tensor([[.5, .5, .04, .08]], dtype=torch.float32)
    distance = torch.zeros((1, 4), dtype=torch.float32)
    base = native.distance2bbox(points, distance, scale)
    changed = distance.clone()
    changed[0, 0] = 2.
    decoded = native.distance2bbox(points, changed, scale)
    require(float(torch.max(torch.abs(decoded - torch.tensor([[.49, .5, .06, .08]])))) < 1e-7,
            'single l edge must shift center and size by native transform')
    changed[0, 0] = -4.
    zero = native.distance2bbox(points, changed, scale)
    changed[0, 0] = -5.
    negative = native.distance2bbox(points, changed, scale)
    require(bool(legal_boxes(base.numpy())[0]) and not bool(legal_boxes(zero.numpy())[0])
            and not bool(legal_boxes(negative.numpy())[0]), 'nonpositive width rejection')
    return {'passed': True, 'tests': ['endpoints', 'plateau_low_index_raw_mass', 'flat', 'nonadjacent',
            'ratio_including_equal', 'strict_and_tied_budget_no_new_eligibility',
            'arbitrary_tie_counterexample', 'query_capacity_one', 'native_single_edge', 'positive_size'],
            'budget_cases': counts, 'cache_opened': False}


def build_candidates(raw, refs, nodes, scale, native):
    """No boxes/GT/labels enter peak or secondary-box construction."""
    raw = torch.as_tensor(raw).detach().cpu()
    refs = torch.as_tensor(refs).detach().cpu()
    require(raw.dtype == refs.dtype == nodes.dtype == torch.float32, 'native FP32 required')
    require(tuple(raw.shape) == (300, 132) and tuple(refs.shape) == (300, 4), 'native input shape')
    require(bool(torch.isfinite(raw).all()) and bool(torch.isfinite(refs).all()), 'native input nonfinite')
    require(bool((refs[:, 2:] > 0).all()), 'nonpositive reference size')
    probability = torch.nn.functional.softmax(raw.reshape(-1, 33), dim=1)
    expected = torch.nn.functional.linear(probability, nodes).reshape(300, 4)
    decoded_original = native.distance2bbox(refs, expected, scale)
    p = probability.reshape(300, 4, 33).numpy().astype(np.float64)
    pairs = np.full((300, 4, 2), -1, dtype=np.int64)
    masses = np.zeros((300, 4, 2), dtype=np.float64)
    ratios = np.zeros((300, 4), dtype=np.float64)
    peaks_all = []
    distances = expected[:, None, :].expand(300, 4, 4).clone()
    for q in range(300):
        row = []
        for e in range(4):
            peaks = local_peaks(p[q, e])
            row.append(peaks)
            for rank, peak in enumerate(peaks[:2]):
                pairs[q, e, rank] = peak['index']
                masses[q, e, rank] = peak['mass']
            if len(peaks) >= 2:
                ratio = peaks[1]['mass'] / peaks[0]['mass']
                require(0 < ratio <= 1, 'secondary ratio exceeds main')
                ratios[q, e] = ratio
                distances[q, e, e] = nodes[peaks[1]['index']]
        peaks_all.append(row)
    boxes = native.distance2bbox(refs[:, None, :].expand(300, 4, 4), distances, scale)
    legal = legal_boxes(boxes.numpy())
    has_two = pairs[:, :, 1] >= 0
    valid = {name: has_two & (masses[:, :, 1] >= threshold) & legal
             for name, threshold in [('tau10', .1), ('tau20', .2)]}
    require(not np.any(valid['tau20'] & ~valid['tau10']), 'threshold subsets')
    return {'decoded_original': decoded_original, 'boxes': boxes, 'valid': valid, 'legal': legal,
            'has_two': has_two, 'peaks': peaks_all, 'peak_indices': pairs,
            'peak_masses': masses, 'ratios': ratios, 'expected_edges': expected, 'pmf': p}


def peak_report(candidate):
    result = {}
    for e, name in enumerate(EDGES):
        numbers = [len(row[e]) for row in candidate['peaks']]
        result[name] = {'peak_count_hist': {str(n): numbers.count(n) for n in sorted(set(numbers))},
                        'has_two': int(candidate['has_two'][:, e].sum())}
        for arm, threshold in [('tau10', .1), ('tau20', .2)]:
            threshold_ok = candidate['has_two'][:, e] & (candidate['peak_masses'][:, e, 1] >= threshold)
            boxes = candidate['boxes'][:, e].numpy()
            result[name][arm] = {'threshold_pass': int(threshold_ok.sum()),
                'legal': int(candidate['valid'][arm][:, e].sum()),
                'nonfinite': int((threshold_ok & ~np.isfinite(boxes).all(-1)).sum()),
                'nonpositive_width': int((threshold_ok & (boxes[:, 2] <= 0)).sum()),
                'nonpositive_height': int((threshold_ok & (boxes[:, 3] <= 0)).sum())}
    for arm in ('tau10', 'tau20'):
        counts = candidate['valid'][arm].sum(1)
        result[arm] = {'valid_secondary_total': int(counts.sum()),
                       'queries_with_secondary': int((counts > 0).sum()),
                       'valid_count_per_query_hist': {str(n): int((counts == n).sum()) for n in range(5)}}
    return result


def add_count_tree(total, increment):
    """Sum peak-frequency reports; every leaf is a count, never a probability."""
    for key, value in increment.items():
        if isinstance(value, dict):
            add_count_tree(total.setdefault(key, {}), value)
        else:
            require(isinstance(value, int), 'peak aggregate must contain counts only')
            total[key] = total.get(key, 0) + value


def score_matrix(original, candidates, valid, gt):
    # Each q retains one identity; variants are reduced on geometry before matching.
    scores = [h.iou(original[:, None, :], gt[None, :, :])]
    for e in range(4):
        matrix = np.full((300, len(gt)), -1., dtype=np.float64)
        indices = np.flatnonzero(valid[:, e])
        if len(indices):
            matrix[indices] = h.iou(candidates[indices, e, None, :], gt[None, :, :])
        scores.append(matrix)
    stacked = np.stack(scores, axis=0)
    return stacked.max(0), stacked.argmax(0)


def evaluate(matrix, witnesses, class_eligible, small):
    counts, pairs, existence, sensitivity = {}, {}, {}, {}
    for group, mask in [('all', np.ones(len(small), dtype=bool)), ('sub4', small)]:
        ids = np.flatnonzero(mask)
        for kind, allow in [('geo', np.ones(matrix.shape, dtype=bool)), ('class', class_eligible)]:
            for tag, threshold in [('05', .5), ('075', .75)]:
                key = group + '_' + kind + tag
                edges = (matrix[:, ids] >= threshold) & allow[:, ids]
                matched = h.match_capacity_one(edges)
                counts[key] = len(matched)
                pairs[key] = [{'query': q, 'gt_index': int(ids[g]),
                               'box_variant': int(witnesses[q, ids[g]]), 'IoU': float(matrix[q, ids[g]])}
                              for q, g in matched]
                existence[key] = ids[np.any(edges, axis=0)].tolist()
                sensitivity[key] = {label: len(h.match_capacity_one((matrix[:, ids] >= threshold + delta)
                                                                   & allow[:, ids]))
                                    for label, delta in [('minus1e-8', -1e-8), ('plus1e-8', 1e-8)]}
    return {'counts': counts, 'matching': pairs, 'existence_gt_indices': existence,
            'threshold_sensitivity': sensitivity}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=pathlib.Path)
    parser.add_argument('--synthetic-only', action='store_true')
    args = parser.parse_args()
    default = ROOT / '172_C0_secondary_peak_synthetic_precheck' if args.synthetic_only else DEFAULT_OUT
    out = (args.out or default).resolve()
    require(out.parent == ROOT.resolve() and out.name.startswith('172_'), 'output restricted to new 172_* directory')
    require(not out.exists(), 'never overwrite existing output')
    start = time.perf_counter()
    resources = h.resource_limits()
    require(resources['priority'] == 'below_normal', 'BelowNormal required')
    out.mkdir()
    execution = {'state': 'running', 'resources': resources, 'model_forwards': 0, 'model_vjps': 0,
                 'model_backwards': 0, 'cache_autograd': 0, 'optimizer_steps': 0,
                 'checkpoint_loads': 0, 'cache_loads': 0}
    hashes = {}
    try:
        paths = [pathlib.Path(__file__).resolve(), PROTOCOL, SOURCE, HELPER,
                 h.NATIVE / 'dfine_utils.py', h.NATIVE / 'box_ops.py']
        hashes = {str(path): file_sha(path) for path in paths}
        require(hashes[str(SOURCE)] == SOURCE_SHA, '168 source changed')
        require(hashes[str(h.NATIVE / 'dfine_utils.py')] == 'ad87ec5a50b1970e76b06f7242298ffa332beaf78c828ac3e8d41ac656ec5f6c', 'native math changed')
        require(hashes[str(h.NATIVE / 'box_ops.py')] == '398fc107fc003b0fabaf36154594d50a8e0aa54195056b343031960c0aff0975', 'native box ops changed')
        native = h.load_native_math()
        h.save(out / '00_synthetic.json', synthetic_tests(native))
        if args.synthetic_only:
            unchanged = {path: file_sha(path) == digest for path, digest in hashes.items()}
            require(all(unchanged.values()), 'source mutation during synthetic precheck')
            execution.update(state='synthetic_only_complete', source_hashes=hashes,
                             source_unchanged=unchanged, elapsed_seconds=time.perf_counter() - start,
                             cache_opened=False, cache_hashed=False)
            h.save(out / '04_execution.json', execution)
            print('synthetic precheck passed; cache not opened or hashed', flush=True)
            return
        # No cache bytes are opened or hashed until all synthetic tests pass.
        hashes[str(CACHE)] = file_sha(CACHE)
        require(hashes[str(CACHE)] == CACHE_SHA, '127 cache changed')
        require(h.psutil.virtual_memory().available > 8 * 2**30, 'RAM guard before cache')
        cache = torch.load(CACHE, map_location='cpu', weights_only=False)
        execution['cache_loads'] += 1
        require(cache['checkpoint_sha256'] == CHECKPOINT_SHA and cache['weight_key'] == 'model_online', 'C0 identity')
        require(cache['model_state_sha256'] == STATE_SHA and cache['mode'] == 'eval', 'C0 state/mode')
        require(int(cache['reg_max']) == 32 and h.scalar(cache['up'], 'up') == .5
                and h.scalar(cache['reg_scale'], 'scale') == 4., 'native support metadata')
        records = cache['records']
        require(len(records) == 4 and [r['image_index'] for r in records] == list(range(4)), 'four fixed images')
        up, scale = torch.tensor([.5], dtype=torch.float32), torch.tensor([4.], dtype=torch.float32)
        nodes = native.weighting_function(32, up, scale)
        require(nodes.dtype == torch.float32 and tuple(nodes.shape) == (33,)
                and bool((nodes[1:] > nodes[:-1]).all()), '33 native nodes')
        totals = {arm: dict.fromkeys(EXPECTED, 0) for arm in ('original', 'tau10', 'tau20')}
        reports, derived, peak_totals = [], [], {}
        gt_total, sub4_total = 0, 0
        for number, record in enumerate(records):
            require(h.psutil.virtual_memory().available > 8 * 2**30, 'RAM guard per image')
            boxes = h.array(record['pred_boxes'], 'pred_boxes', (300, 4))
            logits = torch.as_tensor(record['pred_logits']).detach().cpu()
            require(logits.dtype == torch.float32 and tuple(logits.shape) == (300, 10)
                    and bool(torch.isfinite(logits).all()), 'logit interface')
            require(legal_boxes(boxes).all(), 'cached original boxes illegal')
            gt = h.array(record['target']['boxes'], 'GT')
            labels = h.array(record['target']['labels'], 'labels', integer=True)
            small_tensor = torch.as_tensor(record['original_sub4']).detach().cpu()
            require(small_tensor.dtype == torch.bool, 'original_sub4 bool contract')
            small = small_tensor.numpy().copy()
            require(gt.shape == (len(labels), 4) and small.shape == labels.shape
                    and legal_boxes(gt).all() and (labels >= 0).all() and (labels < 10).all(), 'GT contract')
            require(len(gt) == [146, 280, 418, 58][number]
                    and int(small.sum()) == [46, 121, 271, 15][number], 'fixed per-image cohorts')
            allow, flat = h.allowed_classes(logits)
            saved_flat = torch.as_tensor(record['natural_top300_flat_indices']).reshape(-1).tolist()
            require(flat == saved_flat, 'native top300 differs from cached tie order')
            candidate = build_candidates(record['final_corners'], record['ref_points_initial'], nodes, scale, native)
            parity = float(np.max(np.abs(candidate['decoded_original'].numpy().astype(np.float64) - boxes)))
            require(parity <= 1e-6, 'native expected box parity failed')
            probabilities = logits.sigmoid().numpy().astype(np.float64)
            scores = probabilities[:, None, :] * candidate['ratios'][:, :, None]
            require(np.all(scores <= probabilities[:, None, :]), 'secondary score exceeds own main')
            class_eligible = allow[:, labels]
            matrix0 = h.iou(boxes[:, None, :], gt[None, :, :])
            result = {'original': evaluate(matrix0, np.zeros_like(matrix0, dtype=np.int64), class_eligible, small)}
            for arm in ('tau10', 'tau20'):
                matrix, witnesses = score_matrix(boxes, candidate['boxes'].numpy().astype(np.float64), candidate['valid'][arm], gt)
                require(np.all(matrix >= matrix0), 'oracle lost original geometry')
                result[arm] = evaluate(matrix, witnesses, class_eligible, small)
                result[arm]['perGT_existence_vs_original'] = {}
                for key in EXPECTED:
                    before = set(result['original']['existence_gt_indices'][key])
                    after = set(result[arm]['existence_gt_indices'][key])
                    result[arm]['perGT_existence_vs_original'][key] = {
                        'gained': sorted(after - before), 'lost': sorted(before - after)}
                    require(not before - after, 'existence should be nested')
            for key in EXPECTED:
                require(result['original']['counts'][key] <= result['tau20']['counts'][key]
                        <= result['tau10']['counts'][key], 'matching not nested')
                for arm in totals:
                    totals[arm][key] += result[arm]['counts'][key]
            gt_total += len(gt)
            sub4_total += int(small.sum())
            report = {'image_index': number, 'event': record['event'], 'input_sha256': record['input_sha256'],
                      'GT': len(gt), 'original_sub4': int(small.sum()), 'native_box_max_abs_error': parity,
                      'native_top300_exact': True, 'peaks': peak_report(candidate), 'arms': result}
            h.save(out / f'02_image_{number}.json', report)
            reports.append(report)
            add_count_tree(peak_totals, report['peaks'])
            derived.append({'image_index': number, 'event': record['event'], 'input_sha256': record['input_sha256'],
                'pred_boxes': record['pred_boxes'], 'pred_logits': logits, 'selected_indices': record['selected_indices'],
                'target': record['target'], 'original_sub4': record['original_sub4'],
                'natural_top300_flat_indices': torch.tensor(flat), 'secondary_boxes': candidate['boxes'],
                'valid_tau10': torch.from_numpy(candidate['valid']['tau10']),
                'valid_tau20': torch.from_numpy(candidate['valid']['tau20']),
                'peak_indices': torch.from_numpy(candidate['peak_indices']),
                'peak_masses': torch.from_numpy(candidate['peak_masses']),
                'ratios': torch.from_numpy(candidate['ratios']), 'all_peaks': candidate['peaks']})
            print('completed fixed image', number, 'sub4 class .5',
                  {arm: result[arm]['counts']['sub4_class05'] for arm in totals}, flush=True)
        require(gt_total == 902 and sub4_total == 453, 'fixed panel size')
        require(totals['original'] == EXPECTED, 'original eight counts failed parity')
        increments = [r['arms']['tau10']['counts']['sub4_class05'] - r['arms']['original']['counts']['sub4_class05']
                      for r in reports]
        gain = totals['tau10']['sub4_class05'] - totals['original']['sub4_class05']
        checks = {'net_sub4_class05_gain_at_least_10': gain >= 10,
                  'at_least_2_positive_images': sum(n > 0 for n in increments) >= 2}
        gate = {'passed': all(checks.values()), 'checks': checks, 'gain': gain,
                'per_image_gain': increments, 'positive_images': sum(n > 0 for n in increments),
                'meaning': 'Only a necessary upper-bound resource test. Failure stops this exact K2 version; '
                           'pass permits fixed selector/score-budget test, not training/AP/novelty.'}
        torch.save({'records': derived, 'reg_max': 32, 'up': .5, 'reg_scale': 4., 'nodes': nodes,
                    'checkpoint_sha256': CHECKPOINT_SHA, 'weight_key': 'model_online',
                    'source_cache_sha256': CACHE_SHA, 'source_hashes': hashes,
                    'schema': 'single-edge-secondary-oracle-v1; variant0=main,1..4=l,t,r,b'}, out / '01_candidates.pt')
        unchanged = {path: file_sha(path) == digest for path, digest in hashes.items()}
        require(all(unchanged.values()), 'input/source mutation')
        h.save(out / '03_results.json', {'state': 'complete', 'GT': gt_total, 'original_sub4': sub4_total,
            'counts': totals, 'gate': gate, 'thresholds': {'primary': .1, 'descriptive_only': .2},
            'per_image_counts': [{arm: r['arms'][arm]['counts'] for arm in totals} for r in reports],
            'per_image_peak_reports': [r['peaks'] for r in reports],
            'peak_totals': peak_totals,
            'source_hashes': hashes, 'source_unchanged': unchanged,
            'interpretation': 'query-capacity-one upper bound with original class eligibility; not COCO AP or deployment'} )
        execution.update(state='complete', source_hashes=hashes, source_unchanged=unchanged,
                         elapsed_seconds=time.perf_counter() - start, gate=gate)
        h.save(out / '04_execution.json', execution)
        print('completed; gate', gate['passed'], 'gain', gain, flush=True)
    except Exception:
        execution.update(state='failed', elapsed_seconds=time.perf_counter() - start,
                         source_hashes=hashes, error=traceback.format_exc())
        h.save(out / '04_execution.json', execution)
        raise


if __name__ == '__main__':
    main()
