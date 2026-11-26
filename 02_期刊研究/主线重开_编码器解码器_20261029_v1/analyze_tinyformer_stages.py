"""CPU coverage audit; metrics are diagnostic, never detector AP or a causal bound."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
STAGES = ['encoder_all_valid', 'encoder_topk300', 'decoder_pre', 'decoder_0',
          'decoder_1', 'decoder_2', 'decoder_3', 'native_post300']
SHORT = ['all_enc', 'topk', 'pre', 'd0', 'd1', 'd2', 'd3', 'native']


def xyxy(boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    return np.concatenate([boxes[:, :2] - boxes[:, 2:] / 2,
                           boxes[:, :2] + boxes[:, 2:] / 2], axis=1) * 640


def ious(gt, pred):
    lt = np.maximum(gt[:, None, :2], pred[None, :, :2])
    rb = np.minimum(gt[:, None, 2:], pred[None, :, 2:])
    inter = np.maximum(rb - lt, 0).prod(axis=2)
    ga = np.maximum(gt[:, 2:] - gt[:, :2], 0).prod(axis=1)
    pa = np.maximum(pred[:, 2:] - pred[:, :2], 0).prod(axis=1)
    return inter / np.maximum(ga[:, None] + pa[None, :] - inter, 1e-30)


def coverage(matrix, threshold):
    eligible = matrix >= threshold
    any_hit = eligible.any(axis=1)
    matched = np.zeros(matrix.shape[0], dtype=bool)
    if matrix.size:
        # Cardinality dominates all possible differences in summed eligible IoU.
        profit = eligible * (matrix.shape[0] + 1 + matrix)
        r, c = linear_sum_assignment(-profit)
        matched[r[eligible[r, c]]] = True
    assert matched.sum() <= any_hit.sum()
    return any_hit, matched


def selfcheck():
    gt = np.array([[0., 0, 10, 10], [1., 0, 11, 10]])
    pred = np.array([[.5, 0, 10.5, 10]])
    mat = ious(gt, pred)
    a, b = coverage(mat, .5)
    assert a.sum() == 2 and b.sum() == 1  # Duplicate GT coverage is not two detections.
    assert np.allclose(ious(gt[:1], gt[:1]), 1)
    assert coverage(np.zeros((2, 0)), .5)[0].sum() == 0
    assert coverage(np.zeros((0, 2)), .5)[1].size == 0
    # Maximum IoU alone can prefer fewer eligible pairs; cardinality has priority.
    a, b = coverage(np.array([[1., .5], [.5, 0.]]), .5)
    assert a.sum() == b.sum() == 2
    # Same geometry with a wrong class must lose class-aware coverage.
    assert coverage(np.where(np.array([[False]]), np.ones((1, 1)), 0), .5)[0].sum() == 0


def main():
    selfcheck()
    complete = json.loads((ROOT / 'receipts/study_complete.json').read_text())
    assert complete['status'] == 'STUDY100_TRACE_COMPLETE' and complete['images'] == 100
    out = ROOT / 'analysis'
    out.mkdir(exist_ok=True)
    frozen = json.loads((ROOT / 'receipts/frozen_input_contract.json').read_text())
    counters, patterns, transitions = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    per_gt, per_image, switches, trace_hashes = [], [], [], []
    max_topk_box_diff = 0.
    started = time.monotonic()
    for filename in frozen['file_names_in_existing_order']:
        path = ROOT / 'traces' / (Path(filename).stem + '.npz')
        trace_hashes.append({'file_name': path.name, 'bytes': path.stat().st_size,
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        with np.load(path, allow_pickle=False) as z:
            gt, labels, size, crowd = z['gt_boxes'], z['gt_labels'], z['gt_sqrt_area'], z['gt_crowd']
            masks = {'all': np.ones(len(gt), dtype=bool), 'lt8': size < 8,
                     '8_16': (size >= 8) & (size < 16), '16_32': (size >= 16) & (size < 32),
                     'ge32': size >= 32, 'small_lt32': size < 32,
                     'small_crowded16': (size < 32) & crowd, 'small_isolated16': (size < 32) & ~crowd}
            rows = [{'image': filename, 'uid': str(uid), 'sqrt_area_640_stretch': float(sz),
                     'small_crowded16': int(cr)} for uid, sz, cr in zip(z['gt_uids'], size, crowd)]
            image_row = {'image': filename}
            for key, mask in masks.items():
                image_row[key + '_n'] = int(mask.sum())
            stored_hits, query_gt = {}, {}
            token_lookup = {int(t): i for i, t in enumerate(z['encoder_token_ids'])}
            top_positions = np.array([token_lookup[int(t)] for t in z['topk_token_ids']])
            diff = np.max(np.abs(z['encoder_all_valid_boxes'][top_positions] - z['encoder_topk300_boxes']))
            max_topk_box_diff = max(max_topk_box_diff, float(diff))
            assert diff <= 1e-6
            assert np.array_equal(z['encoder_all_valid_logits'][top_positions], z['encoder_topk300_logits'])
            for name, short in zip(STAGES, SHORT):
                if name == 'native_post300':
                    boxes, pred_labels = z['native_pp_boxes'].astype(np.float64), z['native_pp_labels']
                else:
                    boxes, pred_labels = xyxy(z[name + '_boxes']), z[name + '_logits'].argmax(axis=1)
                geometric = ious(gt, boxes)
                class_aware = np.where(labels[:, None] == pred_labels[None, :], geometric, 0)
                for kind, mat in [('geom', geometric), ('class', class_aware)]:
                    best = mat.max(axis=1) if mat.shape[1] else np.zeros(len(gt))
                    for i, row in enumerate(rows):
                        row[short + '_' + kind + '_best_iou'] = float(best[i])
                    for th, tag in [(.5, '50'), (.75, '75')]:
                        ah, oh = coverage(mat, th)
                        stored_hits[(name, kind, tag)] = ah
                        for i, row in enumerate(rows):
                            row[short + '_' + kind + '_any_' + tag] = int(ah[i])
                            row[short + '_' + kind + '_one_' + tag] = int(oh[i])
                        for stratum, mask in masks.items():
                            c = counters[(stratum, name)]
                            c['n'] = int(c.get('n', 0)) if 'n' in c else 0
                            c[kind + '_any_' + tag] += int(ah[mask].sum())
                            c[kind + '_one_' + tag] += int(oh[mask].sum())
                            image_row[stratum + '_' + short + '_' + kind + '_any_' + tag] = int(ah[mask].sum())
                            image_row[stratum + '_' + short + '_' + kind + '_one_' + tag] = int(oh[mask].sum())
                for stratum, mask in masks.items():
                    counters[(stratum, name)]['n'] += int(mask.sum())
                    counters[(stratum, name)]['images_with_gt'] += int(mask.any())
                if name.startswith('decoder_') and name != 'decoder_pre':
                    # Per-query nearest GT, without one-to-one exclusivity; not identity truth.
                    if len(gt):
                        best_gt = class_aware.argmax(axis=0)
                        score = class_aware[best_gt, np.arange(len(boxes))]
                        query_gt[name] = np.where(score >= .5, best_gt, -1)
            for kind in ['geom', 'class']:
                for tag in ['50', '75']:
                    hits = [stored_hits[(n, kind, tag)] for n in STAGES]
                    for stratum, mask in masks.items():
                        for i in np.flatnonzero(mask):
                            bit = ''.join(str(int(h[i])) for h in hits)
                            patterns[(stratum, kind, tag)][bit] += 1
                        for prev, nxt, ph, nh in zip(STAGES, STAGES[1:], hits, hits[1:]):
                            c = transitions[(stratum, kind, tag, prev, nxt)]
                            c['n'] += int(mask.sum())
                            c['retained'] += int((ph & nh & mask).sum())
                            c['lost'] += int((ph & ~nh & mask).sum())
                            c['gained'] += int((~ph & nh & mask).sum())
                            c['neither'] += int((~ph & ~nh & mask).sum())
            for i in range(3):
                prev, nxt = query_gt['decoder_' + str(i)], query_gt['decoder_' + str(i + 1)]
                for q in np.flatnonzero((prev >= 0) & (nxt >= 0) & (prev != nxt)):
                    switches.append({'image': filename, 'from_layer': i, 'to_layer': i + 1, 'query_slot': int(q),
                                     'from_uid': str(z['gt_uids'][prev[q]]), 'to_uid': str(z['gt_uids'][nxt[q]]),
                                     'from_small': int(size[prev[q]] < 32), 'to_small': int(size[nxt[q]] < 32),
                                     'from_crowded': int(crowd[prev[q]]), 'to_crowded': int(crowd[nxt[q]])})
            per_gt.extend(rows)
            per_image.append(image_row)
    assert len(per_gt) == 4491 and len(per_image) == 100
    # Source-image bootstrap; conditional on this in-sample sample/checkpoint.
    rng = np.random.default_rng(20261029)
    draws = rng.integers(0, 100, size=(2000, 100))
    comparisons = []
    for stratum in ['all', 'small_lt32', 'small_crowded16', 'small_isolated16', 'lt8', '8_16', '16_32']:
        den = np.array([r[stratum + '_n'] for r in per_image])
        for kind in ['geom', 'class']:
            for first, last in [('all_enc', 'topk'), ('topk', 'd3'), ('d0', 'd3'), ('d3', 'native')]:
                a = np.array([r[stratum + '_' + first + '_' + kind + '_any_50'] for r in per_image])
                b = np.array([r[stratum + '_' + last + '_' + kind + '_any_50'] for r in per_image])
                boot = (b - a)[draws].sum(axis=1) / den[draws].sum(axis=1) * 100
                comparisons.append({'stratum': stratum, 'kind': kind, 'iou': .5, 'from': first, 'to': last,
                                    'delta_coverage_percentage_points': float((b.sum() - a.sum()) / den.sum() * 100),
                                    'source_image_bootstrap_95_percentile_ci': np.quantile(boot, [.025, .975]).tolist()})
    summary = {'status': 'CPU_STAGE_COVERAGE_COMPLETE', 'images': 100, 'positive_gt': 4491,
               'source_scope': 'Author trained-on-train checkpoint, unchanged exploratory train100; no generalization or causal stage attribution',
               'coverage_scope': 'No score threshold; class-aware uses per-query argmax except native flat top300 endpoint; unmatched all-encoder head readout is diagnostic only',
               'one_to_one': 'Whole-image maximum eligible cardinality, then sum IoU; stratum counts taken after global pairing',
               'selfchecks': 'PASS duplicate-coverage, cardinality, empty sets, identity IoU and wrong-class cases',
               'topk_subset_max_box_abs_difference': max_topk_box_diff,
               'coverage': [{'stratum': s, 'stage': n, **dict(c)} for (s, n), c in counters.items()],
               'transitions': [{'stratum': s, 'kind': k, 'iou': t, 'from': p, 'to': n, **dict(c)}
                               for (s, k, t, p, n), c in transitions.items()],
               'coverage_patterns': [{'stratum': s, 'kind': k, 'iou': t, 'stage_order': STAGES,
                                      'bit_patterns': dict(c)} for (s, k, t), c in patterns.items()],
               'paired_coverage_comparisons': comparisons, 'nearest_GT_query_switch_events': len(switches),
               'query_switch_warning': 'Nearest GT at IoU>=.5 and same predicted argmax class; no causal or true identity interpretation',
               'bootstrap': {'unit': 'source image', 'draws': 2000, 'seed': 20261029,
                             'limits': 'Conditional descriptive interval; related frames may violate image independence'},
               'analysis_seconds': time.monotonic() - started,
               'analysis_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'training_updates': 0, 'additional_model_forwards': 0}
    for filename, rows in [('coverage_by_gt.csv', per_gt), ('coverage_by_image.csv', per_image), ('nearest_gt_query_switches.csv', switches)]:
        with (out / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['no_events'])
            writer.writeheader()
            writer.writerows(rows)
    (out / 'trace_sha256_manifest.json').write_text(json.dumps(trace_hashes, indent=2))
    (out / 'stage_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'status': summary['status'], 'images': 100, 'positive_gt': 4491,
                      'analysis_seconds': summary['analysis_seconds'], 'query_switch_events': len(switches)}))
    for row in summary['coverage']:
        if row['stratum'] in ['all', 'small_crowded16', 'small_isolated16', 'lt8']:
            print(json.dumps(row))


if __name__ == '__main__':
    main()
