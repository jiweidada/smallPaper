"""Per-edge informativeness of the adaptive range, on the frozen traces.

This is the level at which the candidate actually operates: one radius per
edge. Two mutually exclusive readings of the mechanism are tested, and each
one is fatal to the other if it fails:

  Reading A (range as a search aperture): the aperture should scale with the
    real per-edge error. Measured by P(|mean_edge - gt_edge| <= r), by the
    calibration slope median(error/r), and by rank correlation between r and
    the real error. An aperture narrower than the error distribution cannot
    see the boundary it is supposed to refine.
  Reading B (narrow as confidence): if the narrow range is the point, then the
    three per-edge reads must not collapse, otherwise the 12-point budget is
    mostly redundant. Collapse is reported by audit_innovation_effect_size.py.

Only the frozen old-EMEA traces are read. No training, no CUDA, no GT in
sampling (GT enters the scoring of already-sampled geometry only).
"""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np

EDGES = ('left', 'top', 'right', 'bottom')


def _rank(values):
    """Average ranks for ties, pure Python: this env's BLAS path crashes."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman(x, y):
    """Rank correlation, computed without BLAS."""
    rx, ry = _rank([float(v) for v in x]), _rank([float(v) for v in y])
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return float(num / (dx * dy)) if dx > 0 and dy > 0 else None


def auroc(scores, labels):
    """Rank AUROC; BLAS-free."""
    scores = [float(s) for s in scores]
    labels = [bool(v) for v in labels]
    pos, neg = sum(labels), len(labels) - sum(labels)
    if pos == 0 or neg == 0:
        return None
    ranks = _rank(scores)
    return float((sum(r for r, l in zip(ranks, labels) if l) - pos * (pos + 1) / 2) / (pos * neg))


def main(args):
    root, out = Path(args.traces), Path(args.output)
    if out.exists():
        raise RuntimeError('Preserving previous audit; choose a new output')
    frozen = json.loads(Path(args.result).read_text(encoding='utf8'))['trace_hashes']
    seen, records, hashes = set(), [], []
    for path in sorted(root.glob('*.npz')):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == frozen[path.name], path.name
        hashes.append(digest)
        d = np.load(path)
        image_id = int(path.stem.split('_')[0])
        mean_edge, q3, side = d['mean_edge_px'], d['q3_edge_px'], d['initial_edge_size_px']
        gt_edges = d['gt_edges_crop_px']
        for i in range(300):
            area = float(d['original_gt_area'][i])
            matched = int(d['match_annotation_id'][i]) >= 0
            if not (matched and area < 1024 and not bool(d['truncated'][i])):
                continue
            key = (image_id, int(d['match_annotation_id'][i]), i)
            if key in seen:            # one row per (window, query) is already unique per GT
                continue
            seen.add(key)
            r_a = .5 * (np.asarray(q3[i], float)[:, 2] - np.asarray(q3[i], float)[:, 0])
            r_f = .25 * np.asarray(side[i], float)
            err = np.abs(np.asarray(mean_edge[i], float) - np.asarray(gt_edges[i], float))
            for e in range(4):
                records.append({'image': image_id, 'gt': key[1], 'edge': e,
                                'iou': float(d['match_iou'][i]), 'area': area,
                                'r_a': float(r_a[e]), 'r_f': float(r_f[e]),
                                'err': float(err[e])})
    assert len(hashes) == 170, len(hashes)

    def block(recs, tag):
        ra = np.array([r['r_a'] for r in recs])
        rf = np.array([r['r_f'] for r in recs])
        err = np.array([r['err'] for r in recs])
        iou = np.array([r['iou'] for r in recs])
        return {
            'tag': tag, 'edge_query_rows': len(recs),
            'unique_GT_edges': len({(r['image'], r['gt'], r['edge']) for r in recs}),
            'mean_error_px': float(err.mean()), 'median_error_px': float(np.median(err)),
            'mean_r_a_px': float(ra.mean()), 'median_r_a_px': float(np.median(ra)),
            'mean_r_f_px': float(rf.mean()), 'median_r_f_px': float(np.median(rf)),
            'coverage_error_within_r_a': float(np.mean(err <= ra)),
            'coverage_error_within_r_f': float(np.mean(err <= rf)),
            'calibration_error_over_r_a_median': float(np.median(err / np.maximum(ra, 1e-6))),
            'calibration_error_over_r_f_median': float(np.median(err / np.maximum(rf, 1e-6))),
            'spearman_r_a_vs_error': spearman(ra, err),
            'spearman_r_f_vs_error': spearman(rf, err),
            'spearman_error_vs_r_a_within_small': spearman(ra[err < err.mean()], err[err < err.mean()]),
            'auroc_r_a_for_above_median_error': auroc(ra, err > np.median(err)),
            'auroc_r_f_for_above_median_error': auroc(rf, err > np.median(err)),
            'auroc_r_a_for_iou_below_0.75': auroc(ra, iou < .75),
            'auroc_r_f_for_iou_below_0.75': auroc(rf, iou < .75),
            'error_within_one_P3_cell_fraction': float(np.mean(err <= 8.)),
        }

    small = [r for r in records]
    hard = [r for r in records if r['iou'] < .75]
    easy = [r for r in records if r['iou'] >= .75]
    report = {
        'status': 'per_edge_informativeness_of_adaptive_range_not_AP',
        'traces': len(hashes), 'grid_note': 'P3 cell = 8 native px of the 640 crop view',
        'groups': {'all_small_live_matched': block(small, 'all'),
                   'matched_iou_below_0.75': block(hard, 'gap population'),
                   'matched_iou_at_least_0.75': block(easy, 'well localised')},
        'per_edge': {EDGES[e]: block([r for r in records if r['edge'] == e], EDGES[e])
                     for e in range(4)},
        'reading': [
            'coverage_error_within_r_* = fraction of real per-edge errors the aperture contains',
            'If coverage is far below ~0.8 the aperture is narrower than the error it must refine',
            'spearman_r_a_vs_error near 0 means the layer-1 FDR span does not track real error',
            'auroc 0.5 = no information; >0.5 = larger range for worse error (the wanted direction)',
        ],
        'limitations': ['Frozen old-54-round EMA traces, not a trained new candidate',
                        'One row per (window, query); the same GT can still appear in several windows',
                        'Layer-1 mean error is the layer-1 prediction error, not the final AP gap'],
    }
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--traces', required=True)
    parser.add_argument('--result', required=True)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
