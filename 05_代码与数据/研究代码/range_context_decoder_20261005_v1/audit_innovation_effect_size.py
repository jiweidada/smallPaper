"""Read-only effect-size audit of the per-edge adaptive-range ingredient.

The candidate's only genuinely new ingredient is swapping the fixed per-edge
search radius r_f = 0.25 * initial_reference_side for the distribution-derived
r_a = (q90 - q10) / 2. Everything else (12 added local P3 points, the 4 global
P4 context points, the readout, the insertion point, the weights) is shared
with the fixed-range control. So this audit asks three falsifiable questions
on the already frozen 32-image / 170-window traces:

  Q1 Magnitude: on the P3 grid actually sampled (stride 8 of the 640 crop),
     how far apart are the adaptive and fixed reads? Do the three per-edge
     points even land in different P3 cells?
  Q2 Per-edge value: inside one query, do the four edges really receive
     different ranges, or nearly the same one?
  Q3 Informativeness: the diagnosed failure is matched small objects at
     0.50 <= IoU < 0.75. If the layer-1 FDR span carries information about
     that failure, r_a must be larger for those than for IoU >= 0.75.

No training, no GT in sampling, no CUDA, no modification of frozen artifacts.
"""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np

GRID = 80          # P3 of the 640 crop view: 640 / 8
STRIDE = 8         # native px per P3 cell
EDGE_NAMES = ('left', 'top', 'right', 'bottom')


def kernels(px, py):
    """Bilinear weights of one crop-px point on the 80x80 P3 grid."""
    x = px / STRIDE - .5
    y = py / STRIDE - .5
    x0, y0 = math.floor(x), math.floor(y)
    fx, fy = x - x0, y - y0
    out = {}
    for cx, cy, w in ((x0, y0, (1 - fx) * (1 - fy)), (x0 + 1, y0, fx * (1 - fy)),
                      (x0, y0 + 1, (1 - fx) * fy), (x0 + 1, y0 + 1, fx * fy)):
        out[(cx, cy)] = out.get((cx, cy), 0.) + w
    return out


def kernel_l1(a, b):
    return float(sum(abs(a.get(k, 0.) - b.get(k, 0.)) for k in set(a) | set(b)))


def build_points(mean_edge, radius):
    """12 reads: r=(-1,0,1) along each edge normal, tangential coord = mean centre."""
    cx = .5 * (mean_edge[0] + mean_edge[2])
    cy = .5 * (mean_edge[1] + mean_edge[3])
    pts = []
    for e in range(4):
        r = radius[e]
        for k in (-1., 0., 1.):
            if e in (0, 2):
                pts.append((mean_edge[e] + r * k, cy))
            else:
                pts.append((cx, mean_edge[e] + r * k))
    return np.array(pts, dtype=np.float64)


def query_metrics(mean_edge, q3, initial_side, gt_edges):
    r_a = .5 * (q3[:, 2] - q3[:, 0])
    r_f = .25 * initial_side
    pa, pf = build_points(mean_edge, r_a), build_points(mean_edge, r_f)
    pm = build_points(mean_edge, np.zeros(4))
    row = {}
    # --- Q1: how different is the sampled content? ---
    l1_af, l1_am, l1_fm, same_af = [], [], [], 0
    for i in range(12):
        ka, kf, km = kernels(*pa[i]), kernels(*pf[i]), kernels(*pm[i])
        if set(ka) == set(kf):
            same_af += 1
        l1_af.append(kernel_l1(ka, kf))
        l1_am.append(kernel_l1(ka, km))
        l1_fm.append(kernel_l1(kf, km))
    row['kernel_l1_adaptive_vs_fixed'] = float(np.mean(l1_af))
    row['kernel_l1_adaptive_vs_meanpoint'] = float(np.mean(l1_am))
    row['kernel_l1_fixed_vs_meanpoint'] = float(np.mean(l1_fm))
    row['same_neighbourhood_fraction'] = same_af / 12
    # --- nominal 12-point budget in distinct P3 cells ---
    for tag, pts in (('adaptive', pa), ('fixed', pf)):
        cells = set((math.floor(px / STRIDE), math.floor(py / STRIDE)) for px, py in pts)
        row[f'cells_{tag}'] = len(cells)
        row[f'triplet_one_cell_edges_{tag}'] = sum(
            len({(math.floor(px / STRIDE), math.floor(py / STRIDE))
                 for px, py in pts[e * 3:e * 3 + 3]}) == 1 for e in range(4))
    # --- Q2: radius magnitude and per-edge heterogeneity ---
    row['radius_cells_adaptive_mean'] = float(r_a.mean() / STRIDE)
    row['radius_cells_fixed_mean'] = float(r_f.mean() / STRIDE)
    row['radius_cells_adaptive_per_edge'] = (r_a / STRIDE).tolist()
    row['radius_ratio_adaptive_over_fixed'] = (r_a / np.where(r_f > 0, r_f, np.nan)).tolist()
    pos = r_a[r_a > 0]
    row['edge_radius_maxmin'] = float(pos.max() / pos.min()) if pos.size >= 2 else None
    row['edge_radius_cv'] = float(r_a.std() / r_a.mean()) if r_a.mean() > 0 else None
    row['edges_radius_below_half_cell'] = int((r_a / STRIDE < .5).sum())
    # --- context for Q3, GT used for grouping only ---
    # same metric as the frozen result: mean over the 12 added points of the
    # distance to the nearest GT edge line (axis-aligned box edges).
    def nearest_edge_distance(pts):
        dx = np.minimum(np.abs(pts[:, 0] - gt_edges[0]), np.abs(pts[:, 0] - gt_edges[2]))
        dy = np.minimum(np.abs(pts[:, 1] - gt_edges[1]), np.abs(pts[:, 1] - gt_edges[3]))
        return float(np.minimum(dx, dy).mean())

    row['gt_edge_distance_adaptive_px'] = nearest_edge_distance(pa)
    row['gt_edge_distance_fixed_px'] = nearest_edge_distance(pf)
    return row


def auroc(scores, labels):
    """Rank AUROC of scores separating label True from False."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    if labels.all() or not labels.any():
        return None
    order = scores.argsort()
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    uniq, inv, count = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, ranks)
    ranks = (sums / count)[inv]
    pos, neg = labels.sum(), (~labels).sum()
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def main(args):
    root, out = Path(args.traces), Path(args.output)
    if out.exists():
        raise RuntimeError('Preserving previous audit; choose a new output')
    result = json.loads(Path(args.result).read_text(encoding='utf8'))
    frozen_hashes = result['trace_hashes']
    rows, hashes = [], []
    for path in sorted(root.glob('*.npz')):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == frozen_hashes[path.name], path.name
        d = np.load(path)
        image_id = int(path.stem.split('_')[0])
        mean_edge, q3, side = d['mean_edge_px'], d['q3_edge_px'], d['initial_edge_size_px']
        gt_edges = d['gt_edges_crop_px']
        for i in range(300):
            area = float(d['original_gt_area'][i])
            matched = int(d['match_annotation_id'][i]) >= 0
            iou = float(d['match_iou'][i])
            small_live = matched and area < 1024 and not bool(d['truncated'][i])
            row = {'image': image_id, 'gt': int(d['match_annotation_id'][i]),
                   'area': area, 'iou': iou, 'matched': matched, 'small_live': small_live}
            row.update(query_metrics(np.asarray(mean_edge[i], dtype=np.float64),
                                     np.asarray(q3[i], dtype=np.float64),
                                     np.asarray(side[i], dtype=np.float64),
                                     np.asarray(gt_edges[i], dtype=np.float64)))
            rows.append(row)
        hashes.append({'file': path.name, 'sha256': digest})
    assert len(hashes) == 170 and len({r['image'] for r in rows}) == 32

    groups = {
        'all_matched': lambda r: r['matched'],
        'all_queries_small_live': lambda r: r['small_live'],
        'primary_iou_ge_0.10': lambda r: r['small_live'] and r['iou'] >= .10,
        'strict_iou_ge_0.50': lambda r: r['small_live'] and r['iou'] >= .50,
        'gap_0.50_to_0.75': lambda r: r['small_live'] and .50 <= r['iou'] < .75,
        'good_iou_ge_0.75': lambda r: r['small_live'] and r['iou'] >= .75,
    }

    def balance(selected, key):
        """within_GT_then_equal_GT_then_equal_image mean, as in the frozen verification."""
        gt = defaultdict(list)
        for r in selected:
            if r[key] is not None and np.isfinite(r[key]):
                gt[(r['image'], r['gt'])].append(r[key])
        im = defaultdict(list)
        for (image, _), v in gt.items():
            im[image].append(float(np.mean(v)))
        return float(np.mean([np.mean(v) for v in im.values()]))

    def pooled(selected, key):
        v = [r[key] for r in selected if r[key] is not None and np.isfinite(r[key])]
        return {'mean': float(np.mean(v)), 'median': float(np.median(v)),
                'q10': float(np.percentile(v, 10)), 'q90': float(np.percentile(v, 90)),
                'max': float(np.max(v))} if v else None

    report = {'status': 'effect_size_of_per_edge_range_ingredient_not_AP',
              'traces': len(hashes), 'images': 32, 'query_window_rows': len(rows),
              'grid': f'{GRID}x{GRID} P3 of the 640 crop view, stride {STRIDE} native px',
              'trace_hashes_match_frozen_result': True, 'groups': {}, 'CUDA_initialized': False}
    for name, pred in groups.items():
        sel = [r for r in rows if pred(r)]
        if not sel:
            continue
        entries = {'query_window_rows': len(sel),
                   'unique_GT': len({(r['image'], r['gt']) for r in sel}),
                   'iou_mean': balance(sel, 'iou')}
        for key in ('kernel_l1_adaptive_vs_fixed', 'kernel_l1_adaptive_vs_meanpoint',
                    'kernel_l1_fixed_vs_meanpoint', 'same_neighbourhood_fraction',
                    'cells_adaptive', 'cells_fixed', 'triplet_one_cell_edges_adaptive',
                    'triplet_one_cell_edges_fixed', 'radius_cells_adaptive_mean',
                    'radius_cells_fixed_mean', 'edge_radius_maxmin', 'edge_radius_cv',
                    'edges_radius_below_half_cell', 'gt_edge_distance_adaptive_px',
                    'gt_edge_distance_fixed_px'):
            entries[key] = balance(sel, key)
        flat_a = np.array([r['radius_cells_adaptive_mean'] for r in sel])
        ratio = np.array([v for r in sel for v in r['radius_ratio_adaptive_over_fixed']])
        ratio = ratio[np.isfinite(ratio)]
        entries['radius_cells_adaptive_pooled'] = pooled(sel, 'radius_cells_adaptive_mean')
        entries['radius_ratio_adaptive_over_fixed_median'] = float(np.median(ratio))
        entries['radius_ratio_adaptive_over_fixed_q10'] = float(np.percentile(ratio, 10))
        entries['radius_ratio_adaptive_over_fixed_q90'] = float(np.percentile(ratio, 90))
        entries['radius_ratio_within_20pct_fraction'] = float(np.mean(np.abs(ratio - 1) <= .2))
        entries['queries_all_12_same_neighbourhood_fraction'] = float(
            np.mean([r['same_neighbourhood_fraction'] == 1. for r in sel]))
        entries['queries_kernel_l1_adaptive_fixed_below_0.1_fraction'] = float(
            np.mean([r['kernel_l1_adaptive_vs_fixed'] < .1 for r in sel]))
        entries['edges_triplet_collapsed_fraction_adaptive'] = float(
            np.mean([r['triplet_one_cell_edges_adaptive'] / 4 for r in sel]))
        entries['edges_triplet_collapsed_fraction_fixed'] = float(
            np.mean([r['triplet_one_cell_edges_fixed'] / 4 for r in sel]))
        entries['radius_cells_adaptive_mean_of_means'] = float(flat_a.mean())
        report['groups'][name] = entries

    # --- Q3: does the adaptive span separate the diagnosed gap from good boxes? ---
    gap = [r for r in rows if groups['gap_0.50_to_0.75'](r)]
    good = [r for r in rows if groups['good_iou_ge_0.75'](r)]
    both = gap + good
    labels = [False] * len(gap) + [True] * len(good)
    report['gap_vs_good_discrimination'] = {
        'gap_queries': len(gap), 'good_queries': len(good),
        'radius_cells_adaptive_gap_mean': balance(gap, 'radius_cells_adaptive_mean'),
        'radius_cells_adaptive_good_mean': balance(good, 'radius_cells_adaptive_mean'),
        'radius_cells_fixed_gap_mean': balance(gap, 'radius_cells_fixed_mean'),
        'radius_cells_fixed_good_mean': balance(good, 'radius_cells_fixed_mean'),
        'auroc_adaptive_radius_for_gap_vs_good': auroc(
            [r['radius_cells_adaptive_mean'] for r in both], labels),
        'auroc_fixed_radius_for_gap_vs_good': auroc(
            [r['radius_cells_fixed_mean'] for r in both], labels),
        'auroc_kernel_l1_adaptive_fixed_for_gap_vs_good': auroc(
            [r['kernel_l1_adaptive_vs_fixed'] for r in both], labels),
        'reading': 'AUROC is the probability that a query drawn from gap scores higher '
                   'than one drawn from good; 0.5 means the quantity carries no signal.',
    }
    report['limitations'] = [
        'Frozen old-54-round EMA traces, not a trained new candidate',
        'Kernel L1 bounds how different the read CAN be, not how useful it is',
        'IoU is the frozen match IoU of the old model, not the candidate result',
        'GT enters grouping only, never sampling',
    ]
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--traces', required=True)
    parser.add_argument('--result', required=True)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
