"""Control identifiability of the converged design, on the frozen traces.

The converged design proposes: r = 2*(q90-q10)/2, a box-scale control
(0.25 * updated mean box side), an axis-averaged control (left/right share one
radius, top/bottom another), and the existing fixed control (0.25 * initial
reference side). Before spending GPU rounds it must be shown that these arms
are *distinguishable* at the resolution they are read on.

This script builds the 12 read points of every proposed arm directly from the
frozen per-edge geometry and reports, per pair of arms:
  * mean bilinear-kernel L1 on the P3 grid (how different the reads can be)
  * fraction of queries whose 12 dominant P3 cells are identical
and per arm: radius in P3 cells, per-edge triplet collapse, distinct cells.

Read-only. No training, no CUDA, no frozen artifact modified.
"""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np

from audit_innovation_effect_size import GRID, STRIDE, build_points, kernels, kernel_l1


def arms(mean_edge, q3, initial_side):
    """Every proposed or existing aperture, per edge, in crop px."""
    span = .5 * (q3[:, 2] - q3[:, 0])          # (q90-q10)/2
    width = mean_edge[2] - mean_edge[0]
    height = mean_edge[3] - mean_edge[1]
    lr = .5 * (span[0] + span[2])
    tb = .5 * (span[1] + span[3])
    return {
        'adaptive_k1_current': span,
        'adaptive_k2_proposed': 2 * span,
        'adaptive_k4': 4 * span,
        'scale_0.25_updated_mean_box': .25 * np.array([width, height, width, height]),
        'scale_0.50_updated_mean_box': .50 * np.array([width, height, width, height]),
        'axis_averaged_k2': np.array([lr, tb, lr, tb]) * 2,
        'fixed_0.25_initial_side_current': .25 * initial_side,
    }


def describe(points):
    cells = set((math.floor(px / STRIDE), math.floor(py / STRIDE)) for px, py in points)
    collapsed = sum(len({(math.floor(px / STRIDE), math.floor(py / STRIDE))
                         for px, py in points[e * 3:e * 3 + 3]}) == 1 for e in range(4))
    return cells, collapsed


def main(args):
    root, out = Path(args.traces), Path(args.output)
    if out.exists():
        raise RuntimeError('Preserving previous audit; choose a new output')
    frozen = json.loads(Path(args.result).read_text(encoding='utf8'))['trace_hashes']
    hashes, rows = [], []
    for path in sorted(root.glob('*.npz')):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == frozen[path.name], path.name
        hashes.append(digest)
        d = np.load(path)
        image_id = int(path.stem.split('_')[0])
        mean_edge, q3, side = d['mean_edge_px'], d['q3_edge_px'], d['initial_edge_size_px']
        for i in range(300):
            area = float(d['original_gt_area'][i])
            matched = int(d['match_annotation_id'][i]) >= 0
            small_live = matched and area < 1024 and not bool(d['truncated'][i])
            if not small_live:
                continue
            geom = arms(np.asarray(mean_edge[i], float), np.asarray(q3[i], float),
                        np.asarray(side[i], float))
            entry = {'image': image_id, 'gt': int(d['match_annotation_id'][i]),
                     'iou': float(d['match_iou'][i]), 'points': {}, 'stats': {}}
            for name, radius in geom.items():
                pts = build_points(np.asarray(mean_edge[i], float), radius)
                entry['points'][name] = pts
                cells, collapsed = describe(pts)
                entry['stats'][name] = {
                    'radius_cells_mean': float(radius.mean() / STRIDE),
                    'distinct_cells': len(cells),
                    'collapsed_edges': collapsed,
                }
            rows.append(entry)
    assert len(hashes) == 170

    names = list(rows[0]['points'])
    report = {'status': 'control_identifiability_not_AP', 'traces': len(hashes),
              'query_rows': len(rows), 'grid': f'{GRID}x{GRID} P3, stride {STRIDE} px',
              'per_arm': {}, 'pairwise': {}, 'CUDA_initialized': False}

    def sel(pred):
        return [r for r in rows if pred(r)]

    for tag, pred in (('all_small_live_matched', lambda r: True),
                      ('matched_iou_below_0.75', lambda r: r['iou'] < .75),
                      ('primary_gap_0.50_to_0.75', lambda r: .50 <= r['iou'] < .75)):
        s = sel(pred)
        block, pair = {}, {}
        for name in names:
            block[name] = {
                'radius_cells_mean': float(np.mean([r['stats'][name]['radius_cells_mean'] for r in s])),
                'distinct_cells_mean': float(np.mean([r['stats'][name]['distinct_cells'] for r in s])),
                'collapsed_edges_mean_of_4': float(np.mean([r['stats'][name]['collapsed_edges'] for r in s])),
                'edges_triplet_collapsed_fraction': float(
                    np.mean([r['stats'][name]['collapsed_edges'] / 4 for r in s])),
            }
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                na, nb = names[a], names[b]
                l1 = float(np.mean([np.mean([kernel_l1(kernels(*p1), kernels(*p2))
                                             for p1, p2 in zip(r['points'][na], r['points'][nb])])
                                    for r in s]))
                identical = float(np.mean([
                    describe(r['points'][na])[0] == describe(r['points'][nb])[0] for r in s]))
                pair[f'{na} vs {nb}'] = {'kernel_l1_mean': l1,
                                         'identical_12_cell_set_fraction': identical}
        report['per_arm'][tag], report['pairwise'][tag] = block, pair

    report['reading'] = [
        'kernel_l1_mean near 0 and identical cell sets near 1 means the two arms read '
        'the same content: that contrast cannot attribute anything',
        'collapsed_edges_mean_of_4 = how many of the four edges read a single P3 cell',
    ]
    report['limitations'] = ['Frozen old-54-round EMA layer-1 geometry, not a trained candidate',
                             'Geometry-only: says which contrasts are identifiable, not which works']
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--traces', required=True)
    parser.add_argument('--result', required=True)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
