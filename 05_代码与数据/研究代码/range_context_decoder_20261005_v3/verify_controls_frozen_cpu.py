"""Check new implemented controls against all pinned frozen traces."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
import bootstrap
from bootstrap import ROOT
from range_context import edge_geometry
from engine.deim.dfine_utils import weighting_function
from verify_stage0_cpu import PROJECT

AUDIT = PROJECT / '英文小论文/data/00_实验计划与证据索引/逐边范围实验_20261005_v1/offline_recheck_v1'
sys.path.insert(0, str(AUDIT))
from recheck_geometry import FROZEN, kernel, l1, balanced, avg


def main():
    dest = ROOT / 'implemented_controls_frozen_cpu.json'
    if dest.exists():
        raise RuntimeError('Preserve previous receipt')
    torch.set_num_threads(4)
    support = weighting_function(32, torch.tensor([.5]), torch.tensor([4.]))
    hashes = json.loads((FROZEN / 'completion.json').read_text(encoding='utf8'))['trace_hashes']
    arms = ('adaptive', 'updated_size', 'axis_mean', 'opposite_swap', 'fixed')
    pairs = [('adaptive', tag) for tag in ('updated_size', 'axis_mean', 'opposite_swap')] + [('fixed', 'updated_size')]
    rows, max_mean, max_radius, checked = [], 0., 0., []
    with torch.inference_mode():
        for path in sorted((FROZEN / 'traces').glob('*.npz')):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[path.name]
            d = np.load(path)
            probability = torch.from_numpy(d['p'])[None]
            logits = probability.clamp_min(torch.finfo(torch.float32).tiny).log().flatten(-2)
            ref = torch.from_numpy(d['initial_boxes'])[None]
            geo = {name: edge_geometry(logits, ref, support, 4., name) for name in arms}
            mean_error = float((geo['adaptive'][1][0] - torch.from_numpy(d['first_boxes'])).abs().max() * 640)
            expected_radius = torch.from_numpy(d['q3_edge_px'][:, :, 2] - d['q3_edge_px'][:, :, 0])
            radius_error = float((geo['adaptive'][2]['radius'][0] * 640 - expected_radius).abs().max())
            max_mean, max_radius = max(max_mean, mean_error), max(max_radius, radius_error)
            assert mean_error < .003 and radius_error < .006
            win, wh = d['window'].tolist(), d['image_wh'].tolist()
            for qi in range(300):
                gt, iou = int(d['match_annotation_id'][qi]), float(d['match_iou'][qi])
                if gt < 0 or float(d['original_gt_area'][qi]) >= 1024 or bool(d['truncated'][qi]) or iou < .1:
                    continue
                row = {'image': int(d['image_id']), 'window': path.name, 'query': qi, 'gt': gt, 'iou': iou}
                kernels = {}
                for tag in arms:
                    kernels[tag] = [kernel(p, 8, win, wh) for p in (geo[tag][0][0, qi] * 640).tolist()]
                    ks = kernels[tag]
                    row[tag + '/radius_P3_cells'] = float(geo[tag][2]['radius'][0, qi].mean() * 80)
                    valid_edges = [e for e in range(4) if all(x[1] for x in ks[3 * e:3 * e + 3])]
                    row[tag + '/valid_triplet_same_base_cell_fraction'] = avg([len({x[2] for x in ks[3 * e:3 * e + 3]}) == 1 for e in valid_edges])
                    row[tag + '/distinct_valid_base_cells'] = len({x[2] for x in ks if x[1]})
                for a, b in pairs:
                    ka, kb = kernels[a], kernels[b]
                    distances = [l1(x[0], y[0]) for x, y in zip(ka, kb)]
                    prefix = a + '_vs_' + b + '/'
                    row[prefix + 'kernel_L1'] = avg(distances)
                    row[prefix + 'unordered_valid_base_cell_set_identical'] = float({x[2] for x in ka if x[1]} == {x[2] for x in kb if x[1]})
                    row[prefix + 'all_corresponding_kernels_identical_tol_1e_minus_6'] = float(max(distances) < 1e-6)
                rows.append(row)
            checked.append({'file': path.name, 'sha256': hashes[path.name]})
    groups = {}
    for name, pred in [('small_firstIoU_ge_0.1', lambda r: True), ('small_firstIoU_0.5_to_0.75', lambda r: .5 <= r['iou'] < .75)]:
        chosen = [r for r in rows if pred(r)]
        metrics = {k: balanced(chosen, k) for k in rows[0] if '/' in k}
        groups[name] = {'query_window_rows': len(chosen), 'unique_GT': len({(r['image'], r['gt']) for r in chosen}),
                        'within_GT_then_equal_GT_then_equal_image': metrics}
    assert len(checked) == 170 and not torch.cuda.is_initialized()
    result = {'status': 'implemented_control_geometry_verified_CPU_not_AP', 'trace_hashes_verified': checked,
              'max_mean_box_difference_px': max_mean, 'max_k2_radius_difference_px': max_radius,
              'groups': groups, 'CUDA_initialized': False,
              'sampling': 'align_corners=False, px/8-0.5, crop and original masks, finite grid, zero-padding coefficients',
              'limitations': ['Unordered interpolation base-cell sets are descriptive; equal sets do not imply equal bilinear coefficients, positional keys or attention weights.',
                              'No hard L1>=0.30/identical<=30% gate is frozen; those cutoffs have no calibrated connection to AP sensitivity.',
                              'Kernel L1 is neither AP effect size nor statistical power.',
                              'Averaged radius alignment does not remove conditional-width differences; within-axis controls are still needed.',
                              'Frozen old independent EMA; all examples are previously inspected development validation data.']}
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps({'groups': groups, 'mean_error': max_mean, 'radius_error': max_radius}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
