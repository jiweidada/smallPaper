"""Compare implemented geometry with all frozen 32-image/170-window traces."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import bootstrap
from range_context import edge_geometry
from engine.deim.dfine_utils import weighting_function


def run(args):
    torch.set_num_threads(4)
    root, out = Path(args.traces), Path(args.output)
    if out.exists():
        raise RuntimeError('Preserving previous trace verification')
    support = weighting_function(32, torch.tensor([.5]), torch.tensor([4.]))
    rows, hashes, max_box, max_radius = [], [], 0., 0.
    for path in sorted(root.glob('*.npz')):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        d = np.load(path)
        p = torch.from_numpy(d['p'])[None]
        logits = p.clamp_min(torch.finfo(torch.float32).tiny).log().flatten(-2)
        ref = torch.from_numpy(d['initial_boxes'])[None]
        points, box, stat = edge_geometry(logits, ref, support, 4.)
        fixed, _, _ = edge_geometry(logits, ref, support, 4., 'fixed')
        max_box = max(max_box, float((box[0] - torch.from_numpy(d['first_boxes'])).abs().max() * 640))
        recorded_radius = torch.from_numpy(d['q3_span_px']) * .5
        max_radius = max(max_radius, float((stat['radius'][0] * 640 - recorded_radius).abs().max()))
        distance = (points - fixed).norm(dim=-1)[0] * 640
        collapsed = (stat['radius'][0] == 0).float().mean(-1)
        image_id = int(d['image_id'])
        for i in range(300):
            area = float(d['original_gt_area'][i])
            matched = int(d['match_annotation_id'][i]) >= 0
            rows.append({'image': image_id, 'gt': int(d['match_annotation_id'][i]),
                         'small': matched and area < 1024, 'matched': matched,
                         'primary': matched and area < 1024 and not bool(d['truncated'][i]) and float(d['match_iou'][i]) >= .1,
                         'strict': matched and area < 1024 and not bool(d['truncated'][i]) and float(d['match_iou'][i]) >= .5,
                         'distance': distance[i].numpy(), 'collapsed': float(collapsed[i])})
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        hashes.append({'file': path.name, 'sha256': digest})
    def summarize(selected, gt_balance):
        pool = np.concatenate([r['distance'] for r in selected])
        if gt_balance:
            gt = defaultdict(list)
            for r in selected:
                gt[(r['image'], r['gt'])].append(r)
            im = defaultdict(list)
            for (image, _), values in gt.items():
                im[image].append((np.mean([v['distance'].mean() for v in values]),
                                  np.mean([v['collapsed'] for v in values])))
            distance_mean = np.mean([np.mean([v[0] for v in values]) for values in im.values()])
            collapsed_mean = np.mean([np.mean([v[1] for v in values]) for values in im.values()])
        else:
            gt, im = {}, {r['image']: None for r in selected}
            distance_mean, collapsed_mean = pool.mean(), np.mean([r['collapsed'] for r in selected])
        return {'query_window_rows': len(selected), 'unique_GT': len(gt) if gt_balance else None,
                'images': len(im), 'distance_mean_px': float(distance_mean),
                'distance_mean_P3_cells': float(distance_mean / 8),
                'distance_median_pooled_points_px': float(np.median(pool)),
                'collapsed_edge_fraction': float(collapsed_mean),
                'aggregation_mean': 'within_GT_then_equal_GT_then_equal_image' if gt_balance else 'pooled_all_query_windows',
                'median_warning': 'pooled point median includes repeated windows and the shared central points'}
    assert len(hashes) == 170 and len({r['image'] for r in rows}) == 32
    # Tolerance covers saved float32 probability reconstruction and the native
    # operation-order correction, in crop-native pixel units; not learned gain.
    assert max_box < .003 and max_radius < .003, (max_box, max_radius)
    report = {'status': 'implementation_agrees_with_frozen_geometry_not_AP',
              'windows': len(hashes), 'images': 32, 'all_query_window_rows': len(rows),
              'max_mean_box_difference_native_px': max_box,
              'max_range_radius_difference_native_px': max_radius,
              'source_probability': 'saved probabilities -> log -> softmax; no GT enters sampling',
              'groups': {'all_queries': summarize(rows, False),
                         'small_nontruncated_firstIoU_ge_0.1': summarize([r for r in rows if r['primary']], True),
                         'small_nontruncated_firstIoU_ge_0.5': summarize([r for r in rows if r['strict']], True)},
              'unchanged_trace_hashes': hashes, 'CUDA_initialized': torch.cuda.is_initialized(),
              'limitations': ['Frozen old54-round EMA, not a trained new candidate',
                              'Point distance does not measure feature usefulness or AP']}
    assert not report['CUDA_initialized']
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps({k: v for k, v in report.items() if k != 'unchanged_trace_hashes'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--traces', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
