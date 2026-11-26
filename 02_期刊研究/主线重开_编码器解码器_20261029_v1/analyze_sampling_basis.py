"""A0b CPU interpolation-basis analysis. Basis rank is not feature information rank."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import numpy as np
import torch
import torch.nn.functional as F
from analyze_tinyformer_stages import xyxy, ious

ROOT = Path(__file__).resolve().parent
CHILD = ROOT / 'sampling_observation_v1'


def basis(locations, height, width):
    points = locations.astype(np.float64) * np.array([width, height]) - .5
    origin = np.floor(points).astype(np.int64)
    frac = points - origin
    dx, dy = np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])
    xx = origin[..., 0, None] + dx
    yy = origin[..., 1, None] + dy
    valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
    ww = np.where(dx, frac[..., 0, None], 1 - frac[..., 0, None])
    ww *= np.where(dy, frac[..., 1, None], 1 - frac[..., 1, None])
    ww *= valid
    ids = yy * width + xx
    return ids, ww, points


def rank_metrics(locations, attention, height, width):
    ids, weights, coords = basis(locations, height, width)
    same = ids[..., :, None, :, None] == ids[..., None, :, None, :]
    gram = (same * weights[..., :, None, :, None] * weights[..., None, :, None, :]).sum(axis=(-1, -2))
    trace = np.trace(gram, axis1=-2, axis2=-1)
    rank = trace ** 2 / np.maximum((gram ** 2).sum(axis=(-1, -2)), 1e-30)
    awgram = gram * np.sqrt(attention[..., :, None] * attention[..., None, :])
    atr = np.trace(awgram, axis1=-2, axis2=-1)
    arank = atr ** 2 / np.maximum((awgram ** 2).sum(axis=(-1, -2)), 1e-30)
    diag = np.diagonal(gram, axis1=-2, axis2=-1)
    denom = np.sqrt(diag[..., :, None] * diag[..., None, :])
    cosine = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
    n = locations.shape[-2]
    pairmask = (~np.eye(n, dtype=bool)) & (denom > 0)
    mean_cos = (cosine * pairmask).sum(axis=(-1, -2)) / np.maximum(pairmask.sum(axis=(-1, -2)), 1)
    inside_mass = (attention * weights.sum(axis=-1)).sum(axis=-1)
    return {'normalized_rank': rank / n, 'weighted_normalized_rank': arank / n,
            'basis_pair_cosine': mean_cos, 'level_mass': attention.sum(axis=-1),
            'out_of_bounds_attention_mass': attention.sum(axis=-1) - inside_mass}


def selfcheck():
    same = np.array([[[[.3, .4], [.3, .4], [.3, .4]]]])
    apart = np.array([[[[.1, .1], [.5, .5], [.9, .9]]]])
    weights = np.ones((1, 1, 3)) / 3
    assert np.allclose(rank_metrics(same, weights, 20, 20)['normalized_rank'], 1 / 3)
    assert np.allclose(rank_metrics(apart, weights, 20, 20)['normalized_rank'], 1)
    h, w = 5, 7
    loc = np.array([[.1, .2], [.5, .5], [0, 0], [1, 1], [-.1, .6], [1.2, .5]])
    ids, coeffs, _ = basis(loc, h, w)
    values = np.arange(h * w, dtype=np.float64).reshape(h, w)
    manual = (values.reshape(-1)[np.clip(ids, 0, h * w - 1)] * coeffs).sum(axis=-1)
    grid = torch.tensor(loc * 2 - 1, dtype=torch.float64).reshape(1, 1, -1, 2)
    actual = F.grid_sample(torch.tensor(values).reshape(1, 1, h, w), grid,
                           padding_mode='zeros', mode='bilinear', align_corners=False).flatten().numpy()
    assert np.allclose(manual, actual, atol=1e-12)


def main():
    selfcheck()
    assert json.loads((CHILD / 'completed.json').read_text())['status'] == 'SAMPLING32_OBSERVATION_AND_EQUIVALENCE_PASS'
    records, allqueries = [], []
    image_order = json.loads((ROOT / 'receipts/frozen_input_contract.json').read_text())['file_names_in_existing_order'][:32]
    for filename in image_order:
        with np.load(CHILD / (Path(filename).stem + '.npz'), allow_pickle=False) as z:
            gt, labs, sizes, crowd = z['gt_boxes'], z['gt_labels'], z['gt_sqrt_area'], z['gt_crowd']
            initial_iou = ious(gt, xyxy(z['topk_boxes']))
            query = initial_iou.argmax(axis=1)
            initial_quality = initial_iou[np.arange(len(gt)), query]
            final_iou = ious(gt, xyxy(z['final_boxes']))
            fixed_final = final_iou[np.arange(len(gt)), query]
            any_final = final_iou.max(axis=1)
            for layer in range(4):
                locations, weights = z['locations_' + str(layer)], z['weights_' + str(layer)]
                shapes, pts = z['shapes_' + str(layer)], z['points_' + str(layer)]
                assert locations.shape == (300, 8, 12, 2) and list(pts) == [3, 6, 3]
                assert np.allclose(weights.sum(axis=-1), 1, atol=1e-6)
                combined = defaultdict(lambda: np.zeros(300))
                offset = 0
                for level, ((h, w), count) in enumerate(zip(shapes, pts)):
                    met = rank_metrics(locations[:, :, offset:offset + count], weights[:, :, offset:offset + count], int(h), int(w))
                    offset += count
                    for name in ['normalized_rank', 'weighted_normalized_rank', 'basis_pair_cosine']:
                        combined[name] += (met[name] * met['level_mass']).mean(axis=1)
                    combined['out_of_bounds_attention_mass'] += met['out_of_bounds_attention_mass'].mean(axis=1)
                    combined['level' + str(level) + '_mass'] = met['level_mass'].mean(axis=1)
                refs = z['reference_boxes_' + str(layer)]
                for q in range(300):
                    allqueries.append({'image': filename, 'layer': layer, 'query_slot': q,
                                       'predicted_sqrt_area_640': float(np.sqrt(max(refs[q, 2] * refs[q, 3], 0)) * 640),
                                       **{k: float(v[q]) for k, v in combined.items()}})
                for g in np.flatnonzero(initial_quality >= .25):
                    records.append({'image': filename, 'uid': str(z['gt_uids'][g]), 'layer': layer, 'query_slot': int(query[g]),
                                    'class_id': int(labs[g]), 'sqrt_gt_area_640': float(sizes[g]), 'small_crowded16': int(crowd[g]),
                                    'initial_iou': float(initial_quality[g]), 'fixed_query_final_iou': float(fixed_final[g]),
                                    'any_query_final_iou': float(any_final[g]),
                                    'fixed_query_success50': int(fixed_final[g] >= .5),
                                    'any_query_success50': int(any_final[g] >= .5),
                                    **{k: float(v[query[g]]) for k, v in combined.items()}})
    tables = []
    for layer in range(4):
        for size_label, lo, hi in [('all', 0, float('inf')), ('lt8', 0, 8), ('8_16', 8, 16), ('16_32', 16, 32), ('ge32', 32, float('inf'))]:
            for quality_label, qlo, qhi in [('all_ge25', .25, 1.01), ('25_50', .25, .5), ('50_75', .5, .75), ('ge75', .75, 1.01)]:
                group = [r for r in records if r['layer'] == layer and lo <= r['sqrt_gt_area_640'] < hi and qlo <= r['initial_iou'] < qhi]
                if not group:
                    continue
                row = {'layer': layer, 'size_bin': size_label, 'initial_quality': quality_label,
                       'n': len(group), 'images': len(set(r['image'] for r in group))}
                for status, name in [(0, 'failure'), (1, 'success')]:
                    subset = [r for r in group if r['fixed_query_success50'] == status]
                    row[name + '_n'] = len(subset)
                    for met in ['normalized_rank', 'weighted_normalized_rank', 'basis_pair_cosine', 'level0_mass', 'out_of_bounds_attention_mass']:
                        row[name + '_' + met + '_mean'] = float(np.mean([r[met] for r in subset])) if subset else None
                if row['failure_n'] and row['success_n']:
                    row['failure_minus_success_normalized_rank'] = row['failure_normalized_rank_mean'] - row['success_normalized_rank_mean']
                tables.append(row)
    result = {'status': 'SAMPLING_BASIS_CPU_ANALYSIS_COMPLETE', 'images': 32,
              'gt_with_initial_iou_ge25': sum(r['layer'] == 3 for r in records),
              'all_query_layer_records': len(allqueries), 'GT_query_layer_records': len(records),
              'primary_layer': 3, 'other_layers': 'descriptive only',
              'scope': 'Conditional descriptive observation on fixed first32 train100; selected query may correspond to a neighbour; no causal or feature-information conclusion',
              'metrics': 'Per-level interpolation Gram participation rank normalized by point count, averaged over heads and weighted by native attention level mass',
              'interpretation': 'Failure-minus-success normalized rank below zero is consistent with lower grid-basis diversity, not proof that spreading points or adding P2 improves detection',
              'selfchecks': 'PASS identical/disjoint sample ranks and exact CPU grid_sample bilinear zero-padding reconstruction',
              'tables': tables, 'additional_model_forwards': 0, 'training_updates': 0,
              'analysis_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for filename, rows in [('GT_query_basis.csv', records), ('all_query_basis.csv', allqueries)]:
        with (CHILD / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (CHILD / 'basis_summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'tables'}))
    for r in tables:
        if r['layer'] == 3 and r['initial_quality'] == 'all_ge25':
            print(json.dumps(r))


if __name__ == '__main__':
    main()
