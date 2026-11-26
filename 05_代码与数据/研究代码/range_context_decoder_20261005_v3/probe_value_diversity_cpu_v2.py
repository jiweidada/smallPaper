"""Real frozen P3 diversity and headwise hull bounds; CPU, not trained AP."""
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import bootstrap
from bootstrap import ROOT
from factory import build_model, load_independent_ema_for_cpu_probe, verify_snapshot
from dataset import OriginalCocoImages
from geometry import make_views
from engine.deim.dfine_utils import weighting_function
from range_context import edge_geometry, valid_points, original_coordinates
from verify_stage0_cpu import PROJECT, CHECKPOINT, ANNOTATIONS, IMAGES

FROZEN = PROJECT / '英文小论文/data/00_实验计划与证据索引/分布反馈深入自审_20261004/p0_real_cpu_v3_level_control'
AUDIT = PROJECT / '英文小论文/data/00_实验计划与证据索引/逐边范围实验_20261005_v1/offline_recheck_v1'
sys.path.insert(0, str(AUDIT))
from recheck_geometry import balanced


def sample(feature, points, valid):
    b, q, n, _ = points.shape
    safe = torch.where(valid[..., None], points, torch.zeros_like(points))
    out = F.grid_sample(feature.float(), safe.reshape(b, q * n, 1, 2) * 2 - 1,
                        align_corners=False, padding_mode='zeros')
    out = out.squeeze(-1).transpose(1, 2).reshape(b, q, n, feature.shape[1])
    return torch.where(valid[..., None], out, torch.zeros_like(out))


def diversity(v, valid, prefix, heads=1):
    # [Q,N,C], equal valid-token mean. Projection is untrained if C=128.
    count = valid.sum(1).clamp_min(1)
    pair = valid[:, :, None] & valid[:, None, :]
    pair &= torch.ones(v.shape[1], v.shape[1], dtype=torch.bool).triu(1)
    a, b = v[:, :, None], v[:, None, :]
    l2 = (a - b).norm(dim=-1)
    norms = v.norm(dim=-1)
    scale = ((norms[:, :, None].square() + norms[:, None, :].square()) * .5).sqrt().clamp_min(1e-12)
    cos = F.cosine_similarity(a, b, dim=-1, eps=1e-12)
    pairs = pair.sum((1, 2)).clamp_min(1)
    vh = v.reshape(v.shape[0], v.shape[1], heads, -1)
    mean = (vh * valid[..., None, None]).sum(1) / count[:, None, None]
    distance = (vh - mean[:, None]).norm(dim=-1).masked_fill(~valid[..., None], 0)
    bound = distance.amax(1).square().sum(-1).sqrt()
    rms_norm = ((norms.square() * valid).sum(1) / count).sqrt().clamp_min(1e-12)
    diam = (vh[:, :, None] - vh[:, None, :]).norm(dim=-1)
    diam = diam.masked_fill(~(valid[:, :, None, None] & valid[:, None, :, None]), 0)
    diameter = diam.amax((1, 2)).square().sum(-1).sqrt()
    return {prefix + 'pairwise_cosine_mean': (cos * pair).sum((1, 2)) / pairs,
            prefix + 'pairwise_L2_mean': (l2 * pair).sum((1, 2)) / pairs,
            prefix + 'pairwise_relative_L2_mean': ((l2 / scale) * pair).sum((1, 2)) / pairs,
            prefix + 'uniform_to_headwise_hull_bound_L2': bound,
            prefix + 'uniform_to_headwise_hull_bound_relative_RMSnorm': bound / rms_norm,
            prefix + 'headwise_diameter_bound_relative_RMSnorm': diameter / rms_norm,
            prefix + 'value_RMSnorm': rms_norm,
            prefix + 'near_constant_1e_minus_3_fraction': (bound / rms_norm < 1e-3).float(),
            prefix + 'valid_count_le_one_fraction': (valid.sum(1) <= 1).float()}


def main():
    dest = ROOT / 'real_value_diversity_cpu_v2.json'
    rawdest = ROOT / 'real_value_diversity_query_rows_v2.json'
    if dest.exists() or rawdest.exists():
        raise RuntimeError('Preserve existing probe')
    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(1)
    verify_snapshot()
    model = build_model().eval()
    checkpoint = load_independent_ema_for_cpu_probe(model, CHECKPOINT)
    assert checkpoint['sha256'] == 'cf8bcea8d38e9ee826929f1b6d161a9ceff0ac4851e3a4318e1f197b2312aa97'
    assert hashlib.sha256(ANNOTATIONS.read_bytes()).hexdigest() == '0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd'
    dataset = OriginalCocoImages(IMAGES, ANNOTATIONS)
    manifest = json.loads((FROZEN / 'manifest.json').read_text(encoding='utf8'))
    selected = [manifest['images'][i] for i in (0, 10, 21, 31)]
    hashes = json.loads((FROZEN / 'completion.json').read_text(encoding='utf8'))['trace_hashes']
    reader = model.range_interaction
    support = weighting_function(32, torch.tensor([.5]), torch.tensor([4.]))
    rows, windows, discrepancies = [], [], []
    with torch.inference_mode():
        for im in selected:
            index = next(i for i, entry in enumerate(dataset.images) if entry['id'] == im['id'])
            image_path = dataset.image_root / im['file_name']
            assert hashlib.sha256(image_path.read_bytes()).hexdigest() == im['sha256']
            image, _ = dataset[index]
            full = make_views(image, im['windows'][0])[0]
            global_encoded = model.encoder(model.backbone(full[None]))
            global_context = reader.prepare_context(global_encoded[1])
            for wi, window in enumerate(im['windows']):
                trace_path = FROZEN / 'traces' / f'{im["id"]:04}_{wi:02}.npz'
                assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == hashes[trace_path.name]
                d = np.load(trace_path)
                assert d['window'].tolist() == window
                _, local, crop = make_views(image, window)
                local_encoded = model.encoder(model.backbone(local[None]))
                # Freeze the sampling coordinates from the pinned trajectories.
                # Re-forwarded top-k query indices are not a guaranteed join key:
                # the attempted join failed on image 544 and is retained below.
                probability = torch.from_numpy(d['p'])[None]
                corners = probability.clamp_min(torch.finfo(torch.float32).tiny).log().flatten(-2)
                initial = torch.from_numpy(d['initial_boxes'])[None]
                scale = 4.
                # Bounds/diversity need no query. Routing is only a zero-query,
                # untrained projection probe; not recovered native query states.
                q = torch.zeros(1, 300, reader.width)
                memory = reader.prepare_local(local_encoded[0])
                outputs, reads, masks, metrics = {}, {}, {}, {}
                adaptive_geom = None
                for variant in ('adaptive_range_global_context', 'mean_size_global_context',
                                'axis_mean_global_context', 'opposite_swap_global_context'):
                    reader.variant = variant
                    geo = reader.sampling_geometry(corners, initial, support, scale, crop[None], global_encoded[1].shape[-2:])
                    context = reader.read(global_context, geo['context_points'], geo['context_valid'], q, geo['context_relative'])
                    detail, weights, values = reader.read(memory, geo['local_points'], geo['local_valid'],
                                                         q + reader.context_to_query(context), geo['local_relative'], True)
                    raw = sample(local_encoded[0], geo['local_points'], geo['local_valid'])[0]
                    values = values[0].flatten(-2)
                    reads[variant] = raw
                    masks[variant] = geo['local_valid'][0]
                    outputs[variant] = detail[0]
                    metrics.update(diversity(raw, masks[variant], variant + '/raw_P3/'))
                    metrics.update(diversity(values, masks[variant], variant + '/untrained_V/', heads=reader.heads))
                    if variant == 'adaptive_range_global_context':
                        adaptive_geom = geo
                        diff = (geo['box'][0] - torch.from_numpy(d['first_boxes'])).abs().max() * 640
                        discrepancies.append(float(diff))
                        assert float(diff) < .003, ('frozen_box_disagreement', float(diff))
                        no_context = reader.read(memory, geo['local_points'], geo['local_valid'], q, geo['local_relative'])[0]
                        base_norm = metrics[variant + '/untrained_V/value_RMSnorm']
                        metrics['routing_probe/global_vs_no_context_relative_RMSnorm'] = (detail[0] - no_context).norm(dim=-1) / base_norm
                        metrics['routing_probe/valid_local_fraction'] = masks[variant].float().mean(1)
                        expected_mass = geo['local_valid'].any(-1)[..., None].to(weights.dtype)
                        metrics['routing_probe/weight_sum_error'] = (weights.sum(2) - expected_mass).abs().amax(-1)[0]
                reader.variant = 'adaptive_range_local_context'
                local_geo = reader.sampling_geometry(corners, initial, support, scale, crop[None], global_encoded[1].shape[-2:])
                assert torch.equal(local_geo['local_points'], adaptive_geom['local_points'])
                local_context = reader.read(reader.prepare_context(local_encoded[1]), local_geo['context_points'], local_geo['context_valid'], q, local_geo['context_relative'])
                local_detail = reader.read(memory, local_geo['local_points'], local_geo['local_valid'],
                                           q + reader.context_to_query(local_context), local_geo['local_relative'])[0]
                base_norm = metrics['adaptive_range_global_context/untrained_V/value_RMSnorm']
                metrics['routing_probe/global_vs_local_context_relative_RMSnorm'] = (outputs['adaptive_range_global_context'] - local_detail).norm(dim=-1) / base_norm
                k1_points, _, _ = edge_geometry(corners, initial, support, scale, span_multiplier=1.)
                k1_valid = valid_points(k1_points) & valid_points(original_coordinates(k1_points, crop[None]))
                k1_raw = sample(local_encoded[0], k1_points, k1_valid)[0]
                k1_v = sample(memory, k1_points, k1_valid)[0, :, :, reader.width:]
                metrics.update(diversity(k1_raw, k1_valid[0], 'offline_adaptive_k1/raw_P3/'))
                metrics.update(diversity(k1_v, k1_valid[0], 'offline_adaptive_k1/untrained_V/', heads=reader.heads))
                for variant in ('mean_size_global_context', 'axis_mean_global_context', 'opposite_swap_global_context'):
                    common = masks[variant] & masks['adaptive_range_global_context']
                    delta = (reads['adaptive_range_global_context'] - reads[variant]).square().sum(-1)
                    denominator = reads[variant].square().sum(-1)
                    metrics['contrast/' + variant + '/point_relative_RMS'] = ((delta * common).sum(-1) / (denominator * common).sum(-1).clamp_min(1e-12)).sqrt()
                    metrics['contrast/' + variant + '/untrained_read_relative_RMSnorm'] = (outputs['adaptive_range_global_context'] - outputs[variant]).norm(dim=-1) / base_norm
                for qi in range(300):
                    rows.append({'image': im['id'], 'window_index': wi, 'query': qi,
                                 'gt': int(d['match_annotation_id'][qi]), 'iou': float(d['match_iou'][qi]),
                                 'small_live': bool(int(d['match_annotation_id'][qi]) >= 0 and float(d['original_gt_area'][qi]) < 1024 and not bool(d['truncated'][qi])),
                                 **{key: float(value[qi]) for key, value in metrics.items()}})
                windows.append({'name': trace_path.name, 'sha256': hashes[trace_path.name]})
            print('completed_image', im['id'], 'windows', len(im['windows']), flush=True)
    groups = {}
    keys = [k for k in rows[0] if '/' in k]
    for name, pred in [('all_queries', lambda r: True), ('small_firstIoU_ge_0.1', lambda r: r['small_live'] and r['iou'] >= .1),
                       ('small_firstIoU_0.5_to_0.75', lambda r: r['small_live'] and .5 <= r['iou'] < .75)]:
        chosen = [r for r in rows if pred(r)]
        if name == 'all_queries':
            # All model queries: no matching/GT selection. Equal windows within image.
            byim = defaultdict(list)
            for row in chosen:
                byim[row['image']].append(row)
            aggregate = {k: sum(sum(r[k] for r in values) / len(values) for values in byim.values()) / len(byim) for k in keys}
        else:
            aggregate = {k: balanced(chosen, k) for k in keys}
        groups[name] = {'query_window_rows': len(chosen), 'unique_GT': len({(r['image'], r['gt']) for r in chosen}) if name != 'all_queries' else None,
                        'aggregation': 'within_GT_then_equal_GT_then_equal_image' if name != 'all_queries' else 'equal_windows_within_equal_images',
                        'metrics': aggregate}
    assert not torch.cuda.is_initialized()
    rawdest.write_text(json.dumps(rows, ensure_ascii=False, allow_nan=False), encoding='utf8')
    report = {'status': 'frozen_real_features_CPU_stage0_not_AP', 'selected_images': [im['id'] for im in selected],
              'selection': 'positions 0,10,21,31 of the existing 32-image frozen manifest; same as previous feature probe',
              'checkpoint': checkpoint, 'reader_seed': 1, 'reader_is_untrained': True, 'windows': windows,
              'query_probe': 'zero projected query for routing probes, identical across controls; not native decoder query',
              'geometry': 'pinned saved first-layer probabilities and initial boxes; no current query-index join',
              'max_frozen_mean_box_difference_px': max(discrepancies), 'groups': groups,
              'bound_definition': 'For each head h, B_h=max_i ||v_ih-u_h||_2 over valid tokens, u_h=uniform valid mean. ||concat_h(weighted_h-u_h)|| <= sqrt(sum_h B_h^2). This relaxes attainable query/key logits; it is not task-optimal evidence.',
              'normalization': 'relative RMSnorm divides by sqrt(mean_valid ||concatenated value_i||_2^2). Raw P3 is single-head diagnostic; actual projected V is 4-head.',
              'seconds': time.time() - started, 'CUDA_initialized': False,
              'limitations': ['4 development images / 20 windows; no independent generalization evidence.',
                              'The reader K/V and query projections are randomly initialized. Their geometry bound and routing effects are conditional probes, not trained predictions.',
                              'An attempted native query join failed on image 544 (234.2px maximum box discrepancy); original script/log retained. Cause not established; no tolerance relaxation or image exclusion. Bounds use saved geometry and real fresh features instead.',
                              'A nonzero hull only establishes representational room, not useful global semantic selection.',
                              'Zero-initialized restore is exactly zero. No ratio to zero restore or prediction/AP bound is claimed.',
                              'Near-constant 1e-3 is descriptive, not a validated AP-identifiability threshold.',
                              'Matching is offline first-layer class-agnostic IoU; crop-truncated means window cut, not natural occlusion.',
                              'Bootstrap across these four repeatedly inspected images would not estimate training-seed noise.']}
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    for name, group in groups.items():
        v = group['metrics']
        print(name, group['query_window_rows'], {k: v[k] for k in v if k.startswith('adaptive_range_global_context/') or k.startswith('routing_probe/')}, flush=True)


if __name__ == '__main__':
    main()
