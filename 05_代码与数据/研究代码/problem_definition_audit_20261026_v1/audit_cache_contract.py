"""Audit existing cache observability; no model construction, forward or protocol change."""
import io
import json
import hashlib
import time
from pathlib import Path
import torch

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parent
EN = ROOT.parents[1]
MAIN = EN/'持续寻找研究主线_20261020_v2'
CACHE = EN/'code/unit_instance_completion_20261021_v1/342_codex_stage3_20261022_v1'
START = time.monotonic()
NAMES = ['no_raw_geometry_IoU_ge_0_5',
         'raw_geometry_exists_but_no_GT_class_score_above_0_001',
         'eligible_GT_class_geometry_exists_but_not_in_recorded_top300',
         'recorded_top300_has_GT_class_geometry']


def sha(b):
    return hashlib.sha256(b).hexdigest()


def edges(pb):
    b = pb.float()*640.
    return torch.cat((b[:, :2]-b[:, 2:]/2, b[:, :2]+b[:, 2:]/2), 1)


def iou(g, p):
    wh = (torch.minimum(g[:, None, 2:], p[None, :, 2:])
          -torch.maximum(g[:, None, :2], p[None, :, :2])).clamp(min=0)
    intersection = wh[..., 0]*wh[..., 1]
    ag = torch.prod(g[:, 2:]-g[:, :2], dim=1)
    ap = torch.prod(p[:, 2:]-p[:, :2], dim=1)
    return intersection/(ag[:, None]+ap[None, :]-intersection).clamp(min=1e-12)


def main():
    inputs = []
    results = {}
    for seed in (1, 2):
        files = sorted((CACHE/f'342_cache_val_v2_seed{seed}').glob('342_*.pt'))
        assert len(files) == 548
        unique_counts = []
        row_order_difference_max = 0.
        raw_max_score_min = fullp_score_min = 1.
        min_GT_class_score_any_query = 1.
        groups = {key: [0]*4 for key in ('all_GT', 'tiny_cached_shortside_lt4', 'non_tiny')}
        first = None
        for path in files:
            b = path.read_bytes()
            inputs.append({'path': str(path), 'sha256': sha(b)})
            d = torch.load(io.BytesIO(b), map_location='cpu', weights_only=True)
            raw, logits = d['final300_boxes'], d['final300_logits']
            assert raw.shape == (300, 4) and logits.shape == (300, 10)
            prob = logits.sigmoid()
            p = edges(raw)
            q = d['fullp_query_index'].long()
            labels = d['fullp_labels'].long()
            flat = d['fullp_flatten_index'].long()
            scores = d['fullp_scores']
            assert torch.equal(q*10+labels, flat)
            assert len(flat.unique()) == len(flat)
            assert torch.equal(p[q], d['fullp_boxes'])
            assert torch.equal(logits[q], d['fullp_logits_row'])
            assert torch.equal(prob.flatten()[flat], scores)
            assert bool((scores > .001).all())
            mask = torch.ones(prob.numel(), dtype=torch.bool)
            mask[flat] = False
            assert float(scores.min()) >= float(prob.flatten()[mask].max())
            expected_n = int((prob.flatten().topk(300).values > .001).sum())
            assert len(scores) == expected_n == 300
            row_error = float((p-d['fullp_boxes']).abs().max())
            row_order_difference_max = max(row_order_difference_max, row_error)
            unique_counts.append(len(q.unique()))
            raw_max_score_min = min(raw_max_score_min, float(prob.max(1).values.min()))
            fullp_score_min = min(fullp_score_min, float(scores.min()))
            min_GT_class_score_any_query = min(min_GT_class_score_any_query, float(prob.min()))
            if first is None:
                first = {'image_id': int(d['image_id']), 'raw_queries': 300, 'postprocessed_rows': len(q),
                         'postprocessed_unique_queries': len(q.unique()),
                         'first_12_query_ids': q[:12].tolist(),
                         'all_postprocessed_boxes_exactly_match_raw_by_query_id': True,
                         'same_row_order_max_difference_px': row_error}
            gt, gl = d['gt_boxes_640'].float(), d['gt_labels'].long()
            I = iou(gt, p)
            geometry = I >= .5
            has_geometry = geometry.any(1)
            has_eligible = (geometry & (prob[:, gl].T > .001)).any(1)
            post_has = ((I[:, q] >= .5) & (labels[None, :] == gl[:, None])).any(1)
            assert bool((~post_has | has_eligible).all())
            assert bool((~has_eligible | has_geometry).all())
            category = torch.zeros(len(gt), dtype=torch.long)
            category[has_geometry & ~has_eligible] = 1
            category[has_eligible & ~post_has] = 2
            category[post_has] = 3
            tiny = d['gt_short_side_640_all'] < 4
            for key, selected in (('all_GT', torch.ones(len(gt), dtype=torch.bool)),
                                  ('tiny_cached_shortside_lt4', tiny), ('non_tiny', ~tiny)):
                counts = torch.bincount(category[selected], minlength=4)
                groups[key] = [a+int(b) for a, b in zip(groups[key], counts)]
        assert sum(groups['all_GT']) == 38759
        assert sum(groups['tiny_cached_shortside_lt4']) == 4952
        assert groups['tiny_cached_shortside_lt4'][0] == {1:3715, 2:3727}[seed]
        results[str(seed)] = {
            'files': len(files), 'raw_query_count_every_file': 300, 'raw_class_logits_per_file': [300, 10],
            'postprocessed_rows_every_file': 300,
            'distinct_postprocessed_query_count': {'min': min(unique_counts), 'max': max(unique_counts),
                                                 'mean': sum(unique_counts)/len(unique_counts)},
            'minimum_raw_query_max_class_score': raw_max_score_min,
            'minimum_recorded_postprocess_score': fullp_score_min,
            'minimum_score_over_all_raw_query_class_pairs': min_GT_class_score_any_query,
            'max_same_row_order_coordinate_difference_px': row_order_difference_max,
            'first_image_example': first,
            'coverage_partition_not_TP_or_AP': {k: dict(zip(NAMES, v)) for k, v in groups.items()}}
    source_paths = [MAIN/'360_问题定义与归因_20261026_v1.md',
                    MAIN/'360_问题定义与归因_提示词_20261026_v7.md',
                    CACHE/'342_build_val_v2.py', EN/'code/unit_instance_completion_20261021_v1/stage1_cache_v3.py',
                    EN/'code/dn_pre_frozen_diagnostic_20261020_v1/native/engine/deim/dfine_decoder.py',
                    EN.parent/'tmp/360_v7_fullp_rank.py', EN.parent/'tmp/360_v7_geometry_probe.py']
    sources = [{'path': str(p), 'sha256': sha(p.read_bytes())} for p in source_paths]
    index = json.loads((MAIN/'327_训练实施输入索引.json').read_text(encoding='utf-8'))
    configs = []
    for cp in index['checkpoints']:
        cfg_path = Path(cp['path']).parent.parent/'bundle/resolved.yml'
        cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
        args = cfg['ResidualGeometryDEIM']['recipe']['DFINETransformer']
        assert args['num_queries'] == 300
        configs.append({'seed': cp['seed'], 'path': str(cfg_path), 'sha256': sha(cfg_path.read_bytes()),
                        'num_queries': args['num_queries'], 'query_select_method': args['query_select_method'],
                        'feat_strides': args['feat_strides']})
    out = {'scope': 'Existing-cache field contract and observability audit; no new model experiment',
           'seeds': results, 'decoder_configs': configs, 'source_hashes': sources,
           'input_manifest': inputs, 'elapsed_seconds': time.monotonic()-START,
           'coverage_definition': 'Existence of at least one class-correct geometric candidate per GT; not one-to-one matching, TP, recall or AP.',
           'new_model_forwards': 0, 'training': False, 'old_files_modified': False}
    with (ROOT/'cache_contract_audit.json').open('x', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k not in ('input_manifest', 'source_hashes')},
                     ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
