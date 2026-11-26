"""Read frozen outputs only: error onset and same-query box component oracles.

No model, loss, evaluator, checkpoint execution, or optimizer is constructed.
The three observed boundaries cannot localize an error within the backbone.
"""
from pathlib import Path
import csv
import datetime
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/sod_error_onset_20261010_v1'
OLD = ROOT / 'outputs/small_object_encoder_decoder12_analysis_20261010_v1'
CONTROL = ROOT / 'outputs/small_object_encoder_decoder12_20261010_v5'
READERS = ROOT / '英文小论文/code/small_object_encoder_decoder12_control_analysis_20261010_v1'
sys.path.insert(0, str(READERS))
import geometry_core as geom


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, obj):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def saved(folder, arm):
    p = folder / f'{arm}_per_GT.npz'
    assert sha(p) == read(folder / 'manifest.json')['files'][p.name]
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}, p


def component_oracles(seed, base, annotations, images):
    packet = OLD / f'terminal_collection/seed{seed}/extracted'
    manifest = read(packet / 'manifest.json')['files']
    stages = packet / 'baseline_eval/stages_epoch12'
    ids = base['annotation_id']; row_of = {int(g): i for i, g in enumerate(ids)}
    groups = {int(i): [] for i in images}
    for g in ids:
        groups[int(annotations[int(g)]['image_id'])].append(annotations[int(g)])
    result = {k: np.zeros(len(ids), dtype=np.float64) for k in
              ('center_replaced_IoU', 'size_replaced_IoU', 'dx_over_GT_width',
               'dy_over_GT_height', 'width_ratio', 'height_ratio')}
    result['valid_best_query'] = np.zeros(len(ids), dtype=bool)
    files = {}
    for iid, im in sorted(images.items()):
        p = stages / f'image_{iid:06d}.npz'
        key = p.relative_to(packet).as_posix()
        assert sha(p) == manifest[key]['sha256'], key
        files[key] = manifest[key]['sha256']
        with np.load(p, allow_pickle=False) as z:
            assert z['original_hw'].tolist() == [im['height'], im['width']]
            raw = z['final_boxes'].copy()
        assert raw.shape == (300, 4) and np.isfinite(raw).all()
        rows = np.array([row_of[int(g['id'])] for g in groups[iid]], dtype=np.int64)
        gt = np.array([g['bbox'] for g in groups[iid]], dtype=np.float64).reshape(-1, 4)
        valid = (raw[:, 2:] > 0).all(1)
        matrix = np.zeros((len(gt), 300), dtype=np.float64)
        pred = np.zeros((300, 4), dtype=np.float64)
        pred[valid] = geom.pixel_xywh(raw[valid], [im['height'], im['width']])
        matrix[:, valid] = geom.iou_matrix(gt, pred[valid])
        q = matrix.argmax(1)
        assert np.array_equal(q, base['best_query_within_run'][rows, 2])
        assert np.array_equal(matrix[np.arange(len(gt)), q], base['max_iou'][rows, 2])
        chosen = pred[q]; usable = valid[q]; result['valid_best_query'][rows] = usable
        gt_center = gt[:, :2] + gt[:, 2:] * .5
        pred_center = chosen[:, :2] + chosen[:, 2:] * .5
        center_replaced = np.concatenate((gt_center - chosen[:, 2:] * .5, chosen[:, 2:]), axis=1)
        size_replaced = np.concatenate((pred_center - gt[:, 2:] * .5, gt[:, 2:]), axis=1)
        # Compute aligned-pair IoU directly; never match to a different GT here.
        def aligned_iou(box):
            wh = np.maximum(0, np.minimum(gt[:, :2] + gt[:, 2:], box[:, :2] + box[:, 2:])
                            - np.maximum(gt[:, :2], box[:, :2]))
            inter = wh[:, 0] * wh[:, 1]
            union = gt[:, 2] * gt[:, 3] + box[:, 2] * box[:, 3] - inter
            value = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            value[~usable] = 0
            return value
        result['center_replaced_IoU'][rows] = aligned_iou(center_replaced)
        result['size_replaced_IoU'][rows] = aligned_iou(size_replaced)
        ratios = chosen[:, 2:] / gt[:, 2:]
        errors = (pred_center - gt_center) / gt[:, 2:]
        for k, value in zip(('width_ratio', 'height_ratio', 'dx_over_GT_width', 'dy_over_GT_height'),
                            (ratios[:, 0], ratios[:, 1], errors[:, 0], errors[:, 1])):
            result[k][rows] = value
    return result, files, sha(packet / 'manifest.json')


def describe(a, mask, threshold, ti, components=None):
    support = a['max_iou'] >= threshold
    bits = support[:, 0].astype(int) * 4 + support[:, 1].astype(int) * 2 + support[:, 2].astype(int)
    n = int(mask.sum()); tp = a['native_TP'][:, ti]
    r = {'GT': n, 'native_all_area_TP_filtered_GT': int((mask & tp).sum()),
         'support_pattern_counts': {f'{b:03b}': int((mask & (bits == b)).sum()) for b in range(8)},
         'never_supported': int((mask & (bits == 0)).sum()),
         'supported_early_but_lost_final': int((mask & support[:, :2].any(1) & ~support[:, 2]).sum()),
         'final_geometry_without_native_TP': int((mask & support[:, 2] & ~tp).sum()),
         'final_best_query_already_good_initial': int((mask & (a['final_best_at_initial'] >= threshold)).sum()),
         'initial_best_same_query_good_final': int((mask & (a['initial_best_at_final'] >= threshold)).sum())}
    assert sum(r['support_pattern_counts'].values()) == n
    assert not (mask & tp & ~support[:, 2]).any()
    assert r['never_supported'] + r['supported_early_but_lost_final'] + r['final_geometry_without_native_TP'] + r['native_all_area_TP_filtered_GT'] == n
    if components is not None:
        failed = mask & ~support[:, 2]
        valid = failed & components['valid_best_query']
        center = components['center_replaced_IoU'] >= threshold
        size = components['size_replaced_IoU'] >= threshold
        r['same_final_best_query_GT_replacement_oracle'] = {
            'no_final_geometry_GT': int(failed.sum()), 'usable_best_query_GT': int(valid.sum()),
            'center_only_rescue': int((valid & center & ~size).sum()),
            'size_only_rescue': int((valid & size & ~center).sum()),
            'either_single_component_rescues': int((valid & center & size).sum()),
            'neither_single_component_rescues': int((valid & ~center & ~size).sum()),
            'unusable_query': int((failed & ~components['valid_best_query']).sum()),
            'median_abs_center_error_GT_width_height': [float(np.median(np.abs(components[k][valid]))) if valid.any() else None
                                                       for k in ('dx_over_GT_width', 'dy_over_GT_height')],
            'median_predicted_over_GT_width_height': [float(np.median(components[k][valid])) if valid.any() else None
                                                      for k in ('width_ratio', 'height_ratio')]}
        o = r['same_final_best_query_GT_replacement_oracle']
        assert sum(o[k] for k in ('center_only_rescue', 'size_only_rescue', 'either_single_component_rescues',
                                  'neither_single_component_rescues', 'unusable_query')) == int(failed.sum())
    return r


def main():
    started = time.monotonic(); protocol = read(OUT / 'protocol.json')
    assert not (OUT / 'report.json').exists()
    assert sha(READERS / 'geometry_core.py') == protocol['geometry_reader_sha256']
    ann_path = OLD / 'terminal_collection/seed1/extracted/reference/val_annotations.json'
    assert sha(ann_path) == protocol['annotation_sha256']
    annotation = read(ann_path)
    images = {int(i['id']): i for i in annotation['images']}
    anns = {int(a['id']): a for a in annotation['annotations']}
    assert len(anns) == 38759 and len(images) == 548
    report = {'status': 'SAVED_OUTPUT_ERROR_ONSET_ANALYZED', 'sources': [], 'seeds': [],
              'protocol_sha256': sha(OUT / 'protocol.json'), 'implementation_sha256': sha(__file__),
              'new_model_forwards': 0, 'new_evaluator_calls': 0, 'training_updates': 0, 'training_epochs': 0,
              'scope': protocol['scope']}
    rows_csv = []
    for seed in (1, 2):
        base, source = saved(OLD / f'paired_seed{seed}', 'baseline')
        ids = base['annotation_id']; assert set(map(int, ids)) == set(anns)
        wh = np.array([[anns[int(g)]['bbox'][2]*640/images[int(anns[int(g)]['image_id'])]['width'],
                        anns[int(g)]['bbox'][3]*640/images[int(anns[int(g)]['image_id'])]['height']] for g in ids])
        side = wh.min(1)
        masks = {'all': np.ones(len(ids), bool), 'COCO_small': base['original_area'] <= 1024,
                 'input_under4': side < 4, 'input_4_to_under8': (side >= 4) & (side < 8),
                 'input_8_to_under16': (side >= 8) & (side < 16), 'input_ge16': side >= 16}
        components, hashes, packet_sha = component_oracles(seed, base, anns, images)
        write(OUT / f'seed{seed}_raw_stage_sources.json', {'packet_manifest_sha256': packet_sha, 'files': hashes})
        np.savez_compressed(OUT / f'seed{seed}_box_component_oracles.npz', annotation_id=ids, **components)
        arms = {'baseline': (base, source)}
        arms['fine'] = saved(OLD / f'paired_seed{seed}', 'candidate')
        arms['degraded'] = saved(CONTROL / f'paired_seed{seed}', 'candidate')
        seed_report = {'seed': seed, 'arms': {}}
        for name, (a, p) in arms.items():
            for k in ('annotation_id', 'image_id', 'category_id', 'original_area'):
                assert np.array_equal(base[k], a[k]), k
            report['sources'].append({'seed': seed, 'arm': name, 'path': str(p), 'sha256': sha(p)})
            seed_report['arms'][name] = {}
            for group, mask in masks.items():
                seed_report['arms'][name][group] = {}
                for t, ti in ((.5, 0), (.75, 5)):
                    r = describe(a, mask, t, ti, components if name == 'baseline' else None)
                    seed_report['arms'][name][group][str(t)] = r
                    rows_csv.append(dict(seed=seed, arm=name, group=group, IoU=t,
                        **{k: v for k, v in r.items() if isinstance(v, int)}))
        report['seeds'].append(seed_report)
        print(json.dumps({'seed': seed, 'status': 'LOCAL_SAVED_OUTPUT_ANALYSIS_DONE',
                          'under4_IoU75': seed_report['arms']['baseline']['input_under4']['0.75']}, ensure_ascii=False), flush=True)
    assert 'torch' not in sys.modules
    report['elapsed_seconds'] = time.monotonic() - started
    report['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write(OUT / 'report.json', report)
    with (OUT / 'stage_onset_counts.csv').open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_csv[0])); writer.writeheader(); writer.writerows(rows_csv)
    write(OUT / 'analysis_manifest.json', {'files': {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}})


if __name__ == '__main__':
    main()
