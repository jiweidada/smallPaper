"""Compare saved random-native and enhanced Stem outputs without model execution."""
from pathlib import Path
import hashlib
import json
import datetime
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/random_Stem_area_boundaries12_20261011_v1'
assert not OUT.exists()
OUT.mkdir()
sources = {}
labels = ['native_TP', 'miss_never_supported_at_three_boundaries', 'miss_earlier_supported_lost_at_final', 'miss_final_supported_no_true_class_after_top300_score_filter', 'miss_final_supported_true_class_removed_by_class100', 'miss_final_supported_retained_true_class_no_native_TP', 'miss_uncertain_ranking_reconstruction']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    sources[str(p.relative_to(ROOT))] = sha(p)
    return json.loads(p.read_text(encoding='utf-8'))


def arrays(p):
    sources[str(p.relative_to(ROOT))] = sha(p)
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def reduce_precision(v):
    valid = v[v > -1]
    return float(valid.mean() * 100) if valid.size else None


rows = []
for seed in (1, 2):
    root = ROOT / f'outputs/stem_fixed12_seed{seed}_formal_collection12_20261011_v1'
    verified = {}
    endpoints = {}
    precisions = {}
    for arm in ('original_stem', 'native_stem_scratch'):
        p = root / f'seed{seed}_{arm}'
        verified[arm] = read(p / 'verification.json')
        assert verified[arm]['all548_stage_arrays_reloaded_exact'] is True
        endpoints[arm] = read(p / 'extracted/run/endpoint_epoch12/summary.json')
        precisions[arm] = arrays(p / 'extracted/run/endpoint_epoch12/precision_recall.npz')['precision']
        assert precisions[arm].shape == (10, 101, 10, 4, 3)
        assert endpoints[arm]['overall_percent'] == verified[arm]['candidate_overall_percent']
    full = arrays(ROOT / f'outputs/stem_primary_joint_analysis12_20261011_v1/seed{seed}_original_stem/stage_per_GT.npz')
    full_joint = arrays(ROOT / f'outputs/stem_primary_joint_analysis12_20261011_v1/seed{seed}_original_stem/joint_per_GT.npz')
    random_root = ROOT / f'outputs/stem_random_primary_analysis12_20261011_v1/seed{seed}_native_stem_scratch'
    random = arrays(random_root / 'control_stage_per_GT.npz')
    rank = arrays(random_root / 'control_ranking/per_GT.npz')
    old = read(random_root / 'report.json')
    for key in ('annotation_id', 'image_id', 'category_id', 'original_area'):
        assert np.array_equal(full[key], random[key]) and np.array_equal(full[key], rank[key])
    assert np.array_equal(full['annotation_id'], full_joint['annotation_id'])
    assert len(full['annotation_id']) == 38759
    area = full['original_area']
    masks = {'all': np.ones(len(area), bool), 'small': area <= 1024, 'medium': (area >= 1024) & (area <= 9216), 'large': area >= 9216}
    category_names = {x['class_id']: x['name'] for x in endpoints['original_stem']['categories_percent']}
    class_rows = []
    for area_index, area_label in enumerate(('all', 'small', 'medium', 'large')):
        whole = {arm: reduce_precision(precision[:, :, :, area_index, 2]) for arm, precision in precisions.items()}
        expected_key = {'all': 'AP', 'small': 'APs', 'medium': 'APm', 'large': 'APl'}[area_label]
        assert all(abs(value - endpoints[arm]['overall_percent'][expected_key]) <= 1e-12 for arm, value in whole.items())
        categories = []
        valid_count = 0
        for cid in range(10):
            vals = {arm: reduce_precision(p[:, :, cid, area_index, 2]) for arm, p in precisions.items()}
            assert (vals['original_stem'] is None) == (vals['native_stem_scratch'] is None)
            if vals['original_stem'] is None:
                continue
            valid_count += 1
            difference = vals['original_stem'] - vals['native_stem_scratch']
            categories.append(dict(class_id=cid, name=category_names[cid], random_AP=vals['native_stem_scratch'], enhanced_AP=vals['original_stem'], delta_AP_points=difference))
        delta = whole['original_stem'] - whole['native_stem_scratch']
        assert abs(sum(x['delta_AP_points'] for x in categories) / valid_count - delta) <= 1e-12
        for row in categories:
            row['equal_class_mean_contribution_points'] = row['delta_AP_points'] / valid_count
        class_rows.append(dict(area=area_label, valid_classes=valid_count, random_AP=whole['native_stem_scratch'], enhanced_AP=whole['original_stem'], delta_AP_points=delta, category_decomposition=sorted(categories, key=lambda x: x['equal_class_mean_contribution_points'])))

    observations = []
    for threshold, t_index, label_suffix in ((.5, 0, '50'), (.75, 5, '75')):
        full_labels = full_joint[f'joint_label_{label_suffix}']
        cell = rank[f'cell_{label_suffix}']
        random_labels = np.full(len(area), -1, np.int8)
        mapping = {'native_TP': 0, 'no_true_class_support_after_top300_and_score_threshold': 3, 'true_class_support_removed_by_class100': 4, 'true_class_retained_geometry_without_native_TP': 5, 'uncertain_ranking_reconstruction': 6}
        for name, number in mapping.items():
            random_labels[cell == name] = number
        no_final = cell == 'no_final_geometric_support'
        earlier = (random['max_iou'][:, :2] >= threshold).any(1)
        random_labels[no_final & earlier] = 2
        random_labels[no_final & ~earlier] = 1
        assert (random_labels >= 0).all()
        for index, label in enumerate(labels):
            assert int((random_labels == index).sum()) == old['outcomes'][str(threshold)]['all']['control']['counts'][label]
            assert int((full_labels == index).sum()) == old['outcomes'][str(threshold)]['all']['full_enhanced']['counts'][label]
        fTP, rTP = full['native_TP'][:, t_index], random['native_TP'][:, t_index]
        fSupport, rSupport = full['max_iou'] >= threshold, random['max_iou'] >= threshold
        for area_label, mask in masks.items():
            lost = mask & rTP & ~fTP
            gained = mask & fTP & ~rTP
            counts = {label: int((lost & (full_labels == i)).sum()) for i, label in enumerate(labels)}
            assert sum(counts.values()) == int(lost.sum())
            observations.append(dict(area=area_label, IoU_threshold=threshold, GT=int(mask.sum()),
                native_all_area_TP_filtered_by_GT_area={'random': int((mask & rTP).sum()), 'enhanced': int((mask & fTP).sum()), 'net': int((mask & fTP).sum() - (mask & rTP).sum()), 'gained': int(gained.sum()), 'lost': int(lost.sum())},
                geometric_support_net_change_by_saved_boundary={name: int((mask & fSupport[:, i]).sum() - (mask & rSupport[:, i]).sum()) for i, name in enumerate(('encoder_initial', 'after_first_decoder', 'final_decoder'))},
                lost_TP_enhanced_error_partition=counts,
                lost_TP_lacks_enhanced_initial_support=int((lost & ~fSupport[:, 0]).sum()),
                lost_TP_initial_support_changed_between_models=int((lost & rSupport[:, 0] & ~fSupport[:, 0]).sum()),
                loss_geometry_needed=int((lost & ~fSupport[:, 2]).sum()),
                loss_final_geometry_exists_but_not_native_TP=int((lost & fSupport[:, 2]).sum())))
    rows.append(dict(seed=seed, native_area_AP_class_decomposition=class_rows, saved_boundary_observations=observations))

report = dict(status='TWO_SEED_RANDOM_NATIVE_STEM_AREA_PR_AND_SAVED_BOUNDARIES_ANALYZED', UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), rows=rows,
    comparison='User enhanced Stem minus native Stem randomly initialized;754 non-Stem common initial tensors retained.',
    metric_boundary='Class decomposition is exact native maxDets100 area PR arithmetic. TP/error cohorts use native all-area matching filtered by GT area, and are not area-specific AP contributions.',
    location_boundary='Encoder initial is the earliest saved boundary, not the first internal network module. No cross-model query identity is assumed.',
    area_boundary='Native medium and large area ranges include boundary values1024/9216; edge cohorts may overlap exactly as evaluator intervals do.',
    missing_evidence='No matched per-layer feature/gradient history exists here; module-specific cause and training-gradient conflict remain unproven.',
    new_training_epochs=0,new_model_forwards=0,new_native_evaluator_calls=0,root_proven=False,novelty_proven=False,sources_sha256=sources)
(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
text = ['增强 Stem 与随机原始 Stem：中、大目标及阶段边界核对', '固定第12轮EMA；只分析已保存产物。']
for row in rows:
    text.append(f"\n种子{row['seed']}")
    for area in row['native_area_AP_class_decomposition']:
        text.append(f"{area['area']} AP变化 {area['delta_AP_points']:+.6f}，贡献最小的类别：" + ', '.join(f"{x['name']} {x['equal_class_mean_contribution_points']:+.4f}" for x in area['category_decomposition'][:3]))
    for obs in row['saved_boundary_observations']:
        if obs['area'] in ('medium','large') and obs['IoU_threshold']==.5:
            text.append(f"{obs['area']} IoU50：三边界几何覆盖净变化 {obs['geometric_support_net_change_by_saved_boundary']}；最终原生全尺度TP净变化 {obs['native_all_area_TP_filtered_by_GT_area']['net']}；损失TP原因 {obs['lost_TP_enhanced_error_partition']}")
text.extend(['',report['metric_boundary'],report['location_boundary'],report['missing_evidence']])
(OUT/'中大目标与阶段差异.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
(OUT/'manifest.json').write_text(json.dumps({p.name:sha(p) for p in OUT.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf-8')
print('\n'.join(text))
