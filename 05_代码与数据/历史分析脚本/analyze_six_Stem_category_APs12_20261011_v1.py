"""Arithmetic attribution from saved native PR arrays; no evaluator/model calls."""
from pathlib import Path
import datetime
import hashlib
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/six_Stem_category_APs12_20261011_v1'
ARMS = ('original_stem', 'native_stem_scratch', 'highfreq_rgb')
evidence = {}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def record(path):
    evidence[path.relative_to(ROOT).as_posix()] = dict(bytes=path.stat().st_size, sha256=digest(path))
    return json.loads(path.read_bytes())

def verify_manifest(folder):
    manifest = record(folder / 'manifest.json')
    for name, expected in manifest['files'].items():
        path = folder / name
        assert path.is_file() and digest(path) == expected, path
    return len(manifest['files'])

def reduction_check(cells, native):
    """Bound only FP64 summation arithmetic, never model/replay differences."""
    valid = cells[cells > -1]
    assert valid.size and np.all((valid >= 0) & (valid <= 1))
    local = float(np.mean(valid) * 100)
    reference = math.fsum(float(x) for x in valid.flat) / valid.size * 100
    unit_roundoff = np.finfo(np.float64).eps / 2
    operations = valid.size + 2  # conservative sequential sum + divide + scale
    gamma = operations * unit_roundoff / (1-operations*unit_roundoff)
    bound = gamma * (math.fsum(abs(float(x)) for x in valid.flat) / valid.size * 100)
    assert abs(native-reference) <= bound and abs(local-reference) <= bound
    return dict(saved_native_percent=native, local_reduction_percent=local,
                local_minus_saved=local-native, compensated_sum_reference_percent=reference,
                sequential_FP64_roundoff_bound_points=bound, PR_cells=int(valid.size))

six = ROOT / 'outputs/native_selection_analysis12_20261011_v5/actual_six'
verified_analysis_files = verify_manifest(six)
selection = record(six / 'report.json')
assert selection['seeds'] == [1, 2]
assert selection['new_model_forwards'] == selection['new_evaluator_calls'] == selection['new_training_epochs'] == 0
records = {}
protocol = None
for seed in (1, 2):
    for arm in ARMS:
        folder = ROOT / f'outputs/stem_fixed12_seed{seed}_formal_collection12_20261011_v1/seed{seed}_{arm}'
        formal = record(folder / 'verification.json')
        assert formal['identity']['seed'] == seed and formal['identity']['arm'] == arm
        assert formal['all548_stage_arrays_reloaded_exact'] and formal['first100_inputs_and_pre_forward_RNG_exact']
        assert formal['fixed_endpoint'] == dict(epochs=12, zero_based_epoch=11, weights='EMA')
        endpoint = folder / 'extracted/run/endpoint_epoch12'
        verify_manifest(endpoint)
        summary = record(endpoint / 'summary.json')
        assert summary['identity'] == formal['identity']
        assert summary['overall_percent'] == formal['candidate_overall_percent']
        p = summary['protocol']
        if protocol is None:
            protocol = p
        assert p == protocol
        assert p['max_dets'] == [1, 10, 100] and p['category_ids'] == list(range(10))
        assert p['area_labels'] == ['all', 'small', 'medium', 'large']
        assert p['area_ranges'][1] == [0.0, 1024.0] and len(p['image_ids']) == 548
        arraypath = endpoint / 'precision_recall.npz'
        evidence[arraypath.relative_to(ROOT).as_posix()] = dict(bytes=arraypath.stat().st_size, sha256=digest(arraypath))
        with np.load(arraypath, allow_pickle=False) as saved:
            precision = saved['precision'].copy()
        assert precision.shape == (10, 101, 10, 4, 3)
        assert precision.dtype == np.dtype('float64') and np.isfinite(precision).all()
        native = summary['overall_percent']
        values = {}
        categories = summary['categories_percent']
        assert [x['class_id'] for x in categories] == p['category_ids']
        for metric, area in [('AP', 0), ('APs', 1), ('APm', 2), ('APl', 3)]:
            cells = precision[:, :, :, area, -1]
            valid = cells > -1
            reduction = reduction_check(cells, native[metric])
            calculated = native[metric]
            per_class = []
            for category_index, category in enumerate(categories):
                own = cells[:, :, category_index]
                own_valid = own > -1
                ap = category[metric]
                own_reduction = reduction_check(own, ap) if own_valid.any() else None
                assert own_valid.any() or ap == -1
                per_class.append(dict(class_id=category['class_id'], name=category['name'], AP_percent=ap,
                                      valid_PR_cells=int(own_valid.sum()), valid=bool(own_valid.any()),
                                      local_arithmetic=own_reduction))
            valid_counts = [x['valid_PR_cells'] for x in per_class if x['valid']]
            assert len(set(valid_counts)) == 1, 'Unequal validity needs cell-weighted attribution'
            class_mean = math.fsum(x['AP_percent'] for x in per_class if x['valid']) / len(valid_counts)
            values[metric] = dict(native_percent=calculated, valid_classes=len(valid_counts),
                                  class_mean_percent=class_mean, arithmetic_residual=class_mean-calculated,
                                  categories=per_class, local_arithmetic=reduction)
        # Full small-class PR tensor retained for exact IoU slices in the comparison.
        records[(seed, arm)] = dict(metrics=values, small_PR=precision[:, :, :, 1, -1],
                                    formal_verification_sha256=digest(folder / 'verification.json'),
                                    native_summary_sha256=digest(endpoint / 'summary.json'),
                                    checkpoint_sha256=formal['paired_native_results']['candidate_checkpoint_sha256']
                                    if 'candidate_checkpoint_sha256' in formal['paired_native_results'] else None)

pairs = {}
for seed in (1, 2):
    pairs[str(seed)] = {}
    full = records[(seed, 'original_stem')]
    for left_arm in ('native_stem_scratch', 'highfreq_rgb'):
        left = records[(seed, left_arm)]
        comparison = {}
        for metric in ('AP', 'APs', 'APm', 'APl'):
            a, b = left['metrics'][metric], full['metrics'][metric]
            assert [(x['class_id'], x['valid_PR_cells']) for x in a['categories']] == [(x['class_id'], x['valid_PR_cells']) for x in b['categories']]
            n = b['valid_classes']
            contributions = []
            for x, y in zip(a['categories'], b['categories']):
                assert x['name'] == y['name'] and x['valid'] == y['valid']
                if x['valid']:
                    delta = y['AP_percent'] - x['AP_percent']
                    contributions.append(dict(class_id=y['class_id'], name=y['name'],
                        left_percent=x['AP_percent'], full_percent=y['AP_percent'], delta_points=delta,
                        contribution_to_native_mean_points=delta/n))
            total = b['native_percent'] - a['native_percent']
            summed = math.fsum(x['contribution_to_native_mean_points'] for x in contributions)
            comparison[metric] = dict(native_delta_points=total, summed_class_contribution_points=summed,
                arithmetic_residual=summed-total, improved_classes=sum(x['delta_points'] > 0 for x in contributions),
                worsened_classes=sum(x['delta_points'] < 0 for x in contributions),
                classes=sorted(contributions, key=lambda x:x['contribution_to_native_mean_points'], reverse=True))
        slices = []
        for t, threshold in enumerate(protocol['iou_thresholds']):
            a, b = left['small_PR'][t], full['small_PR'][t]
            assert np.array_equal(a > -1, b > -1)
            slices.append(dict(IoU=threshold,
                left_small_AP_percent=float(np.mean(a[a > -1])*100),
                full_small_AP_percent=float(np.mean(b[b > -1])*100),
                delta_points=float(np.mean(b[b > -1])*100)-float(np.mean(a[a > -1])*100)))
        comparison['small_AP_by_IoU'] = slices
        comparison['scope'] = 'Exact arithmetic decomposition of native macro AP only; no causal attribution to features, queries, geometry, ranking, or TP population counts.'
        pairs[str(seed)][f'{left_arm}_to_original_stem'] = comparison

OUT.mkdir(exist_ok=False)
report = dict(status='VERIFIED_SIX_SAVED_NATIVE_PR_AND_CLASS_ARITHMETIC_WITH_REDUCTION_RESIDUALS',
    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), pairs=pairs,
    records={f'seed{s}_{a}':{k:v for k,v in x.items() if k!='small_PR'} for (s,a),x in records.items()},
    verified_selection_analysis_files=verified_analysis_files, native_evaluator_max_detections_per_image_per_category=100,
    model_postprocessor_output_items=300, metric_units='percentage points',
    small_area_closed_range=[0,1024], all_six_formal_fixed12_and_exact_saved_EMA_reloads_verified=True,
    new_model_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,new_training_epochs=0,
    root_proven=False,novelty_proven=False,
    local_initial_attempt_note='An exact local FP64 mean check rejected scratch seed1 AP: local 7.2957880927895635 versus saved 7.295788092789563. Native saved results remain authoritative; cross-runtime reduction residuals are explicitly recorded and bounded by FP64 arithmetic. No model/replay tolerance is changed and no evaluator is rerun.',
    saved_model_replay_exact_requirement_unchanged=True,
    limitations=['Macro AP is not weighted by class GT counts.',
        'Small-area native evaluation has its own ignore/matching behavior; all-area TP flags stratified by small GT are not this APs population.',
        'AP uses IoU .50:.95 and the full precision-recall curve, so TP50/TP75 changes cannot reconstruct APs.',
        'The highpass input x-avg_pool3(x) versus raw RGB also changes input mean/amplitude/boundaries; positive APs does not isolate a frequency root.',
        'Whole enhanced Stem versus random native Stem also changes structure/parameterization and training trajectory; pretrained BASE acceptance remains separate.'])
with (OUT/'report.json').open('x',encoding='utf8') as f:
    json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
lines=['六组 Stem：固定第12轮 EMA，读取已核验原生 PR 数组；没有运行模型、评价器或新增训练。',
       '原生 AP 使用每图每类 maxDets=100；模型后处理300项不等于原生 AP maxDets=300。',
       '类别贡献是精度指标的算术分解，不是原因分解。小目标命中数量不是 APs 的算术贡献。','']
for seed, data in pairs.items():
    for name,x in data.items():
        lines.append(f'种子{seed} {name}: ΔAP={x["AP"]["native_delta_points"]:+.6f}，ΔAPs={x["APs"]["native_delta_points"]:+.6f}。')
        lines.append('各类别 ΔAPs / 对总体 APs 的贡献（百分点）：')
        for y in x['APs']['classes']:
            lines.append(f'  {y["name"]}: {y["delta_points"]:+.6f} / {y["contribution_to_native_mean_points"]:+.6f}')
        lines.append(f'APs 提升类别{x["APs"]["improved_classes"]}，下降类别{x["APs"]["worsened_classes"]}；算术残差{x["APs"]["arithmetic_residual"]:.3g}。\n')
(OUT/'类别精度分解.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
manifest=dict(files=evidence,report_sha256=digest(OUT/'report.json'),text_sha256=digest(OUT/'类别精度分解.txt'),producer_sha256=digest(Path(__file__)))
with (OUT/'manifest.json').open('x',encoding='utf8') as f:
    json.dump(manifest,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(status=report['status'],pairs={s:{n:{'APs':x['APs']['native_delta_points'],'improved_classes':x['APs']['improved_classes'],'top3':x['APs']['classes'][:3]} for n,x in p.items()} for s,p in pairs.items()}),ensure_ascii=False))
