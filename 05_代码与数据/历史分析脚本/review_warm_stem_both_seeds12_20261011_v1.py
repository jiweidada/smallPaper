"""Read both verified fixed-12 endpoints; preserve cross-seed reversals and native area semantics."""
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np

ROOT = Path.cwd()
read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, value):
    with Path(p).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

def arrays(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}

def verified_report(folder):
    folder = ROOT / folder
    manifest = read(folder / 'manifest.json')
    for name, h in manifest['files'].items():
        assert sha(folder / name) == h, name
    return read(folder / 'report.json'), sha(folder / 'manifest.json')

spec = importlib.util.spec_from_file_location('dual_warm_cached_tools', ROOT / '英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py')
cached = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cached)
protocol, _, _, _, readers = cached.prepared()
seeds = {}
for seed in (1, 2):
    verification_path = ROOT / f'outputs/warm_stem_primary_collection12_20261011_v1/seed{seed}_warm_stem_v1/primary_verification_v1.json'
    v = read(verification_path)
    assert v['status'] == 'PASS_WARM_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS'
    assert v['files_verified'] == 615 and v['actual_stage_images'] == 548
    assert v['all_collected_files_verified'] and v['all_ZIP_entries_CRC_verified']
    assert v['verification_script_sha256'] == sha(ROOT / '英文小论文/code/warm_stem_primary_collection12_20261011_v1/verify_primary_v2.py')
    assert v['identity']['seed'] == seed
    packet = Path(v['downloaded_packet'])
    assert sha(packet / 'primary_manifest.json') == v['primary_manifest_sha256']
    a, analysis_manifest_sha = verified_report(f'outputs/warm_stem_primary_analysis12_20261011_v1/seed{seed}')
    n, order_manifest_sha = verified_report(f'outputs/warm_stem_native_order12_20261011_v1/seed{seed}')
    assert a['identity'] == v['identity'] and a['completed_epochs'] == 12
    assert a['all548_images'] and a['all38759_GT'] and a['all10_categories']
    assert a['native_TP_proxy_conflicts'] == 0 and not a['score_mismatch_images']
    assert a['analysis_sha256'] == sha(ROOT / '英文小论文/code/warm_stem_primary_analysis12_20261011_v1/analyze_v2.py')
    assert n['analysis_sha256'] == sha(ROOT / '英文小论文/code/warm_stem_native_order12_20261011_v1/analyze.py')
    assert n['status'] == 'WARM_AND_CACHED_BASE_NATIVE_ALL_AREA_PR20200_CELLS_AND_GT_PREFIXES_REPRODUCED_EXACTLY'
    assert n['total_precision_cells_reconstructed'] == 20200 and n['native_PR_max_abs_difference'] == 0
    assert n['all_native_GT_flags_exact'] and n['actual_native_stable_ties_preserved']
    assert n['primary_verification_sha256'] == sha(verification_path)
    assert abs(n['delta_AP_points'] - a['delta_points']['AP']) < 1e-10
    base = cached.cached_baseline(seed, protocol, readers['full'])
    warm = arrays(ROOT / f'outputs/warm_stem_primary_analysis12_20261011_v1/seed{seed}/warm_stage_per_GT.npz')
    bv = base['values']
    for k in ('annotation_id', 'image_id', 'category_id', 'original_area'):
        assert np.array_equal(warm[k], bv[k]), k
    small = warm['original_area'] <= 1024
    counts = np.bincount(warm['category_id'][small], minlength=10)
    assert counts.sum() == 26586 and (counts > 0).all()
    pr = {}
    for tag, p in (('BASE12', base['packet'] / 'baseline_eval/endpoint_epoch12'), ('warm12', packet / 'run/endpoint_epoch12')):
        values = arrays(p / 'precision_recall.npz')
        P, R = values['precision'], values['recall']
        assert P.shape == (10, 101, 10, 4, 3) and R.shape == (10, 10, 4, 3)
        raw_TP = R[:, :, 1, 2] * counts[None, :]
        assert np.max(np.abs(raw_TP - np.rint(raw_TP))) < 1e-8
        TP = np.rint(raw_TP).astype(np.int64)
        per_iou = []
        for ti, t in enumerate(np.linspace(.5, .95, 10)):
            precision = P[ti, :, :, 1, 2]
            assert (precision >= 0).all()
            per_iou.append(dict(IoU=float(t), APs=float(precision.mean() * 100),
                native_small_area_TP=int(TP[ti].sum()), class_native_small_area_TP=TP[ti].tolist()))
        assert abs(sum(x['APs'] for x in per_iou) / 10 - a[tag]['APs']) < 1e-9
        pr[tag] = per_iou
    iou_deltas = [dict(IoU=b['IoU'], APs_delta_points=w['APs']-b['APs'],
        native_small_area_TP_delta=w['native_small_area_TP']-b['native_small_area_TP'])
        for b, w in zip(pr['BASE12'], pr['warm12'])]
    contributions = dict(low_0_50_to_0_70=sum(x['APs_delta_points'] for x in iou_deltas[:5])/10,
                         high_0_75_to_0_95=sum(x['APs_delta_points'] for x in iou_deltas[5:])/10)
    assert abs(sum(contributions.values()) - a['delta_points']['APs']) < 1e-9
    strata, stage = {}, {}
    for ti, t in ((0, .5), (5, .75)):
        bi, wi = bv['max_iou'][:, 0] >= t, warm['max_iou'][:, 0] >= t
        bf, wf = bv['max_iou'][:, 2] >= t, warm['max_iou'][:, 2] >= t
        rows = {}
        for name, m in [('shared_initial_support', small & bi & wi),
                        ('warm_new_initial_support', small & ~bi & wi),
                        ('warm_lost_initial_support', small & bi & ~wi)]:
            final, matched = np.zeros((2, 2), dtype=np.int64), np.zeros((2, 2), dtype=np.int64)
            np.add.at(final, (bf[m].astype(int), wf[m].astype(int)), 1)
            np.add.at(matched, (bv['native_TP'][m, ti].astype(int), warm['native_TP'][m, ti].astype(int)), 1)
            assert int(final.sum()) == int(m.sum()) == int(matched.sum())
            rows[name] = dict(GT=int(m.sum()), baseline_final_already_supported=int((m & bf).sum()),
                warm_final_supported=int((m & wf).sum()), BASE_rows_Warm_columns_final_support=final.tolist(),
                BASE_rows_Warm_columns_native_all_area_TP=matched.tolist())
        strata[str(t)] = rows
        stage[str(t)] = {}
        for group in ('COCO_small', 'input_under4'):
            o = a['outcomes'][str(t)][group]
            stage[str(t)][group] = {k: o[k] for k in ('GT', 'native_all_area_TP_gained',
                'native_all_area_TP_lost', 'native_all_area_TP_net', 'supported_GT_by_boundary')}
        assert rows['warm_new_initial_support']['GT'] - rows['warm_lost_initial_support']['GT'] == stage[str(t)]['COCO_small']['supported_GT_by_boundary']['encoder_initial_boxes']['delta']
    people = next(c for c in n['classes'] if c['name'] == 'people')
    people50 = people['by_IoU'][0]['groups']['all_GT_of_class']
    seeds[str(seed)] = dict(BASE12=a['BASE12'], warm12=a['warm12'], delta_points=a['delta_points'],
        native_gate_passed=a['native_gate_passed'], source_files_verified=615, saved_stage_images=548,
        ranking_ambiguous_images=a['ranking_ambiguous_images'], class_APs=a['class_APs_arithmetic'],
        stage=stage, native_small_area_by_IoU=pr, native_small_area_IoU_deltas=iou_deltas,
        exploratory_IoU_range_contributions_to_total_APs_points=contributions, same_GT_strata=strata,
        people50_native_all_area_prefixes=people50,
        people50_prefix1000=next(x for x in people50['prefixes'] if x['rank'] == 1000),
        proof=dict(primary_verification_sha256=sha(verification_path), primary_manifest_sha256=v['primary_manifest_sha256'],
            analysis_manifest_sha256=analysis_manifest_sha, native_order_manifest_sha256=order_manifest_sha,
            baseline_source_report_sha256=base['source_report_sha256'],
            all_area_precision_cells_reproduced=20200, all_area_max_abs_difference=0),
        independent_saved_EMA_replay_verified=v['saved_EMA_independent_replay_verified'],
        formal_terminal_reader_passed=v['formal_terminal_reader_passed'])

consistent_classes = {'positive_both': [], 'negative_both': [], 'mixed_or_zero': []}
for c1, c2 in zip(seeds['1']['class_APs'], seeds['2']['class_APs']):
    assert c1['class_id'] == c2['class_id'] and c1['name'] == c2['name']
    d1, d2 = c1['delta_APs_points'], c2['delta_APs_points']
    label = 'positive_both' if min(d1, d2) > 0 else 'negative_both' if max(d1, d2) < 0 else 'mixed_or_zero'
    consistent_classes[label].append(dict(name=c1['name'], seed1_delta_APs=d1, seed2_delta_APs=d2))
assert seeds['1']['native_gate_passed'] and not seeds['2']['native_gate_passed']
assert seeds['1']['stage']['0.5']['COCO_small']['supported_GT_by_boundary']['encoder_initial_boxes']['delta'] == -135
assert seeds['2']['stage']['0.5']['COCO_small']['supported_GT_by_boundary']['encoder_initial_boxes']['delta'] == 573
result = dict(status='BOTH_WARM_PRIMARY_FIXED12_VERIFIED_CROSS_SEED_REVERSALS_RETAINED_NATIVE_GATE_FAIL',
    seeds=seeds, two_seed_native_gate_passed=False, consistent_class_APs_signs=consistent_classes,
    training_action='Do not extend, retry or select a best epoch for this Warm configuration.',
    changed_next_action='Finish already reviewed original user P2/SiLU/attention fixed12 controls on both hosts. Use each completed endpoint to test class/IoU-specific candidate coverage and final PR jointly; do not transplant the single-seed Warm coverage or ordering explanation as the cause of the original replacement modules. Existing native-location diagnostic remains necessary to separate selection from selected-reference regression.',
    rejected_explanations=[
        'Uniform early geometric coverage gain explains the APs improvement: seed1 APs rises while early IoU0.50 small coverage falls; seed2 reverses both signs.',
        'The people class universally moves later: front1000 native all-area TP changes by+62 in seed1 and-24 in seed2; common matched GT contain both earlier and later movements in both seeds.',
        'Under4 TP gains establish successful overall small-object detection: both seeds gain TP in that cohort, but overall native APs has opposite signs.',
        'A positive two-seed average or seed1 success meets the configured criterion: the criterion is positive APs and nonnegative AP separately in both seeds.'
    ],
    boundaries='Warm is native pretrained Stem plus a gated complete EnhancedStem branch, not the original replacement Stem nor a pure initialization control. Native small-area recall gives exact small-area TP counts, but saved raw matching records cover all areas only. Score-prefix accounting exactly reconstructs all-area AP and must not be labeled an exact APs prefix reconstruction. Shared GT can be compared across runs; query/box identity across runs and a unique classification/decoder/loss cause are not established. Both independent EMA replays and formal terminal readers remain pending.',
    source_sha256=sha(__file__), primary_files_verified_total=1230, native_all_area_PR_cells_reproduced_total=40400,
    new_model_forwards=0, new_BASE_forwards=0, new_evaluator_calls=0, new_native_matching_calls=0,
    new_training_epochs=0, new_optimizer_updates=0, root_proven=False, novelty_proven=False, goal_complete=False)
dest = ROOT / 'outputs/warm_stem_both_seeds_review12_20261011_v1'
dest.mkdir(exist_ok=False)
write(dest / 'report.json', result)
lines = ['预训练Stem＋增强支路：双种子固定12轮审查', '',
         '这是新增残差支路配置，不等于用户原始替换Stem；两个种子的原生终点与各自既有BASE12比较。']
for s, x in seeds.items():
    lines.append('种子%s：AP %.4f→%.4f（%+.4f）；APs %.4f→%.4f（%+.4f）。' % (s, x['BASE12']['AP'], x['warm12']['AP'], x['delta_points']['AP'], x['BASE12']['APs'], x['warm12']['APs'], x['delta_points']['APs']))
lines += ['', '结论：原生双种子门槛未通过。保留种子1正结果和种子2负结果；不延长、不重跑、不挑最好轮。独立EMA复算及正式终点核验仍待完成。',
          '共1230份文件、两个各548张图的阶段数组已核验。两种子各20200个全面积原生PR单元完全重建，最大差值0。', '',
          '现有数字不能支持“增加初始覆盖自然提高APs”：']
for s, x in seeds.items():
    for t in ('0.5', '0.75'):
        o = x['stage'][t]['COCO_small']
        lines.append('种子%s IoU%s：%s；全面积匹配在小GT上的TP净变化%+d。' % (s, t, '；'.join('%s %d→%d（%+d）' % (k, v['baseline'], v['warm'], v['delta']) for k, v in o['supported_GT_by_boundary'].items()), o['native_all_area_TP_net']))
lines += ['', '最小目标子群：输入短边<4像素，5096个GT。种子1/2在IoU0.50的命中净增5/89，在IoU0.75净增8/19；这不是子群AP，也不等于整体APs稳定提高。', '',
          'people类别的全面积原生排序，IoU0.50、分数前1000条：']
for s, x in seeds.items():
    p = x['people50_prefix1000']
    lines.append('种子%s：TP %d→%d（%+d）=新命中%d−丢命中%d＋共同命中GT进入前段%d−离开前段%d。' % (s, p['BASE_TP'], p['Warm_TP'], p['TP_prefix_delta'], p['new_Warm_TP_in_prefix'], p['lost_BASE_TP_in_prefix'], p['common_matched_GT_entering_Warm_prefix'], p['common_matched_GT_leaving_Warm_prefix']))
lines += ['这里保留原生并列分数顺序和匹配；不能把它称为小面积APs的精确前缀解释，也不能单独归咎分类头。', '', '各类别APs变化（百分点）：']
for c1, c2 in zip(seeds['1']['class_APs'], seeds['2']['class_APs']):
    lines.append('%s：种子1 %+.5f；种子2 %+.5f。' % (c1['name'], c1['delta_APs_points'], c2['delta_APs_points']))
lines += ['', '原生小面积AP按IoU变化：']
for s, x in seeds.items():
    lines.append('种子'+s)
    for row in x['native_small_area_IoU_deltas']:
        lines.append('IoU%.2f：AP差值%+.5f；小面积专用匹配TP差值%+d。' % (row['IoU'], row['APs_delta_points'], row['native_small_area_TP_delta']))
lines += ['', '下一步：继续原始P2、SiLU和注意力的双种子固定12轮，完成后用同样的类别、IoU、逐GT及原生PR证据检查收益。当前Warm分析改变的是待验证问题：早期覆盖、最终定位和类别排序必须联合考察；还没有证明唯一根因。',
          '选中300参考框之前的8400位置筛选/回归区别，仍等待既有原生选点诊断。当前未新增基线前向、训练、模型/评估器调用或匹配。']
(dest / '双种子Stem结果与方向审查.txt').write_text('\n'.join(lines)+'\n', encoding='utf8')
write(dest / 'manifest.json', dict(source_sha256=sha(__file__), files={p.name: sha(p) for p in dest.iterdir() if p.is_file()}))
assert 'torch' not in sys.modules
print(json.dumps(dict(status=result['status'], seeds={s: {k: x[k] for k in ('delta_points', 'native_gate_passed', 'exploratory_IoU_range_contributions_to_total_APs_points')} for s, x in seeds.items()}, consistent_classes=consistent_classes), ensure_ascii=False))
