from pathlib import Path
import datetime, hashlib, json

inputs = json.loads(Path('outputs/stem_fixed12_and_selection_progress_20261011_v1_inputs.json').read_bytes())
root = Path('outputs/stem_fixed12_and_selection_progress_20261011_v1')
assert not root.exists()
read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
obs = {}
for family, seeds in inputs['observations'].items():
    obs[family] = {}
    for seed, item in seeds.items():
        raw = item['payload'].encode('utf8'); m = item['transport']
        assert len(raw) == m.get('bytes', m.get('expected_bytes'))
        assert hashlib.sha256(raw).hexdigest() == m.get('sha256', m.get('expected_sha256'))
        value = json.loads(raw); assert value['seed'] == int(seed)
        obs[family][seed] = value
for seed in ('1', '2'):
    d = obs['diagnostic'][seed]
    assert d['source_verified'] and d['queue']['same_live'] and d['failure'] is None and d['log_tail'] == ''
    r = obs['reference'][seed]
    assert r['native_training_completion'] == 'NATIVE_FIT_FINISHED_EXACT12'
    assert r['completed_epochs'] == 12 and r['entered_epochs'] == list(range(12))
    assert r['candidate_exit'] == 0 and r['identity_equals_receipt']
    assert obs['component'][seed]['code_SHA_verified']
    assert obs['warm'][seed]['source_SHA_verified'] and not obs['warm'][seed]['failure']
    assert obs['queue'][seed]['code_SHA_verified'] and not obs['queue'][seed]['failures']

fixture_path = Path('outputs/native_selection_diagnostic12_20261011_v2/parent_binding_cpu_validation_v2/verification.json')
fixture = read(fixture_path)
assert fixture['status'] == 'PASS_ALL_THREE_JSON_BINDINGS_AND_NEGATIVE_GUARD_CASES'
assert len(fixture['cases']) == 7 and fixture['model_forwards'] == 0
diagnostic_code = Path('英文小论文/code/native_selection_diagnostic12_20261011_v2')
manifest_path = Path('outputs/native_selection_diagnostic12_20261011_v2/local_code_manifest.json')
manifest = read(manifest_path)
assert {p.name: sha(p) for p in diagnostic_code.iterdir() if p.is_file()} == manifest
assert sha(diagnostic_code / 'queue.py') == fixture['queue_sha256']
rows = []
full_sources = {}
for seed in (1, 2):
    packet_root = Path(f'outputs/stem_primary_collection_20261011_v1/seed{seed}_original_stem_v1')
    packet = packet_root / 'extracted'
    verification = read(packet_root / 'primary_verification_v1.json')
    assert verification['status'] == 'PASS_DOWNLOADED_EXACT12_PRIMARY_ARTIFACTS_ONLY'
    done = read(packet / 'run/training_completion.json')
    assert done['status'] == 'NATIVE_FIT_FINISHED_EXACT12' and done['completed_epochs'] == 12
    assert done['entered_epochs'] == list(range(12))
    endpoint = packet / 'run/endpoint_epoch12'
    assert sha(endpoint / 'manifest.json') == done['endpoint_manifest_sha256']
    for name, digest in read(endpoint / 'manifest.json')['files'].items():
        assert sha(endpoint / name) == digest
    summary = read(endpoint / 'summary.json')
    assert summary['identity'] == done['identity'] == verification['identity']
    assert summary['epoch_zero_based'] == 11 and summary['native_stats'] == verification['native_stats']
    full = summary['native_stats']; ref = obs['reference'][str(seed)]['native_fixed12_stats']
    rows.append(dict(seed=seed, random_native_AP=100 * ref[0], random_native_APs=100 * ref[3],
        full_enhanced_AP=100 * full[0], full_enhanced_APs=100 * full[3],
        full_minus_random_AP_points=100 * (full[0] - ref[0]),
        full_minus_random_APs_points=100 * (full[3] - ref[3])))
    full_sources[str(seed)] = dict(packet=str(packet), endpoint_manifest_sha256=sha(endpoint / 'manifest.json'),
        native_completion_sha256=sha(packet / 'run/training_completion.json'),
        primary_verification_sha256=sha(packet_root / 'primary_verification_v1.json'))

root.mkdir()
def write(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
for family, seeds in inputs['observations'].items():
    for seed, item in seeds.items():
        (root / f'{family}_seed{seed}.json').write_bytes(item['payload'].encode('utf8'))
write(root / 'transport.json', {f: {s: i['transport'] for s, i in seeds.items()} for f, seeds in inputs['observations'].items()})
result = dict(status='BOTH_FIXED12_NATIVE_PRIMARY_REFERENCES_COMPLETE_DIAGNOSTIC_QUEUES_DEPLOYED',
    local_saved_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), rows=rows,
    full_sources=full_sources, primary_only_independent_saved_EMA_verification_pending=True,
    full_over_random_APs_positive_both_seeds=all(r['full_minus_random_APs_points'] > 0 for r in rows),
    full_over_random_AP_nonnegative_both_seeds=all(r['full_minus_random_AP_points'] >= 0 for r in rows),
    full_over_pretrained_BASE12_gate_passed=False,
    interpretation='Small-target primary APs increased over the random native Stem in both seeds, but total AP decreased in seed1. No stable total-AP gain, root cause or novel mechanism is established. These reference results cannot replace the cached pretrained BASE12 acceptance gate.',
    diagnostic_code_manifest=manifest, diagnostic_CPU_binding_validation_sha256=sha(fixture_path),
    diagnostic_actual_saved_EMA_replays=0,
    diagnostic_job_arms_per_seed=['original_stem', 'native_stem_scratch', 'highfreq_rgb'],
    pending_after_replays='Prepare the six-record offline consumer with v2 identity binding; verify all native stats/GT/stage arrays, then compare spatial-neighborhood ranks and selected-box geometry without treating anchor priors as8400 predicted boxes.',
    new_BASE_forwards=0, new_training_epochs=0, new_optimizer_updates=0,
    frozen_training_source_changed=False, existing_training_process_signals=0, goal_complete=False)
write(root / 'report.json', result)
lines = ['Stem第12轮原生记录对比与选择前诊断进度（2026-10-11）', '',
    '两个随机原始Stem对照都已正常完成12轮。以下为固定第12轮EMA原生评估主记录；独立保存权重复核仍待GPU空闲。',
    'seed | 随机原始Stem AP/APs | 完整增强Stem AP/APs | 增强−随机 AP/APs（点）']
for r in rows:
    lines.append(f"{r['seed']} | {r['random_native_AP']:.4f}/{r['random_native_APs']:.4f} | {r['full_enhanced_AP']:.4f}/{r['full_enhanced_APs']:.4f} | {r['full_minus_random_AP_points']:+.4f}/{r['full_minus_random_APs_points']:+.4f}")
lines += ['', '这组结果支持小目标APs改善的方向，但总体AP在seed1下降、seed2上升，不能宣称稳定提高总体精度。它也没有通过与保留预训练Stem的已有基线比较的正式筛选规则。', '',
    '诊断新增记录：选择前8400个位置的类别分数、真实TopK选中索引、对应位置先验以及原本已计算的300个框偏移。位置先验并非8400个预测框，附近未选中也不能直接等同于目标不可能被检测。',
    '三种配置（完整增强Stem、随机原始Stem、高频分支RGB对照）各两个种子的诊断队列已部署，核实PID及启动时间一致且进程存活。当前尚无实际诊断回放；先等待现有训练和原生保存EMA复核完成、GPU空闲。每组用独立进程，零训练更新，零基线前向。',
    '修正了只校验第一组和进程内运行代码缓存的问题。真实父入口通过三组独立元数据夹具及六项拒绝测试；这些工程测试没有验证模型精度。', '',
    '当前训练：']
for arm, title in [('attention_spatial_mean', 'Stem空间权重对照'), ('early_final_downsample', 'Stem下采样时机对照')]:
    lines.append(f"{title}：{obs['component']['1']['jobs'][arm]['completed_epochs']}/12、{obs['component']['2']['jobs'][arm]['completed_epochs']}/12。")
lines.append(f"保留原Stem并联增强候选：{obs['warm']['1']['completed_epochs']}/12、{obs['warm']['2']['completed_epochs']}/12。")
lines.append('原主干注意力、SiLU、P2三项独立配置：两端仍为0/12，执行队列已核实存活，等待显存容量满足。')
(root / '第12轮对比与诊断进度.txt').write_text('\n'.join(lines) + '\n', encoding='utf8')

def update(old, new, changes):
    value = read(old); value.update(changes)
    value['supersedes'] = Path(old).name; value['goal_complete'] = False
    write(new, value)
evidence = '../stem_fixed12_and_selection_progress_20261011_v1'
write('outputs/native_selection_diagnostic12_20261011_v2/execution_state_v1.json',
    dict(status='BOTH_FRESH_DIAGNOSTIC_QUEUES_LIVE_WAITING_PREDECESSORS', evidence=evidence + '/report.json',
    CPU_validation='parent_binding_cpu_validation_v2/verification.json',
    actual_diagnostic_replays=0, existing_training_process_signals=0, baseline_forwards=0, training_epochs=0,
    queues={s: v['queue'] for s, v in obs['diagnostic'].items()}, goal_complete=False))
update('outputs/native_stem_scratch12_20261011_v3/execution_state_v12.json',
    'outputs/native_stem_scratch12_20261011_v3/execution_state_v13.json',
    dict(evidence=evidence + '/reference_seed1/2.json', fixed12_primary_comparison=evidence + '/report.json',
    positive_exact12_native_completion_both_seeds=True, independent_saved_EMA_reload_pending=True))
update('outputs/warm_stem12_20261011_v1/execution_state_v5.json',
    'outputs/warm_stem12_20261011_v1/execution_state_v6.json',
    dict(evidence=evidence + '/warm_seed1/2.json', actual_new_training_epochs={s: v['completed_epochs'] for s, v in obs['warm'].items()}))
update('outputs/stem_component12_20261011_v1/execution_state_v24.json',
    'outputs/stem_component12_20261011_v1/execution_state_v25.json',
    dict(current_goal_turn='progress', no_progress_streak=0,
    reference_state='../native_stem_scratch12_20261011_v3/execution_state_v13.json',
    warm_candidate_state='../warm_stem12_20261011_v1/execution_state_v6.json',
    latest_compact_status=evidence + '/component_seed1/2.json',
    fixed12_native_primary_full_vs_random=evidence + '/report.json',
    actual_native_selection_diagnostic_state='../native_selection_diagnostic12_20261011_v2/execution_state_v1.json',
    stable_total_AP_gain_over_random_stem_proven=False))
update('outputs/original_stem12_20261011_v1/execution_state_v27.json',
    'outputs/original_stem12_20261011_v1/execution_state_v28.json',
    dict(authoritative_state='../stem_component12_20261011_v1/execution_state_v25.json'))
print(json.dumps(dict(status=result['status'], rows=rows, actual_diagnostic_replays=0, BASE_retrained=False)))
