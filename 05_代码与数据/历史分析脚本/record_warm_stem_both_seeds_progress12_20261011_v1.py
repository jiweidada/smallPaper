"""Persist verified current observations and advance immutable goal ledgers after both native endpoints."""
from pathlib import Path
import datetime, hashlib, json

ROOT = Path.cwd()
read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, o):
    with Path(p).open('x', encoding='utf8') as f:
        json.dump(o, f, ensure_ascii=False, indent=2, allow_nan=False)

inputs_path = ROOT / 'outputs/warm_stem_both_seeds_progress12_20261011_v1_inputs.json'
inputs = read(inputs_path)
obs = {}
for family, seeds in inputs['observations'].items():
    obs[family] = {}
    for seed, v in seeds.items():
        raw = v['payload'].encode()
        assert len(raw) == v['transport']['bytes'] and hashlib.sha256(raw).hexdigest() == v['transport']['sha256']
        obs[family][seed] = json.loads(raw)
for seed in ('1', '2'):
    w, u, d = obs['warm'][seed], obs['user'][seed], obs['diagnostic'][seed]
    assert w['source_SHA_verified'] and w['failure'] is None and w['completed_epochs'] == 12
    assert not w['trainer']['same_live'] and w['controller_completion_status'] == 'COMPLETE_FIXED12'
    assert w['native_training_completion']['identity_equals_receipt']
    assert u['code_SHA_verified'] and u['queue']['same_live'] and not u['failures']
    for arm, j in u['jobs'].items():
        assert 0 <= j['completed_epochs'] <= 12 and j['failure'] is None
        assert j['trainer']['same_live'] and j['controller']['same_live'] and j['controller']['identity_equals_receipt']
        assert j['full_preflight']['status'] == 'PASS_USER_CONFIG_FULL_AMP_LOSS_BACKWARD_NO_UPDATES'
        assert j['full_preflight']['identity_matches_receipt'] and j['first100']['all100_rows_exact']
        assert j['runtime_start']['identity_equals_receipt'] and j['runtime_start']['optimizer_coverage']['all_parameters_exactly_once']
    assert d['source_verified'] and d['queue']['same_live'] and d['failure'] is None
    assert d['live']['phase'] == 'waiting_actual_predecessor_handles_and_GPU_idle' and not d['actual_diagnostic_children']

review_root = ROOT / 'outputs/warm_stem_both_seeds_review12_20261011_v1'
for name, h in read(review_root / 'manifest.json')['files'].items():
    assert sha(review_root / name) == h
review = read(review_root / 'report.json')
assert review['status'] == 'BOTH_WARM_PRIMARY_FIXED12_VERIFIED_CROSS_SEED_REVERSALS_RETAINED_NATIVE_GATE_FAIL'
assert not review['two_seed_native_gate_passed'] and review['primary_files_verified_total'] == 1230
assert review['native_all_area_PR_cells_reproduced_total'] == 40400
collection = ROOT / 'outputs/warm_stem_primary_collection12_20261011_v1'
verifications = {}
for seed in ('1', '2'):
    path = collection / f'seed{seed}_warm_stem_v1/primary_verification_v1.json'
    v = read(path)
    assert v['status'] == 'PASS_WARM_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS'
    assert sha(path) == review['seeds'][seed]['proof']['primary_verification_sha256']
    assert not v['saved_EMA_independent_replay_verified'] and not v['formal_terminal_reader_passed']
    verifications[seed] = dict(path=f'seed{seed}_warm_stem_v1/primary_verification_v1.json', sha256=sha(path),
        files_verified=v['files_verified'], stage_images=v['actual_stage_images'], last_checkpoint_sha256=v['last_checkpoint_sha256'])
for f in ('remote_collection_execution_v1.json', 'remote_collection_metadata_v1.json', 'remote_waiter_completion_v1.json'):
    assert (collection / 'seed1_warm_stem_v1' / f).exists()

counts = {'warm_stem': {s: w['completed_epochs'] for s, w in obs['warm'].items()}}
jobs = {}
for seed, o in obs['user'].items():
    jobs[seed] = {}
    for arm, j in o['jobs'].items():
        counts.setdefault(arm, {})[seed] = j['completed_epochs']
        stats = j['latest_stats']
        jobs[seed][arm] = dict(j, state='training', AP_points=stats[0]*100 if stats else None,
            APs_points=stats[3]*100 if stats else None, compared_to_cached_BASE12=False, fixed12_efficacy_not_claimed=True)
dest = ROOT / 'outputs/warm_stem_both_seeds_progress12_20261011_v1'
dest.mkdir(exist_ok=False)
for family, seeds in inputs['observations'].items():
    for seed, v in seeds.items():
        (dest / f'{family}_seed{seed}.json').write_bytes(v['payload'].encode())
write(dest / 'observation_transports.json', {f: {s: v['transport'] for s, v in seeds.items()} for f, seeds in inputs['observations'].items()})
report = dict(status='BOTH_WARM_PRIMARY12_ANALYZED_NATIVE_GATE_FAIL_ALL_ORIGINAL_USER_JOBS_ACTUALLY_TRAINING',
    local_saved_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    remote_observation_UTC={f: {s: o['UTC'] for s, o in seeds.items()} for f, seeds in obs.items()},
    previous_goal_turn='progress', current_goal_turn='progress', no_progress_streak=0, goal_complete=False,
    actual_completed_epochs=counts, user_jobs=jobs,
    frozen_Warm26_user28_diagnostic6_sources_verified_both_servers=True,
    Warm_primary_packets_downloaded_and_verified=2, primary_files_verified_total=1230, saved_stage_images_per_seed=548,
    native_all_area_PR_cells_reproduced_total=40400, native_all_area_PR_max_abs_difference=0,
    native_Warm_fixed12={s: {k: r[k] for k in ('BASE12', 'warm12', 'delta_points', 'native_gate_passed')} for s, r in review['seeds'].items()},
    two_seed_native_gate_passed=False, independent_saved_EMA_replay_pending=True, formal_terminal_readers_pending=True,
    both_seed_scientific_review='../warm_stem_both_seeds_review12_20261011_v1/report.json',
    review_manifest_sha256=sha(review_root / 'manifest.json'), primary_verifications=verifications,
    changed_next_action=review['changed_next_action'], rejected_explanations=review['rejected_explanations'],
    exact_all_area_score_prefix_accounting_available=True, exact_small_area_APs_score_prefix_accounting_available=False,
    actual_native_location_waiters_verified_live=True, actual_diagnostic_model_children_started=0,
    new_BASE_training_epochs=0, new_BASE_forwards=0, new_model_forwards_for_download_and_analysis=0,
    new_evaluator_calls_for_download_and_analysis=0, new_matching_calls_for_download_and_analysis=0,
    new_optimizer_updates_for_download_and_analysis=0, new_training_epochs_for_download_and_analysis=0,
    existing_process_signals_this_turn=0, configuration_training_retry=False, root_proven=False, novelty_proven=False,
    orchestration_notes='Two local observer helper evaluations lacked enc/dec bindings and failed before remote command dispatch; subsequently invoked with required bindings. The finished primary reader session was already closed; its saved PASS output was verified. No repeated remote scientific execution.')
write(dest / 'report.json', report)
lines = ['本轮实际工作与远程实验进度', '',
    '已取回种子1的188997132字节训练产物，严格读取器审查615份文件和548张图通过。两种子均已有完整原生固定12轮产物。',
    '种子1：AP 13.9276 / APs 8.6675；相对其基线 +0.1502 / +0.2682。',
    '种子2：AP 13.6041 / APs 8.4911；相对其基线 -0.1072 / -0.1458。',
    '原生双种子门槛未通过；这条Warm配置不延长或重跑。独立EMA核验与正式终点读取仍在等已有GPU任务结束。', '',
    '新增逐目标分析：种子1在IoU0.50的初始小目标几何覆盖减少135，而APs上涨；种子2初始覆盖增加573，而APs下降。覆盖数量不是AP的可靠替代指标。',
    '精确原生全面积PR重建：两种子共40400个单元，最大误差0。people分数前1000条、IoU0.50真阳性：种子1 562→624，种子2 597→573。它是全面积匹配，不能冒充小面积APs前缀。',
    '全部原始注意力、SiLU与P2配置都已实际启动，完整模型AMP/损失反传、参数覆盖及前100批输入对照通过；冻结源码保持一致。', '', '远程观察时刻与实际完成轮数：']
for seed, o in obs['user'].items():
    lines.append('服务器种子%s：%s' % (seed, o['UTC']))
    for arm, j in jobs[seed].items():
        digits = '尚无完整轮验证值' if j['AP_points'] is None else '本轮AP %.4f / APs %.4f' % (j['AP_points'], j['APs_points'])
        lines.append('%s：%d/12；%s。' % (arm, j['completed_epochs'], digits))
lines += ['', '未完成配置的数字是当前轮结果，不作为固定12轮有效性结论。原始P2是P2到P3的支路注入，并非新增第四编码器层；注意力更换含结构/初始化差异，不能称纯注意力因果效应。',
    '两个原生选点诊断队列PID和启动时间再次核对仍存活；均等待已拥有的GPU任务结束，未增加基线前向或诊断训练。',
    '下一步根据原始三项的双种子固定终点继续验证改进；现有结果改变待验证问题，尚不能认定唯一错误模块或创新。目标继续。']
(dest / '本轮实验与审查进度.txt').write_text('\n'.join(lines)+'\n', encoding='utf8')
write(dest / 'manifest.json', dict(files={p.name: sha(p) for p in dest.iterdir() if p.is_file()},
    source_sha256=sha(__file__), input_sha256=sha(inputs_path)))
evidence = '../warm_stem_both_seeds_progress12_20261011_v1/report.json'

def advance(directory, version, changes):
    folder = ROOT / 'outputs' / directory
    old = folder / f'execution_state_v{version}.json'
    value = read(old)
    value.update(changes)
    value.update(supersedes=old.name, goal_complete=False)
    write(folder / f'execution_state_v{version+1}.json', value)

advance('warm_stem_primary_collection12_20261011_v1', 1, dict(status='BOTH_WARM_NATIVE_PRIMARY12_PACKETS1230_FILES_VERIFIED_AND_ANALYZED',
    evidence=evidence, completed_seeds=[1, 2], seed1_pending=False, independent_saved_EMA_pending=True,
    formal_terminal_reader_pending=True, primary_verifications=verifications, files_verified_total=1230,
    both_seed_scientific_review='../warm_stem_both_seeds_review12_20261011_v1/report.json'))
advance('warm_stem12_20261011_v1', 16, dict(status='BOTH_PRIMARY_FIXED12_COMPLETE_NATIVE_GATE_FAIL_REPLAY_PENDING', evidence=evidence,
    previous_goal_turn='progress', current_goal_turn='progress', no_progress_streak=0,
    actual_new_training_epochs=counts['warm_stem'], trainers={s: w['trainer'] for s, w in obs['warm'].items()},
    phases={s: w['live']['phase'] for s, w in obs['warm'].items()}, last_remote_observation_UTC={s: w['UTC'] for s, w in obs['warm'].items()},
    native_training_completions={s: w['native_training_completion'] for s, w in obs['warm'].items()},
    primary_packets_state='../warm_stem_primary_collection12_20261011_v1/execution_state_v2.json',
    both_seed_scientific_review='../warm_stem_both_seeds_review12_20261011_v1/report.json',
    both_seed_native_fixed12=report['native_Warm_fixed12'], two_seed_native_gate_passed=False,
    AP_improvement_proven=False, root_or_novelty_proven=False,
    pending='Both fixed12 native primary endpoints complete; preserve opposite signs, no extension/retry; await owned independent EMA/formal closeouts.'))
advance('user_backbone12_20261011_v1', 12, dict(status='ALL_ORIGINAL_USER_CONFIGS_FULL_PREFLIGHT_PASSED_ACTUALLY_TRAINING_BOTH_SERVERS',
    evidence=evidence, saved_local_UTC=report['local_saved_UTC'], jobs=jobs,
    completed_epochs={s: {a: j['completed_epochs'] for a, j in js.items()} for s, js in jobs.items()},
    last_remote_observation_UTC={s: u['UTC'] for s, u in obs['user'].items()}, queue_processes={s: u['queue'] for s, u in obs['user'].items()},
    queue_phases={s: u['live']['phase'] for s, u in obs['user'].items()}, GPU_resource={s: u['live'].get('resource') for s, u in obs['user'].items()},
    full_GPU_preflight_passed_by_seed={s: {a: True for a in js} for s, js in jobs.items()},
    full_GPU_preflight_and_training_pending=False, remaining_GPU_preflights_and_trainings_pending={'1': [], '2': []},
    standalone_attention_seed1_training_started=True, standalone_attention_seed2_training_started=True,
    queued_configurations={'1': [], '2': []}, fixed12_efficacy_not_yet_claimed=True))
advance('native_selection_diagnostic12_20261011_v2', 5, dict(evidence=evidence,
    last_actual_process_observation_UTC={s: d['UTC'] for s, d in obs['diagnostic'].items()},
    actual_live_waiters={s: d['queue'] for s, d in obs['diagnostic'].items()}, actual_diagnostic_children_started=0,
    phase='waiting_actual_predecessor_handles_and_GPU_idle', source6_verified_both_servers=True))
advance('stem_component12_20261011_v1', 35, dict(latest_compact_status=evidence, previous_goal_turn='progress',
    current_goal_turn='progress', no_progress_streak=0, warm_candidate_state='../warm_stem12_20261011_v1/execution_state_v17.json',
    standalone_user_execution='../user_backbone12_20261011_v1/execution_state_v13.json',
    standalone_user_completed_epochs={s: {a: j['completed_epochs'] for a, j in js.items()} for s, js in jobs.items()},
    standalone_user_queue_phases={s: u['live']['phase'] for s, u in obs['user'].items()},
    actual_native_selection_diagnostic_state='../native_selection_diagnostic12_20261011_v2/execution_state_v6.json',
    warm_both_seed_primary_review='../warm_stem_both_seeds_review12_20261011_v1/report.json',
    warm_both_seed_primary_collection='../warm_stem_primary_collection12_20261011_v1/execution_state_v2.json',
    warm_both_seed_native_fixed12=report['native_Warm_fixed12'], Warm_two_seed_native_gate_passed=False,
    exact_native_all_area_prefixes_reconstructed_both_seeds=True, root_or_novelty_proven=False,
    remote_result_observers_verified_live_both_seeds=True,
    remote_result_observer_liveness_scope='Native diagnostic waiter PID/start ticks observed live now on both hosts; separate collection waiter liveness not asserted.',
    existing_process_signals_this_turn=0, old_training_process_signals_this_turn=0,
    interpretation='Both Warm fixed12 primary endpoints and615 files each verified. AP/APs deltas seed1 +0.15024/+0.26822, seed2 -0.10723/-0.14576, two-seed native gate FAIL. Small IoU0.50 early coverage delta -135/+573 reverses endpoint APs signs; no uniform coverage explanation. Exact all-area PR cells40400 and GT prefixes retain people gains62/loss24 at prefix1000. All original user attention/SiLU/P2 configs now train both servers, at0/5/5 and4/9/9 respectively; fixed12 outcome pending. Independent EMA/formal terminal and owned native-location diagnostics await current jobs.'))
advance('original_stem12_20261011_v1', 38, dict(authoritative_state='../stem_component12_20261011_v1/execution_state_v36.json'))
print(json.dumps(dict(status=report['status'], counts=counts, files_verified=1230, native_PR_cells=40400, goal_complete=False), ensure_ascii=False))
