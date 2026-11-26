from pathlib import Path
import datetime, hashlib, json
read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs = read('outputs/selection_collection_deployment_20261011_v1_inputs.json')
obs = {}
for family, seeds in inputs['observations'].items():
    obs[family] = {}
    for seed, item in seeds.items():
        raw = item['payload'].encode('utf8'); t = item['transport']
        assert len(raw) == t['bytes'] and hashlib.sha256(raw).hexdigest() == t['sha256']
        v = json.loads(raw); assert v['seed'] == int(seed); obs[family][seed] = v
for seed in ('1', '2'):
    c = obs['collection'][seed]
    assert c['collector']['same_live'] and c['source_verified'] and c['start'] is not None
    assert c['failure'] is None and c['log_tail'] == '' and not c['packets']
    assert c['live']['phase'] == 'waiting_existing_diagnostics'
    assert obs['deployment'][seed]['collector_same_live'] and obs['deployment'][seed]['code_SHA_verified']
    assert obs['warm'][seed]['source_SHA_verified'] and obs['warm'][seed]['trainer']['same_live'] and not obs['warm'][seed]['failure']
    assert obs['component'][seed]['code_SHA_verified']
    assert obs['queue'][seed]['code_SHA_verified'] and obs['queue'][seed]['queue']['same_live'] and not obs['queue'][seed]['failures']
    assert all(j['completed_epochs'] == 0 for j in obs['queue'][seed]['jobs'].values())

code = Path('英文小论文/code/native_selection_collection12_20261011_v1')
out = Path('outputs/native_selection_collection12_20261011_v1')
manifest = read(out / 'local_code_manifest.json')
assert {x.name: sha(x) for x in code.iterdir() if x.is_file()} == manifest
fixtures_path = out / 'cpu_binding_validation_v1/verification.json'
fixtures = read(fixtures_path)
assert fixtures['status'] == 'PASS_READ_ONLY_COLLECTION_BINDING_AND_ARCHIVE_NAME_FIXTURES'
assert len(fixtures['cases']) == 13 and all(fixtures['cases'].values())
assert fixtures['protocol_sha256'] == sha(code / 'protocol.json') and not fixtures['actual_model_performance_tested']
audit_sources = ['英文小论文/code/stem_component12_20261011_v1/parallel.py',
    '英文小论文/code/native_stem_scratch12_20261011_v3/coordinate.py',
    '英文小论文/code/warm_stem12_20261011_v1/dispatch.py',
    '英文小论文/code/warm_stem12_20261011_v1/finish.py',
    '英文小论文/code/user_backbone12_20261011_v1/dispatch.py',
    '英文小论文/code/user_backbone12_20261011_v1/finish.py',
    '英文小论文/code/native_selection_diagnostic12_20261011_v2/queue.py']
audit = dict(status='SOURCE_REVIEW_OF_EXPLICIT_CLOSEOUT_WAIT_GRAPH',
    source_files_sha256={n: sha(n) for n in audit_sources},
    edges=[['component_closeout', 'full_saved_EMA_closeout'], ['reference_closeout', 'component_closeout'],
           ['warm_closeout', 'component_closeout'], ['warm_closeout', 'reference_closeout'],
           ['user_config_closeout', 'component_closeout'], ['user_config_closeout', 'reference_closeout'],
           ['user_config_closeout', 'warm_closeout'], ['selection_diagnostic', 'user_config_closeout'],
           ['selection_diagnostic', 'component_closeout'], ['selection_diagnostic', 'reference_closeout'],
           ['selection_diagnostic', 'warm_closeout'], ['read_only_collection', 'selection_diagnostic_records']],
    explicit_parent_wait_cycle_observed=False,
    finite_training_and_actual_GPU_idle_still_required=True,
    scheduling_race_freedom_or_future_success_proven=False,
    actual_old_coordinator_states={s: o['old_coordinator_states'] for s, o in obs['deployment'].items()},
    interpretation='The explicit parent-handle waits have no directed cycle. Actual reference and component coordinators were both sleeping/live, not suspended. Training and GPU-idle observations remain necessary; this review is not a proof that concurrent transitions can never race.')
root = Path('outputs/selection_collection_deployment_20261011_v1'); root.mkdir(exist_ok=False)
def write(path, value):
    with Path(path).open('x', encoding='utf8') as f: json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
for family, seeds in inputs['observations'].items():
    for seed, item in seeds.items(): (root / f'{family}_seed{seed}.json').write_bytes(item['payload'].encode('utf8'))
write(root / 'transport.json', {f: {s: i['transport'] for s, i in seeds.items()} for f, seeds in inputs['observations'].items()})
write(root / 'closeout_wait_graph_review.json', audit)
rows = {a: {s: o['jobs'][a]['completed_epochs'] for s, o in obs['component'].items()}
        for a in ('attention_spatial_mean', 'early_final_downsample')}
rows['warm_stem'] = {s: o['completed_epochs'] for s, o in obs['warm'].items()}
for a in ('backbone_silu', 'p2_skip', 'backbone_attention'): rows[a] = {s: o['jobs'][a]['completed_epochs'] for s, o in obs['queue'].items()}
result = dict(status='READ_ONLY_DIAGNOSTIC_COLLECTORS_DEPLOYED_AND_VERIFIED_LIVE_BOTH_SERVERS',
    local_saved_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    remote_observation_UTC={f: {s: v['UTC'] for s, v in seeds.items()} for f, seeds in obs.items()},
    collection_protocol_sha256=sha(code / 'protocol.json'), code_manifest=manifest,
    CPU_binding_and_archive_name_cases=13, CPU_verification_sha256=sha(fixtures_path),
    collectors={s: o['collector'] for s, o in obs['collection'].items()},
    actual_diagnostic_packets_collected=0, actual_training_completed_epochs=rows,
    goal_turn_classification='progress', previous_goal_turn_classification='progress', no_progress_streak=0,
    actual_model_performance_tested_here=False, causal_root_proven=False, novelty_proven=False,
    new_model_forwards_here=0, new_evaluator_calls_here=0, new_optimizer_updates_here=0, new_training_epochs_here=0,
    new_BASE_forwards=0, new_BASE_epochs=0, existing_process_signals=0, frozen_training_or_diagnostic_sources_changed=False,
    pending='Complete authorized fixed12 jobs and existing EMA closeouts; let existing one-shot diagnostics finish. Download positive checksum-bound packets, verify formal terminal bindings and run the prepared six-record consumer.', goal_complete=False)
write(root / 'report.json', result)
lines = ['只读诊断归档部署与实验进度', '',
    '两台服务器的只读归档进程已启动并核实PID/启动时间一致、存活、源码5个文件精确一致、日志无报错。',
    '服务器30310：PID23546；服务器30411：PID25313。都在等待已有诊断，实际归档0组。',
    '只归档通过精确EMA回放的记录，校验权重指纹、12轮完成记录、548张图数组清单、原生指标和GT一致标志。逐文件保存字节数与SHA256，下载后再核验；失败保留部分有效结果，不重试模型。',
    '13项CPU绑定和压缩包路径检查通过，这些检查不是精度实验。', '',
    '当前已完成轮数（30310 / 30411）：',
    f"Stem空间权重消融：{rows['attention_spatial_mean']['1']}/12 / {rows['attention_spatial_mean']['2']}/12。",
    f"Stem下采样时机消融：{rows['early_final_downsample']['1']}/12 / {rows['early_final_downsample']['2']}/12。",
    f"保留预训练Stem并联增强：{rows['warm_stem']['1']}/12 / {rows['warm_stem']['2']}/12。",
    'SiLU、原主干注意力、P2三项独立配置：两端均0/12，队列存活，等待容量。', '',
    '检查了现有收尾父进程的明确等待关系，没有发现相互等待的循环；此静态审查不能保证未来并发切换绝不出现竞态。原始协调器实际处于运行中的等待状态，未对它们发信号。',
    '本轮未新增模型前向、评估器调用、训练更新或基线训练。继续以实际固定第12轮EMA和两个种子判定。']
(root / '诊断归档部署与实验进度.txt').write_text('\n'.join(lines) + '\n', encoding='utf8')
evidence = '../selection_collection_deployment_20261011_v1/report.json'
def update(old, new, changes):
    value = read(old); value.update(changes); value['supersedes'] = Path(old).name; value['goal_complete'] = False
    write(new, value)
write(out / 'execution_state_v1.json', dict(status=result['status'], evidence=evidence,
    collectors=result['collectors'], actual_packets_collected=0, CPU_validation='cpu_binding_validation_v1/verification.json',
    wait_graph_review='../selection_collection_deployment_20261011_v1/closeout_wait_graph_review.json', goal_complete=False))
update('outputs/native_selection_diagnostic12_20261011_v2/execution_state_v2.json',
    'outputs/native_selection_diagnostic12_20261011_v2/execution_state_v3.json',
    dict(read_only_collection_state='../native_selection_collection12_20261011_v1/execution_state_v1.json',
         actual_diagnostic_packets_collected=0, existing_training_process_signals=0))
update('outputs/warm_stem12_20261011_v1/execution_state_v7.json',
    'outputs/warm_stem12_20261011_v1/execution_state_v8.json',
    dict(evidence=evidence, actual_new_training_epochs=rows['warm_stem'], trainers={s: o['trainer'] for s, o in obs['warm'].items()}))
update('outputs/user_backbone12_20261011_v1/execution_state_v3.json',
    'outputs/user_backbone12_20261011_v1/execution_state_v4.json',
    dict(evidence=evidence, queue_processes={s: o['queue'] for s, o in obs['queue'].items()},
         last_remote_observation_UTC={s: o['UTC'] for s, o in obs['queue'].items()},
         GPU_resource={s: o['live']['resource'] for s, o in obs['queue'].items()}))
update('outputs/stem_component12_20261011_v1/execution_state_v26.json',
    'outputs/stem_component12_20261011_v1/execution_state_v27.json',
    dict(current_goal_turn='progress', no_progress_streak=0, latest_compact_status=evidence,
         actual_component_completed_epochs={a: rows[a] for a in ('attention_spatial_mean', 'early_final_downsample')},
         warm_candidate_state='../warm_stem12_20261011_v1/execution_state_v8.json',
         standalone_user_execution='../user_backbone12_20261011_v1/execution_state_v4.json',
         actual_native_selection_diagnostic_state='../native_selection_diagnostic12_20261011_v2/execution_state_v3.json',
         read_only_diagnostic_collection_state='../native_selection_collection12_20261011_v1/execution_state_v1.json'))
update('outputs/original_stem12_20261011_v1/execution_state_v29.json',
    'outputs/original_stem12_20261011_v1/execution_state_v30.json',
    dict(authoritative_state='../stem_component12_20261011_v1/execution_state_v27.json'))
print(json.dumps(dict(status=result['status'], actual_completed_epochs=rows, actual_packets=0, BASE_retrained=False)))
