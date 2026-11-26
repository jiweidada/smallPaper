from pathlib import Path
import datetime, hashlib, json

read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
root = Path('outputs/selection_consumer_preparation_20261011_v1')
assert not root.exists()
inputs = read('outputs/selection_consumer_preparation_20261011_v1_inputs.json')
obs = {}
for family, seeds in inputs['observations'].items():
    obs[family] = {}
    for seed, item in seeds.items():
        raw = item['payload'].encode('utf8'); t = item['transport']
        assert len(raw) == t['bytes'] and hashlib.sha256(raw).hexdigest() == t['sha256']
        value = json.loads(raw); assert value['seed'] == int(seed)
        obs[family][seed] = value
for seed in ('1', '2'):
    assert obs['queue'][seed]['code_SHA_verified'] and obs['queue'][seed]['queue']['same_live'] and not obs['queue'][seed]['failures']
    assert obs['diagnostic'][seed]['source_verified'] and obs['diagnostic'][seed]['queue']['same_live']
    assert obs['diagnostic'][seed]['failure'] is None and not obs['diagnostic'][seed]['actual_diagnostic_children']
    assert obs['warm'][seed]['source_SHA_verified'] and obs['warm'][seed]['trainer']['same_live'] and not obs['warm'][seed]['failure']
    assert obs['component'][seed]['code_SHA_verified'] and obs['monitor'][seed]['monitor_same_live']
    for name in ('backbone_silu', 'p2_skip', 'backbone_attention'):
        assert obs['queue'][seed]['jobs'][name]['completed_epochs'] == 0

consumer = Path('英文小论文/code/native_selection_analysis12_20261011_v2')
protocol = read(consumer / 'protocol.json')
for name, digest in protocol['files'].items(): assert sha(consumer / name) == digest
for name, digest in protocol['dependencies'].items(): assert sha(name) == digest
fixtures_path = Path('outputs/native_selection_analysis12_20261011_v2/cpu_binding_validation_v1/verification.json')
fixtures = read(fixtures_path)
assert fixtures['status'] == 'PASS_SIX_RECORD_BINDING_AND_NATIVE_AREA_BOUNDARY_FIXTURES'
assert len(fixtures['fixtures']) == 14 and all(fixtures['fixtures'].values())
assert not fixtures['actual_model_performance_tested'] and not fixtures['actual_six_record_evidence_consumed']
cached_path = Path('outputs/native_selection_analysis12_20261011_v2/cached_source_validation_v1.json')
cached = read(cached_path)
assert cached['status'] == 'PASS_BOTH_CACHED_BASE_SOURCES_AND_NATIVE_SMALL_AREA_BOUNDARY'
assert fixtures['protocol_sha256'] == cached['consumer_protocol_sha256'] == sha(consumer / 'protocol.json')
assert all(v['GT'] == 38759 and v['COCO_small'] == 26586 and v['area_exactly_1024'] == 0 for v in cached['seeds'].values())

root.mkdir()
def write(path, value):
    with Path(path).open('x', encoding='utf8') as f: json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
for family, seeds in inputs['observations'].items():
    for seed, item in seeds.items(): (root / f'{family}_seed{seed}.json').write_bytes(item['payload'].encode('utf8'))
write(root / 'transport.json', {f: {s: i['transport'] for s, i in seeds.items()} for f, seeds in inputs['observations'].items()})
rows = {arm: {s: obs['component'][s]['jobs'][arm]['completed_epochs'] for s in ('1', '2')}
        for arm in ('attention_spatial_mean', 'early_final_downsample')}
rows['warm_stem'] = {s: obs['warm'][s]['completed_epochs'] for s in ('1', '2')}
for arm in ('backbone_silu', 'p2_skip', 'backbone_attention'): rows[arm] = {s: obs['queue'][s]['jobs'][arm]['completed_epochs'] for s in ('1', '2')}
result = dict(status='SIX_RECORD_OFFLINE_CONSUMER_PREPARED_ACTUAL_REPLAYS_PENDING',
    saved_local_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    remote_observation_UTC={f: {s: v['UTC'] for s, v in seeds.items()} for f, seeds in obs.items()},
    consumer_protocol_sha256=sha(consumer / 'protocol.json'), consumer_source_files=protocol['files'],
    CPU_binding_verification_sha256=sha(fixtures_path), CPU_binding_cases=14,
    cached_source_verification_sha256=sha(cached_path), cached_BASE_reused_without_forwards=True,
    actual_six_diagnostic_records_consumed=0, actual_model_performance_tested_here=False,
    actual_training_completed_epochs=rows,
    queue_phases={s: v['live']['phase'] for s, v in obs['queue'].items()},
    diagnostic_phases={s: v['live']['phase'] for s, v in obs['diagnostic'].items()},
    GPU_resource={s: v['live']['resource'] for s, v in obs['queue'].items()},
    area_boundary_review='Native small area is inclusive at1024; current annotations have zero GT exactly on that boundary. Updated consumer convention changes no GT grouping on this dataset.',
    engineering_invocation_issue='Initial CPU invocation reached output creation after checks but its parent folder did not exist. Created that local parent and reran unchanged frozen CPU source; no model or remote training retry.',
    original_four_record_consumer_modified=False, frozen_remote_training_sources_modified=False,
    existing_remote_process_signals=0, new_BASE_epochs=0, new_BASE_forwards=0,
    new_model_forwards_here=0, new_optimizer_updates_here=0, new_training_epochs_here=0,
    causal_root_proven=False, novelty_proven=False, goal_complete=False)
write(root / 'report.json', result)
lines = ['六组选择诊断分析准备与远程进度', '',
    '审查后部署的SiLU、原主干注意力、历史P2桥接三项配置，各两个种子：目前均0/12，尚未开始训练。两端队列及结果观察进程存活。',
    '两张24GB GPU各有3个训练进程。新增一个任务后的预留峰值为27136MiB，超过24576MiB；当前约10.6GB空闲不等于后续多尺度训练峰值容量。原队列会在现有任务正常结束、容量满足时启动。', '',
    '配置 | 服务器30310 / 30411 已完成轮数']
titles = dict(attention_spatial_mean='Stem空间权重消融', early_final_downsample='Stem下采样时机消融',
              warm_stem='保留预训练Stem并联增强分支', backbone_silu='SiLU替换', p2_skip='原P2桥接', backbone_attention='原主干注意力方案')
for arm, values in rows.items(): lines.append(f"{titles[arm]} | {values['1']}/12 / {values['2']}/12")
lines += ['', '新增离线分析程序：完整增强Stem、随机原始Stem、RGB对照，共六组，逐组核验正式终点报告与独立EMA回放，通过后才比较真实TopK选点、附近类别分数和已选300个框的几何变化。',
    '14项CPU绑定检查通过；已有两种子基线缓存校验通过，未重跑基线模型。当前实际六组诊断回放仍为0，不能把程序检查当成精度验证。',
    '小目标面积上界按原评估器含1024处理。数据里面积恰好1024的GT为0，因此本次约定修正不会改变该数据的分组或AP。',
    '仍需完成训练、原生独立EMA复核、诊断回放和结果下载，再运行此分析。不能凭关联宣称Stem根因或创新成立。']
(root / '远程进度与诊断分析准备.txt').write_text('\n'.join(lines) + '\n', encoding='utf8')
evidence = '../selection_consumer_preparation_20261011_v1/report.json'
def update(old, new, changes):
    value = read(old); value.update(changes); value['supersedes'] = Path(old).name; value['goal_complete'] = False
    write(new, value)
write('outputs/native_selection_analysis12_20261011_v2/execution_state_v1.json',
    dict(status=result['status'], evidence=evidence, consumer_protocol_sha256=result['consumer_protocol_sha256'],
         CPU_binding_cases=14, actual_diagnostic_records_consumed=0, actual_model_performance_tested=False, goal_complete=False))
update('outputs/native_selection_diagnostic12_20261011_v2/execution_state_v1.json',
    'outputs/native_selection_diagnostic12_20261011_v2/execution_state_v2.json',
    dict(evidence=evidence, queues={s: v['queue'] for s, v in obs['diagnostic'].items()}, actual_diagnostic_replays=0,
         offline_consumer_state='../native_selection_analysis12_20261011_v2/execution_state_v1.json'))
update('outputs/user_backbone12_20261011_v1/execution_state_v2.json',
    'outputs/user_backbone12_20261011_v1/execution_state_v3.json',
    dict(evidence=evidence, completed_epochs={s: {arm: v['completed_epochs'] for arm, v in o['jobs'].items()} for s, o in obs['queue'].items()},
         queue_processes={s: o['queue'] for s, o in obs['queue'].items()}, GPU_resource=result['GPU_resource'],
         last_remote_observation_UTC={s: v['UTC'] for s, v in obs['queue'].items()}, saved_local_UTC=result['saved_local_UTC']))
update('outputs/warm_stem12_20261011_v1/execution_state_v6.json',
    'outputs/warm_stem12_20261011_v1/execution_state_v7.json',
    dict(evidence=evidence, actual_new_training_epochs=rows['warm_stem'],
         pending='Complete both fixed12 trainings, saved EMA replay and formal paired results.',
         resource_rule='Both own candidate trainers are active. New standalone configs wait for capacity using frozen reservations.',
         trainers={s: o['trainer'] for s, o in obs['warm'].items()}))
update('outputs/stem_component12_20261011_v1/execution_state_v25.json',
    'outputs/stem_component12_20261011_v1/execution_state_v26.json',
    dict(current_goal_turn='progress', no_progress_streak=0, latest_compact_status=evidence,
         completed_fixed12_arms_both_seeds=['original_stem', 'detail_lowres', 'highfreq_rgb', 'native_stem_scratch'],
         pending_old_training_arms_both_seeds=['attention_spatial_mean', 'early_final_downsample'],
         actual_component_completed_epochs={a: rows[a] for a in ('attention_spatial_mean', 'early_final_downsample')},
         warm_training_started_seeds=[1,2], warm_candidate_state='../warm_stem12_20261011_v1/execution_state_v7.json',
         actual_native_selection_diagnostic_state='../native_selection_diagnostic12_20261011_v2/execution_state_v2.json',
         offline_six_record_consumer_state='../native_selection_analysis12_20261011_v2/execution_state_v1.json',
         standalone_user_execution='../user_backbone12_20261011_v1/execution_state_v3.json',
         interpretation='Fixed12 primary full-versus-random APs is positive in both seeds but total AP is inconsistent. Direct full remains below pretrained BASE12. Warm candidate retains794 native states; full CUDA detector preflight passed and both trainings are active. Exact saved-EMA diagnostics and six-record consumer are prepared, actual diagnostic replays still pending. Root and novelty unproven.',
         pending='Finish authorized fixed12 training and native-owned EMA closeouts; collect six exact diagnostics and run prepared offline consumer.'))
update('outputs/original_stem12_20261011_v1/execution_state_v28.json',
    'outputs/original_stem12_20261011_v1/execution_state_v29.json',
    dict(authoritative_state='../stem_component12_20261011_v1/execution_state_v26.json'))
print(json.dumps(dict(status=result['status'], completed_epochs=rows, actual_diagnostics=0, CPU_cases=14)))
