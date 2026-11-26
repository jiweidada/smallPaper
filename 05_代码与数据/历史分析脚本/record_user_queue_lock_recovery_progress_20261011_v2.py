from pathlib import Path
import datetime
import hashlib
import json

read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs = read('outputs/user_queue_lock_recovery_progress_20261011_v2_inputs.json')
obs = {}
for family, seeds in inputs['observations'].items():
    obs[family] = {}
    for seed, item in seeds.items():
        raw = item['payload'].encode('utf8')
        assert len(raw) == item['transport']['bytes']
        assert hashlib.sha256(raw).hexdigest() == item['transport']['sha256']
        value = json.loads(raw);assert value['seed'] == int(seed)
        obs[family][seed] = value
code = Path('英文小论文/code/user_queue_lock_recovery_20261011_v2')
art = Path('outputs/user_queue_lock_recovery_20261011_v2')
manifest = read(code/'recovery_manifest.json');validation = read(art/'validation.json')
assert validation['status'] == 'PASS_ONE_POSITIVE_AND_NINE_NEGATIVE_ZERO_START_HANDOVER_FIXTURES'
assert sha(art/'validation.json') == manifest['validation_sha256']
assert all(sha(code/n)==h for n,h in manifest['recovery_files'].items())
archive = read(art/'archive.json')
assert sha(art/'recovery_v2.zip') == archive['sha256']
assert (art/'recovery_v2.zip').stat().st_size == archive['bytes']
for seed in ('1','2'):
    h=obs['handover'][seed];q=obs['queue'][seed];d=obs['diag'][seed]
    assert h['status']=='PASS_ZERO_START_QUEUE_HANDOVER_TO_LOCAL_ATOMIC_LOCK'
    assert h['archive_sha256']==archive['sha256'] and h['old_dispatcher_stopped']
    assert h['old_training_process_signals']==0 and h['queued_dispatcher_signals']==1
    assert h['frozen28_source_verified'] and not h['model_or_training_sources_changed']
    assert h['local_atomic_lock_exclusion_passed'] and h['shared_directory_filesystem']=='nfs'
    assert h['post_signal_state']['actual_configurations_started']==h['post_signal_state']['actual_completed_epochs']==0
    assert q['code_SHA_verified'] and q['queue']['same_live'] and not q['failures']
    assert q['queue']['pid']==h['replacement_handle']['pid']
    assert q['queue']['start_ticks']==h['replacement_handle']['start_ticks']
    assert q['queue']['source_sha256']==manifest['recovery_files']['dispatch.py']
    assert d['source_verified'] and d['queue']['same_live'] and d['failure'] is None
    assert not d['actual_diagnostic_children']
    matched=[v for v in d['live']['handles'] if v['pid']==q['queue']['pid']]
    assert len(matched)==1 and matched[0]['same_live'] and matched[0]['start_ticks']==q['queue']['start_ticks']
    assert obs['component'][seed]['code_SHA_verified']
    assert obs['warm'][seed]['source_SHA_verified'] and obs['warm'][seed]['trainer']['same_live']
assert obs['wait_symptom']['2']['queue']['wchan']=='nlmclnt_block'
silu=obs['queue']['2']['jobs']['backbone_silu'];p2=obs['queue']['2']['jobs']['p2_skip']
for j in (silu,p2):
    assert j['trainer']['same_live'] and j['controller']['same_live']
    assert j['controller']['identity_equals_receipt'] and j['runtime_start']['identity_equals_receipt']
    assert j['full_preflight']['status']=='PASS_USER_CONFIG_FULL_AMP_LOSS_BACKWARD_NO_UPDATES'
    assert j['full_preflight']['identity_matches_receipt']
    assert j['full_preflight']['candidate_forwards']==2 and j['full_preflight']['optimizer_updates']==0
    assert [c['side'] for c in j['full_preflight']['cases']]==[640,832]
    assert all(c['finite_gradients'] for c in j['full_preflight']['cases'])
assert silu['audit']['actual_optimizer_steps']>=100 and silu['first100']['all100_rows_exact']

root=Path('outputs/user_queue_lock_recovery_progress_20261011_v2');root.mkdir(exist_ok=False)

def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

for family,seeds in inputs['observations'].items():
    for seed,item in seeds.items():(root/f'{family}_seed{seed}.json').write_bytes(item['payload'].encode('utf8'))
write(root/'transport.json',{f:{s:i['transport'] for s,i in seeds.items()} for f,seeds in inputs['observations'].items()})
jobs={}
for seed,q in obs['queue'].items():
    jobs[seed]={}
    for arm,j in q['jobs'].items():
        live=bool(j['trainer'] and j['trainer']['same_live'])
        steps=j['audit']['actual_optimizer_steps'] if j['audit'] else None
        jobs[seed][arm]=dict(state='training_epoch0' if live and steps is not None else 'trainer_started_epoch0' if live else 'awaiting_GPU_slot',
            completed_epochs=j['completed_epochs'],trainer=j['trainer'],trainer_same_live=live,
            actual_optimizer_steps_lower_bound=steps,optimizer_counter_pending=live and steps is None,
            full_GPU_preflight_passed=bool(j['full_preflight']),full_GPU_preflight=j['full_preflight'],
            first100=j['first100'],runtime_start=j['runtime_start'],failure=j['failure'],
            native_primary_AP_points=None,native_primary_APs_points=None,compared_to_cached_BASE12=False,
            independent_saved_EMA_effect_not_claimed=True)
counts={a:{s:o['jobs'][a]['completed_epochs'] for s,o in obs['component'].items()}
        for a in ('attention_spatial_mean','early_final_downsample')}
counts['warm_stem']={s:o['completed_epochs'] for s,o in obs['warm'].items()}
for arm in ('backbone_silu','p2_skip','backbone_attention'):
    counts[arm]={s:o['jobs'][arm]['completed_epochs'] for s,o in obs['queue'].items()}
full=read('outputs/stem_AP_primary_breakdown12_20261011_v1/report.json')['seeds'][1]
# Values below are the already verified main fixed12 record, not a new replay.
full_AP=7.908137290758578;full_APs=5.160614737271937
comparisons=[]
for arm in ('attention_spatial_mean','early_final_downsample'):
    j=obs['component']['2']['jobs'][arm]
    assert j['completed_epochs']==12 and j['completion']=='NATIVE_FIT_FINISHED_EXACT12'
    assert j['trainer']['same_live'] is False and j['controller']['same_live'] is False
    stats=j['latest_native_stats']
    comparisons.append(dict(seed=2,arm=arm,AP=stats[0]*100,APs=stats[3]*100,
        full_minus_control_AP=full_AP-stats[0]*100,full_minus_control_APs=full_APs-stats[3]*100,
        primary_log_scope_only=True,independent_saved_EMA_replay_pending=True,
        seed1_same_endpoint_pending=True,unique_mechanism_or_BASE_gate_pass_not_claimed=True))
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
report=dict(status='NFS_QUEUE_LOCK_RECOVERED_FROZEN_CODE_RETAINED_SEED2_SILU_AND_P2_TRAINERS_STARTED',
    local_saved_UTC=now,remote_observation_UTC={f:{s:o['UTC'] for s,o in seeds.items()} for f,seeds in obs.items()},
    symptom='Seed2 dispatcher slept in nlmclnt_block on NFS flock after capacity became eligible; no training child or configuration had started.',
    remedy='Validated scheduling-only replacement uses an atomic directory lock on local overlay /tmp. Both old zero-child queues were stopped with exact PID/start-tick checks; all model/recipe/initialization source28 files stay frozen. Original queue handles remain archived, current public handles point to the replacements, and actual diagnostic waiters observe both new owners.',
    configuration_training_retry=False,CPU_recovery_validation=validation['status'],
    actual_completed_epochs=counts,standalone_jobs=jobs,seed2_component_fixed12_primary= comparisons,
    diag_waiters_follow_replacements=True,actual_diagnostic_children=0,
    frozen28_model_training_sources_changed=False,queued_dispatcher_signals=2,old_training_process_signals=0,
    new_BASE_epochs=0,new_BASE_forwards=0,max_epochs_each=12,
    saved_EMA_independent_replay_pending=True,causal_root_proven=False,novelty_proven=False,
    current_goal_turn='progress',no_progress_streak=0,goal_complete=False)
write(root/'report.json',report)
note='''审查后远程执行与排队锁修复

种子2原队列已经有容量，却在NFS文件锁上等待；当时三项新配置均未启动、没有子进程。
已完成1项正例、9项拒绝用例，以及两台服务器本机原子目录锁的互斥/释放验证；只替换未启动队列的调度器，28份模型和训练代码、初始化、12轮规则保持原样。诊断等待器已确认跟随新的实际队列句柄。

当前：
种子1：Stem两个待完成消融9/12；保留预训练的增强Stem5/12；SiLU/P2/主干注意力等待容量。
种子2：Stem四个消融均12/12；保留预训练的增强Stem7/12；SiLU和P2已经并行启动，均完成0/12；主干注意力等待容量。
SiLU：640/832完整模型AMP预检通过，至少100次实际优化更新，首100份输入与基线完全相同。
P2：640/832完整模型AMP预检通过，训练进程与初始化/优化器覆盖记录已确认；100步计数尚待发布。

种子2固定12轮主记录（AP/APs百分数，独立EMA重载待完成）：
完整增强Stem：7.9081 / 5.1606
将空间注意力图换成每图空间均值：7.3830 / 4.4432；完整版本高0.5251 / 0.7174个百分点。
提前最终下采样：6.0622 / 3.9750；完整版本高1.8459 / 1.1856个百分点。
这些仅是种子2的消融结果；种子1尚未到相同终点，不能宣称稳定的唯一机制或超过预训练BASE12。

本次没有重跑基线，没有重试失败训练。只向两台未启动的队列调度进程发送了SIGTERM；现有训练进程均未被停止。
'''
(root/'执行进度与锁修复.txt').write_text(note,encoding='utf8')
write(root/'manifest.json',dict(files={n:sha(root/n) for n in ('report.json','执行进度与锁修复.txt')},
    recovery_source_sha256={n:sha(code/n) for n in ('dispatch.py','lock_support.py','deploy.py','build_validate.py')}))
evidence='../user_queue_lock_recovery_progress_20261011_v2/report.json'

def update(directory,version,changes):
    base=Path('outputs')/directory;old=base/f'execution_state_v{version}.json';r=read(old)
    r.update(changes);r.update(supersedes=old.name,goal_complete=False);write(base/f'execution_state_v{version+1}.json',r)

update('user_backbone12_20261011_v1',6,dict(status=report['status'],evidence=evidence,saved_local_UTC=now,
    jobs=jobs,completed_epochs={s:{a:j['completed_epochs'] for a,j in js.items()} for s,js in jobs.items()},
    queue_processes={s:o['queue'] for s,o in obs['queue'].items()},
    last_remote_observation_UTC={s:o['UTC'] for s,o in obs['queue'].items()},
    queue_phases={s:o['live']['phase'] for s,o in obs['queue'].items()},
    GPU_resource={s:o['live'].get('resource') for s,o in obs['queue'].items()},
    full_GPU_preflight_and_training_pending=True,
    full_GPU_preflight_passed_by_seed={s:{a:bool(j['full_preflight']) for a,j in o['jobs'].items()} for s,o in obs['queue'].items()},
    full_GPU_preflight_or_training_started=True,configuration_training_retry=False,
    queued_dispatcher_signals=2,old_training_process_signals=0,
    scheduling_only_NFS_lock_recovery='../user_queue_lock_recovery_progress_20261011_v2/report.json'))
update('warm_stem12_20261011_v1',10,dict(evidence=evidence,actual_new_training_epochs=counts['warm_stem'],
    trainers={s:o['trainer'] for s,o in obs['warm'].items()},last_remote_observation_UTC={s:o['UTC'] for s,o in obs['warm'].items()}))
update('native_selection_diagnostic12_20261011_v2',3,dict(evidence=evidence,
    predecessor_queue_handles={s:o['queue'] for s,o in obs['queue'].items()},
    actual_waiters_verified_following_replacements=True,actual_diagnostic_children=0))
update('stem_component12_20261011_v1',29,dict(current_goal_turn='progress',no_progress_streak=0,
    latest_compact_status=evidence,actual_component_completed_epochs={a:counts[a] for a in ('attention_spatial_mean','early_final_downsample')},
    warm_candidate_state='../warm_stem12_20261011_v1/execution_state_v11.json',
    standalone_user_execution='../user_backbone12_20261011_v1/execution_state_v7.json',
    standalone_user_completed_epochs={s:{a:j['completed_epochs'] for a,j in js.items()} for s,js in jobs.items()},
    standalone_user_queue_phases={s:o['live']['phase'] for s,o in obs['queue'].items()},
    standalone_user_training_started={'1':[],'2':['backbone_silu','p2_skip']},
    source28_verified_both_servers=True,existing_process_signals_this_turn=2,
    existing_process_signal_scope='Only exact zero-child, never-started queue dispatchers; no existing trainer signals.',
    old_training_process_signals_this_turn=0,
    actual_native_selection_diagnostic_state='../native_selection_diagnostic12_20261011_v2/execution_state_v4.json',
    scheduling_only_NFS_lock_recovery='../user_queue_lock_recovery_progress_20261011_v2/report.json',
    seed2_component_fixed12_primary= comparisons))
update('original_stem12_20261011_v1',32,dict(authoritative_state='../stem_component12_20261011_v1/execution_state_v30.json'))
write(art/'execution_state_v1.json',dict(status=report['status'],evidence=evidence,
    both_remote_local_locks_verified=True,source28_unchanged=True,configuration_training_retry=False,goal_complete=False))
print(json.dumps(dict(status=report['status'],completed_epochs=counts,seed2_SiLU_observed_optimizer_steps=100,
    seed2_component_fixed12_primary=comparisons,old_training_signals=0,queued_dispatcher_signals=2,goal_complete=False)))
