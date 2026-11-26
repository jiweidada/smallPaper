from pathlib import Path
import datetime,hashlib,json

read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
raw=read('outputs/stem_control_latest_progress12_20261011_v1_inputs.json')
obs={}
for family,seeds in raw['observations'].items():
    obs[family]={}
    for seed,item in seeds.items():
        b=item['payload'].encode('utf8');t=item['transport']
        assert len(b)==t['bytes'] and hashlib.sha256(b).hexdigest()==t['sha256']
        v=json.loads(b);assert v['seed']==int(seed);obs[family][seed]=v
for seed in ('1','2'):
    assert obs['component'][seed]['code_SHA_verified']
    assert not any(j['failure'] for j in obs['component'][seed]['jobs'].values())
    assert obs['warm'][seed]['source_SHA_verified'] and obs['warm'][seed]['trainer']['same_live']
    assert obs['warm'][seed]['failure'] is None
    assert obs['queue'][seed]['code_SHA_verified'] and obs['queue'][seed]['queue']['same_live']
    assert not obs['queue'][seed]['failures']
summary=Path('outputs/stem_new_controls_primary_collection12_20261011_v1/analysis_summary.json')
assert sha(summary)=='bc508533ec829d42f1c32a163513bde72514b7b0a8033fcfad51890c9b7b26f2'
root=Path('outputs/stem_control_latest_progress12_20261011_v1');root.mkdir(exist_ok=False)
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
for family,seeds in raw['observations'].items():
    for seed,item in seeds.items():(root/f'{family}_seed{seed}.json').write_bytes(item['payload'].encode('utf8'))
write(root/'transport.json',{f:{s:i['transport'] for s,i in ss.items()} for f,ss in raw['observations'].items()})
jobs={}
for seed,o in obs['queue'].items():
    jobs[seed]={}
    for arm,j in o['jobs'].items():
        live=bool(j['trainer'] and j['trainer']['same_live']);stats=j['latest_stats']
        state='training' if live else 'awaiting_GPU_slot'
        if not live and o['live'].get('phase')=='full_detector_preflight' and o['live'].get('arm')==arm:state='full_detector_preflight'
        jobs[seed][arm]=dict(state=state,completed_epochs=j['completed_epochs'],trainer=j['trainer'],
            controller=j['controller'],full_preflight=j['full_preflight'],first100=j['first100'],runtime_start=j['runtime_start'],
            actual_optimizer_steps_lower_bound=j['audit']['actual_optimizer_steps'] if j['audit'] else None,
            native_primary_AP_points=stats[0]*100 if stats else None,native_primary_APs_points=stats[3]*100 if stats else None,
            compared_to_cached_BASE12=False,fixed12_efficacy_not_claimed=True,independent_saved_EMA_pending=True,failure=j['failure'])
counts={a:{s:o['jobs'][a]['completed_epochs'] for s,o in obs['component'].items()} for a in ('attention_spatial_mean','early_final_downsample')}
counts['warm_stem']={s:o['completed_epochs'] for s,o in obs['warm'].items()}
for seed,js in jobs.items():
    for arm,j in js.items():counts.setdefault(arm,{})[seed]=j['completed_epochs']
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
report=dict(status='FRESH_SOURCE_PINNED_REMOTE_TRAINING_PROGRESS_NOT_FIXED12_EFFICACY',local_saved_UTC=now,
    remote_observation_UTC={f:{s:o['UTC'] for s,o in ss.items()} for f,ss in obs.items()},actual_completed_epochs=counts,
    user_jobs=jobs,queue_phases={s:o['live'] for s,o in obs['queue'].items()},
    scientific_summary_path='../stem_new_controls_primary_collection12_20261011_v1/analysis_summary.json',
    scientific_summary_sha256=sha(summary),verified_primary_control_packets=2,independent_saved_EMA_pending=True,
    new_BASE_epochs=0,new_BASE_forwards=0,new_model_forwards_from_this_observation=0,new_optimizer_updates_from_this_observation=0,
    remote_process_signals_this_turn=0,configuration_training_retry=False,frozen_training_sources_changed=False,
    goal_turn_classification='progress',no_progress_streak=0,goal_complete=False)
write(root/'report.json',report)
evidence='../stem_control_latest_progress12_20261011_v1/report.json'
def update(directory,version,changes):
    base=Path('outputs')/directory;old=base/f'execution_state_v{version}.json';v=read(old)
    v.update(changes);v.update(supersedes=old.name,goal_complete=False);write(base/f'execution_state_v{version+1}.json',v)
update('user_backbone12_20261011_v1',8,dict(evidence=evidence,saved_local_UTC=now,jobs=jobs,
    completed_epochs={s:{a:j['completed_epochs'] for a,j in js.items()} for s,js in jobs.items()},
    queue_processes={s:o['queue'] for s,o in obs['queue'].items()},last_remote_observation_UTC={s:o['UTC'] for s,o in obs['queue'].items()},
    queue_phases={s:o['live']['phase'] for s,o in obs['queue'].items()},GPU_resource={s:o['live'].get('resource') for s,o in obs['queue'].items()},
    first100_passed={s:{a:bool(j['first100'] and j['first100']['all100_rows_exact']) for a,j in o['jobs'].items()} for s,o in obs['queue'].items()},
    remaining_GPU_preflights_and_trainings_pending=True,fixed12_efficacy_not_yet_claimed=True))
update('warm_stem12_20261011_v1',12,dict(evidence=evidence,actual_new_training_epochs=counts['warm_stem'],
    trainers={s:o['trainer'] for s,o in obs['warm'].items()},last_remote_observation_UTC={s:o['UTC'] for s,o in obs['warm'].items()}))
update('stem_component12_20261011_v1',31,dict(current_goal_turn='progress',no_progress_streak=0,latest_compact_status=evidence,
    actual_component_completed_epochs={a:counts[a] for a in ('attention_spatial_mean','early_final_downsample')},
    warm_candidate_state='../warm_stem12_20261011_v1/execution_state_v13.json',
    standalone_user_execution='../user_backbone12_20261011_v1/execution_state_v9.json',
    standalone_user_completed_epochs={s:{a:j['completed_epochs'] for a,j in js.items()} for s,js in jobs.items()}))
update('original_stem12_20261011_v1',34,dict(authoritative_state='../stem_component12_20261011_v1/execution_state_v32.json'))
print(json.dumps(dict(status=report['status'],counts=counts,queue_phase={s:o['live']['phase'] for s,o in obs['queue'].items()},goal_complete=False)))
