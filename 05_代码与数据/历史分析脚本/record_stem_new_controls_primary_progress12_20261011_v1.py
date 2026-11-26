from pathlib import Path
import datetime
import hashlib
import json

read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs=read('outputs/stem_new_controls_primary_progress12_20261011_v1_inputs.json');obs={}
for family,seeds in inputs['observations'].items():
    obs[family]={}
    for seed,item in seeds.items():
        raw=item['payload'].encode('utf8');t=item['transport']
        assert len(raw)==t['bytes'] and hashlib.sha256(raw).hexdigest()==t['sha256']
        v=json.loads(raw);assert v['seed']==int(seed);obs[family][seed]=v
for seed in ('1','2'):
    assert obs['component'][seed]['code_SHA_verified']
    assert not any(v['failure'] for v in obs['component'][seed]['jobs'].values())
    assert obs['warm'][seed]['source_SHA_verified'] and obs['warm'][seed]['trainer']['same_live']
    assert obs['warm'][seed]['failure'] is None
    assert obs['queue'][seed]['code_SHA_verified'] and obs['queue'][seed]['queue']['same_live']
    assert not obs['queue'][seed]['failures']
art=Path('outputs/stem_new_controls_primary_collection12_20261011_v1')
summary=read(art/'analysis_summary.json')
assert summary['status']=='NEW_STOPPED_SEED2_PRIMARY12_PACKETS_VERIFIED_AND_SAME_GT_CONTROL_ANALYSIS_COMPLETED'
for n,h in read(art/'analysis_summary_manifest.json')['files'].items():assert sha(art/n)==h
assert read(art/'analysis_summary_manifest.json')['summary_source_sha256']==sha('outputs/summarize_stem_new_controls_primary12_20261011_v1.py')
assert len(summary['analyses'])==2 and len(summary['both_seed_existing_native_vs_enhanced_under4'])==2
assert summary['unique_root_proven'] is False and summary['independent_saved_EMA_replay_pending']
assert summary['new_BASE_epochs']==summary['new_BASE_forwards']==0
counts={a:{s:o['jobs'][a]['completed_epochs'] for s,o in obs['component'].items()} for a in ('attention_spatial_mean','early_final_downsample')}
counts['warm_stem']={s:o['completed_epochs'] for s,o in obs['warm'].items()}
jobs={}
for seed,o in obs['queue'].items():
    jobs[seed]={}
    for arm,j in o['jobs'].items():
        trainer_live=bool(j['trainer'] and j['trainer']['same_live'])
        stats=j['latest_stats'];steps=j['audit']['actual_optimizer_steps'] if j['audit'] else None
        jobs[seed][arm]=dict(state='training' if trainer_live else 'awaiting_GPU_slot',completed_epochs=j['completed_epochs'],
            actual_optimizer_steps_lower_bound=steps,trainer=j['trainer'],trainer_same_live=trainer_live,
            full_GPU_preflight_passed=bool(j['full_preflight']),full_GPU_preflight=j['full_preflight'],
            first100=j['first100'],runtime_start=j['runtime_start'],failure=j['failure'],
            native_primary_AP_points=stats[0]*100 if stats else None,native_primary_APs_points=stats[3]*100 if stats else None,
            compared_to_cached_BASE12=False,delta_AP_points=None,delta_APs_points=None,
            comparison_scope='Interim learning progress only, not fixed12 efficacy',independent_saved_EMA_effect_not_claimed=True)
        counts.setdefault(arm,{})[seed]=j['completed_epochs']
root=Path('outputs/stem_new_controls_primary_progress12_20261011_v1');root.mkdir(exist_ok=False)

def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

for family,seeds in inputs['observations'].items():
    for seed,item in seeds.items():(root/f'{family}_seed{seed}.json').write_bytes(item['payload'].encode('utf8'))
write(root/'transport.json',{f:{s:i['transport'] for s,i in seeds.items()} for f,seeds in inputs['observations'].items()})
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
report=dict(status='NEW_PRIMARY_CONTROL_DOWNLOADS_AND_BOTH_SEED_ULTRATINY_COMPARISON_COMPLETED_TRAINERS_CONTINUE',
    local_saved_UTC=now,remote_observation_UTC={f:{s:o['UTC'] for s,o in seeds.items()} for f,seeds in obs.items()},
    scientific_summary_path='../stem_new_controls_primary_collection12_20261011_v1/analysis_summary.json',
    scientific_summary_sha256=sha(art/'analysis_summary.json'),actual_completed_epochs=counts,standalone_jobs=jobs,
    verified_new_primary_packets=2,verified_collected_files_each=607,verified_stage_images_each=548,
    source_analysis_arms=['attention_spatial_mean','early_final_downsample'],analysis_seed=2,
    other_seed_same_fixed12_control_pending=True,independent_saved_EMA_replay_pending=True,
    changed_next_action=summary['changed_next_action'],unique_root_proven=False,novelty_proven=False,
    actual_first_observed_boundary='Selected300 regressed encoder initial boxes; not a dense8400 regressed candidate pool',
    under4_high_IoU_negative_subgroup_retained=True,both_seed_under4_native_vs_enhanced_gap_retained=True,
    frozen_training_sources_changed=False,new_BASE_epochs=0,new_BASE_forwards=0,
    optimizer_updates_from_offline_analysis=0,model_forwards_from_offline_analysis=0,
    existing_remote_process_signals_this_turn=0,configuration_training_retry=False,
    previous_goal_turn_classification='progress',current_goal_turn_classification='progress',no_progress_streak=0,goal_complete=False)
write(root/'report.json',report)
write(art/'execution_state_v1.json',dict(status=summary['status'],evidence='../stem_new_controls_primary_progress12_20261011_v1/report.json',
    actual_verified_packets=2,all548_per_packet_verified=True,only_seed2_controls_complete=True,
    both_seed_existing_native_vs_enhanced_under4_examined=True,independent_saved_EMA_pending=True,goal_complete=False))
evidence='../stem_new_controls_primary_progress12_20261011_v1/report.json'

def update(directory,version,changes):
    base=Path('outputs')/directory;old=base/f'execution_state_v{version}.json';v=read(old)
    v.update(changes);v.update(supersedes=old.name,goal_complete=False);write(base/f'execution_state_v{version+1}.json',v)

update('user_backbone12_20261011_v1',7,dict(evidence=evidence,saved_local_UTC=now,jobs=jobs,
    completed_epochs={s:{a:j['completed_epochs'] for a,j in js.items()} for s,js in jobs.items()},
    queue_processes={s:o['queue'] for s,o in obs['queue'].items()},last_remote_observation_UTC={s:o['UTC'] for s,o in obs['queue'].items()},
    queue_phases={s:o['live']['phase'] for s,o in obs['queue'].items()},GPU_resource={s:o['live'].get('resource') for s,o in obs['queue'].items()},
    first100_passed={'1':{},'2':{'backbone_silu':True,'p2_skip':True}},fixed12_efficacy_not_yet_claimed=True))
update('warm_stem12_20261011_v1',11,dict(evidence=evidence,actual_new_training_epochs=counts['warm_stem'],
    trainers={s:o['trainer'] for s,o in obs['warm'].items()},last_remote_observation_UTC={s:o['UTC'] for s,o in obs['warm'].items()}))
update('stem_component12_20261011_v1',30,dict(current_goal_turn='progress',no_progress_streak=0,
    latest_compact_status=evidence,actual_component_completed_epochs={a:counts[a] for a in ('attention_spatial_mean','early_final_downsample')},
    warm_candidate_state='../warm_stem12_20261011_v1/execution_state_v12.json',
    standalone_user_execution='../user_backbone12_20261011_v1/execution_state_v8.json',
    standalone_user_completed_epochs={s:{a:j['completed_epochs'] for a,j in js.items()} for s,js in jobs.items()},
    new_component_primary_scientific_summary='../stem_new_controls_primary_collection12_20261011_v1/analysis_summary.json',
    new_component_primary_state='../stem_new_controls_primary_collection12_20261011_v1/execution_state_v1.json',
    under4_high_IoU_negative_subgroup_retained=True,both_seed_native_pretrained_vs_enhanced_under4_gap_retained=True,
    first_observed_encoder_initial_box_support_is_not_unique_TopK_cause=True,existing_process_signals_this_turn=0,
    existing_process_signal_scope='No remote process signals in this goal turn.',old_training_process_signals_this_turn=0))
update('original_stem12_20261011_v1',33,dict(authoritative_state='../stem_component12_20261011_v1/execution_state_v31.json'))
print(json.dumps(dict(status=report['status'],counts=counts,verified_primary_packets=2,all_class_negative_subgroups_retained=True,
    current_seed2_interim={a:{k:j[k] for k in ('completed_epochs','native_primary_AP_points','native_primary_APs_points')} for a,j in jobs['2'].items()},
    new_BASE_forwards=0,source_changed=False,goal_complete=False)))
