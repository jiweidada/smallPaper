from pathlib import Path
import datetime,hashlib,json

read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs=read('outputs/stem_both_controls_primary_review_progress12_20261011_v1_inputs.json');obs={}
for family,seeds in inputs['observations'].items():
    obs[family]={}
    for seed,item in seeds.items():
        b=item['payload'].encode('utf8');t=item['transport']
        assert len(b)==t['bytes'] and hashlib.sha256(b).hexdigest()==t['sha256']
        o=json.loads(b);assert o['seed']==int(seed);obs[family][seed]=o
for seed in ('1','2'):
    assert obs['component'][seed]['code_SHA_verified']
    for arm in ('attention_spatial_mean','early_final_downsample'):
        j=obs['component'][seed]['jobs'][arm]
        assert j['completed_epochs']==12 and j['completion']=='NATIVE_FIT_FINISHED_EXACT12' and not j['trainer']['same_live'] and j['failure'] is None
    w=obs['warm'][seed];assert w['source_SHA_verified'] and w['trainer']['same_live'] and w['failure'] is None
    q=obs['queue'][seed];assert q['code_SHA_verified'] and q['queue']['same_live'] and not q['failures']
    for a in ('backbone_silu','p2_skip'):
        j=q['jobs'][a];assert j['trainer']['same_live'] and j['controller']['identity_equals_receipt']
        assert j['full_preflight']['identity_matches_receipt'] and j['first100']['all100_rows_exact'] and j['runtime_start']['identity_equals_receipt']
        assert j['runtime_start']['optimizer_coverage']['all_parameters_exactly_once']
reviewroot=Path('outputs/stem_both_controls_primary_review12_20261011_v1')
for n,h in read(reviewroot/'manifest.json')['files'].items():assert sha(reviewroot/n)==h
review=read(reviewroot/'report.json');assert review['source_sha256']==sha('outputs/summarize_both_stem_controls_primary12_20261011_v1.py')
assert len(review['controls'])==4 and not review['other_seed_control_endpoint_pending'] and review['independent_saved_EMA_replay_pending']
randomroot=Path('outputs/stem_random_primary_collection12_20261011_v1')
for n,h in read(randomroot/'analysis_summary_manifest.json')['files'].items():assert sha(randomroot/n)==h
random=read(randomroot/'analysis_summary.json');assert len(random['seeds'])==2 and not random['pretrained_BASE_gate_passed']
root=Path('outputs/stem_both_controls_primary_review_progress12_20261011_v1');root.mkdir(exist_ok=False)
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
for f,ss in inputs['observations'].items():
    for s,i in ss.items():(root/f'{f}_seed{s}.json').write_bytes(i['payload'].encode('utf8'))
write(root/'transport.json',{f:{s:i['transport'] for s,i in ss.items()} for f,ss in inputs['observations'].items()})
jobs={s:{a:dict(state='training' if j['trainer'] and j['trainer']['same_live'] else 'awaiting_GPU_slot',completed_epochs=j['completed_epochs'],
    trainer=j['trainer'],controller=j['controller'],actual_optimizer_steps_lower_bound=j['audit']['actual_optimizer_steps'] if j['audit'] else None,
    full_preflight=j['full_preflight'],first100=j['first100'],runtime_start=j['runtime_start'],
    native_primary_AP_points=j['latest_stats'][0]*100 if j['latest_stats'] else None,
    native_primary_APs_points=j['latest_stats'][3]*100 if j['latest_stats'] else None,
    compared_to_cached_BASE12=False,fixed12_efficacy_not_claimed=True,independent_saved_EMA_pending=True,failure=j['failure'])
    for a,j in o['jobs'].items()} for s,o in obs['queue'].items()}
counts={a:{s:o['jobs'][a]['completed_epochs'] for s,o in obs['component'].items()} for a in ('attention_spatial_mean','early_final_downsample')}
counts['warm_stem']={s:o['completed_epochs'] for s,o in obs['warm'].items()}
for s,js in jobs.items():
    for a,j in js.items():counts.setdefault(a,{})[s]=j['completed_epochs']
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
report=dict(status='FOUR_NEW_PRIMARY_PACKETS_VERIFIED_RANDOM_AND_COMPONENT_BOTH_SEED_ANALYSES_COMPLETED_USER_TRAINERS_CONTINUE',
    local_saved_UTC=now,remote_observation_UTC={f:{s:o['UTC'] for s,o in ss.items()} for f,ss in obs.items()},actual_completed_epochs=counts,user_jobs=jobs,
    scientific_review='../stem_both_controls_primary_review12_20261011_v1/report.json',scientific_review_sha256=sha(reviewroot/'report.json'),
    random_native_scientific_summary='../stem_random_primary_collection12_20261011_v1/analysis_summary.json',random_native_scientific_summary_sha256=sha(randomroot/'analysis_summary.json'),
    new_verified_primary_packets_this_turn=4,new_packet_files={'random_each':606,'seed1_controls_each':607},all_stage_images_each=548,
    both_seed_spatial_attention_consistent_benefit=False,delayed_downsample_within_enhanced_positive_both_seeds=True,
    native_already_downsamples_after_branch_concat=True,independent_saved_EMA_pending=True,root_proven=False,novelty_proven=False,BASE_gate_passed=False,
    changed_next_action=review['changed_next_action'],
    source_frozen_training_files_changed=False,observation_and_collection_new_model_forwards=0,observation_and_collection_new_training_epochs=0,
    observation_and_collection_new_optimizer_updates=0,new_BASE_forwards=0,new_BASE_epochs=0,remote_process_signals_this_turn=0,
    configuration_training_retry=False,previous_goal_turn='progress',current_goal_turn='progress',no_progress_streak=0,goal_complete=False)
write(root/'report.json',report)
evidence='../stem_both_controls_primary_review_progress12_20261011_v1/report.json'
def update(directory,version,changes):
    base=Path('outputs')/directory;old=base/f'execution_state_v{version}.json';v=read(old)
    v.update(changes);v.update(supersedes=old.name,goal_complete=False);write(base/f'execution_state_v{version+1}.json',v)
write(randomroot/'execution_state_v2.json',dict(status=random['status'],supersedes='execution_state_v1.json',evidence=evidence,
    primary_packets_verified_both_seeds=True,files_each=606,images_each=548,independent_saved_EMA_pending=True,goal_complete=False))
write('outputs/stem_seed1_controls_primary_collection12_20261011_v1/execution_state_v2.json',dict(status='BOTH_SEED1_CONTROL_PRIMARY_PACKETS_DOWNLOADED_VERIFIED_AND_SAME_GT_ANALYZED',
    supersedes='execution_state_v1.json',evidence=evidence,files_each=607,images_each=548,independent_saved_EMA_pending=True,goal_complete=False))
update('native_stem_scratch12_20261011_v3',13,dict(primary_collection='../stem_random_primary_collection12_20261011_v1/execution_state_v2.json',
    primary_same_GT_analysis='../stem_random_primary_collection12_20261011_v1/analysis_summary.json',independent_saved_EMA_pending=True,evidence=evidence))
update('user_backbone12_20261011_v1',9,dict(evidence=evidence,jobs=jobs,completed_epochs={s:{a:j['completed_epochs'] for a,j in js.items()} for s,js in jobs.items()},
    last_remote_observation_UTC={s:o['UTC'] for s,o in obs['queue'].items()},queue_processes={s:o['queue'] for s,o in obs['queue'].items()},
    queue_phases={s:o['live']['phase'] for s,o in obs['queue'].items()},GPU_resource={s:o['live'].get('resource') for s,o in obs['queue'].items()},
    first100_passed={s:{a:bool(j['first100'] and j['first100']['all100_rows_exact']) for a,j in o['jobs'].items()} for s,o in obs['queue'].items()},
    fixed12_efficacy_not_yet_claimed=True))
update('warm_stem12_20261011_v1',13,dict(evidence=evidence,actual_new_training_epochs=counts['warm_stem'],
    trainers={s:o['trainer'] for s,o in obs['warm'].items()},last_remote_observation_UTC={s:o['UTC'] for s,o in obs['warm'].items()},
    native_training_completions={s:o['native_training_completion'] for s,o in obs['warm'].items()}))
update('stem_component12_20261011_v1',32,dict(current_goal_turn='progress',no_progress_streak=0,latest_compact_status=evidence,
    completed_fixed12_arms_both_seeds=['original_stem','native_stem_scratch','detail_lowres','highfreq_rgb','attention_spatial_mean','early_final_downsample'],
    pending_old_training_arms_both_seeds=[],actual_component_completed_epochs={a:counts[a] for a in ('attention_spatial_mean','early_final_downsample')},
    both_component_primary_scientific_review='../stem_both_controls_primary_review12_20261011_v1/report.json',
    new_component_primary_scientific_summary='../stem_both_controls_primary_review12_20261011_v1/report.json',
    new_component_primary_state='../stem_seed1_controls_primary_collection12_20261011_v1/execution_state_v2.json',
    new_component_fixed12_scores=[dict(seed=r['seed'],arm=r['arm'],AP=r['control_AP_percent']['AP'],APs=r['control_AP_percent']['APs'],
        full_minus_control_AP=r['full_minus_control_points']['AP'],full_minus_control_APs=r['full_minus_control_points']['APs'],
        primary_packet_verified=True,independent_saved_EMA_pending=True,seed1_same_endpoint_pending=False) for r in review['controls']],
    warm_candidate_state='../warm_stem12_20261011_v1/execution_state_v14.json',standalone_user_execution='../user_backbone12_20261011_v1/execution_state_v10.json',
    standalone_user_completed_epochs={s:{a:j['completed_epochs'] for a,j in js.items()} for s,js in jobs.items()},
    reference_state='../native_stem_scratch12_20261011_v3/execution_state_v14.json',
    random_same_GT_primary_analysis='../stem_random_primary_collection12_20261011_v1/analysis_summary.json',
    interpretation='Both seeds now have complete fixed12 controls and random-native primary packets. Spatial attention gain is inconsistent; delayed fusion downsample is positive within enhanced design, while native also downsamples after branch concat. Full versus random has both-seed APs gain but inconsistent overall AP and remains below pretrained BASE. Warm and original-user configurations continue; independent EMA and selection diagnostics remain pending.',
    existing_process_signals_this_turn=0,old_training_process_signals_this_turn=0))
update('original_stem12_20261011_v1',35,dict(authoritative_state='../stem_component12_20261011_v1/execution_state_v33.json'))
print(json.dumps(dict(status=report['status'],counts=counts,verified_primary_packets_this_turn=4,goal_complete=False)))
