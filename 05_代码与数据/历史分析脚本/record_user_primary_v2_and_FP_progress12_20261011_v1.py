"""Advance the active goal ledger only from actual preserved observations and science."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path.cwd();read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):
    with Path(p).open('x',encoding='utf8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)
def validated(p):
    for n,h in read(p/'manifest.json')['files'].items():assert sha(p/n)==h,n
    return read(p/'report.json')
inputs=read(ROOT/'outputs/user_primary_v2_and_FP_progress12_20261011_v1_inputs.json');obs={}
for family,seeds in inputs.items():
    obs[family]={}
    for seed,v in seeds.items():
        b=v['payload'].encode();assert len(b)==v['transport']['bytes'] and hashlib.sha256(b).hexdigest()==v['transport']['sha256']
        obs[family][seed]=json.loads(b)
seal=read(ROOT/'outputs/user_primary_collection12_20261011_v2_source_seal.json')
for n,h in seal['files'].items():assert sha(ROOT/n)==h,n
cpu=read(ROOT/'outputs/user_primary_collection12_20261011_v2_CPU_review.json');assert cpu['negative_cases']==162 and cpu['positive_cases']==8 and cpu['actual_scientific_endpoints']==0
user=validated(ROOT/'outputs/user_backbone_seed2_review12_20261011_v1')
fp=validated(ROOT/'outputs/native_fp_query_trace_review12_20261011_v1')
assert user['native_all_area_PR_cells_reproduced']==fp['native_all_area_PR_cells_verified']==40400
assert fp['native_PR_max_abs_error']==user['native_PR_max_abs_error']==0
assert not user['two_seed_efficacy_proven'] and not fp['warm_two_seed_gate_passed']
for arm in ('backbone_silu','p2_skip'):
    r=read(ROOT/f'outputs/user_backbone_native_order12_20261011_v2/seed2_{arm}/report.json')
    assert r['total_precision_cells_reconstructed']==20200 and r['precision_cells_each']==10100
    assert r['native_PR_max_abs_difference']==0 and r['all_native_GT_flags_exact'] and r['actual_native_stable_ties_preserved'] and r['all10_categories_and10_IoUs_retained']
    assert r['primary_verification_sha256']==sha(ROOT/f'outputs/user_backbone_primary_collection12_20261011_v2/seed2_{arm}_v1/primary_verification_v1.json')
counts={};live_trainers=[];live_collectors=[];collected=[]
for s,u in obs['observations'].items():
    assert u['code_SHA_verified'] and u['queue']['same_live'] and not u['failures']
    counts[s]={}
    audit=obs['actual_path_audit'][s];deployment=obs['deployment'][s];current=obs['collector_observation'][s]
    assert audit['user28_sources_SHA_verified'] and audit['remote4_sources_SHA_verified'] and current['remote4_SHA_verified']
    for arm,j in u['jobs'].items():
        counts[s][arm]=j['completed_epochs'];c=current['jobs'][arm];d=deployment['jobs'][arm]
        assert c['pid']==d['pid'] and c['start_ticks']==d['start_ticks'] and c['start_receipt_source_exact'] and c['failure'] is None
        assert audit['jobs'][arm]['full_preflight_passed'] and j['failure'] is None
        assert j['first100']['all100_rows_exact'] and j['runtime_start']['identity_equals_receipt'] and j['runtime_start']['optimizer_coverage']['all_parameters_exactly_once']
        if j['completed_epochs']==12:
            assert s=='2' and arm in ('backbone_silu','p2_skip') and not j['trainer']['same_live'] and not j['controller']['same_live']
            assert c['completed'] and c['waiter_completed'] and not c['same_live'] and len(c['v1_failures_preserved'])==2
            assert c['metadata']['native_stats']==j['latest_stats'] and c['metadata']['archive_entries']==618
            collected.append(dict(seed=int(s),arm=arm,archive_sha256=c['metadata']['sha256']))
        else:
            assert j['completed_epochs']<12 and j['trainer']['same_live'] and j['controller']['same_live'] and c['same_live'] and not c['completed']
            live_trainers.append(dict(seed=int(s),arm=arm,pid=j['trainer']['child_pid'],start_ticks=j['trainer']['child_start_ticks']))
            live_collectors.append(dict(seed=int(s),arm=arm,pid=c['pid'],start_ticks=c['start_ticks']))
assert len(live_trainers)==len(live_collectors)==4 and len(collected)==2
dest=ROOT/'outputs/user_primary_v2_and_FP_progress12_20261011_v1';dest.mkdir(exist_ok=False)
for family,seeds in inputs.items():
    for s,v in seeds.items():(dest/f'{family}_seed{s}.json').write_bytes(v['payload'].encode())
write(dest/'transport_proofs.json',{family:{s:v['transport'] for s,v in seeds.items()} for family,seeds in inputs.items()})
report=dict(status='TWO_ORIGINAL_USER_SEED2_ENDPOINTS_DOWNLOADED_VERIFIED_ANALYZED_FOUR_TRAINERS_CONTINUE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_goal_turn='progress',current_goal_turn='progress',no_progress_streak=0,goal_complete=False,completed_epochs=counts,remote_training_observed_UTC={s:v['UTC'] for s,v in obs['observations'].items()},read_only_collectors_observed_UTC={s:v['UTC'] for s,v in obs['collector_observation'].items()},live_actual_training_handles=live_trainers,live_read_only_collectors=live_collectors,primary_packages_collected_and_verified=collected,
actual_user_endpoint_count=2,user_seed2_results={a:{k:v for k,v in x.items() if k in ('BASE12','candidate12','delta_points','native_gate_passed_seed2','other_seed_endpoint_pending','saved_EMA_independent_replay_pending','formal_terminal_reader_pending')} for a,x in user['arms'].items()},
user_seed2_scientific_review='../user_backbone_seed2_review12_20261011_v1/report.json',FP_trace_scientific_review='../native_fp_query_trace_review12_20261011_v1/report.json',source_seal_sha256=sha(ROOT/'outputs/user_primary_collection12_20261011_v2_source_seal.json'),CPU_fixture_review_sha256=sha(ROOT/'outputs/user_primary_collection12_20261011_v2_CPU_review.json'),engineering_negative_fixtures=162,actual_native_preflight_dispatch_records_validated=6,
v1_collection_error='The native dispatcher writes detector_preflight_dispatch.json; v1 incorrectly requested detector_preflight_start.json. Two seed2 collectors failed before writing any archive. All v1 failure receipts and empty folders remain. v2 freezes the corrected path and checks exact dispatch command plus positive preflight completion.',
v2_collection_repair_scope='Versioned read-only packaging and consumer dependency paths only; original28 training files, same actual checkpoints, evaluator, data, epochs and matching are unchanged. No training retry.',
changed_next_action='Continue four already running original-user jobs to their fixed12 endpoints. Prioritize replicating originalP2, which improves nativeAP andAPs in seed2 and early/final small-GT geometry. Preserve SiLU seed2APs failure. Only after bothP2 native gates pass and full-model review consider the separately frozen phase-mean training control. Do not launch naive per-query top1 pruning: it deletes currently matched entries, and no counterfactualAP has been measured.',
small_area_native_score_prefix_matching_not_saved=True,independent_saved_EMA_and_formal_terminal_readers_pending=True,root_proven=False,novelty_proven=False,
new_training_configurations=0,new_training_commands=0,new_BASE_forwards=0,new_model_forwards_for_this_review=0,new_evaluator_calls=0,new_optimizer_updates=0,new_training_epochs_added_by_pipeline=0,existing_process_signals=0,source_sha256=sha(__file__))
write(dest/'report.json',report)
write(dest/'manifest.json',dict(files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}))
main=ROOT/'outputs/stem_component12_20261011_v1';state=read(main/'execution_state_v37.json')
state.update(status='ACTIVE_ORIGINAL_USER_FIXED12_REPLICATION_AND_QUERY_DIAGNOSIS',supersedes='execution_state_v37.json',previous_goal_turn='progress',current_goal_turn='progress',no_progress_streak=0,goal_complete=False,evidence='../user_primary_v2_and_FP_progress12_20261011_v1/report.json',original_user_primary_collection='../user_backbone_primary_collection12_20261011_v2/execution_state_v1.json',original_user_primary_collection_workers_verified_live_both_seeds=False,original_user_live_primary_collection_workers=4,original_user_primary_packages_verified=2,original_user_current_completed_epochs=counts,original_user_seed2_fixed12_results=report['user_seed2_results'],original_user_actual_trainers_verified_live=live_trainers,original_user_seed2_review=report['user_seed2_scientific_review'],native_FP_and_query_trace_review=report['FP_trace_scientific_review'],root_or_novelty_proven=False,remote_result_observer_liveness_scope='Four original-user actual trainers and four v2 CPU collectors verified by PID/start ticks at recordedUTC. Two seed2 actual trainers/controllers and their collectors positively stopped12. Native8400 diagnostic liveness was not refreshed in this turn.',changed_next_action=report['changed_next_action'])
if isinstance(state.get('warm12_primary_native_result'),dict):state['warm12_primary_native_result']['both_seed_primary_training_complete']=True
for k,v in state.items():
    if isinstance(v,dict) and 'both_seed_primary_training_complete' in v and 'warm' in k.lower():v['both_seed_primary_training_complete']=True
write(main/'execution_state_v38.json',state)
wrap=read(ROOT/'outputs/original_stem12_20261011_v1/execution_state_v40.json')
wrap.update(status=state['status'],supersedes='execution_state_v40.json',authoritative_state='../stem_component12_20261011_v1/execution_state_v38.json',goal_complete=False)
write(ROOT/'outputs/original_stem12_20261011_v1/execution_state_v41.json',wrap)
u=read(ROOT/'outputs/user_backbone12_20261011_v1/execution_state_v14.json')
u.update(status=report['status'],supersedes='execution_state_v14.json',evidence='../user_primary_v2_and_FP_progress12_20261011_v1/report.json',completed_epochs=counts,read_only_primary_collection='../user_backbone_primary_collection12_20261011_v2/execution_state_v1.json',first100_passed={s:{a:j['first100']['all100_rows_exact'] for a,j in x['jobs'].items()} for s,x in obs['observations'].items()},fixed12_efficacy_not_yet_claimed=True,seed2_fixed12_native_results=report['user_seed2_results'],two_seed_efficacy_proven=False,training_retry=False,live_actual_training_handles=live_trainers,primary_stage_and_native_order_readers_source_review='../user_primary_collection12_20261011_v2_source_seal.json',goal_complete=False)
write(ROOT/'outputs/user_backbone12_20261011_v1/execution_state_v15.json',u)
collectroot=ROOT/'outputs/user_backbone_primary_collection12_20261011_v2'
write(collectroot/'execution_state_v1.json',dict(status='TWO_PRIMARY_PACKAGES_VERIFIED_AND_FOUR_READ_ONLY_COLLECTORS_LIVE',evidence='../user_primary_v2_and_FP_progress12_20261011_v1/report.json',packages=collected,live_collectors=live_collectors,training_code_changes=0,independent_saved_EMA_and_formal_terminal_readers_pending=True,goal_complete=False))
write(ROOT/'outputs/user_backbone_primary_collection12_20261011_v1/execution_state_v2.json',dict(status='TWO_V1_PATH_FAILURES_PRESERVED_VERSIONED_READ_ONLY_REPAIR_DEPLOYED',supersedes='execution_state_v1.json',authoritative_state='../user_backbone_primary_collection12_20261011_v2/execution_state_v1.json',failure_reason=report['v1_collection_error'],training_retry=False,goal_complete=False))
write(ROOT/'outputs/native_fp_query_trace_review12_20261011_v1/execution_state_v1.json',dict(status=fp['status'],evidence='report.json',goal_complete=False,root_proven=False,novelty_proven=False))
print(json.dumps(dict(status=report['status'],epochs=counts,user_packages=2,live_trainers=4,Main_state='execution_state_v38.json',goal_complete=False)),flush=True)
