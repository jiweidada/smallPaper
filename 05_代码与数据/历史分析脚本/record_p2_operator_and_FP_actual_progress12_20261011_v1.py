"""Record actual new operator/FP science and specific live training handles."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path.cwd();read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):
    with Path(p).open('x',encoding='utf8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)
def validated(folder,index='manifest.json'):
    for n,h in read(folder/index)['files'].items():assert sha(folder/n)==h,n
    return read(folder/'report.json')
inputs=read(ROOT/'outputs/p2_operator_and_FP_actual_progress12_20261011_v1_inputs.json');observed={}
for family,seeds in inputs.items():
    observed[family]={}
    for seed,v in seeds.items():
        b=v['payload'].encode();assert len(b)==v['transport']['bytes'] and hashlib.sha256(b).hexdigest()==v['transport']['sha256']
        observed[family][seed]=json.loads(b)
science=validated(ROOT/'outputs/p2_equivalence_and_user_FP_review12_20261011_v1')
budget=validated(ROOT/'outputs/query_budget_and_P2_gain_scope12_20261011_v1','manifest_v2.json')
assert science['native_all_area_PR_cells_verified']==30300 and science['native_PR_max_abs_error']==0
assert science['trained_P2_equivalence']['original_rank']==64 and science['trained_P2_equivalence']['phase_mean_rank']==16
assert science['trained_P2_equivalence']['cases']==10 and not science['root_proven'] and not science['two_seed_efficacy_proven']
assert budget['images_with_more_GT_than300']==3 and budget['unavoidable_GT_excess_if_final_global_detection_budget300']==34
counts={};trainers=[];collectors=[];completed=[]
for s,u in observed['observations'].items():
    assert u['code_SHA_verified'] and u['queue']['same_live'] and not u['failures']
    c=observed['collector_observation'][s];assert c['remote4_SHA_verified']
    counts[s]={}
    for arm,j in u['jobs'].items():
        counts[s][arm]=j['completed_epochs'];worker=c['jobs'][arm]
        assert j['failure'] is None and worker['failure'] is None and j['first100']['all100_rows_exact']
        if j['completed_epochs']==12:
            assert s=='2' and arm in ('backbone_silu','p2_skip')
            assert not j['trainer']['same_live'] and not j['controller']['same_live'] and worker['completed'] and worker['waiter_completed'] and not worker['same_live']
            completed.append(dict(seed=int(s),arm=arm))
        else:
            assert j['completed_epochs']<12 and j['trainer']['same_live'] and j['controller']['same_live'] and worker['same_live']
            trainers.append(dict(seed=int(s),arm=arm,pid=j['trainer']['child_pid'],start_ticks=j['trainer']['child_start_ticks']))
            collectors.append(dict(seed=int(s),arm=arm,pid=worker['pid'],start_ticks=worker['start_ticks']))
assert len(trainers)==len(collectors)==4 and len(completed)==2
dest=ROOT/'outputs/p2_operator_and_FP_actual_progress12_20261011_v1';dest.mkdir(exist_ok=False)
for family,seeds in inputs.items():
    for s,v in seeds.items():(dest/f'{family}_seed{s}.json').write_bytes(v['payload'].encode())
write(dest/'transport_proofs.json',{family:{s:v['transport'] for s,v in seeds.items()} for family,seeds in inputs.items()})
report=dict(status='ACTUAL_TRAINED_P2_EQUIVALENCE_FOUR_NEW_FP_TRACE_RUNS_AND_QUERY_CAPACITY_EVIDENCE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_goal_turn='progress',previous_turn_evidence='../user_primary_v2_and_FP_progress12_20261011_v1/report.json',current_goal_turn='progress',no_progress_streak=0,goal_complete=False,completed_epochs=counts,training_observed_UTC={s:x['UTC'] for s,x in observed['observations'].items()},collector_observed_UTC={s:x['UTC'] for s,x in observed['collector_observation'].items()},live_actual_training_handles=trainers,live_read_only_collectors=collectors,complete_native_user_endpoints=completed,
operator_and_FP_review='../p2_equivalence_and_user_FP_review12_20261011_v1/report.json',query_budget_and_gain_scope_review='../query_budget_and_P2_gain_scope12_20261011_v1/report.json',operator_and_FP_review_sha256=sha(ROOT/'outputs/p2_equivalence_and_user_FP_review12_20261011_v1/report.json'),query_budget_review_sha256=sha(ROOT/'outputs/query_budget_and_P2_gain_scope12_20261011_v1/report.json'),new_actual_saved_array_scientific_runs=4,actual_remote_trained_branch_scientific_audits=1,actual_trained_branch_CPU_cases=10,native_all_area_PR_cells_verified=30300,
equivalence_checkpoint_sha256=science['trained_P2_equivalence']['checkpoint_sha256'],native_P2_same_nominal_parameters_different_mean_capacity=True,phase_mean_training_launched=False,phase_mean_sole_matched_capacity_attribution_allowed=False,original_P2_two_seed_native_gate_pending=True,
changed_next_action=science['changed_next_action'],small_area_raw_native_score_prefix_matching_not_saved=True,independent_saved_EMA_and_formal_terminal_readers_pending=True,new_training_configurations=0,new_training_commands=0,new_training_epochs=0,new_full_detector_forwards=0,new_BASE_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,existing_process_signals=0,root_proven=False,novelty_proven=False,source_sha256=sha(__file__))
write(dest/'report.json',report)
write(dest/'manifest.json',dict(files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}))
main=ROOT/'outputs/stem_component12_20261011_v1';state=read(main/'execution_state_v38.json')
state.update(status='ACTIVE_ORIGINAL_USER_REPLICATION_AND_P2_INFORMATION_DIRECTION_AUDIT',supersedes='execution_state_v38.json',previous_goal_turn='progress',current_goal_turn='progress',no_progress_streak=0,goal_complete=False,evidence='../p2_operator_and_FP_actual_progress12_20261011_v1/report.json',original_user_current_completed_epochs=counts,original_user_actual_trainers_verified_live=trainers,P2_operator_equivalence_and_FP_review=report['operator_and_FP_review'],P2_query_budget_and_gain_scope_review=report['query_budget_and_gain_scope_review'],phase_mean_training_launched=False,phase_mean_sole_matched_capacity_attribution_allowed=False,changed_next_action=report['changed_next_action'],root_or_novelty_proven=False,remote_result_observer_liveness_scope='Four existing original-user trainers and four v2 CPU collectors verified by PID/start ticks at recordedUTC; two seed2 user endpoints stopped and verified. CPU equivalence worker32170/257492227 positively completed0. Other native diagnostic/EMA-closeout liveness not refreshed in this turn.')
write(main/'execution_state_v39.json',state)
wrapper=read(ROOT/'outputs/original_stem12_20261011_v1/execution_state_v41.json')
wrapper.update(status=state['status'],supersedes='execution_state_v41.json',authoritative_state='../stem_component12_20261011_v1/execution_state_v39.json',goal_complete=False)
write(ROOT/'outputs/original_stem12_20261011_v1/execution_state_v42.json',wrapper)
user=read(ROOT/'outputs/user_backbone12_20261011_v1/execution_state_v15.json')
user.update(status='TWO_USER_SEED2_PRIMARY_ENDPOINTS_VERIFIED_FOUR_ORIGINAL_TRAINERS_LIVE',supersedes='execution_state_v15.json',evidence='../p2_operator_and_FP_actual_progress12_20261011_v1/report.json',completed_epochs=counts,live_actual_training_handles=trainers,P2_operator_equivalence_review=report['operator_and_FP_review'],P2_query_budget_and_gain_scope_review=report['query_budget_and_gain_scope_review'],phase_mean_sole_matched_capacity_attribution_allowed=False,phase_mean_training_launched=False,goal_complete=False)
write(ROOT/'outputs/user_backbone12_20261011_v1/execution_state_v16.json',user)
primaryroot=ROOT/'outputs/user_backbone_primary_collection12_20261011_v2';primary=read(primaryroot/'execution_state_v1.json')
primary.update(status='TWO_PRIMARY_PACKAGES_VERIFIED_FOUR_READ_ONLY_COLLECTORS_LIVE',supersedes='execution_state_v1.json',evidence='../p2_operator_and_FP_actual_progress12_20261011_v1/report.json',live_collectors=collectors,goal_complete=False)
write(primaryroot/'execution_state_v2.json',primary)
write(ROOT/'outputs/p2_linear_equivalence_audit12_20261011_v1/execution_state_v1.json',dict(status='CPU_SEED2_ACTUAL_TRAINED_P2_EQUIVALENCE_AND_CAPACITY_REVIEW_PASSED',evidence='../p2_equivalence_and_user_FP_review12_20261011_v1/report.json',seed2_worker=dict(pid=32170,start_ticks='257492227',exit_code=0),seed1_actual_trained_branch_not_yet_audited=True,new_detector_forwards=0,new_training_epochs=0,training_control_approved=False,goal_complete=False))
print(json.dumps(dict(status=report['status'],epochs=counts,live_trainers=4,Main_state='execution_state_v39.json',goal_complete=False)),flush=True)
