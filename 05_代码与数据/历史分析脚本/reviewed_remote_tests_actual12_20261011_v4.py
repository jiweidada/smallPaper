"""Publish actual reviewed remote training and preserve negative old replay evidence."""
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parent.parent
read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
evidence={}
def record(p):
    p=Path(p);evidence[p.relative_to(R).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p));return read(p)
def captures(p):
    data=record(p);result={}
    for k,v in data.items():
        b=v['payload'].encode();assert len(b)==v['transport']['bytes'] and hashlib.sha256(b).hexdigest()==v['transport']['sha256'];result[k]=json.loads(b)
    return result
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
prior=R/'outputs/reviewed_remote_tests_actual12_20261011_v3';old=record(prior/'report.json');m=record(prior/'manifest.json');assert sha(prior/'report.json')==m['report_sha256']
for n,v in m['files'].items():assert sha(R/n)==v['sha256'],n
c=captures(R/'outputs/reviewed_remote_tests_actual12_20261011_v4_inputs.json')
t=captures(R/'outputs/reviewed_remote_tests_actual12_20261011_v4_training_inputs.json')
collection=captures(R/'outputs/reviewed_remote_tests_actual12_20261011_v4_collection_inputs.json')
low=R/'outputs/p2_lowrank12_20261011_v3';review=record(low/'joint_release_review.json');decision=record(low/'training_decision.json')
assert sha(low/'training_decision.json')==review['training_decision_sha256'] and decision['seeds']==[1,2]
actual=captures(low/'first100_and_actual_training_inputs.json');engineering=captures(low/'engineering_actual_inputs.json')
source=record(low/'source_review.json');assert source['status']=='PASS_NEW_LAUNCH_PROTOCOL_MODEL_INITIALIZATION_AND_EFFICACY_CRITERION_UNCHANGED'
assert {p.name:sha(p) for p in (R/'英文小论文/code/p2_lowrank12_20261011_v3').iterdir()}==source['files']
training=[]
for seed in [1,2]:
    a=actual[f'seed{seed}'];b=engineering[f'seed{seed}'];d=t[f'turn3_lowrank_v3_actual_training_dispatch_seed{seed}'];first=a['files']['first100']['report'];runtime=a['files']['runtime']['report']
    assert a['completed_epochs']==0 and 'failure' not in a['files'] and first['all100_rows_exact']
    assert a['files']['audit']['report']['actual_optimizer_steps']>0 and b['files']['AMP_completion']['report']['exit_code']==0
    assert b['files']['ready']['report']==decision['engineering'][str(seed)] and d['max_epochs']==12 and not d['automatic_retry']
    assert runtime['initialization']['exact_shared_tensors']==794 and runtime['initialization']['parameters']==10238173
    training.append(dict(seed=seed,remote_dispatch=d,observed_remote_UTC=a['UTC'],GPU=a['GPU'],completed_epochs=a['completed_epochs'],
        audit=a['files']['audit']['report'],first100_inputs_and_pre_forward_RNG_exact=True,runtime_start_sha256=a['files']['runtime']['sha256'],
        engineering_ready_sha256=b['files']['ready']['sha256'],AMP_native640_832_passed=True,parameters=10238173,configuration_training_limit=12,
        optimizer_update_observed=True,native_fixed12_result_available=False,training_model_effect_not_proven=True))
negative=record(R/'outputs/p2_seed1_replay_mismatch_collection12_20261011_v1/query_order_diagnostic.json')
assert negative['archive_files_verified']==567 and not negative['exact_replay_passed'] and negative['new_model_forwards']==negative['new_training_epochs']==0
failed=c['turn2_A_P2_exact_replay_mismatch_actual_seed1'];assert failed['verification']['status']=='CANDIDATE_RELOAD_MISMATCH_NO_RETRY'
user=old['user_fixed12_results']
for x in user:
    x['independent_replay_status']='PASS_SAVED_CANDIDATE_RELOAD' if x['seed']==2 else ('CANDIDATE_RELOAD_MISMATCH_NO_RETRY' if x['arm']=='p2_skip' else 'NOT_ATTEMPTED_AFTER_OLD_QUEUE_FAILURE')
stems=[];selectors=[]
for arm in ['original_stem','native_stem_scratch','highfreq_rgb']:
    formal=record(R/f'outputs/stem_fixed12_seed1_formal_collection12_20261011_v1/seed1_{arm}/verification.json')
    assert formal['all548_stage_arrays_reloaded_exact'] and formal['first100_inputs_and_pre_forward_RNG_exact'] and formal['identity']['seed']==1
    packet=record(R/f'outputs/native_selection_collection12_20261011_v2/seed1_{arm}/packet_verification.json')
    assert packet['status']=='PASS_DOWNLOADED_SELECTION_DIAGNOSTIC_PACKET_HASHES_AND_BINDINGS' and packet['files_verified']==1126
    assert packet['native_terminal_manifest_sha256']==formal['terminal_manifest_sha256'] and packet['identity']==formal['identity']
    stems.append(dict(arm=arm,formal_status=formal['status'],candidate12=formal['candidate_overall_percent'],delta_points=formal['delta_AP_metrics_pp'],exact548_reloaded=True))
    selectors.append(dict(arm=arm,status=packet['status'],files=1126,exact_saved_EMA_diagnostic_proof_verified=True,offline_all548_tensor_geometry_consumer_pending=True))
anchor=record(R/'outputs/native_selection_anchor_audit12_20261011_v3/seed1/report.json')
assert not anchor['model_and_generated_arrays_exact'] and anchor['all_EMA_model_generated_nominal_feature_cells_exact']
out=R/'outputs/reviewed_remote_tests_actual12_20261011_v4';out.mkdir(exist_ok=False)
r=dict(status='TWO_HOST_RANK16_P2_FULLY_ENGINEERING_REVIEWED_AND_ACTUAL_TRAINING_STARTED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    previous_immutable_report_sha256=sha(prior/'report.json'),user_fixed12_results=user,all_six_user_fixed12_primary_packets_verified=True,
    A_original_P2_exact_saved_EMA_replay_failed=failed['verification'],A_saved_replay_offline_diagnostic=negative,
    new_lowrank_source_review=source,new_joint_release_review=review,new_lowrank_actual_training=training,
    A_three_Stem_formal_downloaded_verified=stems,A_three_native8400_diagnostic_packets_downloaded_verified=selectors,
    A_anchor_export_corrected_metadata_review=anchor,
    B_P2_fixed_weight_runtime_percent=old['B_P2_fixed_weight_runtime_percent'],B_small_geometry_class=old['B_P2_small_geometry_class'],B_common_location_small_GT=old['B_common_location_small_GT'],
    launch_revision_scope='The original exact EMA mismatch remains a failure and A fixed-weight controls never reached GPU. The independent rank16 training starts from common pretrained states, not that checkpoint. New engineering release replaces a self-imposed weight-intervention prerequisite, while fixed12 BASE efficacy criteria are unchanged.',
    original_training_source_files_changed=False,new_BASE_forwards=0,new_BASE_training_epochs=0,failed_training_retried=False,original_training_process_signals=0,
    withdrawn_unstarted_private_v2_CPU_queue_signals=2,new_configurations_training_started=2,new_lowrank_fixed12_efficacy_proven=False,two_seed_P2_weight_intervention_complete=False,
    two_seed_formal_efficacy_complete=False,root_or_novelty_proven=False,goal_complete=False,current_goal_turn='progress',no_progress_streak=0,
    pending=['Observe both new rank16 configurations through fixed12 EMA; preserve all losses/logs/weights/stages and native comparisons; no extension or newBASE.',
      'Complete strict six-record selector/geometry consumer using checkpoint-bound EMA arrays. A CPU model/generated arrays differ by2.3841858e-7; preserve that discrepancy and the corrected copied metadata. Do not raise a float tolerance or treat generated arrays as actual EMA.',
      'A old SiLU/attention independent savedEMA closeouts were unattempted after the P2 queue failure; primary fixed12 results are already local. Schedule each once without retrying failed P2.',
      'For any rank16 result, report both BASE12 and originalP2 deltas and locate changes using saved stages before selecting the next mechanism.'])
write(out/'report.json',r)
text=['审查后已在两台远程服务器分别启动rank16 P2实验，种子1/2，每配置最多12轮。','新增模型参数10,238,173；P2分支17,440→6,176，少11,264。模型和共同初始化未变动；两台完整CPU和640/832 AMP损失/反向检查通过。','两台已执行实际优化更新，前100个batch输入与前向前RNG核对一致；尚无新的第12轮精度结论。','复用现有同种子BASE12；新增基线训练和前向均为0。','','旧P2数值：A AP14.3448/APs9.0201；B AP14.2248/APs9.0044。','A旧P2严格EMA重放失败：重放AP14.3477/APs9.0224，差+0.0029/+0.0023个百分点。','567份失败产物已下载核验；548图同时存在查询顺序变化和数值变化，不是纯排列。根因未证明，没有重跑旧训练或放宽一致性阈值。','旧A权重干预没进入GPU；两种子权重干预诊断未完成。','新rank16从共同预训练初始化训练，未使用训练后SVD或A旧权重；旧重放失败保留。','新启动条件改为两台真实工程检查通过，最终AP/APs门槛保持原约定。','','A增强Stem、随机原Stem、高频RGB对照的正式12轮产物和三组网络节点记录已下载核验。','A锚点元数据中误复制的原导出比较字段已纠正并保留原文件；CPU模型与重新生成网格有2.3841858e-7差异，六组汇总读数仍待完成。','','继续等待新两组固定第12轮EMA：两种子APs必须高于各自BASE12，AP不得下降；同时单独报告对原P2的变化。']
(out/'审查后远程实验.txt').write_text('\n'.join(text)+'\n',encoding='utf8')
write(out/'manifest.json',dict(files=evidence,report_sha256=sha(out/'report.json'),text_sha256=sha(out/'审查后远程实验.txt'),producer_sha256=sha(__file__)))
state=dict(status=r['status'],current_goal_turn='progress',previous_goal_turn='progress',no_progress_streak=0,goal_complete=False,
    canonical_report='../reviewed_remote_tests_actual12_20261011_v4/report.json',canonical_report_sha256=sha(out/'report.json'),new_lowrank_actual_training=training,
    all_six_user_primary_fixed12_verified=True,A_user_exact_replay_failed_preserved=True,A_original_P2_runtime_controls_not_started=True,
    new_BASE_forwards=0,new_BASE_training_epochs=0,failed_training_retried=False,old_original_training_signals=0,withdrawn_unstarted_private_CPU_queues=2,
    root_or_novelty_proven=False,two_seed_formal_efficacy_proven=False,pending=r['pending'])
for family,num,previous in [('stem_component12_20261011_v1',45,44),('original_stem12_20261011_v1',48,47),('user_backbone12_20261011_v1',22,21),('user_backbone_primary_collection12_20261011_v2',7,6),('p2_subspace_diagnostic12_20261011_v1',5,4)]:
    write(R/f'outputs/{family}/execution_state_v{num}.json',dict(state,supersedes=f'execution_state_v{previous}.json'))
write(low/'execution_state_v1.json',state)
write(R/'outputs/p2_lowrank12_20261011_v2/execution_state_v1.json',dict(status='WITHDRAWN_BEFORE_ANY_TRAINING_NEW_PROTOCOL_V3_ACTUALLY_RUNNING',training_epochs=0,old_exact_replay_failure_preserved=True,canonical_report_sha256=sha(out/'report.json')))
print(json.dumps(dict(status=r['status'],training_started_seeds=[1,2],actual_optimizer_updates=[v['audit']['actual_optimizer_steps'] for v in training],BASE_forwards=0,goal_complete=False)))
