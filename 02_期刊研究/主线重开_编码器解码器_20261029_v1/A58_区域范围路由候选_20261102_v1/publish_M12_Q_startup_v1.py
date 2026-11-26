"""Publish audited M12 and observed Q startup, preserving interim supplements."""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import shutil
from preservation_boundary_A58_v1 import P,B,sha,save_new,begin,finish

R54=B/'A54_场景尺度约束候选_20261101_v1'
WORKROOT=P.parents[2]
def load(p):return json.loads(Path(p).read_text('utf-8'))
def replace(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(tmp,p)
def resolve_recorded(n):
    q=Path(n);return q if q.is_absolute() else WORKROOT/q

mb=load(P/'fixed12_M_bundle_local_verification_v1.json');ms=load(P/'fixed12_M_checkpoint_local_verification_v1.json')
bb=load(P/'fixed12_B_bundle_local_verification_v1.json')
assert mb['status']=='PASS_FIXED12_BUNDLE_COUNTERS_INPUT_FLOW_AND_CPU_MATCHING' and mb['matched_B_flow']
assert ms['status']=='PASS_FIXED12_CPU_TENSOR_DATA_AUDIT'
assert ms['optimizer_step_histogram']=={'9708':489} and ms['EMA_updates']==9708
assert len(ms['states']['raw']['changed_context_state_keys'])==7
rpc=load(R54/'A58_formal_observe_051.json');assert rpc['exit']==0
obs=json.loads(rpc['stdout']);assert obs['source_binding_verified'] and obs['protocol_sha256']==sha(P/'protocol_FROZEN_v1.json')
assert obs['supervisor']['status']=='FORMAL_RUNNING' and obs['supervisor']['arm']=='Q'
assert obs['supervisor']['completed']==['B','M'] and obs['supervisor']['remaining']==[]
for key in ['supervisor_identity','child_identity']:
    assert obs[key]['exists'] and obs[key]['expected_ticks_match'] and obs[key]['state']!='Z'
assert obs['child_identity']['pid']==28541 and obs['child_identity']['start_ticks']=='29895197'
assert obs['arms']['Q']['initialization.json']['all_initial_tensors_identical']
assert obs['arms']['Q']['initialization.json']['common_init_sha256']=='80df462029e32551a018b1f056794347d1739b908df28f52111ca4dfe54d3302'
bm=bb['endpoint_metrics'];mm=mb['endpoint_metrics'];assert obs['arms']['M']['result.json']['endpoint_metrics']==mm
flow_rpc=load(R54/'A58_Q_early_flow_readcheck_v1.json');assert flow_rpc['exit']==0
flow=json.loads(flow_rpc['stdout']);assert flow['status']=='PASS_Q_FIRST_TWO_EPOCH_FLOW_MATCH'
assert all(all(r['matched'].values()) for r in flow['rows'])
for row in flow['rows']:
    for arm in ['B','M']:
        path=P/('fixed12_'+arm+'_remote_v1')/('arm_'+arm)/('epoch%02d_input_flow.json'%row['epoch'])
        assert sha(path)==row['flow_sha256'][arm]
save_new(P/'Q_first_two_epoch_flow_readcheck_v1.json',flow)
# Earlier M8 wording was clarified without changing its scientific values.
clarification=load(P/'A58_M8_interim_count_clarification_preservation_v2.json')
assert sha(Path(clarification['prior_receipt']))==clarification['prior_receipt_sha256']
assert sha(Path(clarification['prior_report_resolution']['snapshot']))==clarification['prior_report_resolution']['sha256']
assert sha(Path(clarification['current_report']['path']))==clarification['current_report']['sha256']
checks={}
for name in ['A58_M4_interim_supplement_preservation_v1.json','A58_M8_interim_supplement_preservation_v1.json']:
    prior=load(P/name)
    for n,row in prior['files'].items():
        file=resolve_recorded(n)
        if str(file)==clarification['prior_report_resolution']['original']:
            file=Path(clarification['prior_report_resolution']['snapshot'])
        assert file.stat().st_size==row['bytes'] and sha(file)==row['sha256'],n
    checks[name]=len(prior['files'])
save_new(P/'M_interim_supplements_chain_verified_v1.json',dict(status='PASS',files_verified=checks,
    M8_wording_resolution='A58_M8_interim_count_clarification_preservation_v2.json',new_model_forwards=0))
begin('A58_M12_Q_startup_v1','A58_B12_M_startup_preservation_receipt_v1.json')
for i in range(41,52):
    rpc=load(R54/('A58_formal_observe_%03d.json'%i));assert rpc['exit']==0
    save_new(P/'formal_observations_v1'/('%03d.json'%i),json.loads(rpc['stdout']))
for old,new in [('A58_fixed12_M_ready_v1.json','fixed12_M_ready_v1.json'),('A58_fixed12_M_small_v1.zip','fixed12_M_small_v1.zip')]:
    target=P/new;assert not target.exists();shutil.copyfile(R54/'A57_downloads'/old,target)
transport=P/'transport_receipts_v6';transport.mkdir(exist_ok=False)
for src in sorted(R54.glob('A58_*.json')):shutil.copyfile(src,transport/src.name)
save_new(P/'fixed12_saved_audit_usage_B_M_v1.json',dict(scope_sha256=sha(P/'fixed12_audit_scope_FROZEN_v1.json'),
    completed_arms=['B','M'],CPU_saved_matching_calls=2,checkpoint_loads=2,initialization_loads=2,
    maximum_each=3,remaining_arm='Q',remaining_CPU_matching_calls=1,remaining_checkpoint_loads=1,
    new_audit_model_forwards=0))
now=datetime.now(timezone.utc).isoformat();q=obs['arms']['Q'];progress=q['progress.json']
f=sum(v.get('journal_counts',{}).get('forward_start',0) for v in obs['arms'].values())
completed=sum(v.get('journal_counts',{}).get('forward_complete',0) for v in obs['arms'].values())
u=sum(v.get('journal_counts',{}).get('optimizer_returned',0) for v in obs['arms'].values())
delta={k:mm[k]-bm[k] for k in ['AP','APs','APs75','APm','APl','ARs']}
limits=dict(APs_vs_B_min=bm['APs']+.5,APs_vs_M_min=mm['APs']+.3,
    combined_APs_min=max(bm['APs']+.5,mm['APs']+.3),AP_min=bm['AP']-.2,
    Q_median_latency_max_ms=mb['latency']['median_ms']*1.2)
report=P/'12_M12终点核验与Q组启动.txt'
with report.open('x',encoding='utf-8') as dst:
    dst.write(f'''A58：B/M固定终点已核验，候选Q正在运行

固定12轮EMA结果（0–100）
B原版：AP {bm['AP']:.9f}，APs {bm['APs']:.9f}，small AP75 {bm['APs75']:.9f}；模型中位前向 {bb['latency']['median_ms']:.6f}ms。
M单侧范围强对照：AP {mm['AP']:.9f}，APs {mm['APs']:.9f}，small AP75 {mm['APs75']:.9f}；模型中位前向 {mb['latency']['median_ms']:.6f}ms。
M−B：AP {delta['AP']:+.9f}pp，APs {delta['APs']:+.9f}pp。该单seed12轮配方没有显示M的小目标AP收益；不外推为所有区域上下文方法无效。
指标沿用冻结COCO-style10类/maxDets300，非VisDrone官方榜单分数。延迟只覆盖同3090、batch1 eager BF16模型，不含解码/后处理；不是机载延迟。

M的实现与保存数据已核查
M实际完成9708次训练更新、10159次整模型前向、30026行事件；前向分为训练9708/验证411/延迟40。
51项归档文件哈希通过，12轮轮序、每轮前8批实际input/RNG和全轮target流与B一致。不是每批全部图像张量的独立哈希验证。
保存预测在CPU重新匹配一次，8项指标及precision/recall数组与原记录差异均为0。
raw/EMA各674个状态张量检查通过，原生anchors设计inf保持初值且与valid_mask对应；58个BN计数均增9708；489个优化器state均step9708；EMA.updates9708。
新增模块7个状态项均已改变，raw残差门{ms['states']['raw']['context_gate']:.9f}、EMA门{ms['states']['EMA']['context_gate']:.9f}。分支确实训练，不能把M的结果归为“新增分支完全未接入”。
M12 checkpoint已下载保留，SHA256：{ms['checkpoint_sha256']}。核查未新增检测器前向或训练更新；B/M各一次保存预测CPU匹配及一次checkpoint数据加载额度已使用，不要重复执行。

Q真实启动
Q进程28541/start29895197、原监督器26447/start29306939均存活且身份一致，冻结源码9项和协议046e2396…未改。
共同初始化与B/M完全相同。已独立读比前2轮的轮序、前8批实际input/RNG和全轮target流，B/M/Q一致。
记录时间{now}，进度文件为Q第{progress['epoch']}轮、第{progress['microbatch']}/809批，已记完整{progress['completed_epochs']}轮。实际事件前缀Q前向开始{q['journal_counts']['forward_start']}、完成{q['journal_counts']['forward_complete']}，优化器返回{q['journal_counts']['optimizer_returned']}次。
Q尚无首次AP，候选收益、均值损害机制和新颖性仍未成立。

Q的冻结门槛（完整精度判定）
APs至少{limits['APs_vs_B_min']:.9f}（B+0.5pp），同时至少{limits['APs_vs_M_min']:.9f}（M+0.3pp）；本次两项合并下限为{limits['combined_APs_min']:.9f}。
AP至少{limits['AP_min']:.9f}（B−0.2pp）。Q/M模型中位延迟≤1.20，即Q模型中位前向≤{limits['Q_median_latency_max_ms']:.9f}ms。
四项同时满足才继续投入；单seed短测通过也不是统计显著性、充分收敛或论文创新证明。未过门槛停止本配方，不补轮、换seed或修改块/topk/系数/内核抢救。

研究问题仍是候选假设
双侧范围路由只改变相同P3空间区域预算下的Q侧表示，M/Q其余计算相同。区间上界只保证最大未归一化点积，不保证attention mass或小目标AP。
几何少数很常见，仍不等于latent query关系必然被均值掩盖；最近文献/代码阅读范围及未清除的重合风险见04与06记录。
目标active。继续同一Q至固定12轮；旧源码、数据、权重、文献索引和裁决保留。没有追加新实验组或训练预算。
''')
current=load(P/'CURRENT_A58.json');current.pop('candidate_Q_trained',None)
current.update(status='A58_B_M12_VERIFIED_Q_RUNNING',observed_utc=now,M_fixed12_metrics=mm,
    M_fixed12_bundle_verification=str(P/'fixed12_M_bundle_local_verification_v1.json'),
    M_fixed12_checkpoint_verification=str(P/'fixed12_M_checkpoint_local_verification_v1.json'),
    M_minus_B_fixed12_pp=delta,required_Q_fixed12=limits,strong_control_M_completed=True,
    candidate_Q_training_started=True,candidate_Q_completed=False,formal_active_arm='Q',
    training_running=True,new_training_queued=False,new_training_queue=[],
    new_detector_forwards=6+f,new_optimizer_updates=6+u,formal_calls_observed=f,
    formal_completed_calls_observed=completed,formal_updates_observed=u,
    formal_completed_epochs_observed=24+len(q.get('completed_epochs',[])),
    formal_supervisor_identity=obs['supervisor_identity'],formal_trainer_identity=obs['child_identity'],
    Q_early_flow_match=str(P/'Q_first_two_epoch_flow_readcheck_v1.json'),
    fixed12_CPU_saved_matching_calls=2,fixed12_checkpoint_loads=2,report=str(report),
    next='Observe originalQ4/8/fixed12 then its remaining frozen saved-data audit; no rescue or extra experiment.',
    limits='B/M complete and audited. M smallAP is not improved in this recipe. Q partial; candidate efficacy, mechanism, generalization and novelty remain unproven.')
replace(P/'CURRENT_A58.json',current)
for name in ['01_候选判别表.json','15_当前状态与结果校验.json']:
    x=load(B/name)
    x.update(status=current['status'],A58_execution=current,latest_report=str(report),latest_update_utc=now,
        pilot_validation_status='B/M fixed12 audited; M−B APs−0.079482pp; originalQ live with matched early flow, efficacy pending')
    if name.startswith('01_'):
        row=next(r for r in x['candidates'] if r['id']==current['candidate']);row.pop('candidate_trained',None)
        row.update(status=current['status'],report=str(report),strong_control_M12_metrics=mm,
            M_minus_B_fixed12_pp=delta,formal_observed_calls=f,formal_observed_updates=u,
            strong_control_M_completed=True,candidate_Q_training_started=True,candidate_Q_completed=False,
            evidence_scope='Audited fixed B/M; original Q matched early training, no Q AP yet')
    else:
        assert x['cap']==276634 and x['model_forwards']<=246157+f and x['training_updates']<=204476+u
        x.update(model_forwards=246157+f,training_updates=204476+u,recorded_utc=now,
            training_update_scope='Global optimizer-returned updates through current A58 original-worker prefix; includes82 discarded capacity updates. Closed historical formal204394 plus current A58 formal journal events.',
            discarded_capacity_optimizer_updates_observed=82,formal_optimizer_updates_observed=204394+u,
            model_forward_scope='Historical closed246151 + A58 capacity6 + original B/M/Q journal starts; excludes module-only fixtures and saved-data CPU audits.',
            current_native_job_terminal=False,counts_terminal=False,training_counts_terminal=False,
            current_phase='A58_B_M12_VERIFIED_Q_LIVE',current_goal_turn='progress: fixedM12 saved-data audits and12-epoch B flow match passed; originalQ3 live with matching first2 flow; candidate verdict pending',
            current_user_task_status='Goal active; both controls complete, final candidate continues under frozen12-epoch budget')
        x.setdefault('authoritative_phase_history',[]).append(dict(phase='A58',utc=now,classification='progress_and_verified_wait',
            M12_metrics=mm,M_minus_B_fixed12_pp=delta,observed_formal_calls=f,observed_formal_updates=u,new_audit_detector_calls=0))
    replace(B/name,x)
save_new(P/'goal_progress_A58_M12_Q_v1.json',dict(previous_goal_turn_classification='verified_wait originalM11',
    current_goal_turn_classification='progress: fixedM12 complete, CPU rematching/tensor/flow audits passed; originalQ3 live and early flow verified',
    goal_status='active',mainline_validated=False,M12_metrics=mm,M_minus_B_fixed12_pp=delta,
    observed_formal_calls=f,observed_formal_updates=u,new_audit_detector_calls=0,report=str(report)))
save_new(P/'A58_turn_handoff_M12_Q_v1.json',dict(goal_status='active',current=current,bridge_session=19671,credentials_recorded=False,
    remote_root='/root/uav_a58_20261102_v1/pilot_v1',last_observer_index=51,
    observer='/root/uav_a54_20261101_v1/A58_uploads/observe_remote_A58_v1.py',observer_sha256=sha(P/'observe_remote_A58_v1.py'),
    global_actual_model_calls=246157+f,global_actual_optimizer_updates_including_capacity=204476+u,global_model_call_cap=276634,
    B_M_terminal='Both completed once and fully audited; do not rerun their matching, exporter, collector or tensor-audit scripts. MatchingM28733/exportM28817 terminal.',
    remaining_audit='Only Q remains: one CPU saved matching and one local checkpoint/initialization load under fixed12_audit_scope_FROZEN_v1.json after native completion.',
    scripts='Uploaded audit_saved_matching_CPU_v1.py --arm Q; exporter export_fixed12_saved_data_v1.py --arm Q; local collect_verify_fixed12_bundle_v1.py --arm Q then audit_fixed12_checkpoint_local_v1.py --arm Q.',
    next='Observe originalQ4/8/12. Verify Q worker and original supervisor terminal/queue empty before final cohort judgment. No training restart or expansion.'))
finish('A58_M12_Q_startup_preservation_receipt_v1.json','B_M12_FULL_DATA_AUDITS_PASS_Q_LIVE',
    observed_formal_model_calls=f,observed_formal_updates=u,new_audit_detector_calls=0,CPU_saved_matching_calls=2)
print(json.dumps(dict(status=current['status'],M12_AP=mm['AP'],M12_APs=mm['APs'],M_minus_B_APs=delta['APs'],
    Q_epoch=progress['epoch'],global_model_calls=246157+f,global_updates=204476+u,goal_status='active',report=str(report)),ensure_ascii=False))
