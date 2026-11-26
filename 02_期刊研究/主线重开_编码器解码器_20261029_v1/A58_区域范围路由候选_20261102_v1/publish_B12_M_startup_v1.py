"""Publish the audited fixed baseline and the independently observed live M arm."""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import shutil
from preservation_boundary_A58_v1 import P,B,sha,save_new,begin,finish

R54=B/'A54_场景尺度约束候选_20261101_v1'
def load(p):return json.loads(Path(p).read_text('utf-8'))
def replace(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(tmp,p)

bundle=load(P/'fixed12_B_bundle_local_verification_v1.json')
state=load(P/'fixed12_B_checkpoint_local_verification_v1.json')
assert bundle['status']=='PASS_FIXED12_BUNDLE_COUNTERS_INPUT_FLOW_AND_CPU_MATCHING'
assert state['status']=='PASS_FIXED12_CPU_TENSOR_DATA_AUDIT'
assert bundle['optimizer_updates']==state['EMA_updates']==9708 and bundle['full_model_forwards']==10159
assert state['optimizer_step_histogram']=={'9708':482}
rpc=load(R54/'A58_formal_observe_022.json');assert rpc['exit']==0
obs=json.loads(rpc['stdout']);assert obs['source_binding_verified'] and obs['protocol_sha256']==sha(P/'protocol_FROZEN_v1.json')
assert obs['supervisor']['status']=='FORMAL_RUNNING' and obs['supervisor']['arm']=='M' and obs['supervisor']['completed']==['B']
for key in ['supervisor_identity','child_identity']:
    assert obs[key]['exists'] and obs[key]['expected_ticks_match'] and obs[key]['state']!='Z'
assert obs['child_identity']['pid']==27421 and obs['child_identity']['start_ticks']=='29595210'
assert not obs['arms']['Q']['directory_exists']
br=obs['arms']['B']['result.json'];m=br['endpoint_metrics'];assert m==bundle['endpoint_metrics']
assert obs['B12_learning_gate.json']['status']=='PASS' and obs['B12_learning_gate.json']['metrics']==m
assert obs['arms']['M']['initialization.json']['all_initial_tensors_identical']
assert obs['arms']['M']['initialization.json']['common_init_sha256']==br['initialization_sha256']
flow_rpc=load(R54/'A58_M_early_flow_readcheck_v1.json');assert flow_rpc['exit']==0
flow=json.loads(flow_rpc['stdout']);assert flow['status']=='PASS_M_FIRST_TWO_EPOCH_FLOW_MATCH'
assert all(all(r['matched'].values()) for r in flow['rows'])
for row in flow['rows']:
    bpath=P/'fixed12_B_remote_v1/arm_B'/('epoch%02d_input_flow.json'%row['epoch'])
    assert sha(bpath)==row['B_flow_sha256']
save_new(P/'M_first_two_epoch_flow_readcheck_v1.json',flow)
begin('A58_B12_M_startup_v1','A58_B8_interim_preservation_receipt_v1.json')
for i in range(18,23):
    rpc=load(R54/('A58_formal_observe_%03d.json'%i));assert rpc['exit']==0
    save_new(P/'formal_observations_v1'/('%03d.json'%i),json.loads(rpc['stdout']))
for old,new in [('A58_fixed12_B_ready_v1.json','fixed12_B_ready_v1.json'),('A58_fixed12_B_small_v1.zip','fixed12_B_small_v1.zip')]:
    target=P/new;assert not target.exists();shutil.copyfile(R54/'A57_downloads'/old,target)
transport=P/'transport_receipts_v5';transport.mkdir(exist_ok=False)
for src in sorted(R54.glob('A58_*.json')):shutil.copyfile(src,transport/src.name)
now=datetime.now(timezone.utc).isoformat()
mc=obs['arms']['M']['journal_counts'];bc=obs['arms']['B']['journal_counts']
f=bc['forward_start']+mc['forward_start'];completed=bc['forward_complete']+mc['forward_complete']
u=bc['optimizer_returned']+mc['optimizer_returned'];progress=obs['arms']['M']['progress.json']
report=P/'09_B12终点核验与M组启动.txt'
with report.open('x',encoding='utf-8') as dst:
    dst.write(f'''A58：固定B12已核验，单侧范围强对照M正在运行

数据/宿主/问题
VisDrone2019-DET train6471/val548 + ECDet-S，共同初始化80df4620…，单卡RTX3090，真实batch8，输入640×640。
候选问题仍为：区域query均值可能掩盖少数小目标的关系；Q/K双侧范围路由在相同预算下能否超过单侧K范围路由M。
B保持原版路径；M/Q共享同一P3上下文模块及148609新增参数，固定4×4空间块、3头、每块每头top4 key区域，所有query位置继续更新。只有路由统计不同，不加新标签或损失。

固定终点事实
B12 EMA：AP{m['AP']:.9f}，AP50{m['AP50']:.9f}，AP75{m['AP75']:.9f}，APs{m['APs']:.9f}，small AP75{m['APs75']:.9f}，ARs{m['ARs']:.9f}（0–100尺度）。
冻结COCO-style10类/maxDets300，38602计分GT、26444小目标；非VisDrone官方榜单分数。
B12原进程正常结束后，原监督器将B列入completed，预定学习门槛AP≥8且APs≥3通过并启动fresh M。没有追加或续跑B。
实际B前向10159次：训练9708、验证411、延迟40；优化器更新9708、事件日志30026行。模型中位前向{bundle['latency']['median_ms']:.6f}ms，仅batch1 eager BF16模型，不含解码/后处理。

终点数据核查
51项归档文件逐字节哈希通过，12轮输入流记录完整，最终原始/EMA checkpoint已下载保留。
一次独立CPU调用同版本pycocotools对保存预测重新匹配，8项指标、precision和recall数组与原记录差异均为0；构造模型0、检测器前向0、优化器更新0。
一次本地CPU张量检查：raw/EMA各674个状态张量形状与dtype一致；除原生decoder.anchors的设计inf外无非有限状态；58个BatchNorm计数均增9708；482个优化器state均step9708；EMA.updates9708。关闭的新模块raw参数与初始化一致，raw/EMA门均为0。
该核查证明本次保存数据的一致性，不能代替另一seed、收敛结果、候选收益或新颖性证明。
B12 checkpoint SHA256：{state['checkpoint_sha256']}。

M的真实启动与匹配范围
记录时间{now}；原监督器26447/start29306939、M训练27421/start29595210均存活且身份一致，9项冻结训练源码与协议哈希不变。
进度文件显示M第{progress['epoch']}轮、第{progress['microbatch']}/809批，完成{progress['completed_epochs']}轮；实际M前向开始{mc['forward_start']}、完成{mc['forward_complete']}，优化器返回{mc['optimizer_returned']}次。
共同初始化SHA与B一致。已独立读比M前2轮的轮序、前8批实际input/target哈希、前8批CPU/CUDA RNG哈希及全轮target流，与B对应记录一致。这不是每一批全部图像张量的逐字节复核。
M尚无首次AP；Q未启动。因此当前只有基线终点，没有候选胜出结论。

后续固定判别
继续原M至12轮，然后按原队列运行fresh Q12。每组最多12轮，不补seed/轮次、改块/topk/系数或内核抢救；异常按原协议停止。
Q12需同时：APs≥{m['APs']+.5:.9f}（B+0.5pp）；Q−M APs≥0.3pp；AP≥{m['AP']-.2:.9f}（B−0.2pp）；Q/M模型中位延迟≤1.20。实际判定使用完整浮点值，4/8轮只观察，不挑最好轮。
最近方法阅读与有限源码排重见04_AdaCluster有界源码核查.txt。未发现所读方法直接等同，不代表新颖性已清除；区间上界本身不是新数学。
标注几何事实及其解释限制见06_标注几何事实与机制边界.txt。框面积少数不能直接证明query特征被均值掩盖。
目标保持active，主线尚未验证。旧数据、代码、权重、文献索引及裁决全部保留。
''')
current=load(P/'CURRENT_A58.json');current.pop('candidate_MQ_trained',None)
current.update(status='A58_B12_VERIFIED_M_RUNNING_Q_UNSTARTED',observed_utc=now,
    baseline_B12_gate_passed=True,baseline_gate_status='PASS fixed12',B_fixed12_metrics=m,
    B_fixed12_bundle_verification=str(P/'fixed12_B_bundle_local_verification_v1.json'),
    B_fixed12_checkpoint_verification=str(P/'fixed12_B_checkpoint_local_verification_v1.json'),
    B_CPU_saved_prediction_matching_calls=1,B_CPU_checkpoint_loads=1,B_original_worker_terminal_verified=True,
    strong_control_M_started=True,candidate_Q_trained=False,formal_active_arm='M',new_training_queue=['Q12'],
    new_detector_forwards=6+f,new_optimizer_updates=6+u,formal_calls_observed=f,
    formal_completed_calls_observed=completed,formal_updates_observed=u,
    formal_completed_epochs_observed=12+len(obs['arms']['M'].get('completed_epochs',[])),
    formal_supervisor_identity=obs['supervisor_identity'],formal_trainer_identity=obs['child_identity'],
    M_early_flow_match=str(P/'M_first_two_epoch_flow_readcheck_v1.json'),report=str(report),
    next='Observe originalM4/8/fixed12; then original queuedQ12. No new experiments or rescue.',
    limits='Fixed baseline fully audited; M partial and Q unstarted. Candidate efficacy, mean-feature harm, convergence and novelty remain unproven.')
replace(P/'CURRENT_A58.json',current)
for name in ['01_候选判别表.json','15_当前状态与结果校验.json']:
    x=load(B/name)
    x.update(status=current['status'],A58_execution=current,latest_report=str(report),latest_update_utc=now,
             pilot_validation_status='B12 AP24.037992/APs14.305469 tensor/journal/CPU rematching audited; M live with matching early flow; Q unstarted')
    if name.startswith('01_'):
        row=next(r for r in x['candidates'] if r['id']==current['candidate'])
        row.update(status=current['status'],report=str(report),baseline_B12_metrics=m,
            formal_observed_calls=f,formal_observed_updates=u,baseline_gate_passed=True,
            strong_control_M_started=True,candidate_trained=False,candidate_Q_trained=False,
            evidence_scope='Audited fixed baseline; M early matched training; no Q efficacy result')
    else:
        assert x['cap']==276634 and x['model_forwards']<=246157+f and x['training_updates']<=204476+u
        x.update(model_forwards=246157+f,training_updates=204476+u,recorded_utc=now,
            current_native_job_terminal=False,counts_terminal=False,training_counts_terminal=False,
            current_phase='A58_B12_VERIFIED_M_LIVE',current_goal_turn='progress: fixedB12 native completion and CPU matching/state/journal audits passed; originalM3 live with matched first2 flow, Q unstarted',
            current_user_task_status='Goal active; await strong control and Q fixed endpoints, no candidate benefit claim')
        x.setdefault('authoritative_phase_history',[]).append(dict(phase='A58',utc=now,classification='progress_and_verified_wait',
            B12_metrics=m,observed_formal_calls=f,observed_formal_updates=u,new_audit_detector_calls=0,CPU_saved_matching_calls=1))
    replace(B/name,x)
save_new(P/'goal_progress_A58_B12_M_v1.json',dict(previous_goal_turn_classification='progress: B8 actual metrics',
    current_goal_turn_classification='progress: fixedB12 completion and full saved-data audits, matched originalM startup; verifiedwait M3',
    goal_status='active',mainline_validated=False,B12_metrics=m,observed_formal_calls=f,observed_formal_updates=u,
    CPU_saved_matching_calls=1,checkpoint_loads=1,new_audit_detector_calls=0,report=str(report)))
save_new(P/'A58_turn_handoff_B12_M_v1.json',dict(goal_status='active',current=current,bridge_session=19671,credentials_recorded=False,
    remote_root='/root/uav_a58_20261102_v1/pilot_v1',last_observer_index=22,
    observer='/root/uav_a54_20261101_v1/A58_uploads/observe_remote_A58_v1.py',observer_sha256=sha(P/'observe_remote_A58_v1.py'),
    global_actual_model_calls=246157+f,global_actual_optimizer_updates_including_capacity=204476+u,global_model_call_cap=276634,
    B_terminal='OriginalB completed once; matching27632 and exporter27716 terminal, bundle and CPU tensor audit already passed. Do not rerun B audits/collectors.',
    future_audit='Scope fixed12_audit_scope_FROZEN_v1.json permits one saved CPU matching and one local tensor audit each for M and Q after original supervisor completion. Reuse same uploaded scripts; B allowance already consumed.',
    uploaded_CPU_match_script='/root/uav_a54_20261101_v1/A58_uploads/audit_saved_matching_CPU_v1.py',
    uploaded_export_script='/root/uav_a54_20261101_v1/A58_uploads/export_fixed12_saved_data_v1.py',
    local_collection='collect_verify_fixed12_bundle_v1.py --arm M or Q; then audit_fixed12_checkpoint_local_v1.py --arm M or Q, only after downloading ready/small/epoch12.',
    next='Read originalM4/8/12, Q conditional. Do not rerun initialization, capacity, geometry, launchers or completed B audit scripts.'))
finish('A58_B12_M_startup_preservation_receipt_v1.json','B12_FULL_DATA_AUDIT_PASS_M_LIVE_Q_UNSTARTED',
       observed_formal_model_calls=f,observed_formal_updates=u,new_audit_detector_calls=0,CPU_saved_matching_calls=1)
print(json.dumps(dict(status=current['status'],B12_AP=m['AP'],B12_APs=m['APs'],M_epoch=progress['epoch'],
    global_model_calls=246157+f,global_updates=204476+u,goal_status='active',report=str(report)),ensure_ascii=False))
