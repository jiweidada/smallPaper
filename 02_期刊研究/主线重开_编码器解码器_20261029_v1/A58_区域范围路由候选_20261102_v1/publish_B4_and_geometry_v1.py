"""Preserve B4 interim evidence and the bounded annotation-only finding."""
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

saved=load(P/'B4_saved_outputs_local_verification_v1.json')
assert saved['status']=='PASS_SAVED_BYTES_METRIC_ARRAYS_GT_PRED_STRUCTURE'
geometry=load(P/'annotation_geometry_local_verification_v1.json')
assert geometry['status']=='PASS_SOURCE_BYTES_AND_AGGREGATE_CONSISTENCY'
rpc=load(R54/'A58_formal_observe_007.json');assert rpc['exit']==0
obs=json.loads(rpc['stdout'])
assert obs['source_binding_verified'] and obs['protocol_sha256']==sha(P/'protocol_FROZEN_v1.json')
assert obs['supervisor']['status']=='FORMAL_RUNNING' and obs['supervisor']['arm']=='B'
for key in ['supervisor_identity','child_identity']:
    assert obs[key]['exists'] and obs[key]['expected_ticks_match'] and obs[key]['state']!='Z'
assert not obs['arms']['M']['directory_exists'] and not obs['arms']['Q']['directory_exists']
b=obs['arms']['B'];event=next(r for r in b['completed_epochs'] if r['epoch']==4)
assert event['EMA_validation']==saved['metrics']
c=b['journal_counts'];f=c['forward_start'];u=c['optimizer_returned'];completed=c['forward_complete']
assert f>=3373 and u>=3236 and f<=10159 and u<=9708
assert b['forward_phases']['validation']==137
begin('A58_B4_and_geometry_v1','A58_formal_startup_preservation_receipt_v1.json')
for i in ['003','004','005','006','007']:
    rpc=load(R54/('A58_formal_observe_'+i+'.json'));assert rpc['exit']==0
    save_new(P/'formal_observations_v1'/(i+'.json'),json.loads(rpc['stdout']))
transport=P/'transport_receipts_v3';transport.mkdir(exist_ok=False)
for src in sorted(R54.glob('A58_*.json')):shutil.copyfile(src,transport/src.name)
now=datetime.now(timezone.utc).isoformat();m=saved['metrics'];progress=b['progress.json']
report=P/'07_B4首评与继续判别.txt'
text=f'''A58：原版B4首评完成，候选对照仍待固定终点

结果事实
VisDrone2019-DET + ECDet-S，原版B第4轮EMA：AP {m['AP']:.6f}，AP50 {m['AP50']:.6f}，AP75 {m['AP75']:.6f}，APs {m['APs']:.6f}，small AP75 {m['APs75']:.6f}，ARs {m['ARs']:.6f}（均为0–100尺度）。
评价覆盖548张图、38602个计分GT，其中26444个小目标；去掉157个class0并集重叠GT后得到154939条预测。指标为冻结COCO-style10类/maxDets300口径，不是VisDrone官方榜单分数。
已下载本次保存的GT、预测、precision/recall数组、指标及输入流；8项指标由保存数组重聚合，最大差{saved['maximum_metric_difference_pp']:.3g}pp。没有重新跑检测器或另一次COCO匹配。B4权重80,418,323字节保留在远端，SHA256：{saved['checkpoint_preserved_remote']['sha256']}。

解释边界
B4已表现出正常的检测学习能力；不提前替代冻结的B12学习门槛。
M（单侧范围强对照）和Q（双侧范围候选）尚未开始，当前没有候选增益、胜出或创新成立的结果。

新增问题频率证据
单次纯标注/图像头统计：训练有效框中小目标207618个，验证计分框中小目标26444个。
在原实验固定32×32输入区域内，小目标自身框与中心所在区域的相交面积不超过区域1/4的比例，训练96.51%、验证90.45%；相交比例中位数为6.93%、9.73%。
验证76.23%的小目标中心区域还有其他计分框中心。因此不能把框外全部当作背景，也不能把几何少数直接等同于query特征被均值抹去。数据事实、推断和待验证机制见06_标注几何事实与机制边界.txt。
该数据统计构造模型0、检测器前向0、优化器更新0、AP评价0；没有修改三组训练或门槛。

观察到的活性前缀
记录时间{now}。原监督器PID26447/start29306939、B训练PID26448/start29306942均存活且身份一致，9项冻结源码及协议哈希核对通过。
进度文件：B第{progress['epoch']}轮、第{progress['microbatch']}/809批，完成{progress['completed_epochs']}轮。实际事件前缀：前向开始{f}、完成{completed}，优化器返回{u}次；含137次第4轮验证前向。正在执行中的调用和更新按前缀分别记录，不把进度文件旧数当全量实时计数。
本记录是一次观察，下一轮应继续读取原进程，不能从旧PID文件判断当前活性。

下一判别
继续原B至固定12轮；B12 AP≥8、APs≥3才依次启动fresh M12、Q12。
Q12需同时Q−B APs≥0.5pp、Q−M APs≥0.3pp、Q−B AP≥−0.2pp，Q/M模型中位前向延迟≤1.20。4/8轮只观察，不挑最好轮数。
每组12轮硬上限，不增seed、轮次、块大小、topk或新系数救结果。保持源代码、初始化、数据顺序和RNG检查。原协议及完整范围见protocol_FROZEN_v1.json。
主线仍是待验证候选；目标active。已有权重、源码、数据、文献索引及旧裁决保留。
'''
with report.open('x',encoding='utf-8') as dst:dst.write(text)
current=load(P/'CURRENT_A58.json')
current.update(status='A58_B4_INTERIM_VERIFIED_B_CONTINUES_MQ_UNSTARTED',observed_utc=now,
    new_detector_forwards=6+f,new_optimizer_updates=6+u,formal_calls_observed=f,
    formal_completed_calls_observed=completed,formal_updates_observed=u,
    formal_completed_epochs_observed=len(b['completed_epochs']),B_interim_metrics={'epoch4_EMA':m},
    annotation_geometry_report=str(P/'06_标注几何事实与机制边界.txt'),
    annotation_geometry_summary=geometry['summary'],baseline_B12_gate_passed=False,
    baseline_gate_status='Pending fixed12; B4 is interim only',candidate_MQ_trained=False,
    formal_supervisor_identity=obs['supervisor_identity'],formal_trainer_identity=obs['child_identity'],
    report=str(report),next='Observe originalB through8/12; B12 frozen learning gate controls unstartedM/Q; no rescue.',
    limits='B4 baseline performance and data geometry verified within saved-array/annotation scope; no candidate efficacy, attention mechanism or novelty proof.')
replace(P/'CURRENT_A58.json',current)
for name in ['01_候选判别表.json','15_当前状态与结果校验.json']:
    x=load(B/name)
    x.update(status=current['status'],A58_execution=current,latest_report=str(report),latest_update_utc=now,
             pilot_validation_status='B4 AP22.567752/APs12.661749 verified from saved arrays; B continues, M/Q unstarted; annotation geometry available')
    if name.startswith('01_'):
        row=next(r for r in x['candidates'] if r['id']==current['candidate'])
        row.update(status=current['status'],report=str(report),interim_B4_metrics=m,
            formal_observed_calls=f,formal_observed_updates=u,candidate_trained=False,
            evidence_scope='OriginalB4 plus annotated spatial geometry; M/Q efficacy and feature-mean harm untested')
    else:
        assert x['cap']==276634 and x['model_forwards']<=246157+f and x['training_updates']<=204476+u
        x.update(model_forwards=246157+f,training_updates=204476+u,recorded_utc=now,
            current_native_job_terminal=False,counts_terminal=False,training_counts_terminal=False,
            current_phase='A58_B4_INTERIM_BASELINE_B_LIVE',
            current_goal_turn='progress: originalB4 saved AP arrays and GT/predictions verified, single annotation-only geometry evidence completed; verified wait originalB6 alive',
            current_user_task_status='Goal active; baseline learns, candidate efficacy still untested; wait fixedB12/M12/Q12')
        x.setdefault('authoritative_phase_history',[]).append(dict(phase='A58',utc=now,classification='progress_and_verified_wait',
            B4_metrics=m,observed_formal_calls=f,observed_formal_updates=u,new_auxiliary_model_calls=0))
    replace(B/name,x)
save_new(P/'goal_progress_A58_B4_v1.json',dict(previous_goal_turn_classification='progress: capacity6 and formal startup',
    current_goal_turn_classification='progress: data-only geometry and B4 actual metrics; verified wait on exact originalB process',
    goal_status='active',mainline_validated=False,B4_metrics=m,observed_formal_calls=f,observed_formal_updates=u,
    new_auxiliary_model_calls=0,report=str(report)))
save_new(P/'A58_turn_handoff_B4_v1.json',dict(goal_status='active',current=current,bridge_session=19671,credentials_recorded=False,
    remote_root='/root/uav_a58_20261102_v1/pilot_v1',observer='/root/uav_a54_20261101_v1/A58_uploads/observe_remote_A58_v1.py',
    observer_sha256=sha(P/'observe_remote_A58_v1.py'),last_observer_index=7,
    global_actual_model_calls=246157+f,global_actual_optimizer_updates_including_capacity=204476+u,
    global_model_call_cap=276634,next='Observe original worker, B8 then fixedB12 gate. Do not rerun capacity, geometry, publishers or launchers. No direct candidate result yet.'))
finish('A58_B4_and_geometry_preservation_receipt_v1.json','B4_AND_DATA_GEOMETRY_VERIFIED_B_LIVE',
       observed_formal_model_calls=f,observed_formal_updates=u,new_auxiliary_model_calls=0)
print(json.dumps(dict(status=current['status'],B4_AP=m['AP'],B4_APs=m['APs'],
    B_current_epoch=progress['epoch'],global_model_calls=246157+f,global_updates=204476+u,
    goal_status='active',report=str(report)),ensure_ascii=False))
