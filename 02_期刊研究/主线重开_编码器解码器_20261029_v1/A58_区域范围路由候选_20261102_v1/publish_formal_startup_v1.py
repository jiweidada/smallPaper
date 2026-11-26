"""Publish an observed training prefix, never infer completion from launch."""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import shutil
from preservation_boundary_A58_v1 import P,B,sha,save_new,begin,finish

R54=B/'A54_场景尺度约束候选_20261101_v1'
def load(p):return json.loads(Path(p).read_text('utf-8'))
def replace(p,x):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(t,p)

rpc=load(R54/'A58_formal_observe_001.json');assert rpc['exit']==0
obs=json.loads(rpc['stdout'])
assert obs['source_binding_verified'] and obs['protocol_sha256']==sha(P/'protocol_FROZEN_v1.json')
assert obs['supervisor']['status']=='FORMAL_RUNNING' and obs['supervisor']['arm']=='B'
for key in ['supervisor_identity','child_identity']:
    assert obs[key]['exists'] and obs[key]['expected_ticks_match'] and obs[key]['state']!='Z'
assert obs['arms']['B']['initialization.json']['all_initial_tensors_identical']
assert not obs['arms']['M']['directory_exists'] and not obs['arms']['Q']['directory_exists']
counts=obs['arms']['B']['journal_counts']
f=counts['forward_start'];u=counts['optimizer_returned'];completed=counts['forward_complete']
assert 0<u<=f<=10159 and completed in [f,f-1]
save_new(P/'formal_startup_observation_v1.json',obs)
begin('A58_formal_startup_v1','A58_formal_allocation_preservation_receipt_v1.json')
transport=P/'transport_receipts_v2';transport.mkdir(exist_ok=False)
for src in sorted(R54.glob('A58_*.json')):shutil.copyfile(src,transport/src.name)
now=datetime.now(timezone.utc).isoformat()
progress=obs['arms']['B']['progress.json']
text=f'''当前候选：区域query均值是否掩盖少数目标的上下文关系？

数据与算法：VisDrone2019-DET，训练6471张、验证548张；pinned ECDet-S，COCO任务共同初始化，输入640×640，真实batch8，单卡RTX3090。原生P3/P4/P5及检测损失保留。
候选只在融合后的P3输出增加同一上下文模块，M/Q均新增148609参数；所有位置继续更新，固定4×4空间区域、3头、每块每头读取top4个key区域。

三组各12轮，从同一初始化重新训练
B：注册的新模块关闭，原版ECDet-S路径。
M：区域均值query对key逐通道范围进行路由，随后读原始细粒度token。
Q：改用query/key两侧逐通道范围的四端点点积上界，其他计算和预算与M相同。
这是候选假设，尚未证明VisDrone的小目标主要受此限制；原版ECDet-S本身并没有被错误描述为已经使用该均值区域路由。

为什么允许小验证
BiFormer提供区域均值路由先例，Quest提供点query对key范围先例，SpargeAttn说明异质块不能直接按均值压缩。AdaCluster已选读方法段并核查4个源码文件；阅读范围内未识别到本候选公式直接等同，但其活动内核没有完全展开，不能称新颖性已清除。细节、阅读范围与链接见04_AdaCluster有界源码核查.txt和02_最近方法与证据矩阵_v1.json。
候选的可能增量只限定为固定区域预算下的双侧范围路由，不声称发明区间上界或稀疏注意力。Q只对最大未归一化点积给出上界，不保证softmax总质量、召回率或AP。

已经完成的技术核查
CPU组件7次前向/2次合成反向，GPU组件2次前向/2次合成反向；它们均不是整模型训练。
一次真实CPU整模型构造通过：原有667个状态张量保持一致，共同初始化可严格加载，注册总参数9951721。
B/M/Q各两步真实batch8容量通过：共6次整模型前向、6次丢弃更新，无AP评价。两步输入及前向前RNG匹配，M/Q新分支在第2步具有非零梯度。该两步显存峰值约3.54GiB，只说明这两个批次可运行；正式训练的目标数会变化。

真实启动状态（只代表本次观察前缀）
观察记录：{now}；服务器时间另保存在formal_startup_observation_v1.json，不用跨主机时钟差估算训练耗时。
监督器PID{obs['supervisor_identity']['pid']}，训练B PID{obs['child_identity']['pid']}，两者start_ticks核对一致。
B实际事件前缀：整模型前向开始{f}、完成{completed}、优化器已返回{u}次；进度文件显示第{progress['epoch']}轮，完成{progress['completed_epochs']}轮。M/Q尚未启动；此时尚无AP。
冻结源码9项与协议哈希核对通过。读取状态没有新增检测器前向。不能把本前缀当作当前实时状态或最终结果，应使用只读观察器继续检查原进程。

固定判别规则
只在4/8/12轮验证，固定第12轮EMA作终点，不挑最好轮数。
B12需AP≥8且APs≥3，否则取消未启动的M/Q，候选保持未验证状态。
若B通过，Q12需同时满足：相对B的APs至少+0.5pp；相对M的APs至少+0.3pp；相对B的AP不低于−0.2pp；Q相对M的模型前向中位延迟不超过1.20倍。
采用同一BF16与TF32关闭设置，原生AdamW、500步warmup与9708步cosine、梯度裁剪0.1及相同EMA。每轮核对顺序、前8批实际输入/RNG和全部target流。
每组上限9708次训练更新、10159次整模型前向（含411次验证、40次延迟）。3组合计30477次前向、29124次更新、36轮；每组最多6小时，合计最多18小时。
非有限数、OOM、源码或匹配核对异常即停队列。不补轮、不换seed、不调块/topk/系数或更改内核救结果。上述门槛是短测投入判据，不是统计显著性、充分收敛或论文创新证明。
指标沿用固定COCO-style10类、maxDets300、原图面积及class0区域union IoA过滤；不是VisDrone官方榜单分数。

后续裁决
达到门槛才继续核查真实小目标关系是否被保留及剩余直接先例；仍不自动宣布论文主线成立。
未达到则停止本实现并保留完整结果，不把它扩大解释为全部区域注意力或小目标检测方向失效。
权重、数据、冻结代码及旧A57全部保留，旧A57 M/Q未训练的事实不变。目标继续保持active。
'''
report=P/'05_正式验证启动与判别说明.txt'
with report.open('x',encoding='utf-8') as dst:dst.write(text)
current=load(P/'CURRENT_A58.json')
current.update(status='A58_FORMAL_B_RUNNING_MATCHED_MQ_CONDITIONAL',observed_utc=now,
    new_detector_forwards=6+f,new_optimizer_updates=6+u,formal_calls_observed=f,formal_completed_calls_observed=completed,
    formal_updates_observed=u,formal_completed_epochs_observed=progress['completed_epochs'],
    new_training_queue=['M12','Q12'],formal_active_arm='B',formal_supervisor_identity=obs['supervisor_identity'],
    formal_trainer_identity=obs['child_identity'],report=str(report),
    next='Read only original worker; validate4/8/12 fixed. B12 learning gate controls unstarted M/Q. No rescue.',
    limits='Training prefix only. No AP observed yet; candidate benefit, mechanism and novelty unproven.')
replace(P/'CURRENT_A58.json',current)
for name in ['01_候选判别表.json','15_当前状态与结果校验.json']:
    x=load(B/name)
    x.update(status=current['status'],A58_execution=current,latest_update_utc=now,latest_report=str(report),
             pilot_validation_status='Formal B live and source/init verified; M/Q conditional on B12; no AP yet')
    if name.startswith('01_'):
        row=next(r for r in x['candidates'] if r['id']==current['candidate'])
        row.update(status=current['status'],report=str(report),formal_observed_calls=f,formal_observed_updates=u,
                   evidence_scope='Actual B training prefix; candidate M/Q not yet trained')
    else:
        assert x['model_forwards']==246157 and x['training_updates']==204476 and x['cap']==276634
        x.update(model_forwards=246157+f,training_updates=204476+u,recorded_utc=now,
                 current_native_job_terminal=False,counts_terminal=False,training_counts_terminal=False,
                 current_phase='A58_FIXED12_B_RUNNING',current_goal_turn='verified_wait: fresh B original process live, exact sources/init and journal prefix verified; M/Q conditional and unstarted',
                 current_user_task_status='Goal active; frozen candidate pilot running; no efficacy claim')
        x.setdefault('authoritative_phase_history',[]).append(dict(phase='A58',utc=now,classification='verified_wait',
                 observed_formal_calls=f,observed_formal_updates=u,completed_epochs=progress['completed_epochs']))
    replace(B/name,x)
save_new(P/'goal_progress_A58_formal_startup_v1.json',dict(classification='progress_then_verified_wait',
    goal_status='active',mainline_validated=False,formal_protocol_sha256=current['formal_protocol_sha256'],
    observed_model_calls=f,observed_optimizer_updates=u,report=str(report)))
save_new(P/'A58_turn_handoff_formal_startup_v1.json',dict(goal_status='active',mainline_validated=False,
    current=current,remote_root='/root/uav_a58_20261102_v1/pilot_v1',bridge_session=19671,
    credentials_recorded=False,observer='/root/uav_a54_20261101_v1/A58_uploads/observe_remote_A58_v1.py',
    observer_sha256=sha(P/'observe_remote_A58_v1.py'),next='Observe original processes; do not rerun launch, preparation or capacity.',
    past_call_baseline=246151,completed_capacity_calls=6,prior_optimizer_updates_including_capacity=204470,
    completed_capacity_updates=6,total_call_cap=276634,formal_per_arm=10159,formal_updates_per_arm=9708))
finish('A58_formal_startup_preservation_receipt_v1.json','FORMAL_B_STARTUP_PREFIX_PRESERVED',
       observed_formal_model_calls=f,observed_formal_updates=u,mainline_efficacy_unproven=True)
print(json.dumps(dict(status=current['status'],observed_formal_calls=f,observed_formal_updates=u,
    B_completed_epochs=progress['completed_epochs'],goal_status='active',report=str(report)),ensure_ascii=False))
