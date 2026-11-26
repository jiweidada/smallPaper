from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
A50=BASE/'A50_编码解码新机制有界筛选_20261101_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
old_path=A50/'05_A50终点与历史保留回执.json'
assert sha(old_path)=='719d69c93ad1b754df5dfdd72a8e47c3dfed708e429b80263a11a0b952b4e9a7'
old=load(old_path)
before=ROOT/'before_A51';before.mkdir(exist_ok=True);resolved={}
for name in ['01_候选判别表.json','15_当前状态与结果校验.json']:
    p=BASE/name;h=old['current_files_sha256'][str(p)];assert sha(p)==h
    dest=before/name;assert not dest.exists();shutil.copy2(p,dest);assert sha(dest)==h
    resolved[str(p)]={'preserved_path':str(dest),'sha256':h}
cap=load(ROOT/'capacity_result.json');assert cap['status']=='PASS' and cap['whole_model_forwards']==9
obs_receipt=load(ROOT/'observer_05_formal_start.json');assert obs_receipt['exit']==0
obs=json.loads(obs_receipt['stdout']);m=obs['arms']['M']['progress']
assert m['status']=='TRAINING' and obs['arms']['M']['process']['identity_matches']
assert obs['supervisor_process']['live'] and obs['supervisor_process']['identity_matches']
plan=load(ROOT/'protocol_FROZEN_v1.json')
assert plan['capacity_protocol_sha256']==cap['protocol_sha256']
new_forwards=9+m['full_model_forwards'];new_updates=6+m['optimizer_updates']
report=ROOT/'07_容量验收与正式启动.txt'
assert not report.exists()
report.write_text('A51已进入有界最小训练验证，主线是否值得继续仍待固定12终点。\n'
 '原版B复用A49完整12轮：AP24.178231/APs14.161092；不是本轮又训练一次B，也不是未见过B结果的预注册。\n'
 'M独立门控交换、Q共同交换量耦合：相同新模块73746参数、相同基础初始化/输入/优化器/损失/评估，各12轮上限。\n'
 '容量PASS：初始B/M/Q预测逐元素一致，新增参数均有限且确实更新；M峰值3.34062GiB，Q3.34019GiB；9前向、6临时更新，临时权重未复用。\n'
 '当前观察M第1轮已实际执行优化器更新，PID7799/start_ticks21400945；监督器7798/start_ticks21400942。Q尚未开始。\n'
 '固定第12轮EMA：Q−B APs≥0.5pp、Q−M APs≥0.3pp、Q−B AP≥−0.2pp。中间4/8轮只观察，不挑最佳轮次。失败无扩容/阈值/seed/层位救援。\n'
 '理论性质在理想算术与给定通道映射下成立。实际容量检查的A^T A误差约4.5e-4到4.9e-4，含TF32/FP32数值舍入；BF16输出另有量化误差。因此不声称实际图逐位守恒，更不声称语义或能量守恒。\n'
 '此前A50完整边界形状匹配只改变12/1720小GT owner，低于冻结门槛，未训练，不把它记成训练失败。A49三组12终点阴性和旧材料保留。\n'
 '总目标继续；当前仅确认候选定义、有限原文差异、代数/容量接入和真实开训，没有确认AP收益或论文新颖性。\n',encoding='utf-8')
entry={'phase':'A51','state':'M12_TRAINING_Q12_PENDING_B12_REUSED','goal_status':'active',
 'mainline_confirmed':False,'selected_for_pilot':True,'report':str(ROOT/'04_候选问题_最近方法_最小验证.txt'),
 'startup_report':str(report),'protocol':str(ROOT/'protocol_FROZEN_v1.json'),
 'protocol_sha256':sha(ROOT/'protocol_FROZEN_v1.json'),'capacity_result':str(ROOT/'capacity_result.json'),
 'observation':str(ROOT/'observer_05_formal_start.json'),
 'new_whole_model_forwards_observed_lower_bound':new_forwards,
 'new_optimizer_updates_including_capacity_observed_lower_bound':new_updates,
 'scientific_status':'UNVALIDATED_FIXED12_NOT_REACHED','B_reused_not_counted_as_new':True,
 'no_retry_no_extra_queue':True}
board_path=BASE/'01_候选判别表.json';state_path=BASE/'15_当前状态与结果校验.json'
board=load(board_path);state=load(state_path);now=datetime.now(timezone.utc).isoformat()
board['candidates'].append({'id':'A51_CROSS_RESOLUTION_PAIRED_EXCHANGE','status':'MINIMAL_FIXED12_PILOT_ACTIVE_NOT_VALIDATED',
 'operation':'learned orthogonal channel alignment + coupled cross-grid exchange versus parameter-matched independent direction gates',
 'history_difference':'noP2/quota/teacher/fine-read-radius/BN/ignore-loss; paired update constraint on existing P3P4P5',
 'evidence':entry})
board.update(status='ACTIVE_A51_ENCODER_PILOT_MAINLINE_NOT_CONFIRMED',goal_status='active',mainline_validated=False,
 mainline_selected_for_pilot=True,selection_stage_complete=False,pilot_validation_status='A51_RUNNING',
 current_priority_candidate='A51_CROSS_RESOLUTION_PAIRED_EXCHANGE',
 current_priority_question='跨分辨率共同更新约束是否优于原版和同参数独立门控交换？',
 current_host_candidate='ECDet-S fixedd8752b75; VisDrone train6471/val548',
 latest_report=str(report),latest_update_utc=now,A51_execution=entry)
state.update(recorded_utc=now,goal_status='active',mainline_validated=False,mainline_selected_for_pilot=True,
 selection_stage_complete=False,current_phase='A51_M12_TRAINING',latest_authoritative_phase='A51_PILOT_STARTUP',
 current_user_task_status='ACTIVE_FIND_MAINLINE_WITH_BOUNDED_TRAINING',
 current_priority_candidate=board['current_priority_candidate'],current_priority_question=board['current_priority_question'],
 current_native_job_terminal=False,model_forwards=144957+new_forwards,cap=plan['cumulative_whole_model_forward_ceiling'],
 full_forward_cap_closed=False,counts_terminal=False,training_counts_terminal=False,
 counters_observed_lower_bounds_while_job_live=True,
 formal_detector_training_updates=117022+m['optimizer_updates'],capacity_only_optimizer_updates=43,
 detector_optimizer_updates_including_capacity=117059+new_updates,training_updates=117059+new_updates,
 formal_training_epochs=144,latest_report=str(report),A51_execution=entry)
state['training_update_scope']='Observed lower bound while A51 live; includes6 capacity updates; B12 reused, no double count.'
save(board_path,board);save(state_path,state)
receipt=ROOT/'08_A51启动与历史保留校验回执.json';assert not receipt.exists()
signed=[board_path,state_path,report,ROOT/'04_候选问题_最近方法_最小验证.txt',ROOT/'05_核心文献矩阵与阅读范围_v1.json',
 ROOT/'protocol_FROZEN_v1.json',ROOT/'capacity_protocol_FROZEN_v1.json',ROOT/'capacity_result.json',ROOT/'math_check_v1.json']
save(receipt,{'status':'PASS','phase':'A51_STARTUP','goal_status':'active','mainline_confirmed':False,
 'previous_A50_receipt':str(old_path),'previous_A50_receipt_sha256':sha(old_path),
 'previous_signed_current_inputs_resolved':resolved,'current_files_sha256':{str(p):sha(p) for p in signed},
 'observed_counts_lower_bounds':{'whole_model_forwards':144957+new_forwards,'optimizer_including_capacity':117059+new_updates},
 'forward_ceiling':165284,'new_training_arms':['M12','Q12'],'B_reused_not_new':True,
 'supervisor_identity':obs['formal_launch'],'no_extra_training_queue':True})
save(ROOT/'CURRENT_A51.json',{'status':entry['state'],'goal_status':'active','mainline_confirmed':False,
 'receipt':str(receipt),'receipt_sha256':sha(receipt),'latest_observation':entry['observation'],
 'bridge_session_if_alive':86970,'remote_root':'/root/uav_a51_20261101_v1',
 'supervisor_pid':7798,'supervisor_start_ticks':21400942,'M_pid':7799,'M_start_ticks':21400945,
 'new_training_arms':['M12','Q12'],'next':'observe fixed4,8,12; archive results; no new method without checking actual terminal outcome'})
print(json.dumps({'status':'A51_STARTUP_PUBLISHED','goal_status':'active','mainline_confirmed':False,
 'receipt_sha256':sha(receipt),'observed_forward_lower_bound':144957+new_forwards,'forward_ceiling':165284}))
