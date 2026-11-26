from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
A51=BASE/'A51_跨分辨率守恒交换_20261101_v1'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
obsp=ROOT/'observer_06_formal_progress.json';o=json.loads(load(obsp)['stdout'])
assert o['supervisor']['status']=='RUNNING' and o['supervisor']['stage']=='M12'
assert o['supervisor_process']['live'] and o['supervisor_process']['identity_matches']
m=o['arms']['M'];assert m['process']['live'] and m['process']['identity_matches']
assert m['progress']['optimizer_updates']>=100 and m['progress']['status']=='TRAINING'
assert all(o['local_frozen_files_unchanged'].values())
init=m['initialization'];assert init['all_native_initial_tensors_identical'] and init['scratch_no_capacity_weights']
assert init['model_parameters']==9809592 and init['extra_parameters']==6480
plan=load(ROOT/'protocol_FROZEN_v1.json')
assert sha(ROOT/'protocol_FROZEN_v1.json')==init['protocol_sha256']=='c13b61bdbc09d81a051108f7a28516f14ae3a3521a81d8247574129ca413d027'
prior=A51/'terminal_A51_preservation_receipt.json';assert sha(prior)=='02835f48befa1033c107edaeb275b0f5eedaded28daa90fb47917762c01d7631'
old=load(prior);preserved=load(ROOT/'00_A52开始与历史保留.json')['preserved_canonical_files']
for name,entry in preserved.items():
    assert sha(Path(name))==sha(Path(entry['preserved_path']))==entry['sha256']==old['current_files_sha256'][name]
cap=load(ROOT/'capacity_v2/capacity_result.json');assert cap['status']=='PASS'
updates=m['progress']['optimizer_updates'];forwards=m['progress']['full_model_forwards'];epochs=m['progress']['completed_epochs']
counts={'whole_model_forwards':165294+forwards,'formal_optimizer_updates':136438+updates,
        'capacity_only_updates':49,'all_optimizer_updates':136487+updates,'formal_epochs':168+epochs}
report=ROOT/'09_A52训练启动与当前状态.txt';receipt=ROOT/'08_A52启动与历史保留回执.json'
assert not report.exists() and not receipt.exists()
text=['A51固定12终点已完整核验并淘汰；总目标仍进行，尚未确认新主线。',
 '当前A52候选：同均值/同方差条件下，离散边界形状是否改善解码器成对关系。',
 '这是旧17号H2未训练备选的具体化；与A50匹配代价、D045供体mask、D072后处理保留实质作用位置差异，不把它称为全新发现。',
 '组别：B复用A49原版固定12；M同矩高斯积分关系；Q原生离散边际关系。M/Q相同6480新增参数、共同初始化、相同输入和目标流、同原生loss/query与预算；两分布计算不是逐操作同成本，时延/显存仅实测描述。',
 '容量v1因把训练输出检查用于eval而在B1次前向后退出，0更新；已保留原锁/源码/错误。仅修复该检查，隔离v2通过9前向/6临时更新，覆盖单图902GT批。故本阶段容量共10前向/6更新，不能漏掉前次错误。',
 'v2初始FP32 B/M/Q预测最大差0，各9个新增参数张量均变化；峰值约%.4fGiB；这些不是AP收益。'%(max(r['peak_allocated_bytes'] for r in cap['results'])/1024**3),
 '实际M已进入第%d轮、第%d/809批，已观察%d正式更新；loss %.6f，进程PID%d身份匹配。Q在M正常完成后才启动，各最多12轮；没有Q之后的训练队列。'%(m['progress']['epoch'],m['progress']['microbatch'],updates,m['progress']['loss'],m['process']['pid']),
 '原生初始化全部张量一致、新模块零输出初始化、容量权重未用于正式训练；第一轮完整顺序/首8批输入及RNG/全轮目标流须等epoch01文件后核验，此时尚未声称通过。',
 '固定12轮EMA三项晋级门槛：Q−B APs≥0.5pp、Q−M APs≥0.3pp、Q−B AP≥−0.2pp。失败不改频段、权重、层位、seed或轮次救援。总目标不会因一次否定结果完成。',
 '当前累计下界'+json.dumps(counts,ensure_ascii=False)+'；完整两组结束上限为185612整模型前向、155903含容量更新、192正式轮。B复用不重复记账。活性以observer_06实际进程身份为准。',
 'AP为冻结COCO式/class0 union/maxDets300，非官方VisDrone挑战分数；已有B/val多次可见，12轮单seed不是统计显著或可投稿证明。',
 '公式、先例、逐项差异与七项前期调研见04；读取边界见07。积分编码/特征函数/关系注意力不是新原理；离散边际也不是校准后验或同一对象概率。',
 '当前唯一工作是观察已冻结M/Q并保留完整终点；在终点裁决前不另加候选或训练队列。']
report.write_text('\n'.join(text)+'\n',encoding='utf-8')
board=load(BASE/'01_候选判别表.json');state=load(BASE/'15_当前状态与结果校验.json');now=datetime.now(timezone.utc).isoformat()
cid='A52_DISTRIBUTIONAL_PAIR_GEOMETRY'
candidate={'id':cid,'status':'FORMAL_M12_TRAINING_Q12_CONDITIONAL_B12_REUSED',
 'problem':text[1],'operator':'Detached physical marginal characteristic kernels as decoder ordinary-query pair bias',
 'M':'same-moment Gaussian relationship','Q':'native categorical relationship','extra_parameters_each':6480,
 'historical_scope':'old17 untrained H2 now concretized, not a renamed trained-negative rescue',
 'report':str(ROOT/'04_A52定向调研与候选协议.txt'),'registry':str(ROOT/'07_核心文献矩阵与历史差异_v1.json'),
 'fixed_protocol':str(ROOT/'protocol_FROZEN_v1.json'),'primary_gates_pp':plan['primary_gates_pp'],
 'efficacy_result':None,'mainline_confirmed':False}
assert not any(c.get('id')==cid for c in board['candidates']);board['candidates'].append(candidate)
execution={'state':'LIVE_FIXED12_M_THEN_Q','remote_root':'/root/uav_a52_20261101_v1','bridge_session_if_alive':33536,
 'supervisor_identity':o['supervisor_process'],'M_identity':m['process'],'Q_not_started_at_observation':True,
 'protocol_sha256':sha(ROOT/'protocol_FROZEN_v1.json'),'new_training_arms':['M','Q'],'B12_reused':True,
 'live_observation':str(obsp),'no_extra_queue_after_Q':True,'first_epoch_input_flow_verified':False}
board.update(status='A52_FORMAL_M12_TRAINING',goal_status='active',mainline_validated=False,mainline_selected_for_pilot=False,
 candidate_selected_for_pilot=True,selection_stage_complete=False,current_priority_candidate=cid,
 current_priority_question=text[1],latest_report=str(report),latest_update_utc=now,A52_execution=execution)
state.update(recorded_utc=now,status='A52_FORMAL_M12_TRAINING',goal_status='active',mainline_validated=False,
 mainline_selected_for_pilot=False,candidate_selected_for_pilot=True,selection_stage_complete=False,
 current_phase='A52_FIXED12_LIVE',latest_authoritative_phase='A52_LIVE_OBSERVATION_NOT_TERMINAL',
 current_priority_candidate=cid,current_priority_question=text[1],latest_report=str(report),A52_execution=execution,
 model_forwards=counts['whole_model_forwards'],cap=185612,formal_detector_training_updates=counts['formal_optimizer_updates'],
 capacity_only_optimizer_updates=49,training_updates=counts['all_optimizer_updates'],
 detector_optimizer_updates_including_capacity=counts['all_optimizer_updates'],formal_training_epochs=counts['formal_epochs'],
 counts_terminal=False,training_counts_terminal=False,full_forward_cap_closed=False,current_native_job_terminal=False,
 counters_observed_lower_bounds_while_job_live=True,
 training_update_scope='A52 live lower bounds; includes all49 lifetime capacity updates and the1 failed eval-guard forward. Reused B not counted twice.')
save(BASE/'01_候选判别表.json',board);save(BASE/'15_当前状态与结果校验.json',state)
signed=[BASE/'01_候选判别表.json',BASE/'15_当前状态与结果校验.json',report,ROOT/'04_A52定向调研与候选协议.txt',
 ROOT/'05_推理输出检查修复与隔离预检修订.json',ROOT/'06_容量通过与正式协议冻结.json',
 ROOT/'07_核心文献矩阵与历史差异_v1.json',ROOT/'protocol_FROZEN_v1.json',
 ROOT/'capacity_v2/capacity_result.json',ROOT/'capacity_v2/capacity_protocol_FROZEN_v1.json']
save(receipt,{'status':'PASS','phase':'A52_STARTUP','goal_status':'active','mainline_confirmed':False,
 'previous_receipt':str(prior),'previous_receipt_sha256':sha(prior),'previous_signed_current_inputs_resolved':preserved,
 'current_files_sha256':{str(p):sha(p) for p in signed},'observed_counts_lower_bounds':counts,
 'whole_forward_cap':185612,'all_optimizer_update_cap':155903,'formal_epoch_cap':192,
 'first_epoch_flow_verification_pending':True,'source_files_unchanged':True,'live_process_identities_verified':True,
 'publication_added_model_operations':0})
save(ROOT/'CURRENT_A52.json',{'status':'A52_FORMAL_M12_TRAINING','goal_status':'active','mainline_confirmed':False,
 'current_priority_candidate':cid,'receipt':str(receipt),'receipt_sha256':sha(receipt),'latest_report':str(report),
 'latest_observation':str(obsp),'remote_root':'/root/uav_a52_20261101_v1','bridge_session_if_alive':33536,
 'supervisor_pid':9930,'supervisor_start_ticks':22133831,'M_pid':9931,'M_start_ticks':22133834,
 'Q_not_started_at_observation':True,'first_epoch_input_flow_verified':False,
 'next':'Observe live M/Q fixed12; verify actual flows; no new candidate or additional queue before terminal judgment.'})
print(json.dumps({'status':'A52_FORMAL_M12_TRAINING','receipt_sha256':sha(receipt),'observed_counts_lower_bounds':counts,'mainline_confirmed':False}))

