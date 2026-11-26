from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
start=load(ROOT/'00_A50开始与历史保留.json')
for original,x in start['prior_canonical_inputs_resolved'].items():
    assert sha(Path(original))==x['sha256']
    assert sha(Path(x['preserved_path']))==x['sha256']
result=load(ROOT/'activation_result.json')
assert result['status']=='STOP_INSUFFICIENT_DISTINCT_ASSIGNMENT' and result['whole_model_forwards']==16
assert result['optimizer_updates']==0 and result['totals']['small_Q_vs_M_changed']==12
board_path=BASE/'01_候选判别表.json';state_path=BASE/'15_当前状态与结果校验.json'
board=load(board_path);state=load(state_path)
now=datetime.now(timezone.utc).isoformat()
for row in board['candidates']:
    if row.get('id')=='A49_B_KNOWN_IGNORE_SUPERVISION':
        row['status']='RETIRED_FIXED12_GATES_FAILED'
        row['terminal_judgment']=str(ROOT.parent/'A49_下一主线有界筛选_20261101_v1'/'fixed_endpoint_judgment_A49.json')
        row['correction_scope']='Repair stale per-candidate ACTIVE label; top-level A49 terminal result already correct. No experiment restarted.'
board['candidates'].append({'id':'A50_FULL_EDGE_SHAPE_MATCHING',
 'status':'STOPPED_ACTIVATION_BELOW_FIXED_GATE_UNTRAINED',
 'question':'native FDR高阶形状用于分配是否提供超出同矩高斯的作用面',
 'result':{'small_Q_vs_M_owner_changes':12,'small_GT':1720,'fraction':12/1720,
           'model_forwards':16,'optimizer_updates':0},
 'negative_scope':'Exact frozen cost only; no AP training failure or rejection of all distribution-based assignment.'})
board.update(status='ACTIVE_NEW_ENCODER_METHOD_SCREEN_MAINLINE_NOT_CONFIRMED',goal_status='active',
 mainline_validated=False,mainline_selected_for_pilot=False,selection_stage_complete=False,
 current_priority_candidate='A51_CROSS_RESOLUTION_CONSERVATIVE_EXCHANGE_UNDER_REVIEW',
 current_priority_question='跨分辨率成对交换约束相对独立门控融合是否有新依据及可承担的训练判别？',
 latest_report=str(ROOT/'04_A50定向调研与筛选结论.txt'),latest_update_utc=now)
board['A50_terminal_result']={k:v for k,v in result.items() if k!='rows'}
board['selection_stage_scope']='A50 exact proposal stopped; broad research objective still active; A51 not yet pilot-approved.'
state.update(recorded_utc=now,goal_status='active',mainline_validated=False,
 mainline_selected_for_pilot=False,selection_stage_complete=False,
 current_phase='A51_BOUNDED_ENCODER_METHOD_REVIEW',latest_authoritative_phase='A50_ACTIVATION_TERMINAL',
 current_user_task_status='ACTIVE_FIND_MAINLINE',current_priority_candidate=board['current_priority_candidate'],
 current_priority_question=board['current_priority_question'],model_forwards=144957,cap=144957,
 full_forward_cap_closed=True,counts_terminal=True,counters_observed_lower_bounds_while_job_live=False,
 latest_report=board['latest_report'])
state['A50_terminal_result']=board['A50_terminal_result']
state['new_ops_A50']={'whole_model_forwards':16,'optimizer_updates':0,'formal_epochs':0}
state['current_native_job_terminal']=True
save(board_path,board);save(state_path,state)
receipt=ROOT/'05_A50终点与历史保留回执.json'
assert not receipt.exists()
signed=[board_path,state_path,ROOT/'04_A50定向调研与筛选结论.txt',ROOT/'activation_result.json',
 ROOT/'activation_protocol_FROZEN_v1.json',ROOT/'math_check_v1.json']
save(receipt,{'status':'PASS','phase':'A50_TERMINAL','goal_status':'active','mainline_confirmed':False,
 'previous_receipt':start['previous_terminal_receipt'],
 'previous_receipt_sha256':start['previous_terminal_receipt_sha256'],
 'previous_signed_current_inputs_resolved':start['prior_canonical_inputs_resolved'],
 'current_files_sha256':{str(p):sha(p) for p in signed},
 'cumulative_counts':{'whole_model_forwards':144957,'optimizer_including_capacity':117059,'formal_epochs':144},
 'new_training_queued':False,'next_candidate':'A51_UNDER_METHOD_REVIEW_NOT_TRAINED',
 'A49_stale_candidate_row_corrected':True})
save(ROOT/'CURRENT_A50.json',{'status':result['status'],'goal_status':'active','mainline_confirmed':False,
 'receipt':str(receipt),'receipt_sha256':sha(receipt),'bridge_session_if_alive':11147,
 'new_training_queued':False,'new_whole_model_forwards':16,'new_optimizer_updates':0})
print(json.dumps({'status':'A50_TERMINAL_PUBLISHED','goal_status':'active','mainline_confirmed':False,
 'receipt_sha256':sha(receipt),'cumulative_forwards':144957}))
