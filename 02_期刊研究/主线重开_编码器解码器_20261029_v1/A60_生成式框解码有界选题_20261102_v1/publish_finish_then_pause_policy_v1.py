"""Carry the latest user boundary into the A60 handoff; do not stop training."""
from datetime import datetime,timezone
from pathlib import Path
import ast,hashlib,json,shutil

P=Path(__file__).resolve().parent
B=P.parent
D=B/'A54_场景尺度约束候选_20261101_v1'
ROOTS=[B/'01_候选判别表.json',B/'15_当前状态与结果校验.json',P/'CURRENT_A60.json']
SNAP=P/'before_finish_then_pause_policy_v1'
HANDOFF=P/'A60_turn_handoff_finish_then_pause_v1.json'
RECEIPT=P/'A60_finish_then_pause_policy_preservation_receipt_v1.json'
POLICY=P/'USER_FINISH_CURRENT_A60_THEN_PAUSE_v1.json'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def meta(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def new(p,v):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    assert all(not x.exists() for x in [SNAP,HANDOFF,RECEIPT])
    board,state,current=map(read,ROOTS)
    parent_path=Path(current['latest_preservation_receipt']);parent=read(parent_path)
    for p in ROOTS:assert meta(p)==parent['files'][str(p)],str(p)
    policy=read(POLICY);assert policy['user_words_verbatim']=='跑完实验就先整理暂停吧'
    assert policy['goal_status_until_condition_met']=='active' and policy['goal_status_after_condition_met']=='paused'
    ast.parse((P/'assemble_final_A60_v1.py').read_text(encoding='utf-8'))
    transport=read(D/'A60_formal_observe_331.json');assert transport['status']=='EXECUTED' and transport['exit']==0
    obs=json.loads(transport['stdout']);m=obs['arms']['M']
    assert obs['source_binding'] and obs['supervisor']['identity_matched'] and obs['supervisor']['live']
    assert obs['supervisor']['pid']==30238 and m['process']['pid']==35385 and m['process']['identity_matched'] and m['process']['live']
    assert obs['arms']['B']['completed_epochs']==12 and obs['arms']['B']['exit']['exit']==0
    assert not obs['arms']['Q']['exists']
    SNAP.mkdir();resolution={}
    for p in ROOTS:
        dest=SNAP/p.name;shutil.copy2(p,dest);assert meta(dest)==parent['files'][str(p)]
        resolution[str(p)]={'resolved_path':str(dest),**meta(dest)}
    starts=sum(x['journal_counts']['started'] for x in obs['arms'].values())
    updates=sum(x['journal_counts']['updates'] for x in obs['arms'].values())
    epochs=sum(x['completed_epochs'] for x in obs['arms'].values())
    for k,old,value in [('global_observed_model_forwards','observed_formal_model_forwards',starts),
        ('global_observed_all_optimizer_updates','observed_formal_updates',updates),
        ('global_observed_formal_optimizer_updates','observed_formal_updates',updates),
        ('global_completed_formal_epochs','completed_formal_epochs',epochs)]:current[k]=current[k]-current[old]+value
    now=datetime.now(timezone.utc).isoformat();status='A60_ORIGINAL_RUNNING_FINISH_ORGANIZE_THEN_USER_PAUSE'
    handoff={'status':status,'goal_status':'active','mainline_validated':False,
        'previous_goal_turn_classification':'verified_wait','this_goal_turn_classification':'progress: latest user completion boundary persisted',
        'recorded_client_utc':now,'recorded_remote_utc':obs['recorded_utc'],'user_instruction':policy,
        'original_supervisor':obs['supervisor'],'original_M':m['process'],'M_current_epoch':m['progress']['epoch'],
        'M_completed_epochs':m['completed_epochs'],'Q_started':False,
        'source_observation':str(D/'A60_formal_observe_331.json'),'bridge_session_id':19671,
        'bridge_script':str(D/'remote_A54.py'),'bridge_receipt_directory':str(D),
        'next_readonly_receipt_hint':'A60_formal_observe_332.json','numbering_rule':'Use actual filenames; previous hints may be consumed.',
        'terminal_archive_scope':str(P/'MQ_fixed12_archive_scope_FROZEN_v1.json'),
        'archive_commands':read(P/'A60_turn_handoff_MQ_archive_prepared_v1.json')['commands'],
        'final_assembly_script':str(P/'assemble_final_A60_v1.py'),'final_assembly_executed':False,
        'final_pause_tool_called':False,'latest_preservation_receipt':str(RECEIPT),
        'remaining_action':'Complete original M/Q fixed12; archive and verify once per arm after exit/reap, organize all B/M/Q outcomes, verify terminal empty queue, then update_goal paused under explicit user instruction. Do not search/start another candidate after this batch.',
        'new_training_started_by_this_update':False,'original_training_stopped_by_this_update':False}
    new(HANDOFF,handoff)
    current.update({'status':status,'goal_status':'active','recorded_utc':obs['recorded_utc'],'recorded_client_utc':now,
        'arms':obs['arms'],'original_supervisor':obs['supervisor'],'source_binding_verified':True,
        'observed_formal_model_forwards':starts,'observed_formal_updates':updates,'completed_formal_epochs':epochs,
        'pause_after_current_batch':policy,'post_A60_training_queue':[],
        'latest_handoff':str(HANDOFF),'latest_preservation_receipt':str(RECEIPT),
        'next_readonly_receipt':'A60_formal_observe_332.json','current_goal_turn_classification':handoff['this_goal_turn_classification']})
    for x in [board,state]:x.update({'status':status,'goal_status':'active','training_running':True,
        'latest_update_utc':now,'A60_execution':current,'pause_after_current_batch':policy,
        'post_A60_training_queue':[],'latest_handoff':str(HANDOFF),'latest_preservation_receipt':str(RECEIPT)})
    row=next(x for x in board['candidates'] if x['id']=='A60_GEOMETRIC_PROGRESS_BOX_DECODER')
    row.update({'status':status,'pause_after_current_batch':str(POLICY),'post_A60_training_queue':[],
        'available_detection_epochs':{a:[e['epoch'] for e in obs['arms'][a]['epochs'] if 'metrics' in e] for a in ['B','M','Q']}})
    state.update({'model_forwards':current['global_observed_model_forwards'],
        'training_updates':current['global_observed_all_optimizer_updates'],
        'detector_optimizer_updates_including_capacity':current['global_observed_all_optimizer_updates'],
        'formal_detector_training_updates':current['global_observed_formal_optimizer_updates'],
        'formal_optimizer_updates_observed':current['global_observed_formal_optimizer_updates'],
        'formal_training_epochs':current['global_completed_formal_epochs'],
        'previous_goal_turn_classification':'verified_wait','current_goal_turn':'PROGRESS_USER_FINISH_A60_THEN_PAUSE'})
    for p,v in zip(ROOTS,[board,state,current]):
        temp=p.with_name(p.name+'.finish_pause_policy_tmp');new(temp,v);temp.replace(p)
    files=dict(parent['files'])
    for p in [*ROOTS,*[Path(x['resolved_path']) for x in resolution.values()],parent_path,POLICY,HANDOFF,
              Path(__file__),P/'assemble_final_A60_v1.py',D/'A60_formal_observe_330.json',D/'A60_formal_observe_331.json']:
        files[str(p)]=meta(p)
    new(RECEIPT,{'status':status,'goal_status':'active','parent_receipt':str(parent_path),
        'parent_receipt_sha256':meta(parent_path)['sha256'],'parent_path_resolution':resolution,
        'unaffected_parent_files_rehashed_this_turn':False,'files':files,'pause_condition_met':False,
        'new_model_calls':0,'new_optimizer_updates':0,'training_protocol_changed':False})
    print(json.dumps({'status':status,'M_epoch':m['progress']['epoch'],'M_completed':m['completed_epochs'],
        'Q_started':False,'pause_requested_after_batch':True,'goal_status_now':'active'},ensure_ascii=False))

if __name__=='__main__':main()
