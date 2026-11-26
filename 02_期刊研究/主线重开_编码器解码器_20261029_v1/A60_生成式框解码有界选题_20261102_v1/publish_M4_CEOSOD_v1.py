"""Publish the original M4 evidence; preserve prior roots and all training files."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, shutil

P=Path(__file__).resolve().parent
B=P.parent
D=B/'A54_场景尺度约束候选_20261101_v1'
ROOTS=[B/'01_候选判别表.json',B/'15_当前状态与结果校验.json',P/'CURRENT_A60.json']
PARENT=P/'A60_closest_prior_v2_preservation_receipt_v1.json'
SNAP=P/'before_M4_CEOSOD_publish_v1'
REPORT=P/'17_M4首评核验与CEOSOD公开方法补读.txt'
HANDOFF=P/'A60_turn_handoff_M4_CEOSOD_v1.json'
RECEIPT=P/'A60_M4_CEOSOD_preservation_receipt_v1.json'
OBSERVATION=P/'formal_live_observation_297_v1.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def meta(p):return {'bytes':p.stat().st_size,'sha256':sha(p)}
def new(p,value):
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    assert all(not p.exists() for p in [SNAP,HANDOFF,RECEIPT,OBSERVATION]) and REPORT.is_file()
    parent=read(PARENT)
    for p in ROOTS:assert meta(p)==parent['files'][str(p)],str(p)
    vpath=P/'M4_saved_metrics_local_verification_v1.json';v=read(vpath)
    assert v['status']=='PASS_ORIGINAL_M4_SAVED_METRICS_AND_FIRST4_RAW_STREAMS'
    assert not v['candidate_Q_tested'] and not v['fixed12_decision_available']
    assert v['new_model_forwards']==v['new_optimizer_updates']==v['new_COCO_evaluations']==0
    assert sha(D/'A60_downloads/M_epoch4_saved_metrics_bundle_v1.zip')==v['archive_sha256']
    supp=P/'02d_CEOSOD公开材料与条件归因边界_v1.json';s=read(supp)
    assert s['status']=='BOUNDED_CEOSOD_AUTHOR_FIGURE_AND_CONFIG_READING'
    assert not s['experiment_scope']['protocol_changed'] and not s['experiment_scope']['user_marked_read']
    for item in read(P/'CEOSOD_public_supplement_v3/fetch_receipt.json')['files']:
        assert item['status']==200 and meta(Path(item['path']))=={k:item[k] for k in ['bytes','sha256']}
    transport=read(D/'A60_formal_observe_297.json')
    assert transport['status']=='EXECUTED' and transport['exit']==0
    obs=json.loads(transport['stdout']);base,middle,candidate=(obs['arms'][a] for a in ['B','M','Q'])
    assert obs['source_binding'] and obs['supervisor']['identity_matched'] and obs['supervisor']['live']
    assert base['exit']['exit']==0 and base['completed_epochs']==12
    assert middle['process']['identity_matched'] and middle['process']['live']
    assert middle['launch']['pid']==35385 and middle['launch']['start_ticks']=='32308346'
    assert middle['completed_epochs']>=4 and middle['epochs'][3]['metrics']==v['M4_metrics']
    assert not candidate['exists']
    for i in range(4):assert middle['epochs'][i]['flow']['raw_stream_sha256']==base['epochs'][i]['flow']['raw_stream_sha256']
    starts=sum(a['journal_counts']['started'] for a in obs['arms'].values())
    updates=sum(a['journal_counts']['updates'] for a in obs['arms'].values())
    epochs=sum(a['completed_epochs'] for a in obs['arms'].values())
    current=read(ROOTS[2]);counter_names=[('global_observed_model_forwards','observed_formal_model_forwards',starts),
        ('global_observed_all_optimizer_updates','observed_formal_updates',updates),
        ('global_observed_formal_optimizer_updates','observed_formal_updates',updates),
        ('global_completed_formal_epochs','completed_formal_epochs',epochs)]
    totals={k:current[k]-current[old]+value for k,old,value in counter_names}
    SNAP.mkdir();resolution={}
    for p in ROOTS:
        dest=SNAP/p.name;shutil.copy2(p,dest);assert meta(dest)==parent['files'][str(p)]
        resolution[str(p)]={'resolved_path':str(dest),**meta(dest)}
    new(OBSERVATION,obs)
    now=datetime.now(timezone.utc).isoformat();status='A60_M4_VERIFIED_CEOSOD_PUBLIC_SUPPLEMENT_ORIGINAL_M_LIVE'
    current.update(totals)
    current.update({'status':status,'goal_status':'active','mainline_validated':False,'candidate_has_pilot_support':False,
        'candidate_selected_for_pilot':True,'recorded_utc':obs['recorded_utc'],'recorded_client_utc':now,
        'timestamp_scope':'Raw server UTC and client UTC are separate; original durations use monotonic.',
        'snapshot_is_live_lower_bound':True,'arms':obs['arms'],'original_supervisor':obs['supervisor'],
        'source_binding_verified':True,'observed_formal_model_forwards':starts,'observed_formal_updates':updates,
        'completed_formal_epochs':epochs,'M4_verified_evidence':v,'CEOSOD_public_method_supplement':str(supp),
        'M_AP_available':True,'candidate_Q_tested':False,'fixed12_comparison_available':False,
        'scope':'M4 saved arrays and first4 raw streams verified; M live, Q unstarted, CEOSOD full paper unread.',
        'previous_goal_turn_classification':'verified_wait: original M was live with no M detection metric',
        'current_goal_turn_classification':'progress: original M4 and CEOSOD public figures/configs verified',
        'latest_report':str(REPORT),'report':str(REPORT),'latest_handoff':str(HANDOFF),
        'next_readonly_receipt':'A60_formal_observe_298.json'})
    new(HANDOFF,{'status':status,'goal_status':'active','mainline_validated':False,
        'previous_goal_turn_classification':'verified_wait','this_goal_turn_classification':'progress',
        'recorded_client_utc':now,'recorded_remote_utc':obs['recorded_utc'],'snapshot':str(OBSERVATION),
        'canonical':str(ROOTS[2]),'report':str(REPORT),'M4_verification':str(vpath),
        'M4_metrics':v['M4_metrics'],'M4_minus_B4_pp':v['M_minus_B_pp'],
        'M_completed_epochs':middle['completed_epochs'],'M_current_epoch':middle['progress']['epoch'],
        'M_process':middle['process'],'M_launch':middle['launch'],'supervisor':obs['supervisor'],
        'Q_started':False,'candidate_Q_tested':False,'protocol_changed':False,'bridge_session_id':19671,
        'bridge_script':str(D/'remote_A54.py'),'bridge_receipt_directory':str(D),
        'next_readonly_receipt_hint':'A60_formal_observe_298.json','numbering_rule':'Derive from actual files.',
        'terminal_archive_preparation':str(P/'MQ_fixed12_archive_scope_FROZEN_v1.json'),
        'remaining_action':'Observe original M8/M12 and Q. After original worker exit/reap, run frozen terminal exporter exactly once per arm; download models/artifacts and verify all12 streams and frozen judgment. No retry or gate change.',
        'research_limit':'CEOSOD full method unread; Q-M also changes within-image time-label synchrony/distribution.',
        'completion_unproven':'Q results and fixed12 comparison remain missing.'})
    board,state=read(ROOTS[0]),read(ROOTS[1])
    shared={'status':status,'goal_status':'active','mainline_validated':False,'training_running':True,
        'latest_update_utc':now,'current_phase':'A60','A60_execution':current,'A60_M4_verified_evidence':v,
        'A60_CEOSOD_public_method_supplement':str(supp),'latest_report':str(REPORT)}
    board.update(shared);state.update(shared)
    state.update({'model_forwards':totals['global_observed_model_forwards'],
        'training_updates':totals['global_observed_all_optimizer_updates'],
        'detector_optimizer_updates_including_capacity':totals['global_observed_all_optimizer_updates'],
        'formal_detector_training_updates':totals['global_observed_formal_optimizer_updates'],
        'formal_optimizer_updates_observed':totals['global_observed_formal_optimizer_updates'],
        'formal_training_epochs':totals['global_completed_formal_epochs'],
        'previous_goal_turn_classification':'verified_wait','current_goal_turn':'PROGRESS_A60_M4_CEOSOD_PUBLIC'})
    for p,value in zip(ROOTS,[board,state,current]):
        tmp=p.with_name(p.name+'.M4_CEOSOD_publish_tmp');new(tmp,value);tmp.replace(p)
    files=dict(parent['files'])
    paths=[*ROOTS,*[Path(x['resolved_path']) for x in resolution.values()],PARENT,REPORT,HANDOFF,OBSERVATION,
        Path(__file__),supp,vpath,P/'verify_M4_saved_metrics_v1.py',D/'A60_formal_observe_296.json',
        D/'A60_formal_observe_297.json',D/'A60_M_epoch4_export_v1.json',D/'A60_M_epoch4_saved_metrics_download_v1.json',
        D/'A60_downloads/M_epoch4_saved_metrics_bundle_v1.zip',D/'A60_downloads/M_epoch4_saved_metrics_ready_v1.json']
    paths+=list((P/'CEOSOD_public_supplement_v3').glob('*'))
    paths+=[x for x in (P/'epoch4_M_saved_metrics_remote_v1').rglob('*') if x.is_file()]
    for p in paths:assert p.is_file(),str(p);files[str(p)]=meta(p)
    new(RECEIPT,{'status':status,'goal_status':'active','mainline_validated':False,
        'parent_receipt':str(PARENT),'parent_receipt_sha256':sha(PARENT),'parent_path_resolution':resolution,
        'parent_roots_rehashed_before_publish':3,'unaffected_parent_files_rehashed_this_turn':False,
        'verification_scope':'Root snapshots and new M4/CEOSOD artifacts; earlier file hashes inherited, not a fresh full-project recheck.',
        'files':files,'new_training_started':False,'training_protocol_modified':False})
    print(json.dumps({'status':status,'M_epoch':middle['progress']['epoch'],'M_completed':middle['completed_epochs'],
        'M_APs4':v['M4_metrics']['APs'],'M_minus_B_APs4':v['M_minus_B_pp']['APs'],'Q_started':False,
        'report':str(REPORT),'receipt_files':len(files)},ensure_ascii=False))

if __name__=='__main__':main()
