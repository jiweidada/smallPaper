from pathlib import Path
import hashlib,json,time

ROOT=Path(__file__).resolve().parent
LIB=ROOT.parents[1]/'文献_相关原始论文_20261004'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

if __name__=='__main__':
    response=json.loads((ROOT/'continuation_status_03.json').read_text(encoding='utf-8'))
    assert response['status']=='EXECUTED' and response['exit']==0
    obs=json.loads(response['stdout']);p=obs['files']['arm_B/progress.json']
    assert obs['baseline_live_identity']['present'] and obs['baseline_live_identity']['same_start_ticks']
    metrics=json.loads((ROOT/'B_epoch04_EMA_metrics.json').read_text())
    history=obs['files']['arm_B/completed_epochs.json']['evaluations']
    assert history[0]['epoch']==4
    for k,v in metrics.items():assert history[0]['metrics'][k]==v,k
    assert metrics['max_detections']==300 and metrics['evaluated_GT']==38602
    fresh={'goal_status':'active','mainline_confirmed':False,'candidate':'A48 instance-balanced training BN moments',
       'last_observed_server_unix':obs['observed_unix'],'reference_only_require_new_live_poll':True,
       'remote_root':'/root/uav_a48_20261101_v1','bridge_exec_session_if_alive':85299,
       'latest_progress':p,'latest_completed_evaluation':{'arm':'B','epoch':4,'metrics':metrics},
       'M_Q_status':'not started at this observation; queued after baseline12',
       'latest_source_report':str(ROOT/'09_续接核查_正式近邻与B4结果.txt'),
       'latest_remote_receipt':str(ROOT/'continuation_status_03.json'),
       'next_action':'Fresh-query same owned job, then compare completed B/M/Q at same fixed12 endpoint. Do not restart or retune.',
       'goal_scope_preserved':'Keep broad mainline search active if this recipe fails; a negative candidate does not complete goal.'}
    save(ROOT/'CURRENT_A48.json',fresh)
    entries=[]
    for name,scope in [('A48_CEASC_CVPR2023_Publisher.pdf','PDF3–4, Figure2, eq2–3 method read and visual check; not full'),
                       ('A48_Region_Normalization_arXiv1911_10375v1.pdf','PDF3–4 and Figure3, §3.2–3.3/eq13–16; not full, preprintv1')]:
        f=LIB/name;assert f.is_file();entries.append({'path':str(f),'sha256':sha(f),'scope':scope})
    entries += [
      {'title':'LearnableBN arXiv2411.18860v1','url':'https://arxiv.org/html/2411.18860v1','scope':'§3.2–3.4 eq1–9 original HTML; PDF local acquisition timed out; distinct from2026 final'},
      {'title':'LearnableBN AAAI2026 publisher final','doi':'10.1609/aaai.v40i9.37626','url':'https://ojs.aaai.org/index.php/AAAI/article/download/37626/41588',
       'scope':'Original cached publisher PDF methods6927–6928 eq1–10 including GCM+EM and dual-stage; not full experiments, local archive unavailable'},
      {'title':'Material Property Prediction Using Graphs Based on Generically Complete Isometry Invariants','doi':'10.1007/s40192-024-00351-9',
       'url':'https://link.springer.com/article/10.1007/s40192-024-00351-9','scope':'Publisher HTML Including PDD Weights eq4–6 and discussion; not full, local PDF archive timed out'}]
    save(LIB/'A48_research_reading_registry_v2.json',{'date':'2026-11-01','previous_registry':str(LIB/'A48_research_reading_registry_v1.json'),
       'previous_registry_sha256':sha(LIB/'A48_research_reading_registry_v1.json'),'new_reading_records':entries})
    names=['09_续接核查_正式近邻与B4结果.txt','continuation_primary_source_records_v1.json','source_completion_v1.json',
           'graph_source_acquisition_v1.json','B_epoch04_EMA_metrics.json','B_epoch04_input_flow.json',
           'B_epoch04_download.json','continuation_status_01.json','continuation_status_02.json','continuation_status_03.json']
    result={'goal_status':'active','mainline_confirmed':False,'previous_turn_classification':'progress',
       'this_turn_classification':'progress_and_verified_wait','new_detector_jobs':0,'new_optimizer_jobs':0,
       'observed_model_training_advances_external_to_tool_call':True,'owned_live_job':obs['baseline_live_identity'],
       'baseline_progress':p,'B_epoch4_metrics_from_raw_and_history_match':True,'candidate_AP_delta_available':False,
       'new_source_evidence':'CEASC and RN PDFs, published LearnableBN method, group-balanced normalization prior',
       'claims_restricted':'No new weighted-BN formula or first equal-group-statistics claim; task-specific application candidate only',
       'prior_handoff_receipt_sha256':sha(ROOT/'08_A48本轮交付校验.json'),
       'protocol_sha256':sha(ROOT/'protocol_FROZEN.json'),'files_sha256':{f:sha(ROOT/f) for f in names}}
    save(ROOT/'10_续接进展校验_v1.json',result)
    print(json.dumps({'status':'RECORDED','goal_status':'active','this_turn':'progress_and_verified_wait',
        'baseline':p,'baseline_epoch4_AP':metrics['AP'],'baseline_epoch4_APs':metrics['APs'],'mainline_confirmed':False}))
