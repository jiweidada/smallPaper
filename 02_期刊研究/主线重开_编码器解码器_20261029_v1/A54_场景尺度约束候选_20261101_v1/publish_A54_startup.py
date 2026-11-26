"""Publish the new candidate and observed live run, preserving all older signed states."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
P=Path(__file__).resolve().parent;BASE=P.parent;ENG=BASE.parent;LIB=ENG/'文献_相关原始论文_20261004'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    receipt=P/'A54_候选与启动保留回执.json';assert not receipt.exists()
    prior=BASE/'A53_原版漏检与机制候选筛选_20261101_v1/A53_原文筛选保留回执.json';old=load(prior)
    assert sha(prior)=='80e4bf3ab80aa4311bfd1d307a8417dbe62eec8744c159bd9a82f42e4c90336f'
    assert all(sha(Path(p))==h for p,h in old['current_files_sha256'].items())
    obs_outer=load(P/'observer_01_startup.json');assert obs_outer['status']=='EXECUTED' and obs_outer['exit']==0
    obs=json.loads(obs_outer['stdout']);assert obs['supervisor_process']['live'] and obs['arms']['M']['process']['live']
    assert all(obs['frozen_files_unchanged'].values())
    init=obs['arms']['M']['initialization'];assert init['scratch_no_capacity_weights'] and init['all_native_initial_tensors_identical']
    cap=load(P/'capacity_v2/capacity_result.json');assert cap['status']=='PASS' and cap['cumulative_A54_capacity_forwards']==9
    before=P/'before_A54';before.mkdir()
    canonical=[BASE/'01_候选判别表.json',BASE/'15_当前状态与结果校验.json'];resolved={}
    for f in canonical:
        target=before/f.name;shutil.copyfile(f,target);assert sha(f)==sha(target)
        resolved[str(f)]={'preserved_path':str(target),'sha256':sha(target)}
    works=[
      ('Meta-YOLO: Metadata-Guided Real-Time Object Detector in Aerial Imagery','WACV2026','A54_MetaYOLO_WACV2026.pdf',
       'https://openaccess.thecvf.com/content/WACV2026/html/Yoon_Meta-YOLO_Metadata-Guided_Real-Time_Object_Detector_in_Aerial_Imagery_WACV_2026_paper.html',
       'Sections3.1–3.3; PDF page3 equations2–3 visually checked. Not all experiments/supplement.',
       'Normalized telemetry and x/y maps modulate FPN deformable convolution offsets.',
       'Scale/metadata conditioning and spatial offset modulation are established. Our candidate has no telemetry input and compares local vs shared scene-scale readout for reference geometry.'),
      ('Learned Two-Plane Perspective Prior Based Image Resampling for Efficient Object Detection','CVPR2023','A54_TwoPlane_CVPR2023.pdf',
       'https://openaccess.thecvf.com/content/CVPR2023/html/Ghosh_Learned_Two-Plane_Perspective_Prior_Based_Image_Resampling_for_Efficient_Object_CVPR_2023_paper.html',
       'Sections3.1–3.6 and part of3.7; PDF page4 equations4–8 visually checked. Not all experiments/supplement.',
       'Plane/vanishing-point geometry generates saliency for input warping, with scenario-specific learned parameters.',
       'Perspective-driven scale priors are established. Our fixed single-image decoder uses a learned log-size residual and shared affine readout, without input warping.'),
      ('Clustered Object Detection in Aerial Images','ICCV2019; arXiv1904.08008v3','A54_ClusDet_arXiv1904_08008.pdf',
       'https://arxiv.org/abs/1904.08008v3','Sections3.1/3.3; PDF page5 ScaleNet Figure4/equation1 visually checked. Not complete paper.',
       'Cluster scale estimation uses visual cluster features and initial detections; clusters are resized for detection and combined.',
       'Local group-scale prediction is established. Our candidate has no crop detection or NMS merger, and tests parameter-matched local vs projected scale-map readout.'),
      ('FlowDet: Unifying Object Detection and Generative Transport Flows','CVPRW2026 LoViF; arXiv2512.16771v1','65_FlowDet_CVPRW2026_LoViF.pdf',
       'https://arxiv.org/abs/2512.16771v1','Re-read local Section3.2.2 and equation9. Earlier reading scope retained; no new full-paper attestation.',
       'Data-derived and per-image GAP/MLP Gaussian statistics are prior options; size-bucketed mixtures are also considered.',
       'Per-image geometry priors are established. Our native query selection is retained, with no random proposal sampling or generative transport changes.')
    ]
    entries=[]
    for title,venue,file,url,scope,facts,diff in works:
        pdf=LIB/file
        # Locate the older FlowDet PDF by its pre-existing text basename if a filename differs.
        if not pdf.exists() and title.startswith('FlowDet'):
            matches=[p for p in LIB.glob('*.pdf') if 'FlowDet' in p.name];assert len(matches)==1,matches;pdf=matches[0]
        assert pdf.is_file();entries.append({'title':title,'publication':venue,'url':url,'pdf':str(pdf),'pdf_sha256':sha(pdf),'reading_scope':scope,'original_method_summary':facts,'candidate_difference_inference':diff,'full_paper_read':False})
    entries.extend([
      {'title':'GSDet: Object Detection in Aerial Images Based on Scale Reasoning','publication':'IEEE TIP2021','DOI':'10.1109/TIP.2021.3073319','url':'https://pubmed.ncbi.nlm.nih.gov/33886468/',
       'reading_scope':'Original-author abstract indexed by PubMed plus Meta-YOLO related-work passage; full text not acquired.',
       'original_abstract_summary':'GSD estimation combined with RoI size provides a physical-size prior for category feature reweighting.',
       'candidate_difference_inference':'A spatial, class-relative field and native reference-size operation differ from the described reweighting, but complete overlap exclusion is not claimed.','full_paper_read':False},
      {'title':'GSDDet: Ground Sample Distance-Guided Object Detection for Remote Sensing Images','publication':'IEEE TGRS2024; DOI2023','DOI':'10.1109/TGRS.2023.3309838','url':'https://ieeexplore.ieee.org/document/10285472',
       'reading_scope':'Author-homepage bibliographic identity and Meta-YOLO secondary description only; publisher returned anti-bot content.',
       'original_method_summary':None,'candidate_difference_inference':'Full-method comparison remains a gap. Not used as evidence of complete novelty clearance.','full_paper_read':False}
    ])
    hist=[{'id':'D041','state':'Historical implementation/protocol invalid, not a universal performance negative','difference':'Class/size score calibration versus shared spatial field readout and geometric initialization.'},
      {'id':'D069','state':'Uncompleted/unverified','difference':'Grid-relative center/fine geometry factorization versus unchanged native box parameterization plus a scene-shared size constraint.'},
      {'id':'A42/A43/A44/A47','state':'Prior stopping/withdrawal decisions retained','difference':'No P2 readout, distillation, scale quotas or GT jitter.'},
      {'id':'A48/A49','state':'Exact fixed12 recipes retired','difference':'No normalization-statistic replacement or ignore-gradient rule.'},
      {'id':'A50/A51/A52','state':'Original scoped screening/training decisions retained','difference':'No FDR-shape matching, cross-resolution exchange or pairwise distribution-geometry bias.'}]
    matrix={'phase':'A54','goal_status':'active','human_read_attested':False,'reading_actor':'assistant','systematic_review':False,
      'entries':entries,'history':hist,'source_download_record':str(P/'01_原文获取记录_v1.json'),
      'remaining_novelty_risk':'Geometry/scale priors and affine projection are established; full GSDDet/GSDet methods remain incomplete. A positive pilot would not itself prove publication novelty.'}
    save(P/'06_最近文献与历史逐项对比.json',matrix)
    progress=obs['arms']['M']['progress'];mf=progress['full_model_forwards'];up=progress['optimizer_updates'];ep=progress['completed_epochs']
    entry={'id':'A54_SCENE_SCALE_FIELD','status':'FORMAL_M12_RUNNING_Q12_PRESPECIFIED','question':'共享空间尺度场约束能否超过普通位置尺度辅助头，改善无人机小目标检测？',
      'operator':'Same pointwise auxiliary scale head; Q replaces local bilinear readout by a fixed ridge affine scene projection, used for ordinary reference-size updates.',
      'data':'VisDrone train6471/val548','host':'ECDet-S','new_parameters_each':6210,'strong_simple_control':'M ordinary pointwise auxiliary scale prediction with identical parameters/initialization/supervision',
      'report':str(P/'05_候选主线与最小验证说明.txt'),'matrix':str(P/'06_最近文献与历史逐项对比.json'),
      'protocol':str(P/'protocol_FROZEN_v1.json'),'protocol_sha256':sha(P/'protocol_FROZEN_v1.json'),'mainline_confirmed':False,
      'evidence_scope':'GT-only scale-coherence statistics, standalone operator checks, natural capacity, observed formal updates. No new detector AP result yet.',
      'next':'Observe the same owned PIDs; fixed12 EMA gates. No rescue sweep; retain prior untrained and negative scopes.'}
    board=load(canonical[0]);assert not any(x.get('id')==entry['id'] for x in board['candidates']);board['candidates'].append(entry)
    board.update(status='A54_FORMAL_M12_RUNNING_Q12_PRESPECIFIED',mainline_validated=False,current_priority_candidate=entry['id'],candidate_selected_for_pilot=True,latest_A54=entry)
    save(canonical[0],board)
    state=load(canonical[1]);assert state['model_forwards']==185612 and state['detector_optimizer_updates_including_capacity']==155903
    phase={'status':'FORMAL_M12_RUNNING_Q12_PRESPECIFIED','goal_status':'active','mainline_confirmed':False,
      'previous_goal_turn_classification':'progress: literature organization completed, retained as a research evidence resource',
      'report':entry['report'],'matrix':entry['matrix'],'protocol':entry['protocol'],'protocol_sha256':entry['protocol_sha256'],
      'capacity_source':str(P/'capacity_v2/capacity_result.json'),'capacity_forward_count_including_harness_failure':9,'capacity_only_optimizer_updates':6,
      'capacity_peak_allocated_GiB':{r['arm']:r['peak_allocated_bytes']/(1<<30) for r in cap['results']},
      'capacity_failure_preserved':str(P/'capacity_v1_result.json'),'observed_M_progress':progress,'observed_Q_state':'not started yet',
      'snapshot':obs,'snapshot_client_receipt':str(P/'observer_01_startup.json'),'bridge_session_if_alive':19671,
      'counts_are_observed_lower_bounds_while_live':True,'candidate_entry':entry}
    state.update(recorded_utc=datetime.now(timezone.utc).isoformat(),status=phase['status'],current_phase='A54_SCENE_SCALE_FIELD_FIXED12',current_priority_candidate=entry['id'],
      current_priority_question=entry['question'],candidate_selected_for_pilot=True,mainline_selected_for_pilot=True,mainline_validated=False,
      latest_authoritative_phase='A54',latest_report=entry['report'],current_user_task_phase='A54_FIXED12_MAINLINE_SEARCH',current_user_task_status='running',
      model_forwards=185612+9+mf,formal_detector_training_updates=155854+up,capacity_only_optimizer_updates=55,
      detector_optimizer_updates_including_capacity=155903+6+up,formal_training_epochs=192+ep,
      counts_terminal=False,counters_observed_lower_bounds_while_job_live=True,A54_execution=phase)
    save(canonical[1],state);save(P/'startup_state_v1.json',phase)
    old_doc=P/'05_候选主线与最小验证说明.txt';text=old_doc.read_text(encoding='utf-8')
    old_doc.write_text('阶段说明：正文写于GPU容量核查之前；最新容量/正式训练状态以CURRENT_A54.json与启动回执为准。\n\n'+text,encoding='utf-8')
    report=f'''A54已完成原生接入与24GB容量核查，M12正式运行，Q12按预定顺序接续。
截至服务器时间{obs['observed_server_unix']}：M第{progress['epoch']}轮已记录{progress['microbatch']}/809批、{up}次正式更新；原生与新增初始化检查通过、10份冻结本地输入SHA不变；Q尚未启动。此时没有新检测AP。
容量累计9次整网前向/6次临时更新，全部临时状态丢弃；M/Q显存峰值分别{cap['results'][0]['peak_allocated_bytes']/(1<<30):.4f}/{cap['results'][1]['peak_allocated_bytes']/(1<<30):.4f}GiB，包含最多902GT/图的实际batch8。
首版容量checker误把训练字段pred_corners用于eval，原版1次前向后退出，0更新。失败完整保留。修正只涉及checker与冗余eval次数，算法代码SHA未变；纠正版8次前向/6更新，总预算仍9/6。
M完整FP32输出与B初始值逐项差0；M/Q的零门参考输入均在真实调用中严格相等。Q未另做重复FP32整网eval，不能写成做过两组独立输出对比。
GT特权统计只支持尺度场的可预测性；实际AP收益、新颖性和跨数据泛化都未确立。M/Q各固定12轮，epoch4/8不挑最优；不通过门槛即保留结果并切换实质不同路线。
当前总目标active，尚未完成主线验证。原数据、代码、权重、旧裁决与A53合法性反例均保留。
'''
    (P/'07_正式启动与证据边界.txt').write_text(report,encoding='utf-8')
    tracked=canonical+[P/n for n in ['00_检索原始记录_v1.json','01_原文获取记录_v1.json','02_场景尺度统计核查_v1.json','03_异类支持核查_v1.json','04_算子CPU核查_v1.json','05_候选主线与最小验证说明.txt','06_最近文献与历史逐项对比.json','07_正式启动与证据边界.txt','protocol_FROZEN_v1.json','scene_scale_prior.py','startup_state_v1.json','capacity_v1_result.json','capacity_v2/capacity_result.json']]
    tracked.append(BASE/'A53_原版漏检与机制候选筛选_20261101_v1/08_合法性条件读出_原生插值反例_v1.json')
    rec={'status':'PASS','phase':'A54_CANDIDATE_AND_FORMAL_STARTUP','goal_status':'active','mainline_confirmed':False,
      'previous_receipt':str(prior),'previous_receipt_sha256':sha(prior),'previous_signed_current_inputs_resolved':resolved,
      'current_files_sha256':{str(f):sha(f) for f in tracked},'owned_live_identity':{'supervisor':obs['supervisor_process'],'M':obs['arms']['M']['process']},
      'progress_this_goal_turn':'Distinct concrete operator, GT-only evidence,13 CPU properties, bounded9/6 native capacity, and actual fixed12 training start.',
      'new_whole_model_forwards_lower_bound':9+mf,'new_optimizer_updates_including_capacity_lower_bound':6+up,'new_formal_epochs_completed_at_observation':ep,
      'next':'Observe the existing supervisor/child, verify all12 input/RNG flows and fixed12 EMA. No repeated capacity or rescue queue.'}
    save(receipt,rec)
    save(P/'CURRENT_A54.json',{'status':phase['status'],'goal_status':'active','mainline_confirmed':False,'current_priority_candidate':entry['id'],
      'receipt':str(receipt),'receipt_sha256':sha(receipt),'latest_report':str(P/'07_正式启动与证据边界.txt'),
      'owned_jobs_live_at_observation':True,'supervisor_pid':obs['supervisor_process']['pid'],'supervisor_start_ticks':obs['supervisor_process']['start_ticks'],
      'M_pid':obs['arms']['M']['process']['pid'],'M_start_ticks':obs['arms']['M']['process']['start_ticks'],'bridge_session_if_alive':19671,'remote_root':'/root/uav_a54_20261101_v1',
      'next':'Poll the same existing PIDs. No restart on observation timeout. Fixed12 M then Q, no extra queue.'})
    print(json.dumps({'status':phase['status'],'receipt_sha256':sha(receipt),'observed_M_updates':up,'cumulative_model_forward_lower_bound':state['model_forwards'],'cumulative_optimizer_update_lower_bound':state['detector_optimizer_updates_including_capacity']},ensure_ascii=False))
if __name__=='__main__':main()
