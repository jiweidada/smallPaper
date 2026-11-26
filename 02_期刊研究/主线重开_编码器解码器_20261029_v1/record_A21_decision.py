"""Audit actual A21 path/asset evidence; prespecification does not allocate training."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,shutil,sys,zipfile
sys.stdout.reconfigure(encoding='utf-8')
R=Path(__file__).resolve().parent;O=R/'A21_共享状态路径隔离与基线协议_v1'
C=['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json']
REPORT='52_A21共享状态路径隔离与诊断研究设计.txt';RECEIPT=R/'53_A21共享状态路径与研究设计核验回执.json'
def rd(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wr(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def archive(name,expected_name,dirname):
    e=rd(O/expected_name);p=O/name;d=O/dirname
    assert sha(p)==e['sha256'] and p.stat().st_size==e['bytes']
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None and set(z.namelist())==set(e['members'])
        for n in z.namelist():assert z.read(n)==(d/n).read_bytes(),n
    return e,d
def main():
    assert not RECEIPT.exists(),'Already recorded; inspect, do not rerun'
    utc=datetime.now(timezone.utc).isoformat();snap=R/'history_before_A21_freeze_v1'
    old=rd(R/'51_A20严格装载与容量实测核验回执.json')
    for row in old['files']:assert sha((snap if row['path'] in C else R)/row['path'])==row['sha256'],row['path']
    eng,d=archive('engineering_evidence.zip','expected_engineering_evidence.json','verified_remote_engineering')
    data,dd=archive('data_control_evidence.zip','expected_data_control_evidence.json','verified_remote_data_controls')
    runtime=rd(d/'terminal_runtime.json')
    for n,h in runtime['file_hashes'].items():assert sha(d/n if (d/n).exists() else O/n)==h,n
    assert runtime['compute_apps']==''
    result=rd(d/'engineering_result.json');protocol=rd(O/'engineering_protocol_FROZEN.json')
    assert (d/'engineering_protocol_FROZEN.json').read_bytes()==(O/'engineering_protocol_FROZEN.json').read_bytes()
    assert result['protocol_sha256']==sha(O/'engineering_protocol_FROZEN.json')
    for row in protocol['files']:assert sha(O/row['path'])==row['sha256'],row['path']
    assert result['status']=='STATE_MEDIATOR_INSTRUMENTATION_PASS_NOT_SCIENTIFIC_EFFECT'
    assert result['complete_model_forwards']==5 and result['optimizer_updates']==0
    assert result['scientific_outcomes_estimated'] is False and result['GT_or_val_predictions_read'] is False
    events=[json.loads(s) for s in (d/'engineering_events.jsonl').read_text().splitlines() if s.strip()]
    assert [x['index'] for x in events if x['event']=='COMPLETE_FORWARD_ENTER']==[1,2,3,4,5]
    assert events[-1]['event']=='ENGINEERING_TERMINAL_PASS'
    stages=result['stages'];assert len(stages)==5
    for i in [1,2]:
        for k in ['logits','boxes']:assert stages[i][k]==stages[0][k]
        assert stages[i]['operator']['merged_state_plus_local_skip']==stages[0]['operator']['merged_state_plus_local_skip']
    assert stages[0]['operator']['inputs']==stages[4]['operator']['inputs']
    for k in ['fixed_read_Q_pre','fixed_rotated_Q','fixed_local_V','fixed_local_D']:
        assert stages[1]['state_observer'][k]==stages[4]['state_observer'][k]
    assert stages[4]['state_observer']['applied_state']==stages[3]['state_observer']['native_state']
    assert stages[4]['state_observer']['applied_mean']==stages[3]['state_observer']['native_mean']
    assert result['source_audit']['AST_insertions']==1 and result['source_audit']['source_file_modified'] is False
    assert result['native_layer_state_shapes']=={'state':[1,4,12,24],'mean':[1,4,1,12]}
    assert sha(d/'engineering_raw_outputs.pt')==result['raw_outputs_sha256']
    train=rd(O/'train_asset_manifest.json');remote_train=rd(dd/'remote_training_data_integrity.json')
    assert train['image_count']==remote_train['images']==6471 and train['valid_GT_count']==remote_train['valid_GT']==343204
    assert train['aggregate_sha256']==remote_train['aggregate_sha256']==data['train_manifest_aggregate']
    assert sha(O/'verify_assets_and_pixel_controls.py')==data['source_script_sha256']
    cohort=rd(O/'scientific_cohort_asset_manifest.json');pixel=rd(dd/'cohort_pixel_control_preparation.json')
    assert len(cohort['cases'])==33 and cohort['train_exact_image_hash_overlap']==0 and cohort['train_filename_prefix_overlap_count']==6
    assert sum(not x['filename_prefix_also_in_train'] for x in cohort['cases'])==27
    assert pixel['primary_cases']==27 and pixel['secondary_cases']==6 and pixel['zero_energy_cases']==0
    assert not pixel['CUDA_initialized'] and not pixel['scientific_outcomes_estimated'] and pixel['new_model_forwards']==0
    assert pixel['rules_sha256']==sha(O/'cohort_pixel_control_rules_FROZEN.json')
    assert (dd/'cohort_pixel_control_rules_FROZEN.json').read_bytes()==(O/'cohort_pixel_control_rules_FROZEN.json').read_bytes()
    for x in pixel['cases']:
        assert x['target_pixels_unchanged'] and x['outside_masks_unchanged']
        assert abs(x['actual_perturbation_norms'][0]-x['actual_perturbation_norms'][1])<=max(1e-7,1e-5*x['common_energy'])
    scientific=rd(O/'scientific_design_PRESPECIFIED.json')
    assert len(scientific['arms'])==11 and scientific['planning_budget_not_allocation']['actual_new_training_or_assay_calls_allocated_by_this_file']==0
    assert scientific['planning_budget_not_allocation']['total_planned_calls_upper_bound']==38832+548+330
    assert scientific['mainline_validated'] is False
    seal={'utc':utc,'status':'SCIENTIFIC_DESIGN_PRESPECIFIED_EXECUTION_SOURCE_PENDING',
       'scientific_design_sha256':sha(O/'scientific_design_PRESPECIFIED.json'),
       'cohort_manifest_sha256':sha(O/'scientific_cohort_asset_manifest.json'),
       'pixel_rules_sha256':sha(O/'cohort_pixel_control_rules_FROZEN.json'),
       'engineering_source_sha256':sha(O/'state_path_bridge.py'),
       'formal_detector_training_updates_at_seal':0,'trained_detector_validation_outcomes_inspected':False,
       'primary_cases':27,'secondary_cases':6,'training_or_assay_execution_allocated':False,
       'new_user_permission_gate':False,'next':'Implement bounded12epoch native-host diagnostic training/data/analysis code, bind complete source/config and allocate before launching; retain full research objective'}
    wr(O/'scientific_design_seal.json',seal)
    pdf=R.parent/'文献_相关原始论文_20261004/Activation_Patching_Best_Practices_ICLR2024_2309.16042v2.pdf'
    text=pdf.parent/'_text'/pdf.with_suffix('.txt').name
    assert pdf.read_bytes().startswith(b'%PDF') and sha(pdf)=='8ba9f928e321ad2f319bf0ac62ae582d8a70befbe88d0f85773302d64252d396'
    assert text.is_file()
    wr(O/'methodology_reference_archive.json',{'title':scientific['methodology_reference']['title'],
      'url':'https://arxiv.org/abs/2309.16042v2','PDF_url':'https://arxiv.org/pdf/2309.16042v2',
      'PDF':str(pdf),'PDF_sha256':sha(pdf),'PDF_bytes':pdf.stat().st_size,'pages':27,
      'text':str(text),'text_sha256':sha(text),'text_extracted':True,
      'assistant_read_scope':'ArXiv/ICLR metadata and abstract only; full methods not read','human_read_attestation':False})
    opfile=R.parent/'前期调研_码率受限航拍小目标_20261028_v1/远端验证_20261029_v1/ssh_operation_receipts.jsonl'
    ops=[json.loads(s) for s in opfile.read_text(encoding='utf-8').splitlines() if s.strip()]
    ops=[x for x in ops if str(x.get('id','')).startswith('A21_')]
    runs=[x for x in ops if x['id']=='A21_execute_frozen_state_mediation_engineering_5_calls_max']
    assert len(runs)==1 and runs[0]['exit_code']==0 and json.loads(runs[0]['stdout'])['result']==result
    last=next(x for x in ops if x['id']=='A21_final_runtime_snapshot_no_model_calls')
    assert last['exit_code']==0 and len(last['stdout'].strip().splitlines())==1
    wr(O/'remote_operation_receipts.json',{'operations':ops,'scope':'A21 engineering/data only, no formal training launched'})
    docs={n:rd(R/n) for n in C};board,search,matrix,state,source=[docs[n] for n in C]
    assert len(board['candidates'])==14 and len(matrix['rows'])==76
    assert state['model_forwards']==255 and state['cap']==260 and state['training_updates']==2 and state['formal_detector_training_updates']==0
    assert not any('2309.16042' in json.dumps(x) for x in matrix['rows'])
    matrix['rows'].append({'id':'ActivationPatching_BestPractices','title':scientific['methodology_reference']['title'],
      'authors':['Fred Zhang','Neel Nanda'],'url':'https://arxiv.org/abs/2309.16042v2','tier':'ICLR2024 methodology reference, not UAV detector',
      'read_scope':'Author/ICLR metadata and abstract only; archived27pagePDF/text, full methods not read',
      'author_conclusion':'In studied LMs, corruption and metric variants yield different patching conclusions',
      'our_inference':'Prespecify/image-control/metric choices and validate instrumentation; patching itself is not a new detector algorithm',
      'unresolved':'Does not establish validity/power/novelty of this UAV assay or any harmful interaction'})
    ev={'utc':utc,'report':str(R/REPORT),'new_complete_model_forwards':5,'new_optimizer_updates':0,'new_suffix_forwards':0,
      'formal_training_epochs':0,'new_fits':0,'core_literature_rows':77,'methodology_addition_abstract_only':True,
      'native_noop_final_logits_boxes_bitwise_equal':True,'self_state_mean_final_logits_boxes_bitwise_equal':True,
      'source_local_read_inputs_preserved_in_hybrid':True,'donor_state_and_mean_exact':True,
      'fixed_query_count':500,'state_shape':[1,4,12,24],'mean_shape':[1,4,1,12],
      'train_images_hash_matched':6471,'train_valid_GT':343204,'train_exact_image_overlap_with_cohort':0,
      'primary_cases_prefix_disjoint':27,'sensitivity_shared_prefix_cases':6,
      'pixel_control_target_protection_and_energy_pass':True,'trained_detector_effect_estimated':False,
      'distinct_algorithm_verified':False,'mainline_validated':False,
      'scientific_design_status':seal['status'],'new_scientific_training_or_assay_calls_allocated':0}
    priority='邻近内容经共享状态路径影响微小目标定位仍为待检验问题。S/M路径隔离及像素/数据控制已验证；科学选择已预先约定。下一步实现并冻结有界12轮诊断基线执行代码及统计代码，训练后先查学习充分性，再按27主/6敏感性样本与11组固定控制初筛。无新算法或已确认缺陷。'
    for c in board['candidates']:
        if c['id']=='N_NONCAUSAL_STATE_INSTANCE_INTERFERENCE':
            c.update(status='STATE_MEDIATION_INSTRUMENTATION_PASS_TRAINED_MECHANISM_UNTESTED',A21_evidence=ev,
              question_revised_before_outcomes='Neighbour pixel content mediated by shared S/M, not pure GT-box token ownership',
              next='Implement/pin bounded12epoch diagnostic execution and prespecified analysis; no posthoc block/phase/mask rescue')
    board.update(A21_evidence=ev,latest_update_utc=utc,latest_report=str(R/REPORT),current_priority_question=priority,mainline_validated=False)
    search['A21_state_path_and_methodology']={**ev,'primary_web_record':str(R/'source_audit_v1/A21_methodology_primary_web_record.json'),
       'query_count':2,'scope':'Methodological reference and native code/data checks; not a novelty clearance search'}
    search['supplement_utc']=utc
    matrix.update(updated_utc=utc,latest_report=str(R/REPORT),latest_empirical_boundary='A21 untrained-host instrumentation5calls only; no trained effects, AP or novel algorithm')
    source['A21_state_mediation']={**ev,'source_patch_AST_insertions':1,'source_author_files_modified':False,
       'engineering_result':str(d/'engineering_result.json'),'scientific_design':str(O/'scientific_design_PRESPECIFIED.json'),
       'scientific_design_seal':str(O/'scientific_design_seal.json'),'data_receipt':str(dd/'remote_training_data_integrity.json')}
    source['next_instance_ownership_question']={'question':priority,'execution_source_pending':True,
      'scientific_design':str(O/'scientific_design_PRESPECIFIED.json'),'new_forward_budget_allocated':0,'new_user_permission_required':False}
    source['updated_utc']=utc
    state.setdefault('authoritative_phase_history',[]).append(state['latest_authoritative_phase'])
    state.update(recorded_utc=utc,model_forwards=260,cap=260,current_goal_turn='progress_A21_native_state_mediation_and_blinded_scientific_design_verified',
      latest_report=str(R/REPORT),current_priority_question=priority,goal_status='ACTIVE',mainline_validated=False,
      cap_scope='260complete calls throughA21:255prior+5engineering;260capCLOSED.72historicalsuffix calls separate; no scientific training/assay allocation yet.')
    state['latest_authoritative_phase']={**ev,'phase':'A21','classification':'progress','goal_status':'active',
      'status':'ENGINEERING_AND_DATA_COMPLETE_SCIENTIFIC_EXECUTION_PENDING','full_model_forwards':260,
      'cumulative_authorized_protocol_cap':260,'full_forward_cap_closed':True,'complete_mainline':False,
      'bounded_phase_complete':True,'actual_cumulative_detector_optimizer_updates':2,'formal_detector_training_updates':0,'next':priority}
    before=R/'history_before_A21_result_update_v1';before.mkdir(exist_ok=False)
    for n in C:shutil.copy2(R/n,before/n)
    for n,x in docs.items():wr(R/n,x)
    task=[p for p in sorted(O.iterdir()) if p.is_file()]
    names=C+[REPORT,'record_A21_decision.py','source_audit_v1/A21_methodology_primary_web_record.json']
    names += [str(p.relative_to(R)).replace('\\','/') for p in task]
    names += [str(p.relative_to(R)).replace('\\','/') for p in sorted(d.iterdir()) if p.is_file()]
    names += [str(p.relative_to(R)).replace('\\','/') for p in sorted(dd.iterdir()) if p.is_file()]
    receipt={'utc':utc,'status':'A21_STATE_PATH_IMPLEMENTATION_DATA_CONTROLS_AND_PRESPECIFIED_DESIGN_VERIFIED_GOAL_ACTIVE',**ev,
      'goal_status':'ACTIVE','cumulative_complete_model_forwards':260,'complete_forward_cap':260,'complete_forward_cap_closed':True,
      'actual_cumulative_detector_optimizer_updates':2,'formal_detector_training_updates':0,'candidate_rows':14,
      'historical_suffix_forwards':72,'historical_CPU_diagnostic_fits':64,'prior_A20_hashes_verified':len(old['files']),
      'prior_canonical_snapshot':str(snap),'engineering_archive_sha256':eng['sha256'],'engineering_archive_members':8,
      'data_control_archive_sha256':data['sha256'],'data_control_archive_members':6,
      'files':[{'path':n,'sha256':sha(R/n),'bytes':(R/n).stat().st_size} for n in names],
      'external_archived_reference_files':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in [pdf,text]],
      'next':'Implement bounded diagnostic training and prespecified evaluator; freeze complete execution code and ledger before any new model call',
      'limits':'No trained detector outcomes, accuracy improvement, convergence, harmful-neighbour cause, distinct increment or final mainline'}
    wr(RECEIPT,receipt)
    print(json.dumps({k:receipt[k] for k in ['status','goal_status','mainline_validated','cumulative_complete_model_forwards',
      'actual_cumulative_detector_optimizer_updates','formal_detector_training_updates','core_literature_rows','primary_cases_prefix_disjoint','sensitivity_shared_prefix_cases']},ensure_ascii=False))
if __name__=='__main__':main()
