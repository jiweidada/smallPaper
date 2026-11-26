"""Reconcile the completed bounded audit, source comparison, and current decision."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,shutil
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'A16_实际特征空间变化审计_v1'
SA=ROOT/'source_audit_v1'
receipt_path=ROOT/'43_A16完成核验与候选降级回执.json'
assert not receipt_path.exists()
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
summary=load(RUN/'analysis_summary.json');verify=load(RUN/'independent_response_verification.json')
capture=load(RUN/'capture_summary.json');download=load(RUN/'capture_download_verification.json')
assert summary['decision']=='NO_SUFFICIENT_ERROR_SPECIFIC_CANCELLATION_SIGNAL_NO_MODULE_TRAINING'
assert not summary['gate_passed'] and not summary['mainline_validated']
assert verify['GT_rows']==853 and verify['Gram_matrices']==2559
assert capture['new_full_model_forwards']==26 and capture['cumulative_full_model_forwards']==252
assert download['actual_exit_code']==0
for row in load(RUN/'freeze_manifest.json')['files']:assert sha(RUN/row['file'])==row['sha256']
utc=datetime.now(timezone.utc).isoformat()
report=ROOT/'42_A16真实特征汇聚核查与候选降级.txt';assert report.exists()
canonical=['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json',
           '15_当前状态与结果校验.json','16_最近工作源码核查.json']
history=ROOT/'history_before_A16_update_v1';history.mkdir()
for name in canonical:shutil.copyfile(ROOT/name,history/name)
shutil.copyfile(RUN/'scope.json',history/'A16_scope_assets_only.json')

code=SA/'A16_FDAM_deit_fdam_210232f.py';raw=code.read_bytes()
tree=load(SA/'A16_FDAM_tree_210232f.json');meta=load(SA/'A16_FDAM_commit_main.json')
commit='210232fc807bcaf55bfe799f848fcac20fbb5427'
assert meta['sha']==commit and not tree['truncated']
entry=next(r for r in tree['tree'] if r['path']=='FDAM_mmseg/mmcv_custom/deit_fdam.py')
assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==entry['sha']
nodes=ast.parse(raw.decode('utf-8')).body
classes=[n for n in nodes if isinstance(n,ast.ClassDef) and n.name in ['GroupDynamicScale','AttentionwithAttInv','Layer_scale_init_Block']]
assert len(classes)==3
fdam={'utc':utc,'status':'VERIFIED_AUTHOR_GIT_BLOB_AND_READ_SELECTED_OPERATORS_NO_EXECUTION',
    'repository':'https://github.com/Linwei-Chen/FDAM','commit':commit,'tree_entries':len(tree['tree']),
    'file':'FDAM_mmseg/mmcv_custom/deit_fdam.py','git_blob_sha1':entry['sha'],'sha256':sha(code),
    'local_file':str(code),'read_classes':[{'name':n.name,'start':n.lineno,'end':n.end_lineno} for n in classes],
    'source_observation':'AttentionwithAttInv forms V-minus-attended-V and input-dependent low/high residual scales. GroupDynamicScale performs input-routed grouped Fourier weighting. The default nonwindow/nonflash block selects AttInv and applies frequency residuals after attention and MLP.',
    'limits':['Only selected released mmseg/DeiT operator bodies and pinned tree inspected.',
        'No claim that every published configuration executes this same default path.',
        'No MaskDINO source closure,author model execution or AP reproduction.'],
    'new_complete_model_forwards':0,'new_training_updates':0}
fdam_receipt=ROOT/'comparison_sources_v1'/'FDAM_A16_operator_source_receipt.json'
write(fdam_receipt,fdam)
rawnames=['A16_spatial_gradient_nearest_discovery.json','A16_attention_variation_primary_open.json',
    'A16_frequency_methods_read1.json','A16_frequency_methods_read2.json',
    'A16_FDAM_primary_source_and_code_entry.json','A16_primary_bibliographic_metadata.json']
assert all((SA/n).exists() for n in rawnames)
newrows=[
 {'id':'How_Do_ViTs_Work','title':'How Do Vision Transformers Work?','authors':['Namuk Park','Songkuk Kim'],
  'url':'https://arxiv.org/abs/2202.06709','tier':'ICLR2022;metadata arXivv4',
  'read_scope':'Abstract,introduction/selected body excerpts and bibliographic metadata;not whole paper or code.',
  'author_method':'Empirical MSA/convolution frequency comparison and stage-wise AlterNet design.',
  'our_consequence':'Generic attention frequency behavior is prior;does not establish native Dome tiny-localization failure.',
  'unresolved':'No direct source-specific non-square MSDA or unknown-query causal localization proof.'},
 {'id':'ViT_Anti_Oversmoothing','title':'Anti-Oversmoothing in Deep Vision Transformers via the Fourier Domain Analysis: From Theory to Practice',
  'authors':['Peihao Wang','Wenqing Zheng','Tianlong Chen','Zhangyang Wang'],
  'url':'https://arxiv.org/abs/2203.05962','tier':'ICLR2022;arXivv1',
  'read_scope':'Body2.2-2.3and3.1-3.2;abstract and metadata;not complete appendix or author code execution.',
  'author_method':'Fourier analysis of deep self-attention,AttnScale and FeatScale frequency rescaling.',
  'our_consequence':'Generic high-frequency preservation or anti-smoothing is not a distinct increment.',
  'unresolved':'Square self-attention powers differ from conditional final-layer non-square MSDA partial response.'},
 {'id':'FDAM','title':'Frequency-Dynamic Attention Modulation for Dense Prediction',
  'authors':['Linwei Chen','Lin Gu','Ying Fu'],'url':'https://arxiv.org/abs/2507.12006','tier':'ICCV2025;read arXivv4',
  'read_scope':'Complete3.1-3.2,abstract/introduction,selected implementation/detection table;selected pinned author operator bodies. Not whole paper,supplement or model reproduction.',
  'author_method':'Attention inversion and dynamic frequency scaling;paper reports dense prediction,object detection and remote sensing applications.',
  'our_consequence':'Generic dynamic high-frequency restoration has direct detection prior;task transfer alone is insufficient.',
  'unresolved':'No FDAM vs Dome AP comparison;A16 tests a source-specific partial response association,not a new algorithm.',
  'source_receipt':str(fdam_receipt),'source_commit':commit}]

b=load(ROOT/canonical[0]);assert len(b['candidates'])==10
b.update(status='NO_CANDIDATE_PASSED_MECHANISM_AND_NOVELTY_GATES_AFTER_A16',mainline_validated=False,
    latest_update_utc=utc,latest_report=str(report),
    bounded_diagnostic_protocol=str(RUN/'protocol.json'),
    current_priority_question='重新筛选encoder、特征交互与decoder的具体瓶颈；A15采样中心与A16错误特异空间抵消均未提供足够依据。尚无已确认研究空白。',
    new_empirical_evidence='A16:37images853tinyGT;119matchedpairs20images. Bad-minus-good spatial cancellation .00378194 CI[-.02414313,.03378660];new-only .00524851 CI[-.02876478,.04576833]. Prespecified gates failed;no module training.',
    next_decision_protocol='A16closed26/26fullpasses,25CPUsourcecoresand10linearchecks,exit0. No threshold/epsilon/case extension. New architecture candidate requires closest-method distinction and a new bounded protocol before any execution.')
b['constraints']['new_experiments_this_turn']='A16nativeobservation26fullpasses;0new suffix/training.25CPUdouble source core checks+10linear checks separately counted.'
b['candidates'].append({'id':'K_ERROR_SPECIFIC_SPATIAL_RESPONSE_CANCELLATION',
    'question':'Does native weighted final-read aggregation cancel more spatial response in poorly localized tiny queries than geometry-matched good queries?',
    'status':summary['decision'],'increment_approved':False,'failure_cause_identified':False,
    'causal_improvement_evidence':False,'nearest':['HowDoVisionTransformersWork','ViT-Anti-Oversmoothing','FDAM'],
    'scope':'Negative prespecified conditional association screen on37official-valimages,not all layers/encoders or all possible feature operators.',
    'next':'No corresponding gradient-preserving/anti-smoothing module training from this gate.'})
b['next_exploratory_question']={'status':'NO_CONFIRMED_CURRENT_ARCHITECTURE_GAP',
    'question':b['current_priority_question'],'no_new_forward_phase_from_this_record':True,
    'distinctions':['Negative conditional association is not proof of encoder information loss.',
        'Generic frequency enhancement is prior;do not relabel it as a new encoder/decoder.',
        'Reopen architecture evidence before defining another finite experiment.']}
b['rejected_generic_novelty_claims'].append('Generic anti-smoothing/high-frequency residual or dynamic frequency scaling,including direct application to UAV detection,without a distinct mechanism beyond AttnScale/FeatScale/FDAM.')
b['A16_evidence']={'summary':str(RUN/'analysis_summary.json'),'independent_verifier':str(RUN/'independent_response_verification.json'),
    'protocol':str(RUN/'protocol.json'),'FDAM_source_receipt':str(fdam_receipt),
    'source_core_cpu_checks':25,'new_complete_model_forwards':26,'new_training_updates':0}
write(ROOT/canonical[0],b)

s=load(ROOT/canonical[1]);s['supplement_utc']=utc
s['query_groups'].append({'id':'A16_actual_spatial_response_and_attention_frequency',
    'strategy':'Targeted primary-source mechanism search then full-method/selected-source comparison;not systematic review.',
    'raw_requests':[str(SA/n) for n in rawnames],
    'access_limit':'CVF FDAM HTML403;arXiv full method sections available and read. No absent-literature novelty claim.'})
s['A16_response_and_frequency_audit']={'utc':utc,'raw_records':[str(SA/n) for n in rawnames],
    'core_literature_added':[r['id'] for r in newrows],'FDAM_code':str(fdam_receipt),
    'native_observation_summary':str(RUN/'capture_summary.json'),'association_summary':str(RUN/'analysis_summary.json'),
    'decision':summary['decision'],'mainline_validated':False}
write(ROOT/canonical[1],s)

m=load(ROOT/canonical[2]);assert len(m['rows'])==57
assert not {r['id'] for r in m['rows']} & {r['id'] for r in newrows}
m['rows']+=newrows;m['updated_utc']=utc;m['latest_report']=str(report)
m['latest_empirical_boundary']='A16conditionalbad-goodcancellationdifferenceinsufficient;genericfrequencyenhancementalreadyprior. No causal encoder loss,new algorithm or AP gain established.'
write(ROOT/canonical[2],m)

state=load(ROOT/canonical[3]);oldphase=state['latest_authoritative_phase']
state['authoritative_phase_history'].append(oldphase)
phase={'utc':utc,'classification':'progress','goal_status':'active','complete_mainline':False,
    'full_model_forwards':252,'cumulative_authorized_protocol_cap':252,'previous_closed_phase_forwards':226,
    'A16_phase_full_cap':26,'A16_phase_full_forwards':26,'A16_suffix_forwards':0,
    'historical_A15_suffix_forwards':72,'A16_CPU_source_core_calls':25,'A16_separate_CPU_linear_calls':10,
    'bounded_phase_complete':True,'actual_process_exit_code':0,'wrapper_pid':11816,'wrapper_start_ticks':'97988149',
    'main_model_training_updates':0,'CPU_diagnostic_linear_models':64,'scalar_response_heads':96,
    'A16_new_fits':0,'A16_decision':summary['decision'],'A16_statistics':summary['statistics'],
    'A16_download_verification':str(RUN/'capture_download_verification.json'),
    'A16_response_verification':str(RUN/'independent_response_verification.json'),
    'A16_archive':{'bytes':80280884,'members':53,'sha256':download['archive_sha256']},
    'latest_report':str(report),'next':b['current_priority_question'],
    'remaining_uncertainty':['Specific architecture-dependent natural failure','Distinct algorithm','Independent generalization','Training feasibility']}
state.update(recorded_utc=utc,goal_status='ACTIVE',mainline_validated=False,diagnostics_complete=True,
    model_forwards=252,cap=252,training_updates=0,latest_report=str(report),latest_authoritative_phase=phase,
    previous_goal_turn='progress_readonly_remaining_prefix_eligibility_25_no_model_run',
    current_goal_turn='progress_A16_primary_frequency_sources_bounded_native_capture_and_verified_actual_response_analysis',
    current_priority_question=b['current_priority_question'])
write(ROOT/canonical[3],state)

src=load(ROOT/canonical[4]);src.update(updated_utc=utc,new_model_forwards=26,training_updates=0,
    scope_this_update='A16:26nativefullpassesundernewfrozencap;25separateCPUauthorcorechecks+10linearchecks. ReadFDAMpinnedoperatorswithoutmodelimport/execution.',
    current_novelty_boundary='Genericattentionanti-smoothing/frequencyrestorationhasAttnScale/FeatScale/FDAMprior;A16doesnotsupporterror-specificfinal-readcancellation. Noapprovednewencoder/decoder.')
src['FDAM_A16_source']=fdam
src['A16_actual_response_audit']={'protocol':str(RUN/'protocol.json'),'summary':str(RUN/'analysis_summary.json'),
    'source_validation_before':str(RUN/'source_validation_before_capture.json'),
    'source_validation_after':str(RUN/'source_validation_after_capture.json'),
    'independent_verification':str(RUN/'independent_response_verification.json'),'mainline_validated':False}
src['next_spatial_update_question']={'status':'DEMOTED_BY_A15_AND_A16_WITHIN_TESTED_SCOPE',
    'next':'No new last-read centre/gradient-preserving module from these gates;reopen concrete architecture evidence.',
    'not_inferred':'Encoder information loss or universal decoder inadequacy.'}
write(ROOT/canonical[4],src)
write(RUN/'scope.json',{'utc':utc,'status':'COMPLETED_NATIVE_RESPONSE_AUDIT_NO_MODULE_TRAINING',
    'prior_assets_only_scope_snapshot':str(history/'A16_scope_assets_only.json'),
    'protocol':str(RUN/'protocol.json'),'summary':str(RUN/'analysis_summary.json'),
    'new_complete_model_forwards':26,'new_suffix_forwards':0,'source_core_calls':25,'separate_linear_calls':10,
    'main_model_updates':0,'mainline_validated':False,'goal_status':'ACTIVE'})
important=[RUN/'protocol.json',RUN/'analysis_summary.json',RUN/'capture_summary.json',RUN/'independent_response_verification.json',
    RUN/'capture_download_verification.json',RUN/'old_patch_packing_receipt.json',RUN/'scope.json',fdam_receipt,
    *[ROOT/n for n in canonical],report]
receipt={'utc':utc,'status':'A16_COMPLETE_VERIFIED_CANDIDATE_DEPRIORITIZED_GOAL_ACTIVE',
    'goal_status':'ACTIVE','mainline_validated':False,'new_complete_model_forwards':26,
    'cumulative_complete_model_forwards':252,'historical_suffix_forwards':72,'new_suffix_forwards':0,
    'CPU_author_core_calls':25,'CPU_linear_projection_calls':10,'actual_exit_code':0,
    'main_model_updates':0,'CPU_diagnostic_fits_unchanged':64,'scalar_response_heads_unchanged':96,
    'core_literature_rows':60,'decision':summary['decision'],
    'files':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size} for p in important],
    'limits':'Completed bounded audit and arithmetic checks,not completed mainline objective,novelty proof,new detector gain or training feasibility.',
    'next':b['current_priority_question']}
write(receipt_path,receipt)
print(json.dumps({k:receipt[k] for k in ['status','cumulative_complete_model_forwards','core_literature_rows','main_model_updates','mainline_validated','decision']}))
