import datetime, hashlib, json, pathlib, shutil

D = pathlib.Path(__file__).resolve().parent
B = D.parent
NOW = datetime.datetime.now(datetime.timezone.utc).isoformat()
REPORT = B / '90_A41选题纠偏_两条候选与暂定主线.txt'
RECEIPT = B / '91_A41暂定主线选择与证据核验回执.json'
CANON = ['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json']

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def write(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1<<20),b''):h.update(c)
    return h.hexdigest()

if RECEIPT.exists():
    raise RuntimeError('A41 already finalized; do not overwrite its published receipt')

old=read(B/'89_A39冻结采集与分组探针独立核验回执.json')
assert len(old['files_sha256'])==200
for p,digest in old['files_sha256'].items():
    assert sha(pathlib.Path(p))==digest, 'Pre-existing A39 evidence mismatch: '+p

snap=D/'canonical_before_A41'
snap.mkdir(exist_ok=True)
remap={}
for name in CANON:
    original=B/name
    dest=snap/name
    if dest.exists():assert sha(dest)==sha(original)
    else:shutil.copy2(original,dest)
    remap[str(original)]=str(dest)
write(D/'historical_A39_canonical_remap.json',{'scope':'Only the five evolving canonical files are remapped; original A39 receipt is untouched.', 'map':remap})

archive=read(D/'paper_archive_receipts.json')
for r in archive['rows']:
    assert r['download_status'] in ['downloaded','existing_file_reused']
    assert sha(pathlib.Path(r['local']))==r['sha256']
    assert sha(pathlib.Path(r['text_local']))==r['text_sha256']

source_manifest=read(B/'A32_轻量新编码器与专家路由原文筛查_v1/selected_source_manifest.json')
source_rows=[r for r in source_manifest['rows'] if r['repo']=='Intellindust-AI-Lab/EdgeCrafter' and r['path'].endswith(('ecvit.py','decoder.py','ecdet_s.yml'))]
assert len(source_rows)==3
for r in source_rows:assert sha(pathlib.Path(r['local']))==r['sha256']
write(D/'reused_pinned_source_verification.json',{'utc':NOW,'status':'PASS','executed_or_imported':False,'rows':source_rows,'locators':{'ecvit.py':['151-172 staged convolutional stem','475-499 final ViT layer mean and resized pyramid'],'decoder.py':['375-382 FDR and class heads plus LQE'],'ecdet_s.yml':['embed_dim 192','num_layers 4','eval_spatial_size 640']}})

protocol={
 'status':'DRAFT_NOT_IMPLEMENTED_NOT_RUN','candidate':'A41-A','scope':'Research selection completed; efficacy and publication-level novelty remain unverified.',
 'dataset':'VisDrone DET official train/val, ten scored classes, ignore/others rules explicit',
 'host':'ECDet-S','source_commit':'d8752b75c9beaecfcdd0e5cd207bae3b6946ae23','input_hw':[640,640],'num_queries':300,'decoder_layers':4,
 'common_initialization':'same official COCO checkpoint and identical saved ten-class classifier initialization',
 'single_change':'3x3 ordered ROI grid from existing stride-4 stem, small zero-output-initialized MLP, residual to final box-head input',
 'roi_box':'detached predicted reference box entering the final decoder layer; never GT at inference; no extra decoder pass',
 'native_semantics':'classification Linear input unchanged in L; original FDR, losses and any LQE coupling remain',
 'arms':{'B':'original','S':'identical detail branch residual for final class and box heads','L':'same branch and parameter count as S; residual for final box head only'},
 'conditional_budget_draft':{'seed':20261031,'epochs_each_max':12,'groups':3,'microbatch':2,'gradient_accumulation':4,'no_split_continuation':True},
 'primary':'COCO-style small-object AP75 in percentage points; original-image area convention must be fixed before execution',
 'secondary':['APs','overall AP','same-device same-precision batch1 latency','peak CUDA training memory'],
 'practical_decision_draft':{'L_minus_B_small_AP75_pp_min':0.5,'L_minus_S_small_AP75_pp_min':0.3,'latency_increase_fraction_approx_max':0.1,'meaning':'resource-priority thresholds, not significance tests'},
 'exit':'Without advantage over same-detail shared branch, stop the localization-only novelty claim; no threshold/layer/loss rescue sweep',
 'capacity_and_runtime':'not measured; check approximately <=22GiB peak and project three arms within a few days before allocating the run',
 'current_allocated_model_calls':0,'current_allocated_optimizer_updates':0,'not_an_approval_request':True
}
write(D/'minimal_validation_DRAFT.json',protocol)

new_papers=[
 {'id':'DESTR_CVPR2022_A41','title':'DESTR: Object Detection With Split Transformer','authors':['Liqiang He','Sinisa Todorovic'],'venue':'CVPR 2022','pages':'9377-9386','url':'https://openaccess.thecvf.com/content/CVPR2022/html/He_DESTR_Object_Detection_With_Split_Transformer_CVPR_2022_paper.html','read_scope':'Conference abstract, selected Method 3.1/3.2 opening paragraphs from archived PDF; not whole method/experiments/code','method_fact':'Separate classification/regression cross-attention, mini-detector initialization, pair self-attention','implication':'Class/reg decoupling is prior work, not the candidate novelty'},
 {'id':'SD_DETR_arxiv2023_A41','title':'Decoupled DETR: Spatially Disentangling Localization and Classification for Improved End-to-End Object Detection','authors':['Manyuan Zhang','Guanglu Song','Yu Liu','Hongsheng Li'],'venue':'arXiv preprint','version':'2310.15955v1','url':'https://arxiv.org/abs/2310.15955v1','read_scope':'HTML Method 3.1,3.2.1,3.2.2 and opening 3.2.3; PDF archived, full experiments not read','method_fact':'Shared self-attention, separate task cross-attention/FFNs, ROIAlign/discriminative-point task query initialization, alignment loss','implication':'Partial decoder sharing and task-specific spatial reads are prior'},
 {'id':'DETRDistill_ICCV2023_A41','title':'DETRDistill: A Universal Knowledge Distillation Framework for DETR-families','authors':['Jiahao Chang','Shuo Wang','Hai-Ming Xu','Zehui Chen','Chenhongyi Yang','Feng Zhao'],'venue':'ICCV 2023','pages':'6898-6908','url':'https://openaccess.thecvf.com/content/ICCV2023/html/Chang_DETRDistill_A_Universal_Knowledge_Distillation_Framework_for_DETR-families_ICCV_2023_paper.html','read_scope':'Conference abstract and selected PDF section 4.2 target-aware feature distillation; not all method/formulas/experiments','method_fact':'Teacher-query/feature correlation masks with quality weighting for target-aware feature KD, plus logits/query-prior KD','implication':'Generic query-guided feature distillation is already prior; backup B has high overlap risk'}
]
archive_map={r['id']:r for r in archive['rows']}
archive_ids={'DESTR_CVPR2022_A41':'DESTR_CVPR2022','SD_DETR_arxiv2023_A41':'SD_DETR_arxiv2310_15955v1','DETRDistill_ICCV2023_A41':'DETRDistill_ICCV2023'}
for p in new_papers:
    r=archive_map[archive_ids[p['id']]]
    p['pdf_archive_receipt']=r
    p['human_read_attestation']=False
write(D/'new_core_records.json',{'rows':new_papers})

selection={
 'phase':'A41','recorded_utc':NOW,'status':'PROVISIONAL_MAINLINE_RECOMMENDED_FOR_SMALL_PILOT',
 'stage_complete':'research selection only','mainline_selected_for_pilot':True,'mainline_validated':False,'innovation_validated':False,'publication_value_established':False,
 'selection_bar':['credible concrete problem','reasoned difference from closest methods','affordable discriminating experiment'],
 'not_required_before_pilot':['prove every prior method ineffective','exhaustive novelty certification','establish full causal mechanism'],
 'recommended_candidate':'A41-A','reserve_candidate':'A41-B',
 'mainline':'面向无人机小目标的紧凑ViT检测器：复用早期空间细节的定位专用解码',
 'priority_question':'定位支路单独读取已有stride-4局部网格，是否比同成本的分类/定位共享细节残差更有效地改善小目标AP75？',
 'report':str(REPORT),'protocol':str(D/'minimal_validation_DRAFT.json'),
 'new_whole_detector_forwards':0,'new_optimizer_updates':0,'new_probe_fits':0,'new_training_jobs':0,
 'cumulative_full_detector_forwards':3395,'cumulative_detector_optimizer_updates':528,
 'ECDet_training_24GB_verified':False,'A39_negative_result_unchanged':True,
 'old_branches_restarted':False,'remote_access_this_phase':False,
 'next':'Optional separate implementation/capacity and three-arm pilot stage; model asset preparation does not change the selected question.',
 'goal_status':'active'
}

board=read(B/CANON[0]); assert len(board['candidates'])==18
board['candidates'] += [
 {'id':'A41-A','status':'RECOMMENDED_FOR_SMALL_PILOT_NOT_VALIDATED','question':selection['priority_question'],'nearest':['EdgeCrafter_A32','DEIMv2_A32','DESTR_CVPR2022_A41','SD_DETR_arxiv2023_A41'],'increment_approved':None,'pilot_recommended':True,'efficacy_validated':False,'evidence':str(REPORT),'next':str(D/'minimal_validation_DRAFT.json')},
 {'id':'A41-B','status':'RESERVE_HIGH_PRIOR_OVERLAP_NO_TRAINING_ALLOCATION','question':'Can instance-matched final-query self-supervision improve early spatial semantic discrimination?','nearest':['RT_DETRv4_A32','DETRDistill_ICCV2023_A41'],'increment_approved':None,'pilot_recommended':False,'efficacy_validated':False,'evidence':str(REPORT),'next':'Check direct overlap only if primary A is retired; do not run both at once'}
]
board.update(status='PROVISIONAL_MAINLINE_SELECTED_A41_NOT_EFFICACY_VALIDATED',mainline_validated=False,mainline_selected_for_pilot=True,current_priority_candidate='A41-A',current_host_candidate='ECDet-S',current_priority_question=selection['priority_question'],next_exploratory_question=selection['priority_question'],remaining_deliverable='Selection-stage report delivered; empirical efficacy and publication-level contribution require separate pilot and later validation.',latest_report=str(REPORT),latest_update_utc=NOW)
board['selection_criteria_A41']=selection['selection_bar']
board['next_decision_protocol']={'status':'DRAFT_NOT_RUN','file':str(D/'minimal_validation_DRAFT.json')}
board['bounded_diagnostic_protocol']=str(D/'minimal_validation_DRAFT.json')
board['A41_evidence']=selection
write(B/CANON[0],board)

search=read(B/CANON[1])
search['A41_selection_bar_and_shortlist']={'utc':NOW,'review_type':'targeted non-systematic research selection','search_terms_in_report':10,'raw_first_three_not_separately_archived':True,'archived_return_subset':str(D/'browser_source_receipts.json'),'new_verified_records':3,'unverified_TSC_lead_not_used_as_gap_evidence':True,'report':str(REPORT)}
write(B/CANON[1],search)

matrix=read(B/CANON[2]); assert len(matrix['rows'])==114
assert not ({r['id'] for r in matrix['rows']} & {r['id'] for r in new_papers})
matrix['rows']+=new_papers
matrix.update(updated_utc=NOW,latest_report=str(REPORT))
matrix['A41_scope_note']='Three primary-source verified records added with partial reading explicitly disclosed; no exhaustive novelty claim.'
write(B/CANON[2],matrix)

state=read(B/CANON[3])
assert state['model_forwards']==3395 and state['training_updates']==528
state.update(recorded_utc=NOW,current_goal_turn=selection,latest_authoritative_phase=selection,mainline_validated=False,mainline_selected_for_pilot=True,current_priority_question=selection['priority_question'],latest_report=str(REPORT))
state['A41_evidence']=selection
state['goal_status']='active'
write(B/CANON[3],state)

audit=read(B/CANON[4])
audit['A41_provisional_mainline_and_source_difference']={'scope':'research-selection rationale, not causal/novelty validation','source':str(D/'reused_pinned_source_verification.json'),'new_papers':str(D/'new_core_records.json'),'specific_difference':'Existing stride-4 stem grid only to final regression residual; no whole decoder split or shared pyramid fusion','closest_prior_is_acknowledged':True,'effectiveness_unverified':True,'classification_score_coupling_through_native_LQE_retained':True,'report':str(REPORT)}
write(B/CANON[4],audit)

for p,digest in old['files_sha256'].items():
    assert sha(pathlib.Path(remap.get(p,p)))==digest,'Historical evidence changed: '+p

files=list(D.rglob('*'))+[REPORT]+[B/name for name in CANON]
selection.update(previous_A39_files_verified_via_five_snapshot_remaps=200,core_matrix_rows=117,candidate_count=20,new_paper_archives_verified=3,reused_source_files_verified=3)
write(D/'selection_decision.json',selection)
files.append(D/'selection_decision.json')
selection['files_sha256']={str(p):sha(p) for p in files if p.is_file()}
write(RECEIPT,selection)
for p,digest in read(RECEIPT)['files_sha256'].items():assert sha(pathlib.Path(p))==digest
print(json.dumps({'status':'PASS','recommended_candidate':'A41-A','mainline_selected_for_pilot':True,'efficacy_validated':False,'matrix_rows':117,'candidates':20,'historical_A39_files':200,'A41_files_hashed':len(selection['files_sha256']),'new_forwards':0,'new_optimizer_updates':0},ensure_ascii=False))
