"""Record new primary encoder sources and verified zero-forward population/cache audit."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,ast
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'A17_新编码器机制与多实例映射审计_v1'
SA=ROOT/'source_audit_v1'
RECEIPT=ROOT/'45_A17完成核验与编码器研究边界回执.json'
assert not RECEIPT.exists()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
pop=load(RUN/'annotation_summary.json');cached=load(RUN/'cached_cell_summary.json');verify=load(RUN/'independent_verification.json')
assert pop['images']==6471 and pop['counts']['tiny_GT']==52210
assert verify['annotation_files_reparsed_and_hashed']==6471 and verify['cached_pairs_checked']==69
assert cached['adequate_matched_cohort'] and not cached['gate_passed'] and not cached['mainline_validated']
assert cached['new_complete_model_forwards']==pop['new_complete_model_forwards']==0
previous=ROOT/'43_A16完成核验与候选降级回执.json'
for f in load(previous)['files']:assert sha(ROOT/f['path'])==f['sha256']
canonical=['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json']
history=ROOT/'history_before_A17_update_v1';history.mkdir()
for name in canonical+[previous.name]:shutil.copyfile(ROOT/name,history/name)
utc=datetime.now(timezone.utc).isoformat();report=ROOT/'44_A17新编码器源码对照与局部多实例核查.txt'
assert report.exists()
targets=load(SA/'A17_source_targets.json');trees={};sources=[]
scopes={
 'qwen2_d2e.py':'Coremaskbuilder/querywrapper/configandfactoryread;notwholepackageexecution.',
 'deepseek_ocr2.py':'SelectedSAM-to-Qwen-to-projectorimagefeaturepathandconstructor;notallVLLMprocessing.',
 'build_linear.py':'DownloadedandGitblobverifiedonly;nocontent-specificmethodclaim.',
 'mambairv2_arch.py':'index_reverse,semantic_neighbor,ASSM,Selective_Scanbodyread;notwholemodelortraining.',
 'detection_lsnet.py':'LKP,LSConv,Block,stem/downsamplingandfeatureoutputsread;notalltraining/weightloading.',
 'ska.py':'Forwardkernelweightindexingandautogradwrapperread;notallbackwardkernelsorexecution.'}
for ident,repo,path,short in targets:
    meta=load(SA/('A17_'+ident+'_commit.json'));tree=load(SA/('A17_'+ident+'_tree.json'))
    assert not tree['truncated']
    p=SA/('A17_'+ident+'_'+short);data=p.read_bytes()
    expected=next(x['sha'] for x in tree['tree'] if x['path']==path)
    assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==expected
    ast.parse(data.decode('utf-8'))
    trees[ident]={'repository':'https://github.com/'+repo,'commit':meta['sha'],'entries':len(tree['tree']),'truncated':False}
    sources.append({'id':ident,'repository_file':path,'local_file':str(p.relative_to(ROOT)),
        'commit':meta['sha'],'git_blob_sha1':expected,'sha256':sha(p),'bytes':len(data),'read_scope':scopes[short]})
audit={'utc':utc,'status':'PINNED_AUTHOR_SOURCE_BLOBS_VERIFIED_SELECTED_BODIES_READ_NO_MODEL_EXECUTION',
    'trees':trees,'files':sources,'source_body_calls':0,'new_complete_model_forwards':0,
    'observations':{
        'DeepSeek_OCR2':'Visual-prefixbidirectional/query-suffixcausalmask;visualsnotqueryconditioned;wrapperoutputs144/256learnedquerytokens. Noargsortinreadencoderwrapper.',
        'MambaIRv2':'HardGumbelroute,sort,onescan,inversegather. Readforwardhasnoevalargmaxbranch;promptmadebeforepermutationandpassedwithoutmatchinggather. Staticdataflowquestion,notanexecutedfailureorglobalpapererrorclaim.',
        'LSNet':'7x7LKPpredicts3x3SKAweights;nominaldetectionstrides8/16/32/64,notDomeP2equivalence;Tritonsharingci%wc.'},
    'limits':['Noauthorencoderinstantiated,weightsdownloaded,trainingorlatencybenchmark.',
              'Paperabstract/selectedsourceclaimsremainboundedbyexplicitreadscope.']}
audit_path=RUN/'source_operator_audit.json';write(audit_path,audit)
raw=[]
for p in sorted(SA.glob('A17_*.json')):
    obj=load(p)
    if isinstance(obj,dict) and 'request' in obj and 'response' in obj:raw.append({'file':str(p.relative_to(ROOT)),'sha256':sha(p),'request':obj['request']})
query_count=sum(len(x['request'].get('search_query',[])) for x in raw)
write(RUN/'search_log.json',{'utc':utc,'review_type':'Targetednon-systematicprimary-sourcearchitectureaudit',
    'search_queries':query_count,'web_request_records':len(raw),'records':raw,
    'limits':['Noexhaustiveliteratureorfirst-methodclaim.','MambaSODdroneprimaryfulltextunread,publisher403/repository404;RGB-Dsaliencypaperisadifferentwork.',
              'VRWKV-DETRpublisherpreviewandothermemoryencodersearchhitsareleadsonly,notfull-methodclearance.']})
newrows=[
 {'id':'DeepSeek_OCR2','title':'DeepSeek-OCR 2: Visual Causal Flow','authors':['Haoran Wei','Yaofeng Sun','Yukun Li'],
  'url':'https://arxiv.org/html/2601.20552v1','tier':'arXivpreprintJan2026',
  'read_scope':'Complete3.1and3.2.1-3.2.4,selected3.3/introduction;corepinnedmask/querysourceandselectedimagecalls.Notwholepaperortraining.',
  'author_method':'SAMtokenizerplusLM-stylevisualprefixandcausalquerysuffix;onlyqueryoutputsfeedLLM.',
  'our_consequence':'Mixedvisibility/learnablevisualqueryreadingareprior;notadense-gridencoderreplacementorverifiedUAVgain.',
  'unresolved':'Spatialownershipforunorderedtinyinstances;single24GBtrainingfeasibility;notconfirmedastheusersoriginalframeworkreference.','source_receipt':str(audit_path)},
 {'id':'MambaIRv2','title':'MambaIRv2: Attentive State Space Restoration',
  'authors':['Hang Guo','Yong Guo','Yaohua Zha','Yulun Zhang','Wenbo Li','Tao Dai','Shu-Tao Xia','Yawei Li'],
  'url':'https://arxiv.org/html/2411.15269','tier':'CVPR2025',
  'read_scope':'Complete4.1-4.3,selectedmotivation/settings;pinnedASSM,Selective_Scanandindexhelpers.Notwholepaperorrestorationexecution.',
  'author_method':'Prompt-conditionedstate-spaceoutputandsemantic-guidedtokensort/inversewithlocalwindowattention.',
  'our_consequence':'Semanticorderingandpromptedstate-spacearenotnew;conditionalcausalclaimsandprovenancealignmentrequirescope.',
  'unresolved':'NotUAVdetectionevidence;readforwardpromptorder/evalroutingneedcare,nomeasuredbugornewmoduleclaimed.','source_receipt':str(audit_path)},
 {'id':'LSNet','title':'LSNet: See Large, Focus Small','authors':['Ao Wang','Hui Chen','Zijia Lin','Jungong Han','Guiguang Ding'],
  'url':'https://arxiv.org/html/2503.23135','tier':'CVPR2025',
  'read_scope':'Complete3.1-3.3,selectedsupplementsettings;pinnedLKP/LSConv/Block/stem/output/SKA.Notwholemodelrun.',
  'author_method':'Large-kernelperceptionpredictssmall-kernelgroupedaggregationweights;hybridlaststageattention.',
  'our_consequence':'See-large/focus-smallispriorandhasdirectUAVtransfer;authorstemhasstride8firstoutput.',
  'unresolved':'NoP2-equivalentmatchedarchitecturecomparisonornewtrainingcapacitymeasure.','source_receipt':str(audit_path)},
 {'id':'PaQ_DETR','title':'PaQ-DETR: Learning Pattern and Quality-Aware Dynamic Queries for Object Detection',
  'authors':['Zhengjian Kang','Jun Zhuang','Kangtong Mo','Qi Chen','Rui Liu','Ye Zhang'],
  'url':'https://arxiv.org/html/2603.06917v1','tier':'CVPR2026indexedversion;arXivv1bodyread',
  'read_scope':'Complete3.1-3.4,abstract/introduction;noauthorcodeorfullpaperexecution.',
  'author_method':'Encoder-conditionedconvexcombinationofsharedpatterns,quality-awareauxiliaryone-to-manyassignmentandpatternregularization.',
  'our_consequence':'Content-conditionedquerypatternsincludingsharedgradientpathsarealreadyprior.',
  'unresolved':'Doesnotestablishourfixedcellfailure;nocausalcomparisonwithlocalcontentunmixing.'},
 {'id':'DQ_DETR_tiny','title':'DQ-DETR: DETR with Dynamic Query for Tiny Object Detection',
  'authors':['Yi-Xin Huang','Hou-I Liu','Hong-Han Shuai','Wen-Huang Cheng'],
  'url':'https://arxiv.org/html/2404.03507v1','tier':'ECCV2024',
  'read_scope':'Complete3.1-3.6and4.2;selectedotherexperiments;noauthorcodeexecution.',
  'author_method':'Categoricalcounting,density-guidedfeatureenhancementandTop-Kquerycontent/positiongenerationwith300/500/900/1500budgets.',
  'our_consequence':'Density-adaptivequeries/encoderconditioningareprior;nototherDQ-DETRhomonyms.',
  'unresolved':'Noauthormodelbenchmarkorindependentcausallocalmultiplicitygain.'},
 {'id':'BUTR','title':'Dynamic Query as Budgeting for Tiny Object Detection',
  'authors':['Yi Zhang','Di Xiong','Ge Gao','Xiangyue Zhang','Yihang Qiu','Yuxuan Zhou','Shuo Chen'],
  'url':'https://arxiv.org/html/2610.33430v1','tier':'arXivpreprint24Oct2026;notassertedpeerreviewed',
  'read_scope':'Complete3and4.1-4.2,selected5.1/tables;algorithmappendixnotcompleteread,nopinnedcodeortraining.',
  'author_method':'DomePAQIoperator/budgetmatchedanalysis;encodermax-sigmoidconfidencesumtimesconservativefactorwithstandardTop-K.',
  'our_consequence':'Needbudget-matchedcontrolswhencomparingnewencodersorqueryinitializers;genericbudgetingsimplificationcovered.',
  'unresolved':'AuthorsDome-LmeasurementsnotourDome-Sreproduction;empiricaldensityorderisnotuniversalmonotonicitytheorem.'},
 {'id':'DCLSConv_UAV','title':'An Edge-Computing-Oriented Small-Object Detection Algorithm for UAV Aerial Images',
  'authors':['Chanchan Zhao','Xiaoyu Gao','Bao Shi','Ziyang Zhang'],'url':'https://doi.org/10.32604/cmc.2026.085282','tier':'ComputersMaterialsandContinua2026',
  'read_scope':'Publishermetadata/abstractandcomplete2.1;notwholepaper,othermodulesorcode.',
  'author_method':'LSNet-inspireddynamiclocalaggregationguidedby7x7/11x11contextinYOLOv11nbackbone,withotherfusion/headcomponents.',
  'our_consequence':'DirectLSNet-styleUAVbackbonetransferalreadyexists;cannotclaimtasktransferalone.',
  'unresolved':'Noauthorcodeclosureorindependentwholemodelperformanceverification.'},
 {'id':'CrowdDet','title':'Detection in Crowded Scenes: One Proposal, Multiple Predictions',
  'authors':['Xuangeng Chu','Anlin Zheng','Xiangyu Zhang','Jian Sun'],
  'url':'https://openaccess.thecvf.com/content_CVPR_2020/html/Chu_Detection_in_Crowded_Scenes_One_Proposal_Multiple_Predictions_CVPR_2020_paper.html','tier':'CVPR2020',
  'read_scope':'Abstract,introductionandpublishermetadataonly;fullmethod/codepending.',
  'author_method':'EachproposalpredictsasetofcorrelatedinstanceswithEMDlossandSetNMS.',
  'our_consequence':'Genericlocalone-to-multiplepredictionisalreadyprior.',
  'unresolved':'Fullmethod/sourcecomparisonrequiredbeforeanencoderinstance-ownershipproposal;notfullycleared.'},
 {'id':'Repulsion_Loss','title':'Repulsion Loss: Detecting Pedestrians in a Crowd',
  'authors':['Xinlong Wang','Tete Xiao','Yuning Jiang','Shuai Shao','Jian Sun','Chunhua Shen'],
  'url':'https://arxiv.org/abs/1711.07752','tier':'CVPR2018;arXivmetadata/abstractread',
  'read_scope':'Abstractandmetadataonly;fullmethod/codepending.',
  'author_method':'Regressionobjectiveattractstargetandrepelssurroundinginstances.',
  'our_consequence':'Genericneighbourrepulsion/localizationdisentanglementisnotnew.',
  'unresolved':'Doesnotproveencodercontentcause;fullnearestcomparisonpending.'}]
matrix=load(ROOT/canonical[2]);assert len(matrix['rows'])==60
assert not {r['id'] for r in matrix['rows']} & {r['id'] for r in newrows}
matrix['rows']+=newrows;matrix.update(updated_utc=utc,latest_report=str(report),
    latest_empirical_boundary='A17fixed-cellpopulationcoincidenceexists,butcachedconditionalmissdifferenceiswide/inconclusiveforcausalprioritization. Genericnewencoder/querymechanismscovered;noapprovedmainline.')
write(ROOT/canonical[2],matrix)
question='相邻同类微小实例的目标归属与位置交互是否在编码阶段混淆？先补齐CrowdDet/Repulsion及最新编码近邻；区别于已结束固定网格计数，尚未认定研究空白。'
b=load(ROOT/canonical[0]);assert len(b['candidates'])==11
b.update(status='NO_CONFIRMED_MAINLINE_AFTER_A17_ENCODER_SOURCE_AND_FIXED_CELL_AUDIT',mainline_validated=False,
    latest_update_utc=utc,latest_report=str(report),current_priority_question=question,
    bounded_diagnostic_protocol=str(RUN/'cached_cell_protocol.json'),
    new_empirical_evidence='Annotation6471trainimages343204validGT;frozen52210tiny,2855sameclassstride4coincidences(5.47%). Cached37images853tiny69pairs10clusters:failure50difference.12589572CI[-.25660428,.46432219],gatefailed.0newmodelcalls.',
    next_decision_protocol='A17zero-forwardauditclosed. No threshold/phase/case relaxation or multiplicity module training. First finish closest-method/source distinction;any new neighbour question needs a fresh frozen protocol and unseen predictions.')
b['constraints']['new_experiments_this_turn']='A17:annotationpopulationandcachedgeometry/sourceauditsonly;0newcomplete/suffix/sourcecorecalls,0fits/training.'
for c in b['candidates']:
    if c['id']=='A_ENCODER_INSTANCE_SEPARABILITY':
        c['A17_scope_note']='A1doesnotproveencoderinstance-ownershipfailure. Fixed-cellconditionalassociationlacksprecision;phase-independentneighbourhypothesisrequiresnewsourceclearanceandprotocol.'
b['candidates'] += [
 {'id':'L_GENERIC_NEW_ENCODER_ORDER_AND_PERCEPTION_SCOPE','status':'INSPIRATION_AND_CONTROLS_ONLY_GENERIC_MECHANISMS_ALREADY_PRIOR',
  'nearest':['DeepSeek-OCR2','MambaIRv2','LSNet','DCLSConvUAV'],'increment_approved':False,
  'scope':'Pinnedselectedoperatorsread,notnewarchitectureperformanceor24GBtrainingvalidation.'},
 {'id':'M_FIXED_CELL_MULTIPLICITY','status':cached['decision'],'increment_approved':False,'failure_cause_identified':False,
  'population_prevalence_sameclass':pop['primary']['same_fraction'],'cached_primary':cached['statistics']['failure50_difference'],
  'scope':'Fixedstride4phase0,previouslyseen37cases;wideCIisnotproofabsentorequivalent. Genericmultipattern/multipredictionalreadyprior.',
  'nearest':['AnchorDETR','CrowdDet(partial)','PaQ-DETR','DQ-DETR','BUTR']}]
b['next_exploratory_question']={'status':'NOT_CONFIRMED_GAP_REQUIRES_CLOSER_METHOD_AND_NEW_UNSEEN_PREDICTION_DIAGNOSIS',
    'question':question,'no_new_forward_phase_from_this_record':True,
    'distinctions':['FixedcellcoincidenceisnotahardDETRcapacitylimit.','WideCIcannotshowequivalence.',
        'Genericsemanticordering/large-smallmixing/multiplepatterns/budgetingareprior.',
        'Aphase-independentneighbouranalysiscannotrescuethefinishedcellgatebyreusingitscasesandsearchingthresholds.']}
b['rejected_generic_novelty_claims'] += ['Directsemanticquerymask/tokenreorderingasnewUAVencoderwithoutbeyond-DeepSeekOCR2/MambaIRv2mechanism.',
    'Directlarge-perception/small-aggregationUAVtransferwithoutbeyond-LSNet/DCLSConvmechanism.',
    'FixedcellcoincidencealoneimpliesDETRrecallceiling;genericone-regionmultiplepredictionsasnovel.']
b['A17_evidence']={'source_audit':str(audit_path),'population':str(RUN/'annotation_summary.json'),
    'cached_comparison':str(RUN/'cached_cell_summary.json'),'verification':str(RUN/'independent_verification.json'),
    'numeric_boundary_audit':str(RUN/'annotation_numeric_boundary_audit.json'),'new_complete_model_forwards':0}
write(ROOT/canonical[0],b)
search=load(ROOT/canonical[1]);search['supplement_utc']=utc
search['query_groups'].append({'id':'A17_encoder_semantic_order_scope_and_local_multiplicity','raw_request_log':str(RUN/'search_log.json'),
    'search_queries':query_count,'new_core_rows':[r['id'] for r in newrows],'review_type':'Targetednon-systematicprimary-sourceaudit'})
search['A17_encoder_and_population_audit']={'utc':utc,'source':str(audit_path),'search_log':str(RUN/'search_log.json'),
    'new_complete_model_forwards':0,'report':str(report),'scientific_mainline':'NOT_VALIDATED'}
write(ROOT/canonical[1],search)
state=load(ROOT/canonical[3]);assert state['model_forwards']==state['cap']==252
state['authoritative_phase_history'].append(state['latest_authoritative_phase'])
phase={'utc':utc,'classification':'progress','goal_status':'active','complete_mainline':False,
    'full_model_forwards':252,'cumulative_authorized_protocol_cap':252,'previous_closed_phase_forwards':252,
    'A17_new_full_cap':0,'A17_new_full_forwards':0,'A17_new_suffix_forwards':0,
    'historical_A15_suffix_forwards':72,'A17_source_operator_calls':0,
    'bounded_phase_complete':True,'local_population_tool_exit_code':0,'local_cached_analysis_tool_exit_code':0,
    'local_final_verifier_tool_exit_code':0,'main_model_training_updates':0,'CPU_diagnostic_linear_models':64,
    'scalar_response_heads':96,'A17_new_fits':0,'A17_primary_source_trees':trees,
    'A17_population_images':6471,'A17_cached_cases':37,'A17_cached_pairs':69,
    'A17_decision':cached['decision'],'A17_statistics':cached['statistics'],
    'latest_report':str(report),'next':question,
    'remaining_uncertainty':['Specificarchitecture-dependentnaturalflaw','Distinctalgorithm','Independentgeneralization','Trainingfeasibility']}
state.update(recorded_utc=utc,goal_status='ACTIVE',mainline_validated=False,diagnostics_complete=True,
    latest_report=str(report),latest_authoritative_phase=phase,
    previous_goal_turn='progress_A16_bounded_native_feature_response_validation_and_candidate_demotion',
    current_goal_turn='progress_A17_primary_new_encoder_sources_and_verified_annotation_population_cached_neighbour_screen',
    current_priority_question=question)
write(ROOT/canonical[3],state)
src=load(ROOT/canonical[4]);src.update(updated_utc=utc,new_model_forwards=0,training_updates=0,
    scope_this_update='A17:pinned3newencodertrees/6blobs,readselectedbodies;annotation/cachedgeometryaudit. Noauthormodel/sourcecoreexecution.',
    current_novelty_boundary='Genericsemanticordering,large-smallmixing,multiplepredictionsandbudgetingalreadyprior. Fixed-cellconditionalmissCIwide;noapprovedencoder/decoderincrement.')
src['A17_new_encoder_sources']=audit
src['A17_local_multiplicity_audit']={'population':str(RUN/'annotation_summary.json'),'cached':str(RUN/'cached_cell_summary.json'),
    'verification':str(RUN/'independent_verification.json'),'mainline_validated':False,'new_complete_model_forwards':0}
src['next_instance_ownership_question']={'status':'UNVERIFIED_SOURCE_DISTINCTION_AND_FRESH_PROTOCOL_REQUIRED','question':question,
    'closest_incomplete':['CrowdDetfullmethod/code','RepulsionLossfullmethod/code'],'new_frozen_protocol_required_before_any_new_model_phase':True}
write(ROOT/canonical[4],src)
write(RUN/'scope.json',{'utc':utc,'status':'COMPLETED_SOURCE_AND_ZERO_FORWARD_ANNOTATION_CACHED_AUDIT',
    'new_complete_model_forwards':0,'new_suffix_forwards':0,'source_operator_calls':0,'new_fits':0,
    'cumulative_complete_model_forwards':252,'main_model_updates':0,'mainline_validated':False,'goal_status':'ACTIVE'})
files=[ROOT/n for n in canonical]+[report,audit_path,RUN/'search_log.json',RUN/'annotation_protocol.json',
    RUN/'annotation_summary.json',RUN/'cached_cell_protocol.json',RUN/'cached_cell_summary.json',
    RUN/'independent_verification.json',RUN/'annotation_numeric_boundary_audit.json',RUN/'scope.json']
receipt={'utc':utc,'status':'A17_COMPLETED_VERIFIED_ZERO_FORWARD_RESEARCH_GOAL_ACTIVE','goal_status':'ACTIVE',
    'mainline_validated':False,'new_complete_model_forwards':0,'cumulative_complete_model_forwards':252,
    'historical_suffix_forwards':72,'new_suffix_forwards':0,'new_fits':0,'main_model_updates':0,
    'source_trees_verified':3,'source_blobs_verified':6,'core_literature_rows':69,
    'search_queries':query_count,'web_request_records':len(raw),
    'primary_annotation_files_reparsed':6471,'cached_pairs_verified':69,'decision':cached['decision'],
    'prior_A16_completion_hashes_verified_before_handoff':True,'prior_canonical_snapshot':str(history),
    'files':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size} for p in files],
    'next':question,'limits':'Completedoneboundedresearchaudit,notentiremainlineobjectiveornewalgorithmvalidation.'}
write(RECEIPT,receipt)
print(json.dumps({k:receipt[k] for k in ['status','source_blobs_verified','core_literature_rows','search_queries','new_complete_model_forwards','cumulative_complete_model_forwards','mainline_validated']}))
