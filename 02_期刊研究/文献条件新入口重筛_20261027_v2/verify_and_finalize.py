"""Read-only arithmetic/provenance verification, then append the new round state."""
import pathlib,json,hashlib,collections,copy,time
B=pathlib.Path(__file__).resolve().parent;R=B.parent;LIB=R/'文献_相关原始论文_20261004'
PARTNER=R/'讨论伙伴_文献条件新入口重筛_20261027_v2'
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for v in iter(lambda:f.read(1024*1024),b''):h.update(v)
 return h.hexdigest()
def save(p,x):pathlib.Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
start=time.monotonic();d=read(B/'JPEG_condition_attempt01/results.json');rows=read(B/'JPEG_condition_attempt01/rows.json');checks={}
assert d['status']=='complete' and len(rows)==7019
for split,n,gt in [('train',6471,343205),('val',548,38759)]:
 rr=[r for r in rows if r['split']==split];assert len(rr)==n and sum(r['GT'] for r in rr)==gt
 assert sum(sum(r['class_GT'].values()) for r in rr)==gt
 assert len({r['signature'] for r in rr})==1
 if split=='train':assert all(r['current_C_D_bytes_equal'] for r in rr)
assert len({r['signature'] for r in rows})==1 and d['total_variation_descriptive']==0 and d['unseen_validation']['images']==d['unseen_validation']['GT']==0
assert d['current_C_D_training_byte_mismatches']==0
checks['all7019_header_and_GT_denominators']=True;checks['same_current_signature_not_claimed_same_visual_quality']=True
for name,h in read(B/'JPEG_condition_attempt01/sha256.json').items():assert sha(B/'JPEG_condition_attempt01'/name)==h
proto=B/'01_自然JPEG编码条件_冻结协议.json';assert sha(proto)==proto.with_suffix('.sha256').read_text(encoding='utf8').strip()
bindings=dict(d['inputs']);checks['JPEG_completed_output_hashes_and_protocol']=True
for p,h in d['inputs'].items():assert sha(p)==h,('source_changed',p)
checks['JPEG_input_bindings_rechecked']=len(d['inputs'])
old_schema=R/'讨论伙伴_候选保留主线取证_20261027_v1/文献成立条件取证_20261027_v1/BASE24_schema_preflight.json';schema=read(old_schema);assert len(schema['rows'])==16
field_counts={}
for letter,seed in [('A',1),('B',2)]:
 rr=[x for x in schema['rows'] if x['letter']==letter];assert len(rr)==8
 assert all(set(x['prediction_schema'])=={'outputs','targets','tag','group','forward_DN_seed'} for x in rr)
 assert all(set(x['prediction_schema']['outputs']['dn_meta'])=={'dn_positive_idx','dn_num_group','dn_num_split'} for x in rr)
 field_counts[str(seed)]={'batches':8,'saved_raw_noisy_bbox_batches':0,'saved_noisy_class_batches':0,'historical_input_collision_result':'NA,not0','GT_exposures_from_previous_fixed_cache':6629}
checks['DN_fields_not_confused_with_collision_counts']=field_counts
for p in [old_schema,PARTNER/'source_ledger.json',PARTNER/'DN_source_ledger.json',PARTNER/'DN_cache_schema_only.json']:
 bindings[str(p)]=sha(p)
for ledger in ['source_ledger.json','DN_source_ledger.json']:
 for p,h in read(PARTNER/ledger)['source_hashes'].items():assert sha(p)==h;bindings[p]=h
checks['partner_input_provenance']=True
papers=[
 {'path':str(LIB/'20261027_条件新入口_Lost_in_Compression_CVPRW2022_Publisher.pdf'),'title':'Lost in Compression: the Impact of Lossy Image Compression on Variable Size Object Detection within Infrared Imagery','url':'https://openaccess.thecvf.com/content/CVPR2022W/PBVS/papers/Bhowmik_Lost_in_Compression_The_Impact_of_Lossy_Image_Compression_on_CVPRW_2022_paper.pdf','version':'CVPRW2022 PBVS official CVF accepted PDF,10pages','sha256':'dd3ac1fcb1d34c89b518249965b7cbcce01cb7805bf0d9bcc8a23e574bdf7550','bytes':1504172,'root_actual_read_scope':'PDF1 andPDF5 section5.2 text/table6,PDF6 caption text;selected1/3/4/5/6 extracted not claimed all read;no root image render','text':str(LIB/'_text/20261027_条件新入口_Lost_in_Compression_selected.txt')},
 {'path':str(LIB/'DN诊断后补查_20261020_v1/DINO_2203.03605v4.pdf'),'title':'DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection','url':'https://arxiv.org/pdf/2203.03605v4','version':'Pinned arXivv4 manuscript reused,not claim latest/accepted-version audit','sha256':'2cda48e74a49e3552aedbf61bc6436767ed3867cb710bbd2b79e48624fd66126','root_actual_read_scope':'Existing extracted PDF7–8 section3.3 plusstartPDF9;Fig3 caption text,no root visual render','text':str(LIB/'_text/DN诊断后补查_20261020_v1_DINO_2203.03605v4.txt')},
 {'path':str(LIB/'NAED复核_20261006_Repulsion_Loss_CVPR2018.pdf'),'title':'Repulsion Loss: Detecting Pedestrians in a Crowd','url':'https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Repulsion_Loss_Detecting_CVPR_2018_paper.pdf','version':'CVPR2018 original reused','root_actual_read_scope':'PDF1–2 introduction/problem premise extracted text,not new method review','text':str(LIB/'_text/NAED复核_20261006_Repulsion_Loss_CVPR2018.txt')},
 {'path':str(LIB/'20261027_条件新入口_DomainBed_arXiv2007.01434v1.pdf'),'title':'In Search of Lost Domain Generalization','url':'https://arxiv.org/pdf/2007.01434v1','version':'arXivv1,2020preprint,27pages;not ICLR accepted version','sha256':'92ea5bb1b5beed6af39f54de49cec2e1acc5450b021410a518cd54cc56e60dd4','root_actual_read_scope':'ExtractedPDF2–4;actually readPDF2–3 problem section2 andstartsection3;no performance/figure claim','partner_actual_read_scope':'ReportsPDF2–4 sections2/3;separate ledger','text':str(LIB/'_text/20261027_条件新入口_DomainBed_selected_pages2-4.txt')},
 {'path':str(LIB/'SOD_2026_PR_DN-TOD.pdf'),'title':'Robust Tiny Object Detection in Aerial Images amidst Label Noise','url':'https://arxiv.org/abs/2401.08056v1','version':'ActualPDFarXiv2401.08056v1,2024-01-16,Elsevier submitted preprint,13pages;filename2026_PR not publication verification','root_actual_read_scope':'PDF1–2 text/problem only;not used to reopen lowfrequency or labelnoise line','text':str(LIB/'_text/20261027_条件新入口_DNTOD_problem_pages1-2.txt')}
]
for x in papers:
 p=pathlib.Path(x['path']);h=sha(p)
 if 'sha256' in x:assert h==x['sha256']
 x['sha256']=h;x['bytes']=p.stat().st_size;x['text_sha256']=sha(x['text']);bindings[x['path']]=h;bindings[x['text']]=x['text_sha256']
failed={'title':'Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming','metadata_url':'https://arxiv.org/abs/1907.07484','attempted_pdf_url':'https://arxiv.org/pdf/1907.07484v3','status':'Metadata-only;attemptedversionnotverified;PDFnotretrieved','reason':'Root web opens transport connection failure;requestsSSL UNEXPECTED_EOF;first compression PDF had succeeded,partial download batch failure preserved in run narrative','used_full_text_or_performance':False}
ledger={'workflow':'academic-research-suite,continued targeted problem-precondition checks;not formal review','papers':papers,'unretrieved':failed,'external_network_scope':'Root normal and connector web attempts failed after initial official search;local originals and official CVF download used;partner primary read scope separately declared','no_novelty_clearance':True}
save(B/'03_原文来源与实际阅读.json',ledger)
save(LIB/'_sources_条件新入口_压缩与条件复核_20261027_v2.json',ledger)
checks['primary_source_versions_and_actual_read_scopes']=True
fallacies={'scope':'Current assistant11type interpretation check,not independent review','coverage':'11/11','Simpson':'No outcome-group trend claimed;allimages sharethetested signature.No subgroup reversal conclusion.','Ecological':'Image-level header identity not used to infer individual object visibility or lossy damage.','Berkson':'Whole fixedtrain/val source population read;no selecting high-error objects.','Collider':'No conditioned performance association/cause claimed.','Base_rate':'6471/548images,343205/38759GT,0GTunseen-domain performanceNA;0/8DNfieldavailability not collisionrate.','Regression_to_mean':'No extreme-case/checkpoint selection or new performance intervention.','Survivorship':'Allsourceimages included;oneknowntrain degenerateGT disclosed;NAanddownloadfailures retained.','Look_elsewhere':'Exploratory repeatedproject/val context;all entry outcomes reported,no significance/independent confirmation.','Forking_paths':'JPEGsignature andcounts frozen beforeexecution;no alternativequalitythreshold scan;no synthetic degradation rescue.','Correlation_not_causation':'No compressionharm,domainfailure,DNharm,methodgain or novelty inferred fromcondition records.','Reverse_causality':'No confidence/error pressures interpreted ascauses;DNinput provenance unresolved.'}
save(B/'07_统计解释与失败范围检查.json',fallacies)
verification={'status':'PASS','verification_level':'Arithmetic/provenance only,not projectcausal/newmethod evidence','checks':checks,'seconds':time.monotonic()-start,'new_model_forwards':0,'training':0}
save(B/'06_独立算术与来源核验.json',verification)
statep=R/'raw_EMA逐GT对照_20261026_v1/work_state_v2.json';old=read(statep);assert old['version']==7
snapshot=B/'previous_work_state_v7.json';assert not snapshot.exists();save(snapshot,old)
stage={'version':1,'client_date':'2026-10-27','authority':str(B/'05_本轮问题地图与结论.txt'),'scope':'Second literature-condition entry round,not defaultingto lowfrequency;not whole-project exhaustion','new_problem_cards':[],'confirmed_new_innovation_mainlines':0,'entry_statuses':{'natural_JPEG_encoding_domain':'Exact declared heterogeneity condition absent,all7019onesignature;not no degradation','crowd_neighbor_drift':'History duplication/no distinct current proposition;not generalized negative','ignore_region_protocol':'Already audited and declaredCOCO target;no new interface issue','domain_identity':'Prerequisite notestablished,not absenceof domains','DN_negative_input':'Historical input/role consequence notclosed,NA notnegative'},'cost':{'primary_source_images':7019,'current_training_copy_files':6471,'compressed_image_files_hashed':13490,'RGB_decodes':0,'new_encoded_images':0,'new_random_generation':0,'model_constructions':0,'checkpoint_loads':0,'model_forwards':0,'model_backwards_VJP':0,'optimizer_steps':0,'new_CPU_evaluations':0,'training':0,'BN_updates':0,'GPU_remote_test_data_byte_reads':0},'discussion_thread_id':'01a17b5a-0a45-7921-9591-3ac370e081b5','discussion_title':'两个检测问题的研究讨论','latest_cursor':'574151d0-7511-4597-9f3b-5eab2bbcf277:49','consensus':str(B/'04_双方公开讨论_共识与保留意见.txt'),'private_reasoning_saved':False,'new_round_complete':'Bounded condition screening complete,no qualifiednewmainline;broader exploration open','old_lowfrequency_card_default':False,'verification':str(B/'06_独立算术与来源核验.json')}
save(B/'research_state.json',stage)
current=copy.deepcopy(old);current.update({'version':8,'client_date':'2026-10-27','latest_user_instruction':'继续按新入口重新找，与另一对话讨论；低频不作默认；先核文献条件，再原因/方法缺口；不训练、不提前提模块/loss；无证据主线如实说','current_objective':'Find evidence-supported concrete project problems from primary literature conditions;this round0qualifiednewcards/mainlines','current_authority':stage['authority'],'current_problem_card':None,'current_problem_cards':[],'hypothesis_ledger':str(B/'research_state.json'),'previous_stage_completed_counts':old.get('completed_current_counts'),'completed_current_counts':{'scope':'Current original JPEGquantization/sampling identity;no new model inference','train_images':6471,'val_images':548,'train_GT_annotations':343205,'val_GT':38759,'encoding_signatures_combined':1,'val_unseen_encoding_images':[0,548],'val_unseen_encoding_GT':[0,38759],'current_C_D_training_bytes_mismatch':[0,6471],'DN_raw_input_available_batches_seed1':[0,8],'DN_raw_input_available_batches_seed2':[0,8],'DN_historical_input_collision_seed1':'NA','DN_historical_input_collision_seed2':'NA'},'previous_stage_cost':old.get('latest_stage_cost'),'latest_stage_cost':stage['cost'],'last_wait_cursor':stage['latest_cursor'],'discussion_status':'Publicconsensus0newcards/0mainlines;JPEGconditionabsent,historyduplicateentrieswithdrawn,domain/DNnotestablished;no defaultlowfrequency','historical_unresolved_cards':list(dict.fromkeys([p for p in old.get('historical_unresolved_cards',[])+([old['current_problem_card']] if old.get('current_problem_card') else []) if p])),'previous_state_snapshot':str(snapshot),'confirmed_innovation_mainline_count_this_round':0,'bounded_stage_status':stage['new_round_complete'],'lowfrequency_is_default_mainline':False,'no_training_or_module_loss_proposal':True})
save(statep,current)
for p in [B/'03_原文来源与实际阅读.json',LIB/'_sources_条件新入口_压缩与条件复核_20261027_v2.json',snapshot]:bindings[str(p)]=sha(p)
outputs=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(B.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_manifest.json']
manifest={'client_date':'2026-10-27','authority':stage['authority'],'inputs':[{'path':p,'sha256':h} for p,h in sorted(bindings.items())],'outputs':outputs,'new_problem_cards':0,'confirmed_new_mainlines':0,'no_default_lowfrequency':True,'mutable_state_pointer':str(statep),'mutable_state_version_at_delivery':8,'verification':str(B/'06_独立算术与来源核验.json'),'project_search_not_declared_exhausted':True,'cost':stage['cost']}
save(B/'delivery_manifest.json',manifest)
assert all(sha(x['path'])==x['sha256'] for x in outputs)
print(json.dumps({'status':'PASS','inputs_bound':len(bindings),'outputs_bound':len(outputs),'state_version':8,'new_problem_cards':0,'confirmed_mainlines':0,'seconds':time.monotonic()-start}))
