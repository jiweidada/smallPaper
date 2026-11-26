import datetime,hashlib,json,pathlib,shutil,sys
sys.stdout.reconfigure(encoding='utf-8')
D=pathlib.Path(__file__).resolve().parent;B=D.parent;LIB=B.parent/'文献_相关原始论文_20261004'
NOW=datetime.datetime.now(datetime.timezone.utc).isoformat()
CAN=['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json']
REPORT=B/'74_A32新编码器筛查与邻近实例证据边界.txt'
RECEIPT=B/'75_A32架构原文与共格关联核验回执.json'
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,o):pathlib.Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not REPORT.exists() and not RECEIPT.exists(),'Already published;do not rerun.'
prior=read(B/'73_A31缓存控制与独立复算回执.json')
for p,h in prior['files_sha256'].items():assert sha(p)==h,p
assert len(prior['files_sha256'])==76
lock=read(D/'geometry_one_shot.lock');assert lock['plan_sha256']==sha(D/'geometry_exposure_plan_FROZEN.json')
g=read(D/'geometry_result.json');v=read(D/'geometry_independent_verification_and_covariates.json')
assert g['status']=='COMPLETED_EXPLORATORY_GEOMETRY_ASSOCIATION'
assert v['status']=='PASS_INDEPENDENT_NATIVE_EVENTS_CENTERS_AND_PRIMARY_STATISTICS'
assert g['new_model_calls']==g['new_optimizer_updates']==0
for planfile in ['geometry_exposure_plan_FROZEN.json','geometry_covariate_plan_FROZEN.json']:
 for row in read(D/planfile)['inputs'].values():assert sha(row['path'])==row['sha256'],row['path']
sources=read(D/'selected_source_manifest.json')['rows']+read(D/'additional_source_manifest.json')['rows']
assert len(sources)==25
for row in sources:
 data=pathlib.Path(row['local']).read_bytes()
 assert hashlib.sha256(data).hexdigest()==row['sha256']
 assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==row['git_blob_sha1']
backup=D/'canonical_before_A32';backup.mkdir(exist_ok=False)
for name in CAN:shutil.copy2(B/name,backup/name)
prior_remapped={str(backup/pathlib.Path(p).name) if pathlib.Path(p).parent==B and pathlib.Path(p).name in CAN else p:h for p,h in prior['files_sha256'].items()}
for p,h in prior_remapped.items():assert sha(p)==h

specs=[
 dict(id='SPAR_Det_WACV2026_A32',stem='A32_SPAR_Det_WACV2026',title='SPAR-Det: Segmentation-guided and Prior-Aided Routing for Small Object Detection',authors=['Seungchan Kwon','Gyuil Lim','Youngjoon Han'],url='https://openaccess.thecvf.com/content/WACV2026/papers/Kwon_SPAR-Det_Segmentation-guided_and_Prior-Aided_Routing_for_Small_Object_Detection_WACV_2026_paper.pdf',tier='WACV2026 officialCVF',read_scope='MainSec3.1-3.4,Sec4.1andselectedTable1; supplementSec1.1,2.1andselected2.6-2.7. SelectedCAHF/RoIhead/VisDroneconfig/registry statically read;not fullpaper or reproduction. PDFstructuralpreflight UNAVAILABLE, no trusted PDFpage anchors claimed.',author_method='Segmentation-guided channel/spatial gating,Gaussian/semantic geometric supervision,and RoI-statistics top4-of8 MoE; CascadeRCNN/DetectoRS/RFLA setup.',our_consequence='GenericMoE,foreground enhancement and geometry guidance in UAV/SOD are prior art. Pinned public VisDrone config/registry selects a learned flattened-RoI gate rather than the paper8-statistics gate.',unresolved='Exact manuscript-to-public-code execution equivalence not established; author code not run.'),
 dict(id='DEIMv2_A32',stem='A32_DEIMv2_arXiv2509_20787v5',title='Real-Time Object Detection Meets DINOv3',authors=['Shihua Huang','Yongjie Hou','Longfei Liu','Xuanlong Yu','Xi Shen'],url='https://arxiv.org/abs/2509.20787v5',tier='arXivv5,27Oct2026;no new venue claim',read_scope='CompleteMethod2 via primaryHTML/localtext,selectedTable1-3 and paragraph on scale-dependent improvements; selectedSTA/ViT/config source. Not fullpaper,training or checkpoint reproduction.',author_method='DINOv3 or distilled compactViT,parallel STA detailCNN fused with resized intermediate tokens;simplified decoder and objectCopyBlend.',our_consequence='Shallow detail fusion and compact DINOv3-to-detector transfer are prior art. Its Sconfig selects blocks3/7/11; paper example indices are not a frozen recipe guarantee.',unresolved='Published COCO gains do not establish a UAV mechanism or our new operator; newhost training memory unmeasured.'),
 dict(id='EdgeCrafter_A32',stem='A32_EdgeCrafter_arXiv2603_18739v5',title='EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation',authors=['Longfei Liu','Yongjie Hou','Yang Li','Qirui Wang','Youyang Sha','Yongjun Yu','Yinzhi Wang','Peizhe Ru','Xuanlong Yu','Xi Shen'],url='https://arxiv.org/abs/2603.18739v5',tier='arXivv5,22Oct2026;manuscript/project state TMLR08/2026',read_scope='CompleteSec3.1-3.3;selectedSec4.2.2/Tables8-9,11,AppendixA.1/A.2.1; selectedECViT andS/baseconfig source. Otherdense-task methods/experiments not fully read; no modelrun.',author_method='Detection-specialized teacher distillation;four stride2 convolutions;mean lateViT tokens then bilinear8/16/32pyramid;hybrid encoder and4-layer300-query decoder.',our_consequence='Provides an inspectable compact architecture baseline. Its stem ablation improvesCOCOAPsmall; heavierSTA has no clear totalAP benefit in its matchedM-scale comparison.',unresolved='Nominal16pixel representation is not an instance-capacity bound. No task-adapted VisDrone checkpoint or single24GB training capacity verified.'),
 dict(id='RT_DETRv4_A32',stem='A32_RT_DETRv4_arXiv2510_25257v2',title='RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models',authors=['Zijun Liao','Yian Zhao','Xin Shan','Yu Yan','Chang Liu','Lei Lu','Xiangyang Ji','Jie Chen'],url='https://arxiv.org/abs/2510.25257v2',tier='arXivv2,8Oct2026',read_scope='CompleteMethod3.1-3.4 from primaryHTML;selectedSec4.2Tables2-7;teacher/encoder files archived and selected,not fulltraining-source closure or paper reproduction.',author_method='Training-only DSI aligns AIFI F5 to frozenVFM features with cosine loss;GAM modulates weight using epochwise gradient-norm ratios.',our_consequence='Generic VFM semantic injection,cosine distillation and gradient-weight balancing are prior art; this is not a new inference encoder.',unresolved='Author information-bottleneck argument is motivation,not proof of our UAV mechanism; no task-adapted reproduction.'),
 dict(id='QueryDet_CVPR2022_A32',stem='A32_QueryDet_CVPR2022',title='QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection',authors=['Chenhongyi Yang','Zehao Huang','Naiyan Wang'],url='https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html',tier='CVPR2022 officialCVF',read_scope='CompleteSec3.2andselected3.1/3.3 from primaryPDFtext/primaryweb;not fullpaper,results,sourceaudit or reproduction.',author_method='Coarse small-object query heatmaps recursively select finerFPN positions; sparseconv heads reuse denseheadweights;dense training with level-loss rebalance.',our_consequence='Generic coarse-to-fine sparse high-resolution reading is prior art,including VisDrone application.',unresolved='Does not establish whether same-cell neighboring targets are merged in a compactViT; proposed exactoperator comparison remains open.')
]
records=[]
for s in specs:
 s=dict(s);stem=s.pop('stem');meta_path=LIB/(stem+'.source.json');meta=read(meta_path)
 assert sha(meta['file'])==meta['sha256'] and sha(meta['text_file'])==meta['text_sha256']
 meta.update(read_scope=s['read_scope'],read_scope_updated_utc=NOW,read_by='assistant',human_read_attested=False)
 reading=LIB/'_text'/(stem+'_reading.txt')
 if reading.exists():meta.update(alternate_reading_text=str(reading),alternate_reading_text_sha256=sha(reading))
 write(meta_path,meta)
 s.update(local_pdf=meta['file'],pdf_sha256=meta['sha256'],access_record=str(meta_path),read_by='assistant',human_read_attested=False);records.append(s)
supp_path=LIB/'A32_SPAR_Det_WACV2026_supplement.source.json';supp=read(supp_path)
assert sha(supp['file'])==supp['sha256'];supp.update(read_scope='Sec1.1/Algorithm1,Sec2.1and selected2.6-2.7;not fullsupplement. No PDFpage anchor trusted.',read_by='assistant',human_read_attested=False,read_scope_updated_utc=NOW);write(supp_path,supp)
write(D/'new_core_records.json',records)
write(D/'discovery_only_records.json',[dict(title='DPT: Deformable Patch-based Transformer for Visual Recognition',url='https://arxiv.org/abs/2107.14467',read_scope='Primaryabstract andmetadata only;no fullmethod/code read.',consequence='Generic adaptive patch positions/scales are prior;exact operator novelty is not cleared.')])

audit=dict(recorded_utc=NOW,selected_source_blobs=25,author_code_imported_or_executed=False,
 ECDetS=dict(commit='d8752b75c9beaecfcdd0e5cd207bae3b6946ae23',source='ecdetseg/engine/edgecrafter/ecvit.py',operators=dict(stem='ConvPyramidPatchEmbed151-172:three normalized stride2convs then stride2projection,totalstride16;overlapping receptivefields',token_features='VisionTransformer326-340;registertoken participates butreturn_layers omitit',fusion='ViTAdapter475-499:mean selectedblocks10/11;bilinear scales2/1/0.5',projection='Sproj_dimNone builds oneprojector forcoarsestmap;HybridEncoder acceptsproj_featsdirectly withno input_proj. Laterlateral/FPNblocks have convolutions. Do not claim everybranch gets a separate pre-encoder1x1.',decoder_config='S/baseconfig:192width,4layers,300queries,8/16/32strides'),runtime='No encoder/head/decoder execution;no weight or environment closure proved'),
 DEIMv2S=dict(commit='f005f50ce565e63f6c298feccb51901a98c3d035',source='engine/backbone/dinov3_adapter.py',operators='Parallel detailCNN produces8/16/32features; resized selectedViT3/7/11tokens concatenate detailfeatures,then1x1+SyncBN. Distinct from ECDet late-feature mean.'),
 SPARDet=dict(commit='11b63c75a04bfe70c8e4b1742d572f58f707e9f7',paper='8-statisticRoI gate;top4-of8',config_entry='VisDrone Cascade thirdbboxhead namesConvFCBBoxHead_MoE; bbox_heads/__init__.py imports convfc_bbox_head_custom',selected_path='custom.py263-299:learnedflattenedRoI gate,top4-of8;all expert FCoutputs computed before gather;shared path.',alternate_path='custom_static9.py contains separate9-statistic task gates and computesall expert outputs;not selected by inspected bbox_heads/__init__.',limitation='Static selected-entry mismatch,not an executed full dependency audit or claim that every author configuration behaves thisway.'),
 RTDETRv4=dict(commit='55fefaaed7efe2a5f72d0a18fd4e05965e35c292',scope='Teacher/encoder source archived;primary method3.1-3.4read;not fulltraining implementation audit.'),
 novelty_boundaries=['Adaptivepatches,DINOsemantictransfer,detailfusion,MoEandcoarse-to-fine sparse reading are not new by themselves.','Moveexistingoperatorstoanewstage orrenameencoder/decoder does not establish a contribution.','No evidence yet establishes an independent of visibility architecture mechanism.'])
write(D/'source_operator_audit.json',audit)

tensions=[
 dict(pair_id='CP-A32-001',paper_a='DEIMv2_A32',paper_b='EdgeCrafter_A32',candidate_basis='shared construct/outcome/measure',overlap_topic='Need for a parallel detail-fusion path',a_finding='STA complements coarseViT semantics with image details.',a_evidence_pointer='DEIMv2v5,Method2,SpatialTuningAdapter',b_finding='Mean latefeature fusion matches heavierSTA totalAP in ECDetM;nativeConvStem helpsAPsmall.',b_evidence_pointer='EdgeCrafterv5,Sec4.2.2,Tables8-9',pair_assessment='conditional_difference',resolution_status='flagged_unresolved',scholar_confirmation='pending'),
 dict(pair_id='CP-A32-002',paper_a='TinyFormer',paper_b='EdgeCrafter_A32',candidate_basis='shared RQ subtopic',overlap_topic='High-resolutiondetailpaths versus efficientcompactViT',a_finding='SSA/PBM preserve shallowdetail;publishedsmallobjectgains.',a_evidence_pointer='TinyFormerv1,Sec3.1-3.2and4.3-4.4;existingmatrixrecord',b_finding='CompactConvStem and latefeature averaging are competitive under its taskdistillationrecipe.',b_evidence_pointer='EdgeCrafterv5,Sec3.1-3.3and4.2.2',pair_assessment='conditional_difference',resolution_status='flagged_unresolved',scholar_confirmation='pending'),
 dict(pair_id='CP-A32-003',paper_a='EdgeCrafter_A32',paper_b='RT_DETRv4_A32',candidate_basis='shared RQ subtopic',overlap_topic='Task-aligned VFM semantictransfer',a_finding='Distill compactbackbone before downstream adaptation.',a_evidence_pointer='EdgeCrafterv5,Sec3.3',b_finding='Align deepAIFI features during detectortraining withno extra inferencepath.',b_evidence_pointer='RTDETRv4v2,Sec3.2-3.4',pair_assessment='no_material_conflict',resolution_status='not_applicable',scholar_confirmation='pending')]
write(D/'cross_paper_tensions.json',dict(cross_paper_tensions=tensions,coverage_note='Scoped advisory scan:five newlyread core papers plus TinyFormer context;threecandidatepairs;not allpairwise orcross-neighborhood conflicts checked. Bibliographiccoupling is inclusiononly. Scholarconfirmation pending;no causal resolution of differing recipes claimed.'))

question='控制可见性、截断和尺寸后，密集邻近小目标是否仍存在由特征采样或实例交互造成的额外可分辨性损失？先以可见内容固定的配对诊断检验，再判断新编码器/实例级解码读出是否有可区分增量。'
draft=dict(status='DRAFT_MECHANISM_DIAGNOSIS_NOT_ALLOCATED_OR_FROZEN',recorded_utc=NOW,question=question,
 evidence=dict(primary_exposed=1470,total_small=4127,primary_O_stratified_RD=g['primary_stratified']['O_CACHED_NATIVE'],posthoc_O_fine_visibility=v['posthoc_sensitivity']['fine_area_occlusion_truncation']['O_CACHED_NATIVE']),
 competing_hypotheses=['Hvisibility:neighbor association is primarilyvisibility/size/contextdifficulty,not tokenmerging.','Harchitecture:atfixedvisiblepixelsandrelativelayout,feature sampling/interaction still induces extra instability for nearbysmallinstances.'],
 next_control='Use sourceD0task-adapted YOLO26s first. Integer-translate alreadyletterboxed visible image content inside sufficient existinguniformpadding;keep everyrealpixel andrelativeGTlayout/visibility unchanged. Comparezero,sub-strideandfullstride-period controls withnativeoutputs restored tooriginalcoordinates. Native gridphase/position/border effects still needseparatecontrols;this is not an automatic causal proof.',
 prerequisites=['Auditexisting source/runtimeandcoordinate APIs;immutable checkpoint/inputhashes.','Selectcases bygeometry/paddingonly,never by observedmissesorbestshift.','Freeze signedshifts,casecount,repeatidentitytest,reportallarms andstrictmodel-callcap in a NEW protocol before any forward.','Verify exactrealpixelpreservation,labelarea/category/visibility invariance,and zerooffsetoutput fidelity.','Use source-alignedignoredregions andsamemisreportbudget;do not compareonlysameconfidence thresholds.','Existing100images remain exploratory;freshconfirmation requirespriorimage-use audit.'],
 architecture_option='ECDetS is a source-inspected compacthost option;no task-adaptedVisDroneweights or measured24GB trainingmemory. Never compareCOCO80-classheaddirectlytoD0VisDrone10classAP.',
 candidate_increment='Onlyif mechanism survives:investigate instance-conditioned finefeature reading underfixedquery/highresolutionbudget. GenericP2/STA,morequeries,adaptivepatches,MoEandprobability-centertransport are not claimednovel.',
 closest_before_algorithm=['TinyFormer/DEIMv2/EdgeCrafter','QueryDet andDPTfulloperator','CrowdDet/RelationDETR/EASE/DualR','ExistingLPS/shift-equivariantViT andAeroPinWorld ifphase-based remedy considered','Instance-maskedcrossattention/box-supervisedinstanceownership literature before any ownershipclaim'],
 model_calls_allocated=0,optimizer_updates_allocated=0,new_algorithm_approved=None,automatic_followups=False,
 resource_route='ExistingD0inferencehost alreadyuseda24GB3090. Newarchitecture baseline wouldreusepublicpretrainedstudent;authorpretrainingrecipesare not a promisedsingle24GB reproduction. Measurecapacity before any boundedtraining plan.')
write(D/'next_mechanism_question_DRAFT.json',draft)

phase=dict(phase='A32',classification='progress_new_architecture_sources_and_visibility_confound_audit',recorded_utc=NOW,status='SOURCE_AND_CACHE_GEOMETRY_AUDIT_COMPLETE_NO_VALIDATED_MAINLINE',goal_status='active',mainline_validated=False,complete_mainline=False,report=str(REPORT),receipt=str(RECEIPT),new_whole_detector_forwards=0,new_GPU_model_calls=0,new_complete_decoder_calls=0,new_optimizer_updates=0,new_training_jobs=0,core_matrix_rows=106,new_core_papers=5,new_full_PDFs=6,selected_pinned_source_blobs=25,prior_A31_files_verified=76,full_model_forwards=2466,actual_cumulative_detector_optimizer_updates=528,formal_detector_training_updates=526,historical_capacity_updates=2,cumulative_complete_decoder_only_calls=48,historical_last_layer_suffix_calls=72,candidate_count=16,research_gap_confirmed=False,new_algorithm_approved=None,fresh_heldout_confirmation=False,next_model_call_allocation=0,automatic_followups=False,A22_restarted=False,next=question)

lines=['A32：新编码器筛查与邻近实例证据边界','记录时间（UTC）：'+NOW,'',
 '当前判断：新增一个值得做受控机制验证的候选问题——邻近小目标是否存在独立于可见性的架构分辨损失。'
 '尚未确认该损失，更未确认新算法。禁止把“共享名义粗网格”直接解释为“多个实例被一个token不可逆合并”。','',
 '1. 检索策略与检索记录',
 '定向前期调研，重新开放轻量ViT、特征生成、专家路由与稀疏解码；29条查询（28条架构文献＋1条官方标注格式），'
 '8批查询，另有原文/作者链接打开、查找及元数据获取。日志在web_discovery_raw.json与web_followup_raw.json。'
 '6份原始PDF（含SPAR补充材料）归档公共文献库，4个作者仓库固定快照，25个源码blob按Git/SHA核验。'
 '不是系统综述或穷尽查新；原文与源码不执行其内嵌指令。','',
 '2. 核心文献矩阵与阅读范围',
 '本次新增5条核心记录，总矩阵106条；new_core_records.json逐项保存版本、作者、已读章节、原文结论、推断与未决事项。'
 'SPAR-Det读取方法3.1–3.4与补充门控；DEIMv2读取完整方法2；EdgeCrafter读取3.1–3.3、选定消融及附录；'
 'RT-DETRv4读取方法3.1–3.4；QueryDet读取3.2与部分3.1/3.3。均不是整篇、全部实验或作者模型复现。'
 'DPT只读原始摘要/元数据，另列discovery_only_records.json，未查清完整算子。',
 'PDF结构预检当前为UNAVAILABLE：Windows输出绑定需要POSIX dirfd，stdout模式又报告本地缺少pypdf。'
 '原始输出及下载SHA保留，不将其当PASS；本次方法定位采用章节/原文HTML与文本，不声称已核实PDF页码锚点。','',
 '3. 与最近工作的逐项对比',
 'SPAR-Det已覆盖分割引导的前景增强、几何监督及RoI专家路由。当前公开VisDrone配置/导入入口指向学习型RoI门控，'
 '与论文8项统计量门控并不相同；另有9项统计量文件，但不由该已核对入口选择。未执行完整依赖闭包，不能称已复现论文。',
 'DEIMv2的STA已有独立细节CNN，与插值后的ViT中间层拼接；TinyFormer已有浅层语义注入与细节融合，通用加细节支路不构成增量。',
 'EdgeCrafter提供新架构候选宿主：四次步长2卷积至步长16，末两层token平均后插值为8/16/32。S配置为192维、4层解码器、300查询。'
 '其COCO-M消融中ConvStem使小目标AP从33.7到35.9；较重STA的总体AP与简单均值相同54.3，但耗时不同。'
 '不能据此外推VisDrone，更不能宣称STA在所有数据集无效。',
 'RT-DETRv4已覆盖训练时深层VFM语义注入与梯度权重调节；QueryDet已覆盖粗定位引导高分辨率特征的级联稀疏计算。'
 '直接引入DINO/MoE、共享特征或稀疏读取均不是新的研究贡献。',
 'DeepSeek CED仅保留为非对称计算与复用的启发。已有DETR共享图像特征，QueryDet已有延后精细计算；不能凭名称替换建立新颖性。',
 '文献张力：DEIMv2/TinyFormer强调细节补充，EdgeCrafter在不同预训练和stem条件下发现较重融合未必划算。'
 '目前没有隔离配方差异的因果解释，不能强行调和或挑选有利结论。cross_paper_tensions.json列3对候选关系，均为作者待确认的建议，不声称遍历全部冲突。','',
 '4. 有证据支持的观察与尚未确认的空白',
 '使用A31忽略区域处理后的100张开发图，4489个有效GT，其中输入面积<1024的小目标4127个。'
 '本次只使用标注和既有YOLO26预测缓存，0次模型调用、0次训练。两分支各在总FP=926的全局阈值上作唯一IoU0.5匹配。',
 '暴露定义：另一个有效小目标中心落在同一个16像素名义网格，且落入不同8像素细格；主网格相位预先设为(0,0)，另报三个相位。'
 '这不是实际CNN感受野或一个网格只能表示一个实例的容量定理。',
 '主相位暴露1470/4127=35.62%，涉及77张图；单次预测分支暴露组漏检858/1470=58.37%，未暴露组972/2657=36.58%。'
 '三个相位对照暴露数为1430、1444、1397，原始漏检率差均同方向；不能挑选最优相位。','',
 '分层后的漏检率差（暴露减未暴露，百分点；配对图像bootstrap 1000次）：',
 '分层 | 共同支持GT | 常规分支差值[95%区间] | 单次预测分支差值[95%区间]',
 ]
modes=[('图像×类别×3档面积',g['primary_stratified']),('图像×类别×更细面积',v['posthoc_sensitivity']['fine_area']),('再加入遮挡与截断',v['posthoc_sensitivity']['fine_area_occlusion_truncation'])]
for label,z in modes:
 a=z['M_NATIVE_NMS_REFERENCE'];o=z['O_CACHED_NATIVE'];ca=a.get('bootstrap95',a.get('paired_image_bootstrap95'));co=o.get('bootstrap95',o.get('paired_image_bootstrap95'))
 lines.append(f"{label} | {o['eligible_GT']} | {100*a['weighted_risk_difference']:.4f} [{100*ca[0]:.4f},{100*ca[1]:.4f}] | {100*o['weighted_risk_difference']:.4f} [{100*co[0]:.4f},{100*co[1]:.4f}]")
lines.extend([
 '更细面积、遮挡与截断敏感性是在看到初始正关联后提出、但在自身结果计算前冻结的事后探索，不能伪装为事前检验。'
 '最后两分支区间均包含零。共同支持人群从2855降至1668，差值不能直接相减来量化“遮挡解释了多少”。'
 '图像级重抽样也不是飞行序列级独立确认；100图已反复使用。',
 '【实测】邻近小目标是数量可观的困难群体，粗分层下关联明显，严格可见性控制后证据变弱。'
 '【推断】它值得一个可见内容固定的机制诊断。'
 '【待验证假设】编码器/特征交互/解码读取还会产生独立于遮挡的实例分辨损失。'
 '【未成立】由此确认token混合、证明新编码器缺陷或获得可投稿的新算子。','',
 '5. 候选增量、可行性与主要风险',
 '候选问题保留，算法尚不批准。若机制得到支持，可考察固定查询数和高分辨率预算下的实例条件细粒度特征读取；'
 '还须区别于QueryDet、DPT、CrowdDet及已有查询关系、实例掩码注意力等。单纯扩大网格、增查询、P2/STA、改概率读取点或拼MoE不列为创新。',
 'VisDrone＋YOLO26s继续作为已训练诊断基线；ECDet-S是源码已审的轻量新宿主选项，尚无已验证的VisDrone适配权重与24GB训练容量。'
 '可以复用公开蒸馏学生权重，作者从头教师/蒸馏配方不等于本机数天可重现。不能直接拿COCO80类头与VisDrone10类AP比较。',
 '主要风险：遮挡及残余尺寸混杂、共同支持样本减少、名义网格不等于信息瓶颈、预训练与模型配方混淆、重复开发集选择偏差，及最近工作未排尽。','',
 '6. 最小验证路线与已完成核验',
 '本轮冻结了网格暴露定义、相位、分层、误报口径和重抽样；完成1176360次独立成对网格检查、16508个GT×相位核对。'
 '另以独立标量匹配重算33016个双分支GT×相位事件，核对坐标、原始标签加入与主统计，最大主统计差2.78e-17。'
 '完整结果、冻结输入与代码分别在geometry_result.json、geometry_exposure_plan_FROZEN.json和独立核验回执。',
 '下一协议仍为DRAFT：在足够原有letterbox均匀填充内整数移动全部真实像素，保持目标可见内容、相对布局、面积和遮挡不变；'
 '设零位移、子步长和整周期控制并恢复输出坐标，区分网格采样与可见性。必须先核对边界/位置编码影响、冻结组别及独立新调用上限，再决定执行。'
 '若考虑相位补救，必须先与LPS、已有等变ViT等比较；观察到不稳定也不自动产生新算法。',
 '本轮没有远端访问、模型权重下载、模型前向或新训练；不重启旧DEIM/A22。不自动调度后续。累计完整检测前向2466、架构优化器更新528、完整DETR解码器重放48，均未增加。','',
 '7. 可核验参考文献与交付文件',
])
for r in records:lines.append(r['title']+'；'+r['url'])
lines.extend(['官方标注字段：https://aiskyeye.com/evaluate/results-format-2024/','原文方法/源码审计：'+str(D/'source_operator_audit.json'),'完整文献与阅读范围：'+str(D/'new_core_records.json'),'统计独立核验与事后敏感性：'+str(D/'geometry_independent_verification_and_covariates.json'),'下一问题草案：'+str(D/'next_mechanism_question_DRAFT.json'),'目标保持active；尚无经验证主线、获准新算法或新增模型预算。'])
REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')

board=read(B/CAN[0]);assert len(board['candidates'])==16
board.update(status='SEARCHING_NO_VALIDATED_MAINLINE_A32',mainline_validated=False,latest_update_utc=NOW,latest_report=str(REPORT),current_priority_candidate=None,current_priority_question=question,next_exploratory_question=question,A32_evidence=phase,A32_selection_note='保留独立于可见性的邻近实例分辨损失问题；严格控制后区间跨零，未证实新编码器缺陷或新算子。ECDetS只作为新宿主选项，0调用预算。')
write(B/CAN[0],board)
search=read(B/CAN[1]);search.update(supplement_utc=NOW,A32_compact_encoder_MoE_and_visibility=dict(phase=phase,queries=29,literature_queries=28,annotation_queries=1,query_batches=8,primary_PDFs=6,pinned_source_blobs=25,logs=[str(D/'web_discovery_raw.json'),str(D/'web_followup_raw.json')],reading_records=str(D/'new_core_records.json'),source_audit=str(D/'source_operator_audit.json')));write(B/CAN[1],search)
matrix=read(B/CAN[2]);assert len(matrix['rows'])==101
assert not any(r.get('id') in {s['id'] for s in records} for r in matrix['rows'])
matrix['rows'].extend(records);matrix.update(updated_utc=NOW,latest_report=str(REPORT),latest_empirical_boundary='A32:nominalcoarse-gridneighbor association weakens after finer size/visibility control;CIcrosseszero. No independent of visibility architecture mechanism or newoperator established.');write(B/CAN[2],matrix)
state=read(B/CAN[3]);assert state['model_forwards']==2466 and state['training_updates']==528
state.update(recorded_utc=NOW,goal_status='active',mainline_validated=False,previous_goal_turn=state['current_goal_turn'],current_goal_turn=phase,latest_authoritative_phase=phase,latest_report=str(REPORT),current_priority_question=question,A32_evidence=phase);write(B/CAN[3],state)
source=read(B/CAN[4]);source.update(updated_utc=NOW,A32_compact_encoder_and_prior_boundary=dict(phase=phase,audit=str(D/'source_operator_audit.json'),tensions=str(D/'cross_paper_tensions.json'),mechanism_draft=str(D/'next_mechanism_question_DRAFT.json'),runtime_or_algorithm_approval=False));write(B/CAN[4],source)
evidence=[p for p in D.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[REPORT]+[B/f for f in CAN]
for r in records:evidence.extend([pathlib.Path(r['local_pdf']),pathlib.Path(r['access_record'])])
evidence.extend([pathlib.Path(supp['file']),supp_path]);evidence.extend(LIB.glob('_text/A32_*.txt'))
receipt=dict(phase);receipt.update(geometry_result=str(D/'geometry_result.json'),independent_verification=str(D/'geometry_independent_verification_and_covariates.json'),primary_O_excess_miss_rate=g['primary_stratified']['O_CACHED_NATIVE'],posthoc_O_fine_visibility=v['posthoc_sensitivity']['fine_area_occlusion_truncation']['O_CACHED_NATIVE'],previous_A31_evidence_remapped_sha256=prior_remapped,files_sha256={str(p):sha(p) for p in sorted(set(evidence))});write(RECEIPT,receipt)
for p,h in receipt['files_sha256'].items():assert sha(p)==h
print(json.dumps(dict(status='A32_PUBLISHED_NO_VALIDATED_MAINLINE',evidence_files=len(receipt['files_sha256']),prior_verified=len(prior_remapped),core_rows=len(matrix['rows']),selected_sources=25,new_model_calls=0,report=str(REPORT)),ensure_ascii=False))
