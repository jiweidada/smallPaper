from pathlib import Path
import datetime,hashlib,json,shutil,sys
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent;B=D.parent;L=B.parent/'文献_相关原始论文_20261004'
NOW=datetime.datetime.now(datetime.timezone.utc).isoformat()
REPORT=B/'82_A36类别与定位质量分数分解及主线边界.txt'
RECEIPT=B/'83_A36原生分类头与独立指标核验回执.json'
CAN=['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not REPORT.exists() and not RECEIPT.exists()
prior=read(B/'81_A35原文源码与原生前缀核验回执.json')
for p,h in prior['files_sha256'].items():assert sha(p)==h,p
assert len(prior['files_sha256'])==115
manifest=read(D/'execution_inputs_manifest.json')
for name,h in manifest['remote_files_sha256'].items():assert sha(D/name)==h,name
for name,key in [('analyze_scores.py','analysis_sha256'),('source_operator_audit.json','source_audit_sha256'),
 ('pinned_A30_independent.py','pinned_A30_sha256'),('bootstrap_draws_FROZEN.npz','bootstrap_sha256'),
 ('coco_reference.py','coco_reference_sha256'),('verify_score_archive.py','archive_verifier_sha256'),
 ('verify_score_metrics.py','metric_verifier_sha256')]:assert sha(D/name)==manifest[key],name
result=read(D/'score_analysis_result.json');av=read(D/'archive_verification.json');iv=read(D/'independent_score_metric_verification.json')
terminal=read(D/'remote_terminal_observation.json');complete=read(D/'remote_complete.json')
assert complete['head_linear_calls']==12 and not result['quality_harm_question_promoted']
assert sum(result['gates'].values())==3 and result['paired_image_intervals']['APsmall'][0]<0
assert iv['matched_arrays_checked']==1736 and iv['metric_max_abs_error']<1e-12
assert terminal['worker_exit_code']==0 and terminal['observed_start_ticks'] is None and not terminal['exception_exists']
assert read(D/'source_pin_verification.json')['status']=='PASS_ALL4_SELECTED_PINNED_AUTHOR_SOURCE_FILES'
state=read(B/CAN[3]);matrix=read(B/CAN[2]);board=read(B/CAN[0])
assert state['current_goal_turn']['phase']=='A35' and state['model_forwards']==3168 and state['training_updates']==528
assert state['complete_decoder_only_replays_cumulative']==61 and len(matrix['rows'])==112 and len(board['candidates'])==17
ids=['GFL_V2','D_FINE','CLSC_DETR','Dome_DETR']
assert all(sum(r['id']==x for r in matrix['rows'])==1 for x in ids)

reading=[
 dict(id='GFL_V2',file='06_GFLv2_CVPR2021_distribution-statistics_LQE__arXiv-2011.12885.pdf',
      text='06_GFLv2_CVPR2021_distribution-statistics_LQE__arXiv-2011.12885.txt',scope='RecheckedcompleteSec3.2jointproduct/DGQP,selectedpriorcontext;notwholepaperorfullauthorcode.',
      source_conclusion='DGQPusesTopkprobabilitiesandmean;classandqualityrepresentationsmultiply;thejointscoreissupervisedbyQFL.',
      our_consequence='Order-statisticsqualityestimation andjointclassification-qualityscoresareexistingmethods.'),
 dict(id='D_FINE',file='105_D_FINE_ICLR2025_Publisher.pdf',text='105_D_FINE_ICLR2025_Publisher.txt',
      scope='ExistingFDR/GO-LSDpaperreadingpreserved;A36addsselectedderivedDomeLQE/VFL/configsource,notanewfullD-FINEpaperread.',
      source_conclusion='InspectedDome-derivedheadaddsonequalityoffsettoclasslogits;sourceconfigusesVFLoncombinedoutputs.',
      our_consequence='DomefunctionalablationisnotanoriginalD-FINEbenchmarkorproofthatitsrawbranchisacalibratedclassifier.'),
 dict(id='CLSC_DETR',file='01_CLSC_DETR_2026_UAV_candidate_ranking__arXiv-2608.21457.pdf',
      text='01_CLSC_DETR_2026_UAV_candidate_ranking__arXiv-2608.21457.txt',scope='Priorcomplete2.1-2.2preserved;A36rechecks2.2scoreprotectionandselected2.1;notauthorcodeorfullexperiments.',
      source_conclusion='Classificationmaxscore,marginandentropydetermineareliabilitycue;itprotectsscoresfromoverlystronggeometricqualitypenalty.',
      our_consequence='Genericclass-reliabilityprotectionorcross-layerqualityscorecalibrationisalreadydirectUAVprior.')]
library=[]
for row in reading:
    row['local_pdf']=str(L/row['file']);row['pdf_sha256']=sha(L/row['file'])
    row['local_text']=str(L/'_text'/row['text']);row['text_sha256']=sha(L/'_text'/row['text'])
    row['read_by']='assistant';row['human_read_attested']=False
    library += [L/row['file'],L/'_text'/row['text']]
write(D/'existing_core_read_scope_update.json',dict(rows=reading,scope='Selectedexistingoriginalsources;noadditionalcorepaperorhumanreadattestation.'))

question=('当前没有已确认主线。A36不足以推进“LQE最终质量修正压低小目标分数”的解释，'
 '不围绕它调权重、阈值或尺寸子集。下一步对照PiDiViT、DEIMv2STA、ECDet深层特征生成与SPAR等原文/源码，'
 '明确细尺度目标/背景信息如何进入最终query；再设计同一目标、匹配背景上的成对语义判别诊断，'
 '区分图像证据不足、编码损失、融合或查询聚合损失。它区别于既有中心位置线性探针，目前尚非已测失效或批准算法。')
draft=dict(status='SOURCE_COMPARISON_AND_PAIRED_SEMANTIC_PROBE_DRAFT_NO_ALLOCATION',question=question,
 current_dataset='VisDrone;existingtaskadaptedDome-S/YOLO26sarediagnosticreferences,notacommittednewarchitecture.',
 first_source_targets=['PiDiViTfullmethod/sourcepending','DEIMv2STAselectedpinnedsourceavailable','EdgeCrafterECDetselectedpinnedsourceavailable','SPAR-Detpaperandselectedimplementationlimits'],
 comparison_axes=['Spatialresolutionanddepthbefore/afterfusion','Instanceconditioningversusquery-independentfusion',
                  'Useoftarget/backgroundcontrastevidence','Extraannotationsandteacherrequirements','Same-parameter/computecontrols'],
 falsifiable_question='Doencoderfine-scalevaluesdiscriminatesomemissedsmallforegroundfrommatchedbackgroundwhilethefinalqueryfeaturedoesnot,underindependenttrain/evaluationsplits?',
 possible_answers=['Bothdiscriminate:inspectscore/readouttrainingratherthanassertencoderloss.',
   'Finevaluesdiscriminatebutquerydoesnot:investigatefusion/aggregationandadistinctquery-readoperator.',
   'Neitherdiscriminates:inspectvisibility,encodinganddomainadaptation;nostandalonereadoutclaim.',
   'Querydiscriminatesbetter:contextmaybenecessary;localcontrastcouldremoveusefulevidence.'],
 constraints=['Controlobjectsize,class,occlusion/truncationandinitialgeometry;donotcallrawboxoverlaprecoverablesemantics.',
   'Distinguishsemanticforeground/backgroundprobe fromoldA1center-locationprobe.',
   '12reusedvalcontextsareexploratory;afreshpropertrain/heldoutdesignisneededbeforepromotion.',
   'Requireaconcretenewconnection/constraintwithclosest-workcomparison;newmathematicalprimitiveisnotmandatory,butrenamingorstackingknownmodulesisnotenough.',
   'CompactViTVisDroneweightsand24GBtrainingcapacityarenotyetverified.'],
 new_model_call_allocation=0,new_training_allocation=0,new_algorithm_approved=None)
write(D/'next_encoder_query_information_DRAFT.json',draft)
phase=dict(phase='A36',classification='progress_source_score_decomposition_and_two_arm_counterfactual_ranking_screen',
 recorded_utc=NOW,status='QUALITY_OFFSET_HARM_NOT_PROMOTED_AFTER_NATIVE_HEAD_AND_COCO_REFERENCE_AUDIT',goal_status='active',
 mainline_validated=False,complete_mainline=False,report=str(REPORT),receipt=str(RECEIPT),
 new_whole_detector_forwards=0,new_image_encoder_forwards=0,new_GPU_model_calls=0,new_optimizer_updates=0,new_training_jobs=0,
 new_complete_decoder_calls=0,new_partial_prefix_decoder_calls=0,new_native_head_linear_calls=12,
 independent_float64_linear_reconstructions=12,new_LQE_forward_calls=0,
 head_native_call_cap=12,head_native_call_cap_closed=True,A35_37_cap_remains_closed=True,
 independent_COCO_evaluate_and_accumulate_calls=8,recovery_matching_only_calls=2,
 metric_export_initial_exit_code=1,metric_export_recovery_exit_code=0,
 full_model_forwards=3168,actual_cumulative_detector_optimizer_updates=528,formal_detector_training_updates=526,historical_capacity_updates=2,
 cumulative_complete_decoder_only_calls=61,historical_last_layer_suffix_calls=72,
 contexts=12,arms=2,bootstrap_draws=1000,matched_FP_budget=111,
 raw_minus_native_APsmall=result['RAW_minus_NATIVE']['APsmall'],raw_minus_native_APsmall_paired95=result['paired_image_intervals']['APsmall'],
 raw_minus_native_small_TP=result['RAW_minus_NATIVE']['small_TP'],quality_harm_question_promoted=False,
 research_gap_confirmed=False,new_algorithm_approved=None,core_matrix_rows=112,new_core_papers=0,new_full_PDFs=0,
 existing_core_records_updated=ids,selected_pinned_source_files=4,candidate_count=17,prior_A35_files_verified=115,
 worker_pid=15311,worker_start_ticks='102412936',worker_exit_code=0,worker_process_observed_absent=True,
 automatic_followups=False,next_model_call_allocation=0,A22_restarted=False,new_heldout_dataset_confirmation=False,next=question)

lines=['A36：类别与定位质量分数分解及主线边界','记录时间（UTC）：'+NOW,'',
 '判断：当前仍未找到经证据确认的小论文主线。原生最终分类头的严格两组对照没有支持“去掉LQE即可稳定改善小目标检测”；'
 '当前不推进最终质量修正压分解释。下一步回到编码表征与query聚合的信息判别问题，先做原文/源码对照。','',
 '1. 检索策略与记录',
 '沿用非系统定向前期调研。6条质量估计/尺度/校准检索式，辅以GFLV2/D-FINE原始HTML、既有原始PDF和Dome作者源码。'
 '检索返回页不等于全文已读；第三方或其他任务的发现结果不用于论证本任务新颖性。原始记录：'+str(D/'search_record.json'),'',
 '2. 核心文献与阅读范围',
 '本轮新增核心论文0，矩阵仍112条；补充或澄清GFLV2、D-FINE派生源码、CLSC-DETR及Dome记录。'
 'GFLV2完整3.2再次核对；CLSC沿用此前完整2.1–2.2并复核2.2；D-FINE论文既有范围保留，新增的是Dome派生实现检查。'
 '没有把读取选段记为整篇或人类已读。无可用通过的PDF结构预检回执，使用章节/文本定位，不提供可信页码锚点。','',
 '3. 与最近工作的逐项比较',
 '维度 | GFLV2 | 当前Dome派生head | CLSC-DETR',
 '质量信息 | 各边概率Top-k及均值 | 同类分布统计MLP | 跨层局部候选几何支持与本层质量',
 '分数结合 | 类别概率×IoU表示 | 各类logit＋共享标量质量修正，再sigmoid | 按类别可靠性调节几何质量的惩罚强度',
 '监督/作用 | 联合分数QFL | 所查配置为combined pred_logits的VFL | 原文2.2已有分类max/margin/entropy保护规则',
 '本轮限制 | 不复现GFLV2训练 | 不持有作者完整训练运行记录；不等于原版D-FINE基准 | 未核对作者完整运行代码及其质量标量的精确提取路径',
 '因此，简单加入定位质量评分、分数保护、跨层几何一致性或重标分数，不能直接当作新算法。','',
 '4. 原文结论、推断与待验证问题',
 '【源码事实】最终类别分支为Linear；LQE从四边概率分布的Top-k和均值计算一个标量，加到12个类别logit。'
 '所查源配置使用VFL；Dome-S文件中的gamma0.8属于学习率调度，不能误当分类损失gamma。',
 '【数学解释边界】联合训练下，类别bias统一加g、质量头末层bias减g，最终和保持不变。'
 '因此两部分的绝对值不具备单独校准概率的唯一解释。以下对照比较固定权重下的功能消融和排序，而不是“纯分类概率与真实IoU”。',
 '【冻结问题】在框、查询状态和特征不变时，移除最终质量修正能否在完整COCO小目标AP和同误报预算下稳定改善检测？'
 '只有NATIVE与RAW_CLASS两组，未做修正权重、阈值或尺寸子集搜索。',
 '【评估】12个反复用于开发的VisDrone官方val上下文；固定Top500查询；sigmoid后类别对Top500，再过滤有效标签1–10和忽略区域，不补满。'
 '原图面积≤1024为小目标；源码对齐忽略区域预处理＋COCO101@500，不称完整官方VisDrone评分。'
 '每组重新匹配GT；1000次配对图像重抽样重排全部分数，FP预算111时各组逐次重估阈值。',
 '指标 | NATIVE(%) | RAW_CLASS(%) | RAW减NATIVE(百分点)']
for k in ['AP','APsmall','AP50','AP75','APsmall75']:
    lines.append(f"{k} | {100*result['point']['NATIVE'][k]:.4f} | {100*result['point']['RAW_CLASS'][k]:.4f} | {100*result['RAW_minus_NATIVE'][k]:+.4f}")
ci=result['paired_image_intervals']['APsmall']
lines += [f"主要APsmall差的95%区间为[{100*ci[0]:+.4f},{100*ci[1]:+.4f}]个百分点；未达到事前+0.5个百分点门槛，区间下界亦未大于0。",
 '固定FP111：NATIVE为382个小目标TP、总TP621；RAW_CLASS为386个小目标TP、总TP621。'
 '这是净小目标TP增加4，并不等于仅新增4个目标而无其他目标交换；其重抽样区间为[-3,28]。',
 '5项推进门槛中3项通过、2项失败，当前质量压分问题不晋级。'
 '有限样本没有证明效应为零，也没有排除质量分支在训练中的作用；本轮仅改变最终推理分数。','',
 '5. 候选增量、可行性与风险',
 '本轮不批准新的质量估计或分数校准增量。CLSC等已直接覆盖保护可靠类别分数的思路；当前对照也缺少足够任务收益。'
 '551个漏检小目标中的几何候选与低分观察仍保留为探索性描述，不能自动转化为“语义信息存在但被编码器/质量头抹掉”。',
 '新的候选问题是成对比较细尺度编码值和最终query的前景/背景判别信息。'
 '需要识别同一目标上的信息损失并与现有细节接入方法区分；它尚未被本轮实验验证。'
 '算法贡献不要求创造全新数学原语，但必须有具体连接或约束、已测失效与最近方法和等预算对照的可区分证据。','',
 '6. 最小验证与核验情况',
 '完成12次原生PyTorch分类Linear调用；框、特征及查询状态来自A35冻结缓存。完整模型、编码器、解码器及GPU调用均0，训练更新0。'
 '数值核验另做12个float64投影重建；这与原生head调用分别记账。原生12次上限已封闭。',
 '归档1770820字节、21项，内部20项哈希通过；36个原生框/logit/概率数组与A35逐位一致。'
 '独立float64投影最大误差4.64e-6，sigmoid最大误差6.74e-8；所有6000个query均未改变类别argmax。',
 '独立COCO引擎核对1736组匹配/忽略/分数数组、基准点及3个预定重抽样的5项AP指标，最大差2.78e-16；1000次种子序列、FP前缀和区间已核验。',
 '实现修复记录：8次COCO评估与累积完成后，参考导出因NumPy int64序列化失败。'
 '保留原始失败文件并恢复其中完整的8组指标；另用相同输入补做2次基准匹配导出，不重算原8组指标，不重跑模型。'
 '原始冻结源码、主要指标和门槛均保留；修复说明和两份导出文件一并归档。',
 '分类头worker前台退出码0；PID15311/start_ticks102412936已观察到不存在，无exception。'
 '架构累计完整模型前向仍3168、优化器更新528、完整DETR组件重放61；历史部分前缀/后缀独立记账。'
 '本轮仅证明缓存分类头诊断可在CPU执行，没有证明新的视觉编码器训练符合24GB上限。',
 '后续最小步骤：先比较PiDiViT差分特征、DEIMv2STA、ECDet深层特征生成及SPAR的条件和算子；'
 '再冻结同一目标、面积/类别/遮挡/截断及几何质量匹配背景的成对语义诊断，明确训练与独立验证划分。'
 '当前下一模型调用与训练配额均0，没有自动启动新实验。','',
 '7. 可核验参考文献与文件',
 'GFLV2：https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Generalized_Focal_Loss_V2_Learning_Reliable_Localization_Quality_Estimation_for_CVPR_2021_paper.pdf',
 'D-FINE（所查早期版本）：https://arxiv.org/html/2410.13842v1',
 'Dome-DETR：https://arxiv.org/abs/2505.05741；所查源码pin2dde3bc1946a3e9fad9abd0612b59fc39bd6b861。',
 'CLSC-DETR：https://arxiv.org/abs/2608.21457',
 '冻结协议：'+str(D/'score_protocol_FROZEN.json'),
 '完整结果：'+str(D/'score_analysis_result.json'),
 '独立指标核验：'+str(D/'independent_score_metric_verification.json'),
 '下一步草案：'+str(D/'next_encoder_query_information_DRAFT.json'),'',
 '状态：目标继续active，当前未找到已确认主线。']
REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')
backup=D/'canonical_before_A36';backup.mkdir(exist_ok=False)
for n in CAN:shutil.copy2(B/n,backup/n)
remapped={}
for p,h in prior['files_sha256'].items():
    q=Path(p);q=backup/q.name if q.parent==B and q.name in CAN else q
    assert sha(q)==h;remapped[str(q)]=h
board.update(status='SEARCHING_NO_VALIDATED_MAINLINE_A36',mainline_validated=False,latest_update_utc=NOW,latest_report=str(REPORT),
    current_priority_candidate=None,current_priority_question=question,next_exploratory_question=question,A36_evidence=phase,
    bounded_diagnostic_protocol=str(D/'score_protocol_FROZEN.json'),
    next_decision_protocol=dict(phase='afterA36',status='SOURCE_AND_PROBE_DRAFT_NO_MODEL_ALLOCATION',file=str(D/'next_encoder_query_information_DRAFT.json')))
write(B/CAN[0],board)
search=read(B/CAN[1]);search.update(supplement_utc=NOW,A36_score_source_and_head_audit=dict(phase=phase,search_log=str(D/'search_record.json'),
    reading=str(D/'existing_core_read_scope_update.json'),source=str(D/'source_operator_audit.json'),result=str(D/'score_analysis_result.json')))
write(B/CAN[1],search)
for row in matrix['rows']:
    if row['id'] in ids:
        row['A36_scope_update']=dict(read_by='assistant',human_read_attested=False,report=str(REPORT),native_score_explanation_not_promoted=True)
    if row['id']=='GFL_V2':row['A36_scope_update'].update(rechecked='Sec3.2jointproductandDGQP',Dome_difference='InspectedDomeaddslogitsandjointVFL;notidenticaltotheGFLV2probabilityproduct.')
    if row['id']=='D_FINE':row['A36_scope_update'].update(derived_source='DomeLQE,LinearheadandVFL/configselectedsource;notoriginalD-FINEbenchmark.')
    if row['id']=='CLSC_DETR':
        row['previous_unresolved_before_A36']=row['unresolved']
        row['unresolved']='Exactauthorqualityscalarpathandruntimecodenotclosed;genericclassification-reliabilityprotectionisalreadyprior. A36currentraw-scoreablationdoesnotestablishanewqualitygap.'
    if row['id']=='Dome_DETR':row['A36_scope_update'].update(source_pin='2dde3bc1946a3e9fad9abd0612b59fc39bd6b861',source_read='FinalLinear/LQEforward,VFLandselectedconfig;checkpointheadfunctionalablation12reusedcontexts.')
matrix.update(updated_utc=NOW,latest_report=str(REPORT),latest_empirical_boundary='A36:RAWminusNATIVEAPsmall+0.0355pp,CIcrosses0;qualityoffsetharmscreenfailed,noencodersemanticcausalornewalgorithmproof.')
write(B/CAN[2],matrix)
state.update(recorded_utc=NOW,goal_status='active',mainline_validated=False,previous_goal_turn=state['current_goal_turn'],current_goal_turn=phase,
    latest_authoritative_phase=phase,latest_report=str(REPORT),current_priority_question=question,A36_evidence=phase,
    native_classification_head_linear_calls_A36=12,independent_float64_projection_reconstructions_A36=12,
    current_native_job_terminal=dict(phase='A36',pid=15311,start_ticks='102412936',observed_absent=True,status='COMPLETED',worker_exit_code=0,cap_closed=True))
write(B/CAN[3],state)
source=read(B/CAN[4]);source.update(updated_utc=NOW,A36_joint_score_and_LQE_diagnosis=dict(phase=phase,
    source_audit=str(D/'source_operator_audit.json'),pinned_source=str(D/'source_pin_verification.json'),
    score_result=str(D/'score_analysis_result.json'),independent_metrics=str(D/'independent_score_metric_verification.json'),new_algorithm_approved=None))
write(B/CAN[4],source)
files=[p for p in D.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+library+[REPORT]+[B/n for n in CAN]
receipt=dict(phase);receipt.update(scientific_result=result,archive_verification=av,independent_metric_verification=iv,
    metric_export_recovery=read(D/'reference_export_recovery_amendment.json'),previous_A35_evidence_remapped_sha256=remapped,
    files_sha256={str(p):sha(p) for p in sorted(set(files))})
write(RECEIPT,receipt)
for p,h in receipt['files_sha256'].items():assert sha(p)==h,p
for p,h in remapped.items():assert sha(p)==h,p
print(json.dumps(dict(status='A36_PUBLISHED_QUALITY_HARM_QUESTION_NOT_PROMOTED',evidence_files=len(receipt['files_sha256']),
    previous_A35_files_verified=115,core_matrix_rows=112,whole_detector_forwards=3168,optimizer_updates=528,
    complete_decoder_only_calls=61,new_native_head_linear_calls=12,mainline_validated=False,report=str(REPORT)),ensure_ascii=False))
