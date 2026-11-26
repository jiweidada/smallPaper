from pathlib import Path
import datetime,hashlib,json,shutil,sys
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent;B=D.parent;L=B.parent/'文献_相关原始论文_20261004'
NOW=datetime.datetime.now(datetime.timezone.utc).isoformat()
REPORT=B/'80_A35查询路由最近方法与早期置信度诊断.txt'
RECEIPT=B/'81_A35原文源码与原生前缀核验回执.json'
CAN=['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

assert not REPORT.exists() and not RECEIPT.exists()
initial_matrix=read(B/CAN[2]);initial_board=read(B/CAN[0])
assert len(initial_matrix['rows'])==106 and len(initial_board['candidates'])==16
new_titles={x['title'].casefold() for x in read(D/'pre_model_source_audit.json')['closest_work']}
assert (new_titles&{x['title'].casefold() for x in initial_matrix['rows']})=={'dynamic query as budgeting for tiny object detection'}
prior=read(B/'79_A34剩余76图独立复核回执.json')
for name,h in prior['files_sha256'].items():assert sha(name)==h,name
assert len(prior['files_sha256'])==603
manifest=read(D/'execution_inputs_manifest.json')
for n,h in manifest['remote_files_sha256'].items():assert sha(D/n)==h,n
for n,key in [('query_retention_analysis.py','analysis_sha256'),('bootstrap_draws_FROZEN.npz','bootstrap_sha256'),
              ('verify_collected_archive.py','archive_verifier_sha256'),('verify_retention_scalar.py','independent_verifier_sha256'),
              ('pre_model_source_audit.json','pre_model_source_audit_sha256')]:assert sha(D/n)==manifest[key],n
r=read(D/'query_retention_analysis_result.json');av=read(D/'archive_verification.json');iv=read(D/'independent_retention_verification.json')
taxonomy=read(D/'exploratory_miss_taxonomy_result.json');taxonomy_iv=read(D/'independent_taxonomy_verification.json')
lqe=read(D/'lqe_source_and_checkpoint_metadata_audit.json')
assert taxonomy['counts']['small']['native_misses']==551 and taxonomy['counts']['small']['raw_geometry_low_trueclass_weak_foreground']==377
assert taxonomy_iv['missed_GT_rows_checked']==620 and taxonomy_iv['new_model_calls']==0
terminal=read(D/'remote_terminal_observation.json');cfg=read(D/'native_prefix_protocol_FROZEN.json')
assert r['point']['1']['small']['native_supported_GT']==382 and r['point']['1']['small']['attrited_GT']==0
assert r['point']['1']['larger']['native_supported_GT']==239 and r['point']['1']['larger']['attrited_GT']==0
assert not r['question_worth_actual_pruned_decoder_followup'] and sum(r['gates'].values())==2
assert iv['max_statistic_error']==0 and iv['primary_columns_checked']==1001
assert av['counts']['decoder_entries']==37 and av['counts']['complete_three_layer_calls']==13
assert terminal['worker_exit_code']==0 and not terminal['same_process_live'] and not terminal['exception_exists']
state=read(B/CAN[3]);assert state['current_goal_turn']['phase']=='A34'
assert state['model_forwards']==3168 and state['training_updates']==528
assert state['complete_decoder_only_replays_cumulative']==48

audit=read(D/'pre_model_source_audit.json');core=[];library_files=[]
stems={
 'QR_DETR_ACCV2024_A35':'A35_QR_DETR_ACCV2024',
 'SQ_DETR_ApplSci2025_A35':'A35_SQ_DETR_AppliedSciences2025',
 'BUTR_2610_33430_A35':'A35_BUTR_arXiv2610_33430v1',
 'DHQ_DETR_RemoteSensing2025_A35':'A35_DHQ_DETR_RemoteSensing2025',
 'AQSFormer_Electronics2024_A35':'A35_AQSFormer_Electronics2024',
 'DifferentialTransformer_ICLR2025_A35':'A35_DifferentialTransformer_arXiv2410_05258v2'}
manual_authors={
 'QR_DETR_ACCV2024_A35':['Tharsan Senthivel','Ngoc-Son Vu'],
 'BUTR_2610_33430_A35':['Yi Zhang','Di Xiong','Ge Gao','Xiangyue Zhang','Yihang Qiu','Yuxuan Zhou','Shuo Chen'],
 'DifferentialTransformer_ICLR2025_A35':['Tianzhu Ye','Li Dong','Yuqing Xia','Yutao Sun','Yi Zhu','Gao Huang','Furu Wei']}
scopes={
 'DHQ_DETR_RemoteSensing2025_A35':'CompleteSec3.3.2 and selectedSec3.3.1 prose;not all distributiondecoder,losses,figures or experiments.',
 'AQSFormer_Electronics2024_A35':'Sec3.3Algorithm1 and surrounding methodparagraphs;not2D-RoPE,deformable module or full experiments.',
 'DifferentialTransformer_ICLR2025_A35':'Sec2.1prose/equations andSec2.2;notFigure2pseudocode,fullLLMexperiments or author source.'}
for x in audit['closest_work']:
    row=dict(id=x['id'],title=x['title'],url=x.get('primary','https://doi.org/'+x['doi']) if 'doi' in x else x['primary'],
             tier='primary_PDF_selected_methods' if x['id'] in stems else 'publisher_partial_not_full_method',
             read_scope=scopes.get(x['id'],x['read_scope']),author_method=x.get('original_method',x.get('original_claim')),
             our_consequence=x['our_inference'],unresolved=x.get('unresolved','Exactauthortraining/runtime not reproduced;noournewalgorithm validated.'),
             read_by='assistant',human_read_attested=False)
    if 'doi' in x:
        mp=D/'metadata'/('Crossref_'+x['doi'].replace('/','_')+'.json');m=read(mp)['message']
        assert m['title'][0].casefold()==x['title'].casefold()
        row.update(doi=x['doi'],authors=[' '.join(v for v in [a.get('given'),a.get('family')] if v) for a in m['author']],metadata_record=str(mp))
    else:row['authors']=manual_authors[x['id']]
    if x['id'] in stems:
        meta=L/(stems[x['id']]+'.source.json');source=read(meta)
        assert sha(source['file'])==source['sha256'] and sha(source['text_file'])==source['text_sha256']
        source.update(read_scope=row['read_scope'],read_by='assistant',human_read_attested=False,read_scope_recorded_utc=NOW,
                      page_anchor_status='No trustedpageanchors;selectedsections/textlocators only.')
        if 'BUTR' in x['id']:source['extraction_note']='pdftotext reported a font-type warning;budget equations cross-checked against selected verifiedauthor source.'
        write(meta,source)
        row.update(local_pdf=source['file'],pdf_sha256=source['sha256'],access_record=str(meta))
        library_files += [Path(source['file']),Path(source['text_file']),meta]
    if x['id']=='BUTR_2610_33430_A35':row.update(source_pin=x['source_pin'],source_read_scope=x['source_read_scope'],venue_status='arXivv1;repositoryNeurIPSbadge not separately verified asacceptance evidence.')
    core.append(row)
assert len(core)==7
new_rows=[];updated_rows=[]
for row in core:
    matches=[x for x in initial_matrix['rows'] if x['title'].casefold()==row['title'].casefold()]
    if matches:
        assert len(matches)==1 and matches[0]['id']=='BUTR'
        previous=matches[0];row.update(id='BUTR',previous_read_scope=previous['read_scope'],tier=previous['tier'],
            read_scope='PriorcompleteSec3/4.1-4.2 andselected5.1/tables;A35addscompleteAppendixBAlgorithm1,selectedAppendixA/Table7 andselectedpinnedauthor source.Notfullpaperortrainingreproduction.')
        updated_rows.append(row)
    else:new_rows.append(row)
assert len(new_rows)==6 and len(updated_rows)==1
write(D/'new_core_records.json',dict(rows=new_rows,updated_rows=updated_rows,scope='6newrecordsand1existingBUTRread/sourceupdate;noexhaustiveness or fullreadingclaim.'))
write(D/'core_dedup_before_publish.json',dict(previous_rows=106,new_rows=6,updated_existing_ids=['BUTR'],final_rows=112,
      duplicate_title_not_added='Dynamic Query as Budgeting for Tiny Object Detection'))

next_question=('查询早期置信度保留候选Q未通过诊断，不追加其剪枝或门控实验。'
 '下一步先把Dome-S最终类别logit与LQE定位质量修正拆开，结合已有GFLV2/D-Fine等原文判断低分发生在哪个环节；'
 '在排除质量评分影响后，再决定是否值得研究新的目标/背景编码或分类读出。'
 '不能把低分或非零质量头参数直接当作已确认失效机制。当前没有已确认主线或批准的新算法。')
draft=dict(status='SOURCE_QUESTION_DRAFT_NO_MODEL_ALLOCATION',question=next_question,
    candidate_promoted=False,new_model_call_allocation=0,new_training_allocation=0,
    immediate_source_and_cache_action='ReadcurrentLQE/criterion andoriginalquality-estimationmethods;freeze a head-only score decomposition protocol before anyadditional learnedcomponent calls.',
    secondary_literature_lead=dict(title='When Pixel Difference Patterns Meet ViT: PiDiViT for Few-Shot Object Detection',
              primary='https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_When_Pixel_Difference_Patterns_Meet_ViT_PiDiViT_for_Few-Shot_Object_ICCV_2025_paper.pdf',
              read_scope='Searchabstract/intro snippets only;complete methods/source pending;not a new core record.'),
    exclusions=['A34 phase-neighbor branch remains closed.','A35 score-retention screen is not retuned to smaller budgets or different thresholds.',
                'Knowncenter-difference,attention subtraction or momentannihilation propertyalone isnotnovelty.',
                'A16spatialresponse cancellation findings do not establish semanticbackgroundinterference.'],
    needed=['Separateclassifierevidencefromlocalizationqualitycorrection at thepinnedcheckpoint.',
            'IncludeCOCOAPsmallacrossIoU0.5:0.95 andAP75 if later testingquality-headablation;do notpromoteonlyIoU0.5support gains.',
            'Exactclosestoperators and scoped equivalences.','Actualimage evidence of the proposed failure.','Distinctalgorithm difference and single24GB validation route.'])
write(D/'next_readout_source_question_DRAFT.json',draft)
phase=dict(phase='A35',classification='progress_primary_query_routing_audit_and_negative_native_support_screen',recorded_utc=NOW,
 status='SOURCE_AND_NATIVE_PREFIX_AUDIT_COMPLETE_NO_VALIDATED_MAINLINE',goal_status='active',mainline_validated=False,complete_mainline=False,
 report=str(REPORT),receipt=str(RECEIPT),new_whole_detector_forwards=0,new_GPU_model_calls=0,new_optimizer_updates=0,new_training_jobs=0,
 new_complete_decoder_calls=13,new_partial_prefix_decoder_calls=24,new_decoder_layer_calls=75,CPU_decoder_entries=37,
 CPU_decoder_component_cap=37,CPU_decoder_component_cap_closed=True,cumulative_complete_decoder_only_calls=61,
 historical_last_layer_suffix_calls=72,full_model_forwards=3168,actual_cumulative_detector_optimizer_updates=528,
 formal_detector_training_updates=526,historical_capacity_updates=2,cached_contexts=12,GT_rows=1241,
 primary_small_TP_support=382,primary_small_support_attrition=0,primary_larger_TP_support=239,primary_larger_support_attrition=0,
 posthoc_small_native_misses=551,posthoc_small_raw_geometry_but_weak_validclass_score=377,
 posthoc_taxonomy_is_confirmatory=False,LQE_effect_on_real_scores_measured=False,
 matched_FP_budget=111,paired_bootstrap_draws=1000,actual_pruned_decoder_effect_measured=False,
 Q_promoted=False,research_gap_confirmed=False,new_algorithm_approved=None,new_core_papers=6,existing_paper_read_scope_updated='BUTR',core_matrix_rows=112,
 new_full_PDFs=6,verified_selected_author_blobs=8,search_groups=17,search_query_count=57,candidate_count=17,
 prior_A34_files_verified=603,worker_pid=15120,worker_start_ticks='102159569',worker_exit_code=0,worker_process_observed_absent=True,
 worker_exit_scope='ForegroundSSHcommand,not a detachedlaunchercode.',automatic_followups=False,next_model_call_allocation=0,
 A22_restarted=False,A34_phase_branch_reopened=False,new_heldout_dataset_confirmation=False,next=next_question)
write(D/'cross_paper_tensions.json',dict(rows=[
 dict(papers=['BUTR','DHQ-DETR'],type='Scope-sensitive empirical tension',source='BUTRAppendixA/Table7 versusDHQ3.3.2',
      statement='BUTR reports nooverallAPadvantage forDDQ-styleinitialization atmatchedK initsseparatelytrained comparisons;DHQ motivatesgeometry/confidenceinitialization.',
      resolution='Differentmodels,training anddatasets;not a directcontradiction orproofthatallgeometryselection isuseless.'),
 dict(papers=['QR-DETR','SQ-DETR','AQS-IDETR'],type='Differentquerylifecycle assumptions',
      statement='Learnedbypass andpermanentcurrentscorepruning aredifferent;AQS-IDETR remains partialreading.',
      consequence='OurfixedsigmoidTop250supportscreen isnot a faithfulAQS/QRreproduction anddoesnotmeasure actualpruningAP.'),
 dict(papers=['DifferentialTransformer','PiDiViT_discovery_only'],type='Noveltyboundarylead',
      statement='Genericattentiondifference orpixelcontrastcannotbeclaimednew;PiDiViTcompleteoperatorcomparisonpending.',
      consequence='No newmomentconstrainedorcontrastdecoderapproved.')],
 independent_Scholar_citation_network_check='Not completed;targetedprimaryreviewonly.'))

lines=['A35：查询路由最近方法与原生早期置信度诊断','记录时间（UTC）：'+NOW,'',
 '判断：当前仍未找到可立项的具体小论文主线。已补齐查询路由、裁剪、质量筛选与预算机制的最近先例；'
 '新候选Q“早期置信度裁剪误删微小目标最终支持”没有通过冻结诊断，不推进相应门控/保留增量。','',
 '1. 检索策略与检索记录',
 '本轮为非系统定向检索：从新编码器、线性/差分注意力、基础视觉表征，到查询退出/恢复、当前分数与后续价值。'
 '17组工具记录含57条检索式；另有原文打开/查找、Crossref元数据和作者Git源码访问。'
 '原始返回含第三方发现页，方法结论仅使用归档原文与作者源码；检索命中不等于已读。'
 '检索记录：'+str(D/'search_record.json'),
 '取得6篇原始PDF；AQS-IDETR仍仅出版者摘要、引言、方法开头和结论片段。'
 '出版者403/429及公开原始PDF取得路径均保留。没有完成穷尽式Scholar引文网络核查或系统综述。','',
 '2. 核心文献矩阵增补（7项最近工作：新增6条、更新已有BUTR1条；总计112条）',
 '文献 | 已核对机制 | 阅读范围与限制',
 'QR-DETR，ACCV2024 | 学习门控＋gather/scatter，跳过SA、CA或整层 | 4.1–4.2、部分3；作者运行源码未闭合，公式有示意省略，不称复现。',
 'SQ-DETR，AppliedSciences2025 | 各层按当前前景分数Top-k，删除后不再参与 | 3.1–3.2；其softmax公式与本host的sigmoid不同。',
 'BUTR，arXiv2610.33430v1 | 编码器最大sigmoid分数之和→保守查询数→标准TopK | 4.1–4.2、附录B及部分3/附录A；8个源码blob核验，未运行作者代码。',
 'AQS-IDETR，IVC2026 | 推理时按置信度曲线膝点逐步裁剪 | 只有出版者摘要/引言等片段；完整膝点算法与实验细节未读。',
 'DHQ-DETR，RemoteSensing2025 | HQQS在初始化前以框和分数做类别无关NMS0.8，再TopK | 完整3.3.2、部分3.3.1；不是后续层的查询完成价值预测。',
 'AQSFormer，Electronics2024 | GT驱动9近邻、IoU均值＋标准差阈值的训练正负样本分配 | 3.3算法1和周围方法；不能与推理裁剪AQS-IDETR混同。',
 'DifferentialTransformer，ICLR2025 | 两个softmax图相减、共享value、学习lambda和逐头归一化 | 方法2.1正文/公式与2.2；语言模型结论不是航拍检测证据。',
 '新增论文均未记为人类已读。PDF结构预检仍无可用通过回执，采用章节/文本定位，不提供可信页码锚点。'
 'BUTR文本抽取有字体警告，其预算公式与作者源码交叉核对。','',
 '3. 与最近工作的逐项比较',
 '初始化数量：BUTR根据编码器证据确定K；本诊断固定500，不把数量效应混入后续保留问题。',
 '初始化多样性：DHQ与DDQ类方法已用几何去重；普通增加NMS、质量分数或重新排序不能直接成为增量。',
 '生命周期：QR已有保留位置并跳层的机制；SQ永久删去低分查询；AQS-IDETR声称无需重训进行逐层裁剪。'
 '本轮不实现新的跳层/恢复规则，只检验在现有完整解码轨迹中，保留当前Top250是否仍保有最终正确检出的支持。',
 '监督目标：若未来要研究“继续计算的价值”，必须与当前类别置信度、质量分数和QR检测监督门控区分；'
 '当前还未证明这种差异在实际任务上必要，更没有批准算法。',
 '源码边界：BUTR作者pin0c69d53f3329fe97e96fe83a7f352dc8d7c4a1a5中，预算含validmask、1e-6下限和范围裁剪；'
 'S-VisDrone配置为250–500、3层。批量解码取本批最大预算；本次单图缓存诊断不评价其批量效率。','',
 '4. 研究空白、观测与推断',
 '【有原文依据的待核查问题】AQS-IDETR的引言认为高质量查询通常早期出现并持续占优；'
 '这种当前分数的持续性是否适用于航拍小目标，可用原始轨迹检查。已有论文并未因此被证明留有可写空白。',
 '【预设诊断】12个既有官方val开发上下文，有效GT均≤186；固定500个初始查询，标记第一层前250名。'
 '250高于每图GT数并留至少64槽余量，从而降低仅因容量不足而丢目标的干扰。'
 '本轮按原图面积≤1024定义小目标，沿用A27口径；与A33/A34网络输入小目标及相位分组不同。',
 '【匹配口径】原生sigmoid/类别对Top500后有效标签1–10、源码对齐忽略区域过滤；总FP前缀预算111。'
 '同分整块接纳，1000次配对图像重抽样逐次重估全局FP前缀。原图COCO小目标933个，全部有效GT1241个。',
 '【支持丢失定义】仅对完整原生解码已正确检出的GT，查看是否所有同类、IoU≥0.5且在接受前缀内的最终查询，都未进入早期Top250。'
 '存在备份查询即不计丢失。查询状态和完整解码结果保持原样，因此这不是实际裁剪后的召回率或AP。',
 '【新观测】第一层：小目标支持丢失0/382，较大目标0/239；第二层同样为0。'
 '主对比小目标减较大目标为0；图像重抽样区间[0,0]。'
 '零事件的经验重抽样会产生这个退化区间，不能据此认定其他图像或实际剪枝风险精确为零。',
 '【条件核查】图像×类别×最终置信度分箱的共同支持包含198个小目标和191个较大目标，11图29层；调整后差同为0。'
 '5项推进门槛中，仅样本支持与调整后同方向两项通过；效应大小及正区间条件失败。',
 '【判断】没有依据推进Q，也不能从该结果宣称任何实际裁剪算法精度无损。'
 '不改用更少保留槽位、其他阈值或子集挽救假设。原始12图已反复用于开发，结果不是独立盲测或序列外泛化。','',
 '【事后描述，非事前检验】551个漏检小目标中，113个没有原始框IoU≥0.5；377个有几何候选但所有有效类别分数仍低于当前阈值；'
 '54个有几何候选且其他类别分数较强；7个原始真类分数达阈值但未成为原生匹配。'
 '这些描述没有证明语义信息可恢复或背景干扰成立。按归一化坐标的独立标量实现核对620个全部漏检和310000对IoU，分型一致。',
 '【新的区分要求】Dome源码LQE把基于四边分布统计的质量分数加到各类logit；已训练checkpoint的末层权重和偏置并非全零。'
 '因此不能直接把最终低分解释为类别证据弱，也不能从非零参数断言质量修正有害。需要先做分数分解，当前尚未执行该分解。','',
 '5. 候选算法增量、可行性与主要风险',
 'Q当前不推进。普通门控、跳层、恢复、质量评分或数量预算已有直接先例；训练未来价值预测器也必须先有实际必要性与可区分差异。'
 '当前最小风险是把已有架构换名，或把未改变查询状态的支持统计误报为实际剪枝提升。',
 '新的局部目标/背景读出是备选原文线索；优先拆分类别证据与定位质量修正，尚无已确认机制或批准算法。'
 'PiDiViT仅发现片段，差分注意力与低阶矩约束不能凭名称组成创新。','',
 '6. 最小验证与已执行情况',
 '已冻结并完成37次CPU原生解码前缀：12图各1/2/3层前缀，加1次3层重复。'
 '其中完整3层13次，部分前缀24次，共75次解码层调用；完整图像前向0、GPU调用0、训练更新0。',
 'Host为Dome-DETR-S作者VisDrone权重＋固定Top500重放，不是整套PAQI推理复现。'
 '原始分类头与LQE头全部来自同一已训练checkpoint；未重置早期头，未改写解码数值代码。',
 '归档31710574字节、27项，内部26项哈希通过；24个最终头数组逐位等于A27参照，24组前缀特征、36组排名及5个重复数组独立核对。'
 '标量实现核对1241个GT、192443次IoU计算及全部1000次主要重抽样，统计最大差0。',
 '前台SSH执行返回worker退出码0，PID15120/start_ticks102159569已观察到不存在，无exception；37上限封闭，无自动续跑。'
 '原架构累计完整前向仍3168、优化器更新528；完整DETR组件重放累计61，新增部分前缀24与历史末层后缀72分别记账。'
 '该结果只证明这种CPU诊断可执行，没有验证新视觉编码器在24GB上的训练资源。','',
 '7. 可核验参考文献与文件',
 'QR-DETR：https://openaccess.thecvf.com/content/ACCV2024/html/Senthivel_QR-DETR__Query_Routing_for_Detection_Transformer_ACCV_2024_paper.html',
 'SQ-DETR：https://doi.org/10.3390/app15147686',
 'BUTR：https://arxiv.org/abs/2610.33430v1；源码：https://github.com/zhangyi-magic/BUTR/tree/0c69d53f3329fe97e96fe83a7f352dc8d7c4a1a5',
 'AQS-IDETR（部分阅读）：https://doi.org/10.1016/j.imavis.2026.106097',
 'DHQ-DETR：https://doi.org/10.3390/rs17030514',
 'AQSFormer：https://doi.org/10.3390/electronics13234591',
 'DifferentialTransformer：https://arxiv.org/abs/2410.05258v2',
 '冻结方案：'+str(D/'native_prefix_protocol_FROZEN.json'),
 '结果：'+str(D/'query_retention_analysis_result.json'),
 '独立复算：'+str(D/'independent_retention_verification.json'),'',
 '下一步：'+next_question]
REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')

backup=D/'canonical_before_A35';backup.mkdir(exist_ok=False)
for name in CAN:shutil.copy2(B/name,backup/name)
remapped={}
for name,h in prior['files_sha256'].items():
    p=Path(name);q=backup/p.name if p.parent==B and p.name in CAN else p
    assert sha(q)==h;remapped[str(q)]=h
board=read(B/CAN[0]);assert len(board['candidates'])==16
board['candidates'].append(dict(id='Q_QUERY_CONFIDENCE_VS_COMPLETION',status='DEPRIORITIZED_NO_NATIVE_SUPPORT_ATTRITION_IN_FROZEN_SCREEN',
    question=cfg['question'],nearest=[x['title'] for x in core[:6]],increment_approved=False,causal_evidence=False,
    evidence='12reusedcontexts;FP111;0/382smalland0/239largernativesupportsattritedbyfirstlayerTop250;actualpruneddecoderAPnotmeasured.',
    next='No queryretentiongate orsmallerbudgetsearch fromthisscreen;reopenonlywith newindependent mechanism evidence.'))
board.update(status='SEARCHING_NO_VALIDATED_MAINLINE_A35',mainline_validated=False,latest_update_utc=NOW,latest_report=str(REPORT),
    current_priority_candidate=None,current_priority_question=next_question,next_exploratory_question=next_question,A35_evidence=phase,
    existing_reference='YOLO26sD0sourcebaseline;Dome-SauthorweightswithfixedTop500nativecontextforquerydiagnosis;newencodertrainingnotestablished.',
    bounded_diagnostic_protocol=str(D/'native_prefix_protocol_FROZEN.json'),
    next_decision_protocol=dict(phase='afterA35',status='SOURCE_DRAFT_NO_MODEL_ALLOCATION',file=str(D/'next_readout_source_question_DRAFT.json')))
write(B/CAN[0],board)
search=read(B/CAN[1]);search.update(supplement_utc=NOW,A35_primary_and_native_query_audit=dict(
    phase=phase,search_log=str(D/'search_record.json'),sources=str(D/'new_core_records.json'),source_audit=str(D/'pre_model_source_audit.json'),
    partial_reading_warning='AQS-IDETRfullmethodunavailable;PiDiViTdiscoveryonly.'))
write(B/CAN[1],search)
matrix=read(B/CAN[2]);assert len(matrix['rows'])==106
assert not ({x['id'] for x in new_rows}&{x['id'] for x in matrix['rows']})
for row in new_rows:assert row['title'].casefold() not in {x['title'].casefold() for x in matrix['rows']}
matrix['rows']+=new_rows
for row in updated_rows:
    index=next(i for i,x in enumerate(matrix['rows']) if x['id']==row['id']);matrix['rows'][index]=row
assert len(matrix['rows'])==112
matrix.update(updated_utc=NOW,latest_report=str(REPORT),
    latest_empirical_boundary='A35:nativeunchanged-outputsupportscreen0/382smallattrition;noactualpruningAPeffectornewalgorithmestablished.')
write(B/CAN[2],matrix)
state.update(recorded_utc=NOW,goal_status='active',mainline_validated=False,previous_goal_turn=state['current_goal_turn'],current_goal_turn=phase,
    latest_authoritative_phase=phase,latest_report=str(REPORT),current_priority_question=next_question,A35_evidence=phase,
    complete_decoder_only_replays_A35=13,complete_decoder_only_replays_cumulative=61,partial_prefix_decoder_calls_A35=24,
    current_native_job_terminal=dict(phase='A35',pid=15120,start_ticks='102159569',observed_absent=True,status='COMPLETED',worker_exit_code=0,cap_closed=True))
write(B/CAN[3],state)
source=read(B/CAN[4]);source.update(updated_utc=NOW,A35_query_lifecycle_and_native_prefix=dict(
    phase=phase,pre_model_audit=str(D/'pre_model_source_audit.json'),author_source=str(D/'selected_butr_source_verified.json'),
    fidelity=str(D/'archive_verification.json'),novel_algorithm_approved=None))
write(B/CAN[4],source)
files=[p for p in D.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+library_files+[REPORT]+[B/n for n in CAN]
receipt=dict(phase);receipt.update(scientific_result=r,archive_verification=av,independent_scalar_verification=iv,
    exploratory_miss_taxonomy=taxonomy,independent_taxonomy_verification=taxonomy_iv,LQE_static_operator_and_parameter_metadata=lqe,
    previous_A34_evidence_remapped_sha256=remapped,files_sha256={str(p):sha(p) for p in sorted(set(files))})
write(RECEIPT,receipt)
for name,h in receipt['files_sha256'].items():assert sha(name)==h,name
for name,h in remapped.items():assert sha(name)==h,name
print(json.dumps(dict(status='A35_PUBLISHED_QUERY_RETENTION_CANDIDATE_NOT_PROMOTED',evidence_files=len(receipt['files_sha256']),
    prior_A34_files_verified=603,core_matrix_rows=112,whole_detector_forwards=3168,optimizer_updates=528,
    complete_decoder_only_calls=61,new_partial_prefix_calls=24,mainline_validated=False,report=str(REPORT)),ensure_ascii=False))
