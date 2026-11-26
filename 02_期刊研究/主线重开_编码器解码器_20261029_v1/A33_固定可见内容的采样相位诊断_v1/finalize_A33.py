from pathlib import Path
import datetime,hashlib,json,shutil,sys
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent;B=D.parent;LIB=B.parent/'文献_相关原始论文_20261004'
NOW=datetime.datetime.now(datetime.timezone.utc).isoformat()
REPORT=B/'76_A33固定像素相位诊断与方向性证据.txt';RECEIPT=B/'77_A33原生相位实验与独立核验回执.json'
CAN=['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not REPORT.exists() and not RECEIPT.exists()
previous=read(B/'75_A32架构原文与共格关联核验回执.json')
for p,h in previous['files_sha256'].items():assert sha(p)==h,p
cfg=read(D/'phase_protocol_FROZEN.json');complete=read(D/'remote_complete.json');r=read(D/'native_phase_analysis_result.json');av=read(D/'archive_verification.json');iv=read(D/'independent_primary_verification.json');terminal=read(D/'remote_terminal_observation.json')
assert complete['counts']['whole_detector_forwards']==169 and not terminal['same_process_live'] and terminal['exception'] is None
assert r['mechanism_question_promotion']==False and iv['max_contrast_error']==0
for row in cfg['local_inputs'].values():assert sha(row['path'])==row['sha256']
backup=D/'canonical_before_A33';backup.mkdir(exist_ok=False)
for name in CAN:shutil.copy2(B/name,backup/name)
remapped={str(backup/Path(p).name) if Path(p).parent==B and Path(p).name in CAN else p:h for p,h in previous['files_sha256'].items()}
for p,h in remapped.items():assert sha(p)==h
meta_path=LIB/'A33_AeroPinWorld_Electronics2026.source.json';meta=read(meta_path);assert sha(meta['file'])==meta['sha256']
scope='CompleteSec3.1-3.5;selected3.6.1-3.6.2,Table6and4.10.2-4.10.4/Conclusion5 frompublisherPDFtext. Full17pagepaper/alllosses/results/authorcode notread or reproduced. No trustedPDFpage-anchorclaim.'
meta.update(read_scope=scope,read_scope_updated_utc=NOW,read_by='assistant',human_read_attested=False);write(meta_path,meta)
primary=read(D/'Aero_Crossref_primary_metadata.json')['message']
source_audit=dict(recorded_utc=NOW,paper='AeroPinWorld',DOI='10.3390/electronics15071364',authors=[a['given']+' '+a['family'] for a in primary['author']],read_scope=scope,local_pdf=meta['file'],pdf_sha256=meta['sha256'],source='https://mdpi-res.com/d_attachment/electronics/electronics-15-01364/article_deploy/electronics-15-01364.pdf',operator='Adopts pinwheel-shaped convolution:four asymmetric-padding horizontal/vertical stride2probes,channelconcatenationandconvolutionalfusion;replacesfirsttwo backboneandtwo bottom-upheadreductions;head unchanged.',paper_setting='COCOsource24epoch adaptation ofYOLOWorldv2;zero-shotVisDroneval1280andUAVDT640;not D0closed-setYOLO26.',novelty_consequence='Genericphase-awarestride2UAVdownsampling is directprior. LPSalreadycarriesphaseindices todecoder,shift-equivariantViTalreadyredesignstokenization/merging/positions. Currentdiagnosticdoesnotreproducethosemodels.',implementation_scope='No Aeroauthorcodeidentifiedorimported;baselineYOLOWorldrepoandPConvreferencealone do not prove exactimplementationclosure.')
write(D/'closest_phase_operator_followup.json',source_audit)
question='24图保持可见像素不变的初筛已测到一般相位不稳定；邻近实例分粗格/共粗格的方向性对比为正但区间跨零。若继续，仅在此前未做相位干预的其余76图固定复核同一对比，不换指标或挑位移；算法与新编码器缺陷仍未成立。'
phase=dict(phase='A33',classification='progress_controlled_native_pixel_preserving_phase_screen',recorded_utc=NOW,status='NATIVE_PHASE_SCREEN_COMPLETE_PRIMARY_DIRECTIONAL_CI_UNCERTAIN',goal_status='active',mainline_validated=False,complete_mainline=False,report=str(REPORT),receipt=str(RECEIPT),new_whole_detector_forwards=169,new_GPU_model_calls=169,new_extra_native_O_decodes=168,new_all_native_head_decodes=337,new_optimizer_updates=0,new_standalone_TAL_calls=0,new_complete_decoder_calls=0,new_training_jobs=0,images=24,input_small_GT=725,arms=14,matched_FP_budget=222,bootstrap_draws=1000,mechanism_question_promotion=False,research_gap_confirmed=False,new_algorithm_approved=None,full_model_forwards=2635,actual_cumulative_detector_optimizer_updates=528,formal_detector_training_updates=526,historical_capacity_updates=2,cumulative_complete_decoder_only_calls=48,historical_last_layer_suffix_calls=72,core_matrix_rows=106,new_core_papers=0,existing_paper_read_scope_updated='AeroPinWorld_partial',new_full_PDFs=1,candidate_count=16,A33_cap=169,A33_cap_closed=True,A30_101_cap_remains_closed=True,historical_39092_cap_remains_closed=True,worker_pid=14687,worker_start_ticks='101778872',worker_process_observed_absent=True,worker_exit_code_observed=None,launcher_exit_code=0,automatic_followups=False,next_model_call_allocation=0,A22_restarted=False,next=question)
lines=['A33：固定像素相位诊断与方向性证据','记录时间（UTC）：'+NOW,'',
 '判断：一般相位不稳定已在本冻结模型中观察到；“粗格中的邻近实例额外干扰”这一具体解释仍不确定。'
 '预设5项推进条件中，主要方向性对比的95%区间包含零，故机制候选不推进为已确认主线，更没有新算法。','',
 '1. 检索策略与记录',
 '复核LPS与等变ViT的相位/位置恢复先例，并取得AeroPinWorld完整公开PDF。网页429的失败与公开出版者CDN成功均保留；'
 'Crossref元数据和PDF作者、题名一致。该文由原先摘要/片段范围更新为方法3.1–3.5等已读范围，不增加为重复核心条目。','',
 '2. 核心文献与阅读范围',
 'AeroPinWorld：Jie Li、Mingze Guan、Jincheng Xu、Xun Du、Haonan Chen、Yang Liu；Electronics2026,15(7),1364；DOI10.3390/electronics15071364。'
 '已读3.1–3.5、部分3.6、Table6、4.10.2–4.10.4及结论；不是整篇或作者代码复现。'
 'LPS/等变ViT沿用现有已读方法范围，当前网页再次核实原始题名与任务边界。','',
 '3. 与最近工作的逐项对比',
 'AeroPinWorld已将风车形卷积的四路偏移探测与融合用于四处下采样；其研究是COCO训练到航拍数据的零样本开放词汇迁移，'
 '不能与本次闭集YOLO26数值直接比较。LPS和等变ViT已有相位选择、索引传递和解码恢复；通用相位模块或改名新编码器不构成创新。'
 '本轮只检验同一已训练检测器的受控失效，不复现上述模型，也不验证DeepSeek机制已成功迁移。','',
 '4. 本轮观测与研究空白边界',
 '从原100张开发图按预先固定文件名SHA顺序选24张；所有图上下原有填充≥32。原RGB图只按历史代码缩放一次，'
 '随后在已有均匀填充中做网络输入y方向0、±8、±16、±32的整数移动。真实像素、目标相对布局、面积及遮挡不变。'
 '物理画面的绝对位置及到输入边界距离仍改变，所以不能把任何变化全部归因于单一混叠算子。',
 '725个输入面积<1024的小目标中，78个在±8移动时从“共16粗格、分8细格”转为分粗格，76个反向变化，分别覆盖17/15张图。'
 '粗格为名义几何标签，不是实际感受野或单token容量限制。',
 '所有分支/条件都在总FP预算222（floor(9.26×24)）下作全局分数前缀匹配；同分整块处理。1000次配对图像重抽样中逐次重估各条件阈值。'
 '指标是官方源码对齐的忽略区域预处理＋COCO101@500，不称完整官方VisDrone复现。','',
 '主要方向性对比＝拆开组的检出变化－合并组的检出变化；变化均以±8相对±32的对称均值定义：']
for k,label in [('M','常规分支＋NMS'),('O','单次预测分支')]:
 z=r['contrasts'][k];ci=z['paired_reestimated_FP_bootstrap95'];fi=z['flip8_minus_flip32_bootstrap95']
 lines.append(f"{label}：{100*z['separation_minus_merge_contrast']:.4f}个百分点，95%区间[{100*ci[0]:.4f},{100*ci[1]:.4f}]；±8比±32额外检出翻转{100*z['flip8_minus_flip32']:.4f}个百分点，区间[{100*fi[0]:.4f},{100*fi[1]:.4f}]。")
lines.extend(['方向性对比的两分支区间均跨零；一般相位翻转增加较明确，不能据此宣称邻近实例混合机制已确认。'
 '725个目标和24图只够初筛，缺乏方向性证据不等于已证明效应为零。','',
 '14项描述性结果（同一24图；不得与此前100图AP作差）：','分支_位移 | 输入小目标AP(%) | 小目标TP@FP222 | 实际FP'])
for name,m in r['metrics'].items():lines.append(f"{name} | {100*m['AP_input_small']:.4f} | {m['small_TP']} | {m['actual_FP']}")
lines.extend(['GT逐目标并集与逐图最优位移均使用真值选择，且条件阈值不同；不得报告为可部署召回增益或同FP算法上限。'
 '单次预测分支对应GT并集475、逐图最优464，常规分支484/468；这些数仅描述现象。','',
 '5. 候选增量、可行性与风险',
 '仍不批准算法。只有在特定实例方向性效应得到复核后，才讨论固定预算的实例条件细节读取；须先排除已有采样、实例注意力、查询竞争方法。'
 '当前数据不能证明ViT token混合，不能证明LPS/PConv无效，也不能证明新编码器有益。重复开发集、有限样本、边界效应和全局评分变化均限制解释。',
 '本轮24GB3090运行169次前向，无训练；PyTorch峰值分配0.14356GiB、保留0.1875GiB，仅是此推理程序的分配统计，不代表新架构训练容量。','',
 '6. 最小方案、执行与核验',
 '24×7＋1重复，共169次完整前向；168次额外原生O解码，累计原生head解码337；0优化器更新、0单独TAL调用、0新DETR解码重放。'
 '24个零移动的双头原始值、解码框及最终原图预测与A30旧缓存严格相同；末次原生端到端重复逐位相同；参数及源码/权重哈希保持。',
 '183122348字节归档SHA与远端一致，191个文件、190项内部SHA核验通过；本地再比144组原始/解码数组、48组逐图分支记录及168组像素保真。'
 '独立标量实现复核14条件的固定FP事件及原样本＋3个重抽样的方向性统计，共40600个GT判别，最大差0；抽样种子1000次一致。',
 '原生采集55.36秒，含归档63.03秒。PID14687/start_ticks101778872已被观察到退出，无exception，完成文件已核验。'
 '启动器退出码为0；脱离启动器运行的worker未单独持久化退出码，故不把启动器的0当worker退出码。169上限已封闭，无自动重试/后续队列。',
 '下一可用安全动作是保留当前结果，在原100图中尚未做相位干预的其余76图复核同一预设方向，不换指标、不挑位移。'
 '这需要独立新协议和新调用上限，不能动用本次或历史剩余预算；目前下一调用配额仍为0，未启动扩展。','',
 '7. 可核验文献和文件',
 'AeroPinWorld：https://doi.org/10.3390/electronics15071364',
 'LPS：https://arxiv.org/abs/2210.08001v1',
 '等变ViT：https://arxiv.org/abs/2305.16316v2',
 '冻结协议：'+str(D/'phase_protocol_FROZEN.json'),
 '完整结果：'+str(D/'native_phase_analysis_result.json'),
 '独立核验：'+str(D/'independent_primary_verification.json'),
 '当前累计架构分支完整前向2635、优化器更新528、完整DETR解码器重放48。目标继续active，未找到经验证主线。'])
REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')
board=read(B/CAN[0]);board.update(status='SEARCHING_NO_VALIDATED_MAINLINE_A33',mainline_validated=False,latest_update_utc=NOW,latest_report=str(REPORT),current_priority_candidate=None,current_priority_question=question,next_exploratory_question=question,A33_evidence=phase,A33_selection_note='一般相位不稳定有数据；邻近实例方向性对比CI跨零，未通过事前推进门槛。允许考虑剩余76样本复核，须新协议；新算法仍无。');write(B/CAN[0],board)
search=read(B/CAN[1]);search.update(supplement_utc=NOW,A33_phase_primary_and_native_diagnostic=dict(phase=phase,Aero_access=str(D/'Aero_access_receipt.json'),source_audit=str(D/'closest_phase_operator_followup.json'),primary_verification=str(D/'independent_primary_verification.json')));write(B/CAN[1],search)
matrix=read(B/CAN[2]);assert len(matrix['rows'])==106
found=[row for row in matrix['rows'] if row.get('id')=='AeroPinWorld_partial'];assert len(found)==1;row=found[0]
row.update(previous_read_scope=row['read_scope'],read_scope=scope,authors=source_audit['authors'],doi='10.3390/electronics15071364',local_pdf=meta['file'],pdf_sha256=meta['sha256'],access_record=str(meta_path),author_method=source_audit['operator'],our_consequence='Directphase-awareUAVdownsampling prior;the settingiszero-shotYOLOWorldtransfer,notourclosed-setphase diagnosis.',unresolved='Exactauthorcode/weightsnotclosed;publishedmethodnotreproduced;noournewalgorithm validated.',read_by='assistant',human_read_attested=False)
matrix.update(updated_utc=NOW,latest_report=str(REPORT),latest_empirical_boundary='A33:pixelpreservingnativephasescreen found generalinstability;primary neighborseparationcontrast CIcrosseszero;no mechanismornewalgorithmvalidated.');write(B/CAN[2],matrix)
state=read(B/CAN[3]);assert state['model_forwards']==2466 and state['training_updates']==528
state.update(recorded_utc=NOW,goal_status='active',mainline_validated=False,model_forwards=2635,previous_goal_turn=state['current_goal_turn'],current_goal_turn=phase,latest_authoritative_phase=phase,latest_report=str(REPORT),current_priority_question=question,A33_evidence=phase,current_native_job_terminal=dict(phase='A33',pid=14687,start_ticks='101778872',observed_absent=True,status='COMPLETED',worker_exit_code=None,cap_closed=True));write(B/CAN[3],state)
source=read(B/CAN[4]);source.update(updated_utc=NOW,A33_phase_operator_and_native_control=dict(phase=phase,closest_work=str(D/'closest_phase_operator_followup.json'),frozen_protocol=str(D/'phase_protocol_FROZEN.json'),archive_verification=str(D/'archive_verification.json'),novel_algorithm_approved=None));write(B/CAN[4],source)
files=[p for p in D.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[REPORT]+[B/name for name in CAN]+[Path(meta['file']),Path(meta['text_file']),meta_path]
receipt=dict(phase);receipt.update(archive_verification=av,independent_primary_verification=iv,scientific_result=r,previous_A32_evidence_remapped_sha256=remapped,files_sha256={str(p):sha(p) for p in sorted(set(files))});write(RECEIPT,receipt)
for p,h in receipt['files_sha256'].items():assert sha(p)==h
print(json.dumps(dict(status='A33_PUBLISHED_PRIMARY_GATE_NOT_PASSED',evidence_files=len(receipt['files_sha256']),model_forwards=2635,training_updates=528,mainline_validated=False,report=str(REPORT)),ensure_ascii=False))
