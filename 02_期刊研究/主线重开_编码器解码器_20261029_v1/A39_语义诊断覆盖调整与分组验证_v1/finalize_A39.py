from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,sys
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent;B=D.parent
REPORT=B/'88_A39细尺度与最终查询判别对照及候选去留.txt'
RECEIPT=B/'89_A39冻结采集与分组探针独立核验回执.json'
CAN=['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json']
NOW=datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x,exclusive=False):
    with Path(p).open('x' if exclusive else 'w',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
assert not REPORT.exists() and not RECEIPT.exists()
prior=read(B/'87_A38冻结采集与独立覆盖核验回执.json')
for p,h in prior['files_sha256'].items():assert sha(p)==h,p
assert len(prior['files_sha256'])==227
for n,h in read(D/'execution_inputs_manifest.json')['files_sha256'].items():assert sha(D/n)==h,n
result=read(D/'analysis/semantic_analysis_result.json');support=read(D/'independent_support_verification.json')
metrics=read(D/'independent_metric_verification.json');archive=read(D/'archive_verification.json')
collection=read(D/'downloaded_evidence/collection_result.json');control=read(D/'analysis/control_feature_verification.json')
assert result['probe_fit_entries']==metrics['verified_fitted_models']==7
assert not result['all_gates_passed'] and support['primary_val_pairs']==199
assert abs(result['fine_minus_query_AUC']+.04563016085452387)<1e-15
assert metrics['bootstrap_references_checked']==5000 and metrics['max_reconstructed_prediction_error']==0
asset=read(B/'A40_ECDet新编码器宿主与训练入口核查_v1/checkpoint_download_receipt.json')
assert asset['status']=='OFFICIAL_COCO_CHECKPOINT_DOWNLOADED_LOCALLY_NOT_LOADED' and sha(asset['file'])==asset['sha256']
log=Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文\前期调研_码率受限航拍小目标_20261028_v1\远端验证_20261029_v1\ssh_operation_receipts.jsonl')
ssh=[]
for line in log.open(encoding='utf-8'):
    try:r=json.loads(line)
    except ValueError:continue
    if str(r.get('id','')).startswith('A39_') and 'action' in r:ssh.append(r)
assert len(ssh)==13
last=next(r for r in ssh if r['id']=='A39_final_native_terminal_observation_13');assert last['exit_code']==0
terminal=json.loads(last['stdout'])
assert not terminal['wrapper_process_present'] and not terminal['collector_process_present']
assert terminal['worker_exit']['worker_exit_code']==0 and terminal['worker_exit']['author_entries']==113
write(D/'ssh_selected_receipts.json',dict(rows=ssh),True);write(D/'remote_terminal_observation.json',terminal,True)
phase=dict(phase='A39',recorded_utc=NOW,classification='progress_prospective_coverage_and_negative_paired_semantic_screen_independently_verified',
    status='FINE_OVER_QUERY_ADVANTAGE_NOT_SUPPORTED_CANDIDATE_R_DEPRIORITIZED',goal_status='active',
    mainline_validated=False,complete_mainline=False,research_gap_confirmed=False,new_algorithm_approved=None,
    report=str(REPORT),receipt=str(RECEIPT),new_whole_detector_forwards=113,whole_detector_cap=113,whole_detector_cap_closed=True,
    old_A38_cap114_remains_closed=True,reused_A38_caches=112,new_unique_images=112,combined_unique_images=224,
    train_images=119,val_images=105,train_prefix_groups=64,val_prefix_groups=64,train_pairs=508,all_val_pairs=444,
    primary_val_pairs=199,primary_val_images=60,primary_val_prefix_groups=45,
    primary_QUERY_AUC=result['primary_AUC']['QUERY'],primary_FINE_AUC=result['primary_AUC']['FINE'],
    primary_CONTRAST_AUC=result['primary_AUC']['CONTRAST'],fine_minus_query_AUC=result['fine_minus_query_AUC'],
    fine_minus_query_CI95=result['fine_minus_query_paired_image_CI95'],gates=result['gates'],all_gates_passed=False,
    semantic_effect_estimated=True,new_data_probe_fit_entries=7,new_synthetic_ridge_fits=0,new_optimizer_updates=0,new_training_jobs=0,
    new_native_head_only_calls=0,new_complete_decoder_only_calls=0,first_control_float64_head_reconstruction=1,
    independent_all_case_float64_head_reconstructions=224,independent_bootstrap_references=5000,
    full_model_forwards=3395,actual_cumulative_detector_optimizer_updates=528,formal_detector_training_updates=526,
    historical_capacity_updates=2,cumulative_complete_decoder_only_calls=61,
    full_new_first_query_feature_repeat_verified=True,inherited_A38_query_full_repeat_unavailable=True,
    peak_cuda_allocated_bytes=collection['peak_cuda_allocated_bytes'],native_query_range_observed=[min(collection['query_counts']),max(collection['query_counts'])],
    source_host_is_compact_ViT=False,source_host='OriginaltaskadaptedDome-SHGNetv2B0CNN+hybridencoder',
    previous_A38_files_verified=227,archive_files=137,archive_manifest_files_verified=136,independent_pair_rows_verified=952,
    new_core_papers=0,core_matrix_rows=114,candidate_count=18,new_search_queries=9,
    next_host='ECDet-S compactViT;asset/entry/capacityverificationonly,notamainline',
    next_host_weight_local_verified=True,next_host_weight_bytes=asset['bytes'],next_host_weight_sha256=asset['sha256'],
    next_host_instantiated=False,next_host_training_24GB_verified=False,
    next_model_call_allocation=0,next_optimizer_update_allocation=0,next_data_probe_fit_allocation=0,
    automatic_remote_followups=False,A22_restarted=False,old_DEIM_restarted=False,compression_branch_reopened=False,
    next='停止本CNN宿主下R候选的阈值/lambda/尺度/分组补救；先闭合ECDet-S官方权重严格装载、完整训练入口、VisDrone10类映射及24GB容量门槛，再从真实紧凑ViT基线查具体编码失效。尚未找到可立项主线。')
text=f'''A39：细尺度memory与最终query的判别对照及候选去留

结论
预设的“细尺度特征比最终query更容易区分弱分数小目标/难背景”优势没有得到支持。当前候选R停止作为优先方向，不据此添加细节读取、差分补偿或分数融合模块。本轮仍未找到足以立项的具体问题和算法增量；无人机航拍小目标检测是方向，新主线尚未确认。

一、前瞻性覆盖调整与固定协议
A38在未拟合数据探针时只有78对主验证样本，低于原100对门槛。本轮沿已记录的、只依据数量的草案固定取样：保留64个训练前缀，纳入全部64个可用验证前缀，各组按原SHA256规则取最多2帧。单帧组保留1帧，不按标签或表征表现选图。
实际119张训练图、105张验证图；复用112份旧缓存，新增55张训练图与57张验证图。新阶段113次冻结前向=112张新图+1次首张新图重复，旧A38的114预算保持关闭。没有检测器优化更新或新检测器训练。新增首图的18项数组，包括query、细尺度、差分特征和原生预测，全部逐位重复一致，模型state摘要未变。
保留原特征、匹配、lambda1、RFF种子、原生0.5支持阈值、100对主验证门槛及推进要求；有相邻帧后按64个验证文件名前缀整组重采样。前缀只是分组代理，不是已核验的真实航次；数据未用于拟合探针的条件，不等于作者权重/整个旧项目从未见过它们。

二、研究对象与样本
宿主为Dome-S的HGNetv2-B0CNN加混合编码器，未测新的紧凑ViT。原生250–500查询机制保留。特征比较为：最终分类Linear之前的256维query、同一最终预测框内的细尺度memory7x7均值、内外差分对照，以及几何/原生分数和固定RFF512控制。
GT只用于推理后的诊断匹配，不作为一方取样框。主分析是原图面积<=1024、无遮挡/未截断、原始框一对一IoU>=0.5、原生TopQ在分数0.5下无正确类别IoU0.5支持且可匹配背景的目标。背景排除所有真实类别1–11和忽略区域。它不是全部漏检或官方检测AP。
训练508对来自100张图、60个前缀；主验证199对来自60张图、45个前缀。其余没有主验证例对的前缀仍参与64组重采样。全部验证匹配例对444对；主分析199对不能与它们混用。
7次数据探针拟合均只使用训练组。数据探针拟合属于学习，区别于检测器优化更新0；bootstrap没有重复拟合。

三、结果与预设去留
表征/探针                 训练AUC       主验证AUC
最终QUERY                 0.943270      0.741749
细尺度均值FINE            0.930486      0.696119
内外差分CONTRAST          0.918807      0.670614
几何/原生分数GEOMETRY     0.519712      0.531729
QUERY固定RFF512           0.983841      0.651903

主效应FINE-QUERY=-0.04563016，即-4.563个AUC百分点；1000次前缀整组bootstrap的95%区间为[-0.10145313,+0.00647487]，即[-10.145,+0.647]个百分点。
区间跨0，不能说QUERY在总体上显著优于FINE；本次区间也没有支持事先要求的FINE至少+5个百分点优势。细尺度AUC>=0.75、FINE-QUERY>=0.05、区间下限>0这三个关键要求均未通过。未按表现更换lambda、阈值、图像/类别子组或选择次要差分结果挽救假设。
差分对照主验证AUC0.6706也未提供推进依据。RFF训练AUC较高、验证较低，说明这一固定非线性控制泛化有限，不能代表所有可能的query读出器。
正确类别检查为次要描述：QUERY准确率0.5678、FINE0.5126；预定支持类别上的宏平均分别0.5031、0.3314。未为这些次要差值预注册显著性结论，不能据此把反向“query监督细尺度”直接提升为新主线。

四、独立核验与边界
新增证据包167398216字节，SHA256={archive['archive_sha256']}；137个归档成员、136项清单hash核验通过；112份旧缓存hash也核验一致。原生日志记录1–113完成，worker退出码0，wrapper16133与collector16134均已观察不存在。
952行匹配记录逐项核验，包括二维忽略区域并集、所有query背景排除、类别11处理、Hopcroft–Karp匹配基数、原生TopQ来源和每对的支持状态。全部224图的原始分类头float64重构最大误差1.8327e-6。
7个已拟合探针均核验训练统计量、正常方程和预测值；预测重构最大误差0。独立的二分计数AUC复算5000个bootstrap参考值、全部主结果及区间，完全一致。验证器未新增拟合或原始网络调用。
新首图426个query的细尺度/差分NumPy读取误差分别至多1.4035e-5/1.3574e-5。旧A38首query完整重复特征未保存的限制保留；A39新首图的完整特征重复已验证，不能倒写为旧A38也完成该项。
上述为固定CNN宿主、固定样本和线性/RFF探针的判别可访问性结果，不是信息论擦除、注意力因果损伤、所有读出方式、紧凑ViT缺陷或检测AP收益证明。区间只反映给定训练探针后的评估组重采样，不包括训练随机性或模型选择偏差。

五、研究决策
【观察】指定细尺度均值和差分表征未优于最终query；已有query仍有一定可读判别信号。
【推断】当前证据不足以支撑在该宿主上增加细节补偿或局部差分读取模块，候选R不再优先。
【没有证明】并未证明语义完整保留、所有编码器都无需改进、所有非线性局部读取无效，也未找到新的算法增量。
【后续限制】停止这条候选的同数据lambda/阈值/分组补救。新工作应从真正紧凑ViT的任务适配基线和明确编码失效出发。

六、新编码器宿主的准备状态
下一技术宿主选为官方ECDet-S/EdgeCrafter，源代码固定提交d8752b75c9beaecfcdd0e5cd207bae3b6946ae23，重核5个所选文件。原README链接的COCO检测权重已核对发布资产元数据并先下载到本地：39938420字节，SHA256={asset['sha256']}。
这只是模型资产准备，未装载为模型、未执行前向/反向、未开始训练，也不构成论文创新。还需检查完整作者训练入口、原权重严格装载、VisDrone10类与忽略0/其他11的映射、类别头初始化，以及单卡24GB真实容量。不能以COCO类别输出直接替代任务适配基线。
既有DeepSeek机制与共享状态源码对照仅作为后续设计参照；必须先有具体失败证据，不能以改名、共享状态或替换骨干当作增量。
本轮另保存9条定向查询与2次网页打开记录。Le-DETR已在矩阵中，不重复计数；TSC-DETR仅为语义引导方向的检索线索，其完整原文和同名仓库关联未闭合，不据此断言完整方法或官方开源。核心矩阵仍114条。
官方ECDet模型入口参考：https://github.com/Intellindust-AI-Lab/EdgeCrafter/tree/d8752b75c9beaecfcdd0e5cd207bae3b6946ae23
原模型链接：https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_s.pth
最近方法与阅读范围沿用84_A37局部差分与前景语义最近工作核查.txt及核心矩阵；原文方法、我们的观察、推断和待验证假设分列。

当前状态：goal active；候选R不推进；新主线未确认。A39113次采集和7次数据拟合预算关闭，下一模型/训练分配0。无旧DEIM/A22重启或压缩分支实验。
'''
with REPORT.open('x',encoding='utf-8') as f:f.write(text)
backup=D/'canonical_before_A39';backup.mkdir(exist_ok=False)
for name in CAN:shutil.copy2(B/name,backup/name)
remapped={}
for p,h in prior['files_sha256'].items():
    q=Path(p);q=backup/q.name if q.parent==B and q.name in CAN else q
    assert sha(q)==h;remapped[str(q)]=h
board=read(B/CAN[0]);card=next(r for r in board['candidates'] if r['id']=='R_FINE_MEMORY_QUERY_SEMANTIC_GAP')
card.update(status='NOT_PROMOTED_AFTER_NEGATIVE_PAIRED_ACCESSIBILITY_SCREEN_A39',evidence=str(D/'analysis/semantic_analysis_result.json'),
    causal_evidence=False,increment_approved=False,next='No currenthostlambda/threshold/view/size/classrescuegrid;move to task-adaptedcompactViThost readiness.')
question='新主线尚未确认；先核验ECDet-S真正紧凑ViT宿主的权重、完整训练入口、VisDrone类别适配与24GB容量，再定位具体编码失效。'
board.update(status='SEARCHING_NO_VALIDATED_MAINLINE_A39',latest_update_utc=NOW,mainline_validated=False,
    latest_report=str(REPORT),A39_evidence=phase,current_priority_candidate=None,current_priority_question=question,
    next_exploratory_question=question,current_host_candidate='ECDet-Sasset/entry/capacitygate,notanewmainline',
    bounded_diagnostic_protocol=str(D/'semantic_protocol_FROZEN.json'),
    next_decision_protocol=dict(status='COMPACT_VIT_HOST_ASSET_PREPARED_NO_MODEL_ALLOCATION',file=str(D/'next_compact_ViT_host_DRAFT.json'),
        actual_asset_receipt=str(B/'A40_ECDet新编码器宿主与训练入口核查_v1/checkpoint_download_receipt.json')))
write(B/CAN[0],board)
search=read(B/CAN[1]);search.update(supplement_utc=NOW,A39_paired_semantic_negative_and_next_encoder=dict(phase=phase,
    search_log=str(D/'post_screen_direction_search_raw.json'),next_host=str(D/'next_compact_ViT_host_DRAFT.json'),new_core_papers=0))
write(B/CAN[1],search)
matrix=read(B/CAN[2]);assert len(matrix['rows'])==114
for row in matrix['rows']:
    if row['id']=='Dome_DETR':row['A39_scope_update']=dict(combined_caches=224,primary_pairs=199,probe_fits=7,
        fine_minus_query_AUC=result['fine_minus_query_AUC'],CI=result['fine_minus_query_paired_image_CI95'],candidate_not_promoted=True,not_compact_ViT=True)
    if row['id']=='EdgeCrafter_A32':row['A39_asset_scope_update']=dict(selected_source_files_rehashed=5,
        checkpoint_downloaded_locally=True,checkpoint_sha256=asset['sha256'],model_instantiated=False,training24GB_verified=False)
matrix.update(updated_utc=NOW,latest_report=str(REPORT),latest_empirical_boundary='A39FINE-QUERYAUC-4.563pp,64group95CI[-10.145,+0.647]pp;keyadvantagegatesfailed;noViTcausalornewalgorithmproof.')
write(B/CAN[2],matrix)
state=read(B/CAN[3]);state.update(recorded_utc=NOW,goal_status='active',mainline_validated=False,
    previous_goal_turn=state['current_goal_turn'],current_goal_turn=phase,latest_authoritative_phase=phase,
    latest_report=str(REPORT),current_priority_question=question,A39_evidence=phase,model_forwards=3395,training_updates=528,
    current_native_job_terminal=dict(phase='A39',wrapper_pid=16133,collector_pid=16134,both_observed_absent=True,
        worker_exit_code=0,whole_detector_entries=113,cap_closed=True,data_probe_fits=7,probe_fit_cap_closed=True))
write(B/CAN[3],state)
source=read(B/CAN[4]);source.update(updated_utc=NOW,A39_paired_semantic_accessibility=dict(phase=phase,
    source_preservation=str(D/'native_operator_preservation.json'),independent_support=str(D/'independent_support_verification.json'),
    independent_metrics=str(D/'independent_metric_verification.json'),new_algorithm_approved=None,next_host=str(D/'next_compact_ViT_host_DRAFT.json')))
write(B/CAN[4],source)
asset_files=[B/'A40_ECDet新编码器宿主与训练入口核查_v1/checkpoint_download_receipt.json',Path(asset['file'])]
files=[p for p in D.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[REPORT]+[B/n for n in CAN]+asset_files
receipt=dict(phase);receipt.update(scientific_result=result,archive_verification=archive,independent_support=support,
    independent_metrics=metrics,control_verification=control,terminal_observation=terminal,
    previous_A38_evidence_remapped_sha256=remapped,files_sha256={str(p):sha(p) for p in sorted(set(files))})
write(RECEIPT,receipt,True)
for p,h in receipt['files_sha256'].items():assert sha(p)==h,p
for p,h in remapped.items():assert sha(p)==h,p
print(json.dumps(dict(status='A39_PUBLISHED_CANDIDATE_R_NOT_PROMOTED',evidence_files=len(receipt['files_sha256']),
    old_A38_files_verified=227,new_native_entries=113,data_probe_fits=7,whole_model_forwards=3395,
    fine_minus_query_AUC=result['fine_minus_query_AUC'],mainline_validated=False,next_model_allocation=0),ensure_ascii=False))
