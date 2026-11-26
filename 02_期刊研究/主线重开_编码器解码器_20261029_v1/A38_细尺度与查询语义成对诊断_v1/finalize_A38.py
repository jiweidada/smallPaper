from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,sys
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent;B=D.parent
REPORT=B/'86_A38细尺度与查询语义诊断的样本覆盖核查.txt'
RECEIPT=B/'87_A38冻结采集与独立覆盖核验回执.json'
CAN=['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json']
NOW=datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x,exclusive=False):
    with Path(p).open('x' if exclusive else 'w',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
assert not REPORT.exists() and not RECEIPT.exists()
prior=read(B/'85_A37原文源码及诊断边界核验回执.json')
for p,h in prior['files_sha256'].items():assert sha(p)==h,p
assert len(prior['files_sha256'])==75
for m in ['execution_inputs_manifest.json','export_recovery_execution_manifest.json']:
    for name,h in read(D/m)['files_sha256'].items():assert sha(D/name)==h,name
av=read(D/'archive_verification.json');iv=read(D/'independent_support_verification.json')
result=read(D/'analysis/semantic_analysis_result.json');control=read(D/'analysis/control_feature_verification.json')
plan=read(D/'coverage_followup_sample_DRAFT.json')
assert iv['primary_val_pairs']==78 and result['probe_fit_entries']==0
assert not (D/'analysis/probe_fit_entries.jsonl').exists()
assert av['whole_detector_entries']==114 and not av['full_query_feature_repeat_available']
log=Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文\前期调研_码率受限航拍小目标_20261028_v1\远端验证_20261029_v1\ssh_operation_receipts.jsonl')
ssh=[]
for line in log.open(encoding='utf-8'):
    try:r=json.loads(line)
    except ValueError:continue
    if str(r.get('id','')).startswith('A38_') and 'action' in r:ssh.append(r)
assert len(ssh)==27
last=next(r for r in ssh if r['id']=='A38_final_native_terminal_observation_27')
assert last['exit_code']==0
terminal=json.loads(last['stdout'])
assert not terminal['wrapper_process_present'] and not terminal['collector_process_present']
assert terminal['exit_record']['worker_exit_code']==0 and terminal['exit_record']['combined_author_entries']==114
write(D/'ssh_selected_receipts.json',dict(rows=ssh),True)
write(D/'remote_terminal_observation.json',terminal,True)
phase=dict(phase='A38',recorded_utc=NOW,classification='progress_native_feature_collection_and_independently_verified_sample_support_failure',
    status='COLLECTION_COMPLETE_WITH_EXPORT_DEVIATION_SUPPORT_GATE_FAILED_NO_SEMANTIC_EFFECT_ESTIMATE',
    goal_status='active',mainline_validated=False,complete_mainline=False,research_gap_confirmed=False,
    report=str(REPORT),receipt=str(RECEIPT),
    new_whole_detector_forwards=114,new_GPU_whole_detector_entries=114,whole_detector_cap=114,whole_detector_cap_closed=True,
    original_failed_export_process_entries=2,recovery_process_entries=112,original_worker_exit_code=1,recovery_worker_exit_code=0,
    full_query_feature_repeat_available=False,export_deviation_disclosed=True,
    new_optimizer_updates=0,new_training_jobs=0,new_native_head_only_calls=0,new_complete_decoder_only_calls=0,
    new_diagnostic_data_probe_fits=0,synthetic_analytic_ridge_fits=2,
    first_control_float64_head_reconstruction=1,all_case_float64_head_reconstructions=112,
    full_model_forwards=3282,actual_cumulative_detector_optimizer_updates=528,formal_detector_training_updates=526,
    historical_capacity_updates=2,cumulative_complete_decoder_only_calls=61,
    unique_images=112,train_images=64,val_images=48,train_pairs=290,all_val_pairs=198,primary_val_pairs=78,
    minimum_primary_val_pairs=100,train_images_with_pairs=54,val_images_with_primary_pairs=28,
    semantic_AUC_estimated=False,semantic_effect_estimate=None,semantic_CI=None,
    native_query_range_observed=[315,424],native_query_range_source=[250,500],
    recovery_peak_cuda_allocated_bytes=611366912,recovery_peak_scope='Only112-callrecoveryprocess;failedprocessdidnotexportpeak.',
    source_host='FrozenoriginalDome-SHGNetv2B0CNN+hybridencoder,notcompactViT',
    previous_A37_files_verified=75,archive_files=154,archive_manifest_files_verified=153,independent_pair_rows_verified=488,
    new_core_papers=0,core_matrix_rows=114,candidate_count=18,
    next_design_status='COUNT_ONLY_PROSPECTIVE_COVERAGE_DRAFT_NO_ALLOCATION',
    prospective_train_images=119,prospective_val_images=105,prospective_train_prefixes=64,prospective_val_prefixes=64,
    prospective_cached_images_reused=112,prospective_new_unique_images=112,possible_later_whole_model_cap=113,
    next_model_call_allocation=0,next_data_probe_fit_allocation=0,next_training_allocation=0,
    new_algorithm_approved=None,automatic_remote_followups=False,A22_restarted=False,old_DEIM_restarted=False,
    next='先固定只依据覆盖数量的前瞻性样本调整协议和64前缀整组bootstrap，再决定新阶段采集。旧A38114cap保持关闭；保持100对门槛及原判别指标。未估计语义差距，不能宣称候选成立或失败。')
text=f'''A38：细尺度memory与最终query语义诊断的样本覆盖核查

当前结论
本轮完成112张图像的冻结特征采集，但预设样本支持门槛没有通过：验证组只有78对符合主分析条件的样本，低于预定100对。因此没有拟合数据探针，没有得到AUC、效应差或置信区间，不能判断细尺度memory是否优于最终query。
这不是“主线已找到”，也不是该机制被否定。当前候选仍是同一预测区域下的细尺度判别证据与查询读取之间是否存在可重复差距；新算法与论文增量尚未确认。

一、冻结设计与实际执行
沿用A37固定的64张训练图与48张验证图，来源均为VisDrone，文件名前缀不跨拟合/评估组；验证排除了此前反复使用的12个前缀。没有用标签或探针表现挑选图片。全项目从未见过这些图像、作者权重从未用验证集选模型、以及真实航次完全独立，均未得到证明。
宿主是原始Dome-S公开VisDrone权重，HGNetv2-B0CNN加混合编码器，800x800作者变换，FP32/eval/冻结。保留原生250–500查询机制，本批实际315–424；本轮不能证明新紧凑ViT存在同样问题。
细尺度读取来自投影后的融合memory首层256x200x200；query来自最后分类Linear之前。两种表征共享同一个预测框，GT只在推理后用于诊断匹配。真值框不作为一方的特权裁剪输入。
主分析针对原图面积<=1024、无遮挡且未截断、可与原始query框一对一IoU>=0.5匹配、并在原生TopQ输出于分数0.5下无正确类别IoU0.5支持的小目标。它不是全部漏检目标，也不是官方VisDrone AP。
背景必须避开全部真实类别1–11及忽略区域，并与正例的预测框大小、长宽比和原生分数匹配。原生12类平铺TopQ预算先截断，再识别有效类别1–10，不补满无效类别占用的名额。

二、采集、故障与方案偏离
累计实际114次完整检测器前向，原预算未扩大，检测器优化更新0。首个进程完成2次前向后，导出字典中的file元数据字段与NumPy函数的目标文件参数重名，保存第一张完整特征时失败。失败进程及退出码1被保留。
在尚无验证图推理、数据探针拟合或语义效应估计时，记录了一次导出恢复方案：保留原脚本、冻结协议、阈值、指标、分组、seed和失败材料，使用标准NPZ写入器保持数组字段/类型/数值；通过原先预留的首图重复调用补存首图，再处理剩余111张，共112次。恢复进程退出码0，和前2次合计114次。
这包含一次重新实例化相同冻结权重的过程，偏离了原先不重启的程序约束；重复首图也被移到了前面。原第一次query特征未保存，无法声称完成原定的完整query逐位重复检查。报告不把这一缺失写成通过。
已实际验证：补存的原生预测、完整细尺度memory图和分类头参数，与此前保存的原件逐位一致；恢复进程的模型state摘要前后一致。两端NPZ导出回读样例通过。分析前还加入了仅将NumPy标量转成JSON内置标量的导出保护；原冻结分析、匹配、取样和指标文件hash均保持不变。

三、实际样本覆盖
                                    训练64图     验证48图
可用的可见小目标GT                    1291         898
一对一几何匹配覆盖                    1188         830
找到匹配难背景的正/负例对               290         198
其中无原生正确类别支持的例对            137          78
训练匹配对分布在54张图；主验证78对分布在28张图。只有主验证例对数未达到预设100，其余样本支持条件通过。程序在该门槛处结束，数据探针拟合次数为0。
几何覆盖只说明存在可匹配的原始候选框，不能解释为语义已保留、分类可恢复或检测召回。主验证78对是经过可见性、几何与背景匹配条件筛选后的诊断集合。

四、独立核验范围
下载证据包166088002字节，SHA256={av['archive_sha256']}；154个归档成员的清单及153项文件hash核验通过。两份前向日志共同记录1–114，完成计数1–114；112张样本均有成功导出事件。
独立计算复核488行匹配记录，使用二维网格面积核对忽略区域并集、另行检查全query背景排除、以Hopcroft–Karp核查两阶段匹配基数，并检查每对的类别、可见性、分数分箱、原生支持及TopQ来源。独立计数仍为训练290对、验证主分析78对。
首图374个query的NumPy双线性读取与导出细尺度均值最大差1.1151e-5，内外差分最大差1.1381e-5；全部112图的float64原始分类Linear重构最大差1.6467e-6。它们验证字段与计算一致性，不证明语义差距、信息擦除或新算法收益。
两次解析性合成ridge样例属于前置数值检查，不是数据探针拟合；本轮真实数据探针0次。缓存Linear矩阵重构不是新的原始网络前向。
恢复采集进程CUDA峰值分配611366912字节，约583MiB；旧失败进程未导出峰值统计。当前24GB卡已完成这一冻结推理采集，不等于已验证新编码器端到端训练的显存或收敛。

五、候选主线与最近工作边界
本轮没有进行新的文献方法排重；沿用A37的114条矩阵。PiDiViT已有候选区域差分/多尺度增强，Unmasking the Tiny已有借助分类语义恢复弱前景，STA、ECDet、SPAR与QueryDet也覆盖细节接入、区域路由和粗到细读取。不能以这些组成部分或弱分数现象本身充当新颖性。
由于本轮没有估计两种表征的差距，也没有取得UnmaskingTiny完整方法，不能写“现有方法没有解决此缺口”。当前仅有可证伪问题与可靠的覆盖数量。
原文方法与链接见84_A37局部差分与前景语义最近工作核查.txt；本轮自己的观察和数值核验见A38目录，不能与原文结论混为一谈。

六、下一步的前瞻性草案，当前未执行
在未得到任何数据探针表现之前，准备了一次只依据覆盖数量的取样调整草案：保留64个训练前缀，纳入全部64个可用验证前缀，每组按原SHA256规则取最多2张，有单张者保留1张。标签、特征差距和探针表现不参与选择。
草案得到119张训练图、105张验证图；复用本轮112份缓存，额外112张图。若下一阶段核定执行，可能预算为112张新图各1次及1次预定的新图完整query重复控制，共113次。当前分配为0，不追加A38的114预算。
新增相邻帧意味着后续置信区间必须按64个验证前缀整组重采样。前缀仍只是保守分组代理，不能称为已验证航次。原100对主验证门槛、判别指标、阈值、特征对照、lambda与7次数据拟合上限均保留；新采集、分组bootstrap和停止规则须先在独立阶段固定。覆盖再次不足时不自动追加图片或放宽阈值。
该调整在比较结果尚未出现时提出，不能作为主线有效证据。完成后仍须与完整最近方法对比，并用真正可部署的算法对照和检测指标判断研究价值。

当前状态：目标active；主线未确认；A38114次原生采集已结束且预算关闭；数据拟合0；没有新检测器训练、旧DEIM/A22重启或压缩分支实验。
'''
with REPORT.open('x',encoding='utf-8') as f:f.write(text)
backup=D/'canonical_before_A38';backup.mkdir(exist_ok=False)
for name in CAN:shutil.copy2(B/name,backup/name)
remapped={}
for p,h in prior['files_sha256'].items():
    q=Path(p);q=backup/q.name if q.parent==B and q.name in CAN else q
    assert sha(q)==h;remapped[str(q)]=h
board=read(B/CAN[0]);card=next(r for r in board['candidates'] if r['id']=='R_FINE_MEMORY_QUERY_SEMANTIC_GAP')
card.update(status='SUPPORT_GATE_FAILED_NO_SEMANTIC_EFFECT_ESTIMATE',evidence=str(D/'independent_support_verification.json'),
    causal_evidence=False,increment_approved=False,next='Prospectivecount-onlycoverageprotocolbeforeanynewcalls;do notlower100pairgateorclaimnull/positivegap.')
question='同框细尺度memory与最终query的判别差距尚未估计；先固定在无探针表现时提出的覆盖调整，再决定是否继续验证。'
board.update(status='SEARCHING_NO_VALIDATED_MAINLINE_A38',latest_update_utc=NOW,mainline_validated=False,
    latest_report=str(REPORT),A38_evidence=phase,current_priority_question=question,
    bounded_diagnostic_protocol=str(D/'semantic_protocol_FROZEN.json'),
    next_decision_protocol=dict(status='COUNT_ONLY_COVERAGE_DRAFT_NO_ALLOCATION',file=str(D/'coverage_followup_sample_DRAFT.json')))
write(B/CAN[0],board)
search=read(B/CAN[1]);search.update(supplement_utc=NOW,A38_semantic_coverage=dict(phase=phase,new_search_queries=0,
    protocol=str(D/'semantic_protocol_FROZEN.json'),source_taps=str(D/'native_source_tap_audit.json'),report=str(REPORT)))
write(B/CAN[1],search)
matrix=read(B/CAN[2]);assert len(matrix['rows'])==114
for r in matrix['rows']:
    if r['id']=='Dome_DETR':r['A38_scope_update']=dict(frozen_native_entries=114,unique_images=112,
        head_reconstruction_cases=112,support_val_pairs=78,semantic_effect_measured=False,export_deviation_disclosed=True)
matrix.update(updated_utc=NOW,latest_report=str(REPORT),latest_empirical_boundary='A38native114entries,captured112images;primary78pairs<100gate;datafit0,noAUCestimateandnohypothesisacceptance/rejection.')
write(B/CAN[2],matrix)
state=read(B/CAN[3]);state.update(recorded_utc=NOW,goal_status='active',mainline_validated=False,
    previous_goal_turn=state['current_goal_turn'],current_goal_turn=phase,latest_authoritative_phase=phase,
    latest_report=str(REPORT),current_priority_question=question,A38_evidence=phase,model_forwards=3282,training_updates=528,
    current_native_job_terminal=dict(phase='A38',wrapper_pid=15965,collector_pid=15966,both_observed_absent=True,
        worker_exit_code=0,original_export_failed_worker_exit_code=1,whole_detector_entries=114,cap_closed=True))
write(B/CAN[3],state)
source=read(B/CAN[4]);source.update(updated_utc=NOW,A38_native_semantic_feature_collection=dict(phase=phase,
    source_taps=str(D/'native_source_tap_audit.json'),independent_support=str(D/'independent_support_verification.json'),
    export_amendment=str(D/'export_recovery_amendment_FROZEN.json'),semantic_effect_measured=False,new_algorithm_approved=None))
write(B/CAN[4],source)
files=[p for p in D.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[REPORT]+[B/n for n in CAN]
receipt=dict(phase);receipt.update(archive_verification=av,independent_support_verification=iv,
    control_verification=control,scientific_result=result,terminal_observation=terminal,
    previous_A37_evidence_remapped_sha256=remapped,files_sha256={str(p):sha(p) for p in sorted(set(files))})
write(RECEIPT,receipt,True)
for p,h in receipt['files_sha256'].items():assert sha(p)==h,p
for p,h in remapped.items():assert sha(p)==h,p
print(json.dumps(dict(status='A38_PUBLISHED_SUPPORT_GATE_FAILED_NO_EFFECT_ESTIMATE',files=len(receipt['files_sha256']),
    previous_A37_files_verified=75,whole_forwards=3282,new_native_entries=114,data_probe_fits=0,
    primary_val_pairs=78,minimum_pairs=100,next_model_allocation=0),ensure_ascii=False))
