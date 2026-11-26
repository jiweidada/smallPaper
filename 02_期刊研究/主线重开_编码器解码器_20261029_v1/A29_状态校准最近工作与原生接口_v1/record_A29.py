"""Publish the P down-selection and cached source-image evidence; no mainline claim."""
from pathlib import Path
import datetime,hashlib,json,shutil,sys
sys.stdout.reconfigure(encoding='utf-8')
R=Path(__file__).resolve().parent;N=R.parent
NAMES=['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json']
REPORT=N/'68_A29状态校准候选降级与原始图像漏检筛查.txt'
RECEIPT=N/'69_A29最近方法与缓存漏检核验回执.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    p=Path(p);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def main():
    assert not REPORT.exists() and not RECEIPT.exists()
    check=read(R/'source_and_cache_verification.json');assert check['status']=='PASS_PRIMARY_BYTE_SCOPE_AND_ALL1718_CACHED_MISS_BINS'
    prior=read(N/'67_A28原文源码与候选范围核验回执.json')
    for p,h in prior['files_sha256'].items():assert sha(p)==h,p
    back=R/'canonical_before_A29';back.mkdir(exist_ok=False)
    for name in NAMES:shutil.copy2(N/name,back/name)
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();decision=read(R/'P_decision.json');rows=read(R/'new_core_records.json')['rows'];search=read(R/'search_strategy_and_log.json')
    cache=read(R/'D0_cache_candidate_audit_result.json');math_result=read(R/'formula_and_geometry_result.json');native=read(R/'selected_source_manifest.json')['rows']
    board=read(N/NAMES[0]);matrix=read(N/NAMES[2]);assert len(board['candidates'])==16 and len(matrix['rows'])==92
    assert not {r['id'] for r in rows}&{r['id'] for r in matrix['rows']}
    phase={'phase':'A29','classification':'progress_closest_prior_downselection_and_existing_source_prediction_reanalysis','recorded_utc':now,'status':'P_DEPRIORITIZED_NO_VALIDATED_MAINLINE_WITH_SOURCE_CACHE_DIAGNOSIS','goal_status':'active','mainline_validated':False,'complete_mainline':False,'report':str(REPORT),'receipt':str(RECEIPT),
      'previous_goal_turn_classification':'no_progress: preceding answer only checked and restated current status;this phase resumes pending A29 and adds source,algebra and cache evidence.',
      'new_whole_detector_forwards':0,'new_encoder_calls':0,'new_complete_decoder_calls':0,'new_native_scan_component_calls':0,'new_optimizer_updates':0,'new_training_jobs':0,'new_GPU_model_calls':0,'new_AP_evaluations':0,
      'cached_source_images_reanalyzed':100,'cached_missed_small_GT_independently_reaggregated':1718,'new_affine_coefficient_identity_cases':12,'new_geometry_counterexample_cases':4,
      'new_core_papers':5,'core_matrix_rows':97,'candidate_count':16,'new_full_PDFs_archived':5,'new_verified_author_git_blobs':9,
      'P_status':decision['status'],'P_novelty_cleared':False,'P_native_fidelity_verified':False,'P_real_UAV_defect_measured':False,'P_task_benefit_measured':False,'P_approved_as_mainline':False,'next_native_scan_allocation':0,'A22_restarted':False,
      'full_model_forwards':prior['full_model_forwards'],'actual_cumulative_detector_optimizer_updates':prior['actual_cumulative_detector_optimizer_updates'],'formal_detector_training_updates':prior['formal_detector_training_updates'],'historical_capacity_updates':prior['historical_capacity_updates'],'cumulative_complete_decoder_only_calls':prior['cumulative_complete_decoder_only_calls'],'historical_last_layer_suffix_calls':prior['historical_last_layer_suffix_calls'],'full_forward_cap_closed':True,'counts_terminal':True,
      'counter_scope':'Preserved architecture-probe counters;older D0/DA 100-epoch training belongs to its own prior branch and is not counted as new architecture execution.',
      'new_mainline_candidate_approved':None,'next':decision['next_priority']}
    candidate=next(c for c in board['candidates'] if c['id']==decision['candidate_id']);candidate.update(status=decision['status'],A29_decision=decision,A29_evidence=phase)
    board.update(status='SEARCHING_NO_VALIDATED_MAINLINE_A29',mainline_validated=False,latest_update_utc=now,latest_report=str(REPORT),current_priority_question=decision['next_priority'],A29_evidence=phase)
    board['current_priority_candidate']=None;board['A29_selection_note']='P降级为方法对照。当前只有新的缓存诊断问题，尚无替代算法获准作为主线；编码器/特征交互/解码器选择继续开放。'
    write(N/NAMES[0],board)
    log=read(N/NAMES[1]);log['A29_closest_prior_and_source_cache_followup']={'search_log':str(R/'search_strategy_and_log.json'),'phase':phase,'verification':str(R/'source_and_cache_verification.json')};log['supplement_utc']=now;write(N/NAMES[1],log)
    matrix['rows'].extend(rows);matrix.update(updated_utc=now,latest_report=str(REPORT),latest_empirical_boundary='A29: no new model or AP evaluation. Existing D0 source-image cache reanalysis is descriptive development evidence,not encoder causality or novel-algorithm validation. P is deprioritized.');write(N/NAMES[2],matrix)
    state=read(N/NAMES[3]);state.update(goal_status='active',mainline_validated=False,latest_report=str(REPORT),latest_authoritative_phase=phase,current_goal_turn=phase,current_priority_question=decision['next_priority'],A29_evidence=phase)
    state['authoritative_phase_history'].append({'phase':'A29','receipt':str(RECEIPT),'status':phase['status']});write(N/NAMES[3],state)
    audit=read(N/NAMES[4]);audit.update(updated_utc=now,A29_closest_calibration_and_source_cache={'phase':phase,'operator_comparison':str(R/'closest_work_operator_comparison.json'),'source_manifest':str(R/'selected_source_manifest.json'),'cache_result':str(R/'D0_cache_candidate_audit_result.json'),'cache_independent_verification':str(R/'D0_cache_independent_verification.json'),'novelty_boundary':'Step/spectrum calibration,geometry skips,write-gain absorption and generic classification/localization alignment are known mechanisms. No new encoder/decoder contribution has been validated.'});write(N/NAMES[4],audit)
    pins={x['repo']:x['commit'] for x in native}
    c=cache['exclusive_bin_counts'];low=c['B_same_class_overlap_score_005_to_025']+c['C_same_class_overlap_score_below_005']
    table='\n'.join(f"  {k}: {v}" for k,v in c.items())
    threshold='\n'.join(f"  conf={x['confidence']}: smallTP={x['small_TP']},smallFN={x['small_FN']},FP={x['FP']}" for x in check['historical_D0_threshold_table'])
    refs='\n\n'.join(f"{r['title']}\n  {r['url']}\n  版本/状态：{r['tier']}\n  阅读范围：{r['read_scope']}\n  原始PDF：{r['local_pdf']}\n  PDF SHA256：{r['pdf_sha256']}" for r in rows)
    report=f'''A29 状态校准候选降级与原始图像漏检筛查
记录时间：{now}
Material Passport：academic-research-suite；定向文献研究＋本地既有预测再分析。原始论文/源码字节、D0备份与分箱已核验；算法有效性和可投稿主线未获验证。阅读由助手完成，无人工全文阅读声明。

当前裁决：P“变分辨率选择性状态更新校准”降级为方法对照，不再优先追加原生扫描接入或训练。没有查证它与某篇完整方法完全相同；降级依据是经典算子等价、已有直接近邻、面积缩放规则的二维几何反例，以及缺少真实航拍机制和检测收益。整体目标保持active，尚无替代算法被确认是小论文主线。

1. 检索策略与检索记录
本阶段8批、29条定向查询，外加原始页面与作者仓库核查。检索从SSM重离散化与分辨率适配展开，补到长度校准、Hilbert几何和状态写入等价变换，再以已有YOLO26s原图预测核查下一步应研究的失败类型。不是系统综述，未检索到同名算法不构成首创证明。
检索完整返回及查询字符串分别存于本目录 search_strategy_and_log.json、web_discovery_raw.json、score_prior_discovery_raw.json。技术结论依赖原作者/官方会议/arXiv；博客、日更列表和审稿意见不作技术依据。DeciMamba及GATECH-EIC LongMamba仅作发现记录，没有在本阶段标为完整方法已读；jzhang38同名工程仓库不与LongMamba论文混同。

2. 核心文献矩阵与阅读范围
核心矩阵由92增至97条，本阶段新增5篇原始PDF，均保留版本、哈希和逐节阅读范围；没有把下载全文等同于读完全篇。
FractalMamba++须使用2505.14062v4：其方法已变为Hilbert序列化、FHSC与FA-RoPE。PDF页眉仅标提交TMM，不称已录用。它已直接讨论高分辨率扫描中的信息衰减，并用几何跳连与位置调制应对；COCO是训练后的Mask R-CNN检测，并非极端分辨率零样本检测验证。
MambaExtend（ICLR2025）冻结原模型，校准各层步长；其training-free仍包含梯度或零阶优化。Mamba Modulation（NeurIPS2025）沿用该校准思路改变谱；作者公开完整序列前向用 -exp(scale*A_log)，不能误写成简单scale*A。两者的语言模型结果均不能替代航拍检测证据。
Norm or Direction?的所读部分显示，特征范数与前景信息不是一回事，移除SS2D重训后仍可能有背景高范数。其分割读出测试用sqrt(C)*x/||x||保持幅度尺度，说明只比较范数不足以定位SSM缺陷。LinQuant本阶段仅读摘要及部分吸收变换小节，其等价变换保持原算子，不能说它已实现P的变分辨率策略。

3. 与最接近工作的逐项对比
令a=delta*A，b=delta*B*u，其中delta已经过bias和softplus。
原生选择性扫描：exp(a)*h+b。
MambaExtend所核查校准分支：exp(s*a)*h+s*b。
Mamba Modulation所核查前向：exp(delta*(-exp(s*A_log)))*h+b。该式保持Euler输入项，没有在已读分支中发现P式写入补偿；这不是全仓库或所有推理路径的等价性声明。
P草案：exp(rho*a)*h+b*expm1(rho*a)/expm1(a)。
关键等价：对固定系数，定义连续系统的常值驱动v=a*b/expm1(a)，精确积分rho时长即得到P。因此P是已知仿射映射的连续生成元再离散化，公式和输入增益折叠本身不构成新原理。
冻结后12个固定系数案例的两种计算形式最大误差为{math_result['max_affine_identity_error']}。这是条件代数核验，没有调用原生扫描或预训练模型，也没有证明二维图像不变性。
另有4个二维距离反例：参考方形网格H宽，水平相邻位置在行扫描距离1、列扫描距离H；两轴放大r倍后，对应位置距离为r与r²H。rho=1/r²仅把列距离恢复为H，行距离变为1/r。不存在同一个标量同时满足rho*r=1与rho*r²=1（r>1）。这排除面积比例具有精确各向同性几何保持含义，不能据此断言真实AP一定下降。
FractalMamba++直接改变几何路由并学习状态注入；S4ND已有连续二维SSM分辨率适配；Mamba-3已有输入离散化改进；LinQuant已有等价因子放置。P即使以后有收益，也需要狭义应用差异与强对照，现阶段不足以独立推进。
源码固定：Mamba Modulation {pins['gnepul-ace/mamba_modulation']}；MambaExtend {pins['ArminAzizi98/LongContextMamba']}。9个Git blob逐一核验，仅静态阅读；未导入/运行下载源码。详见 selected_source_manifest.json 和 closest_work_operator_comparison.json。

4. 有依据的未决问题与本地观测
[原文结论] 长度/谱校准、高分辨率扫描记忆衰减和二维几何改造已有工作；一般性的分类—定位对齐和质量评分也已被TOOD、GFL/GFLv2研究。不能把这些一般问题重命名为创新。
[既有基线] D0是YOLO26s、Ultralytics8.4.191、固定第100轮EMA。它在原始数据JPEG上训练，区别于追加压缩增强的DA。训练早已完成，本阶段没有重启。源图预测是unfused one2many+multi-label NMS、640输入、maxDet500，采用内部crowd口径；不是官方VisDrone榜单分数，也不自动代表原生one2one行为。
[本阶段新再分析] 对已用于开发的100图、37645条保存预测，历史4129个输入面积<32²的有效GT在conf0.25/IoU0.5下有2411个TP、1718个FN。先冻结分箱，再按原始letterbox ratio/pad转换GT；1718个FN全部用独立标量IoU实现重算通过，排除crowd。互斥分箱按固定优先顺序如下：
{table}
其中{low}个漏检GT至少有一个类别正确、IoU>=0.5但分数低于0.25的保存框；这不是可恢复召回上界，因为同框可覆盖多个GT、匹配和误报也会改变。421个属于类别正确但只在IoU>=0.1附近有分数>=0.05候选；剩余类别不能被简单解释为编码器抹去目标。
历史阈值敏感性表：
{threshold}
从0.25降低到0.05，smallTP由2411增至2931，同时FP由1200增至5628。这说明降阈值需要支付误报代价，不能凭低分正确框数量宣称可部署收益。
[我的推断] 下一步值得优先核查的是“已有可定位证据为何没有形成足够可靠的分类排序/空间读出”，并与定位误差分开；尚不能据此选择某个新编码器或归因于状态记忆。
[待验证假设] 相同FP和候选预算下，现有检测头是否存在能由具体结构干预纠正的微小目标读出缺口；该缺口是否超出已有质量评分/对齐方法。当前没有可称为文献已确认的新研究空白。

5. 候选算法增量、可行性与主要风险
P：降级，不分配原计划最多24次原生扫描核验。保留数学草案和A28证据，不改冻结文件；不把经典公式延伸包装成新架构。
新编码器/特征交互/解码器：仍开放，但本阶段不命名新模块或直接启动候选训练。下一项只是读出机制诊断，不把“增加置信度分支”“融合回归特征”“加局部门控”自动视为创新，TOOD/GFL已有一般性覆盖。
可承担宿主已有：D0在单卡batch8、640输入完成100轮，历史耗时约3.05小时，allocated峰值约4.06GiB；这支持复用当前对照，不证明任何新骨干/检测头都能在24GB训练。D0 checkpoint、212462766字节备份及6项解压后原始文件与压缩包逐字节SHA重新核对通过。
主要风险：分数低只是困难目标的症状；定位/分类/匹配和后处理混杂；100图已反复用于开发，不能称独立验证；缓存只含NMS后候选，无法定位编码器或分配器；单seed与内部口径不足以支撑期刊结论。

6. 最小后续验证方案（草案，当前分配0）
先静态核对D0固定版本的检测头、正样本分配、分类目标及one2many/one2one实际路径，对照TOOD/GFL与YOLO26已有机制。只有具体差异明确后，才冻结同checkpoint、同像素输入的原生接口采集方案，记录NMS前各层位置/分类/回归输出和候选流转；不从缓存缺失直接推断编码器缺失。
机制评价固定FP与候选预算，先排除全局阈值、尺度/类别评分及更多计算解释，再决定有没有必要改编码器、交互或读出结构。若只有换阈值、换现成骨干、范数或合成误差改善，就不晋级。
任何新的真实前向、训练、完整AP和独立seed均需先写明病例、组别、预算、停止条件和指标再执行。已获得的一般远端研究授权继续有效，不增加人为审批门槛；本报告未启动任务或设置自动队列。先审计全部图像使用记录，再选择预测盲的确认集；当前100图仅用于开发。新模型单卡资源须实测。
参见 next_readout_diagnostic_DRAFT.json。O单向查询隔离维持不晋级，N未验证，A22与旧DEIM保持终止；压缩/恢复不追加。

7. 可核验参考与链接
{refs}

一般性评分/对齐边界（本阶段只核对原始摘要；GFL/GFLv2已在既有核心矩阵）：
TOOD: Task-Aligned One-Stage Object Detection
  https://openaccess.thecvf.com/content/ICCV2021/html/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.html
Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection
  https://papers.nips.cc/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html
Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection
  https://arxiv.org/abs/2011.12885

核验材料：{R/'source_and_cache_verification.json'}
本阶段新增模型/扫描/编码器/解码器前向0、优化器更新0、GPU调用0、AP重评0；既有100图预测再分析和12+4项代数/几何核验已明确单列。架构分支历史计数保持2365整模型调用、528优化器更新、48完整decoder-only调用；它们不包含更早D0/DA分支训练，也不把本阶段缓存计算冒充模型实验。
寻找可投稿主线的目标未完成，继续active。当前没有算法被标为已找到的主线。
'''
    REPORT.write_text(report,encoding='utf-8')
    remap={str(back/Path(p).name) if Path(p).parent==N and Path(p).name in NAMES else p:h for p,h in prior['files_sha256'].items()}
    for p,h in remap.items():assert sha(p)==h,p
    paths=[p for p in R.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for row in rows:paths.extend([Path(row['local_pdf']),Path(row['access_record']),Path(row['local_pdf']).parent/'_text'/(row['stem']+'.txt')])
    paths.extend(N/n for n in NAMES);paths.extend([REPORT,N/'66_A28新架构原文对照与参考保持状态更新候选.txt',N/'67_A28原文源码与候选范围核验回执.json'])
    files={str(p):sha(p) for p in sorted(set(paths),key=str)}
    receipt=dict(phase);receipt.update(source_verification=check,algebra_and_geometry_result=str(R/'formula_and_geometry_result.json'),D0_cache_result=str(R/'D0_cache_candidate_audit_result.json'),previous_A28_evidence_files_verified_with_canonical_backup=len(remap),previous_A28_evidence_remapped_sha256=remap,files_sha256=files)
    write(RECEIPT,receipt)
    for p,h in files.items():assert sha(p)==h,p
    assert len(read(N/NAMES[2])['rows'])==97
    print(json.dumps({'phase':'A29','status':phase['status'],'verified_evidence_files':len(files),'prior_A28_files_verified':len(remap),'core_matrix_rows':97,'candidate_count':16,'new_model_calls':0,'new_optimizer_updates':0,'goal_status':'active','report':str(REPORT)},ensure_ascii=False))
if __name__=='__main__':main()
