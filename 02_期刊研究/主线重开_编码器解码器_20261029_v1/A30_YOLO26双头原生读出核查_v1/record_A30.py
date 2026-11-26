"""Publish terminal native-head evidence with explicit non-completion boundary."""
from pathlib import Path
import datetime,hashlib,json,shutil,sys
sys.stdout.reconfigure(encoding='utf-8')
R=Path(__file__).resolve().parent;N=R.parent;D=R/'downloaded_evidence'
NAMES=['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json']
REPORT=N/'70_A30原生双头验证与小目标语义读出边界.txt';RECEIPT=N/'71_A30双头任务与原始支持核验回执.json'
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
    old=read(N/'69_A29最近方法与缓存漏检核验回执.json')
    for p,h in old['files_sha256'].items():assert sha(p)==h,p
    av=read(R/'archive_verification.json');mv=read(R/'independent_metric_verification.json');rv=read(R/'raw_support_verification.json')
    assert av['status'].startswith('PASS_') and av['terminal_process_absent'] and av['remote_exit_code']==0
    assert mv['status'].startswith('PASS_') and mv['arms_verified']==8 and rv['native_GT_branch_rows_checked']==8982
    assert rv['raw_result_sha256']==sha(R/'raw_support_result.json')
    manifest=read(D/'evidence_manifest.json')
    for p,h in manifest.items():assert sha(D/p)==h,p
    for name,h in read(R/'execution_inputs_manifest.json').items():assert sha(R/name)==sha(D/name)==h,name
    back=R/'canonical_before_A30';back.mkdir(exist_ok=False)
    for name in NAMES:shutil.copy2(N/name,back/name)
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();result=read(D/'result.json');raw=read(R/'raw_support_result.json');cfg=read(R/'native_dual_head_protocol_FROZEN.json');nextplan=read(R/'next_semantic_readout_question_DRAFT.json');newrows=read(R/'new_core_records.json')['rows'];pin=read(R/'pinned_source_manifest.json');fid=read(D/'cached_prediction_fidelity.json');start=read(D/'start.json')
    board=read(N/NAMES[0]);matrix=read(N/NAMES[2]);assert len(board['candidates'])==16 and len(matrix['rows'])==97
    assert not {x['id'] for x in newrows}&{x['id'] for x in matrix['rows']}
    delta=result['native_O_minus_M_input_small_AP']*100;ci=[x*100 for x in result['native_O_minus_M_paired_image_bootstrap95']]
    phase={'phase':'A30','classification':'progress_native_dual_head_controls_and_failure_scope','recorded_utc':now,'status':'NATIVE_DUAL_HEAD_DIAGNOSIS_COMPLETE_NO_VALIDATED_NEW_ALGORITHM','goal_status':'active','mainline_validated':False,'complete_mainline':False,'report':str(REPORT),'receipt':str(RECEIPT),'previous_goal_turn_classification':'progress:A29deprioritizedclassicalcalibrationP andindependentlyaudited1718cachedsmallmisses.',
      'new_whole_detector_forwards':101,'new_GPU_model_calls':101,'new_standalone_native_TAL_calls':200,'new_extra_native_head_inference_decodes':200,'all_native_head_inference_decodes_including_full_forwards':301,'new_complete_decoder_calls':0,'new_native_scan_component_calls':0,'new_optimizer_updates':0,'new_training_jobs':0,'new_task_postprocess_arms':8,'new_COCO_eval_calls':10,'paired_image_bootstrap_draws':1000,
      'fresh_source_images':100,'matched_FP_primary_budget':1200,'native_O_minus_M_input_small_AP_pp':delta,'native_O_minus_M_paired_image95_pp':ci,'research_gap_confirmed':False,'new_algorithm_approved':None,
      'full_model_forwards':old['full_model_forwards']+101,'actual_cumulative_detector_optimizer_updates':old['actual_cumulative_detector_optimizer_updates'],'formal_detector_training_updates':old['formal_detector_training_updates'],'historical_capacity_updates':old['historical_capacity_updates'],'cumulative_complete_decoder_only_calls':old['cumulative_complete_decoder_only_calls'],'historical_last_layer_suffix_calls':old['historical_last_layer_suffix_calls'],
      'new_protocol_cap':101,'new_protocol_cap_closed':True,'historical39092cap_remains_closed':True,'counts_terminal':True,'A30_pid':start['pid'],'A30_start_ticks':start['start_ticks'],'A30_terminal_process_absent':True,'A30_exit_code':0,'A22_restarted':False,'automatic_followups':False,'next_model_call_allocation':0,
      'new_core_papers':1,'core_matrix_rows':98,'candidate_count':16,'verified_pinned_source_blobs':8,'new_full_PDFs_archived':1,'source_input_and_checkpoint_unchanged':True,'new_heldout_confirmation_performed':False,'next':nextplan['question'],
      'counter_scope':'Architecture-probe counters only. OlderD0/DA100epochtraining is historicalandseparate;YOLOheaddecode/TAL calls are not whole-detector or DETRdecoder calls.'}
    board.update(status='SEARCHING_NO_VALIDATED_MAINLINE_A30',mainline_validated=False,latest_update_utc=now,latest_report=str(REPORT),current_priority_candidate=None,current_priority_question=nextplan['question'],A30_evidence=phase)
    board['next_decision_protocol']={'phase':'afterA30','status':'DRAFT_NO_ALLOCATION','file':str(R/'next_semantic_readout_question_DRAFT.json')}
    board['A30_selection_note']='没有新增获准算法。换分支、交换框/分数、加NMS或直接加正样本均不列为新主线；下一问是局部语义读出与单次预测排他约束，须先排除最近工作并做控制。'
    write(N/NAMES[0],board)
    log=read(N/NAMES[1]);log.update(supplement_utc=now);log['A30_native_head_and_readout_diagnostic']={'phase':phase,'public_source_manifest':str(R/'pinned_source_manifest.json'),'literature_discovery':str(R/'literature_discovery_raw.json'),'version_correction':str(R/'version_scope_correction.json'),'new_papers':1,'new_experiment_protocol':str(R/'native_dual_head_protocol_FROZEN.json')};write(N/NAMES[1],log)
    matrix['rows'].extend(newrows);matrix.update(updated_utc=now,latest_report=str(REPORT),latest_empirical_boundary='A30nativeheadfactorialdiagnostic:101forwards,no updates;all8internalCOCOAP andfixedFP independentlyrecomputed. Nativeheadgap is small/uncertain;rawlow-score support is not causal,novel or encoder-loss evidence.');write(N/NAMES[2],matrix)
    state=read(N/NAMES[3]);state.update(goal_status='active',mainline_validated=False,model_forwards=phase['full_model_forwards'],latest_report=str(REPORT),latest_authoritative_phase=phase,current_goal_turn=phase,current_priority_question=nextplan['question'],A30_evidence=phase,current_native_job_terminal=True)
    state['cap_scope']='Historical39092forwardcap staysclosed with unusedbudgetunavailable. SeparateA30cap101wasfullyconsumedandclosed;no newmodelcallsallocated. training/DETRdecoder counters unchanged.'
    state['authoritative_phase_history'].append({'phase':'A30','receipt':str(RECEIPT),'status':phase['status']});write(N/NAMES[3],state)
    source=read(N/NAMES[4]);source.update(updated_utc=now,A30_YOLO26_native_head_and_source={'phase':phase,'source_audit':str(R/'source_operator_audit.json'),'version_scope_correction':str(R/'version_scope_correction.json'),'native_result':str(D/'result.json'),'independent_metrics':str(R/'independent_metric_verification.json'),'raw_support':str(R/'raw_support_result.json'),'raw_verification':str(R/'raw_support_verification.json')});write(N/NAMES[4],source)
    metric=result['metrics'];M=metric['Mbox_Mscore_NMS'];O=metric['Obox_Oscore_TOPK'];SM=raw['summary']['M'];SO=raw['summary']['O']
    table='\n'.join(f"  {k}: inputsmallAP={100*v['input_area_AP']['AP_small_500']:.6f}%,APall={100*v['input_area_AP']['AP500']:.6f}%,FP1200时smallTP={next(x['small_TP'] for x in v['same_FP'] if x['FP_budget']==1200)}" for k,v in metric.items())
    same='\n'.join(f"  FP预算{x['FP_budget']}: M={x['small_TP']}个smallTP（实际FP{x['actual_FP']}）,O={y['small_TP']}个smallTP（实际FP{y['actual_FP']}）" for x,y in zip(M['same_FP'],O['same_FP']))
    refs='\n'.join(f"  {x['path']}\n    {x['url']}" for x in pin['files'])
    report=f'''A30 原生双头验证与小目标语义读出边界
记录时间：{now}
Material Passport：academic-research-suite / experiment-agent；定向原文/源码核查＋冻结权重推理诊断；验证状态为源码、字节、运行身份、8组AP/固定FP及缓存属性核验通过，不是新算法或主线有效证明。版本A30。

当前结论：尚未找到可正式定题的小论文主线。A30排除了一个必要混淆：低分可定位小目标现象并非只存在于此前one2many评价中。两个原生分支在同误报预算下接近，而可定位低分候选大多在终态TAL中已有有效正样本。因此不能把“换one2one头”“补正样本数量”“交换分类分数/回归框”包装成新贡献，也不能由此归因编码器已经丢失语义。新的编码器、交互和解码器选择继续开放，P仍降级。

1. 检索策略与记录
本阶段一批4条定向检索，查询YOLO26/STAL/featurealignment与tinyobject/anchorpoint对齐；另打开官方文档与第三方概述摘要，再沿官网链接核实原作者报告。完整查询/返回见literature_discovery_raw.json。第三方论文与索引对STAL的描述不作为本次固定版本行为依据，研究不是系统综述或排他性查重。
原作者文献是Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models，arXiv2606.03748v1（2026-06-02）；它区别于Hidayatullah/Tubagus的第三方概述2602.14582。新增原始PDF一篇及逐节阅读记录，矩阵97增至98条。8个源码blob与公开Git tree核对，其中7个来自既有固定源码、1个NMS文件新取；不是8个新算法。

2. 核心文献与版本核查
作者报告已提出双头一致任务分配、直接框回归、ProgressiveLoss、STAL与MuSGD，且表5/6已有权重日程和STAL参考尺寸消融，表7已经分别报告NMS与E2E路径。因此“对比两头”“改变扩展尺寸”“更多正样本”不是本轮的首创。
本次实际版本固定Ultralytics8.4.191，commit2004cba021c68a46429be460e26954ab171b2862。YAML的end2endTrue只是该冻结配置默认值；当前官网文档描述one2many+NMS为精度默认路径、one2one为可选nmsFalse路径。前面口头“默认one2one”已经收窄为版本限定，报告始终明确分支，没有升级或改动冻结代码。
另一个明确版本差异：作者报告式5按边长<s_min触发候选扩展；本次固定源码按边长<中间stride16触发并扩至16。不能将论文的较早触发规则、COCO/MuSGD配方与本研究D0/AdamW100轮配方视为相同复现。
阅读范围：{newrows[0]['read_scope']}。没有声称读完31页或复现作者COCO训练。原始PDF与SHA详见new_core_records.json及公共文献库sidecar。

3. 最近机制源码对照与受控验证
[源码] DetectionModel实际选择E2ELoss，不是相邻的旧E2EDetectLoss。M头topk10；O头先topk7再topk2=1。两头分类均为对归一化任务对齐软目标的BCE，而非忽视定位质量的硬标签分类。任务分数使用p^0.5*非负CIoU^6，随后有候选筛选、冲突消解、二次topk与软目标归一化。候选区域放大不等于每个GT最终必有有效监督。
[源码] 两头共用同一backbone/neck；训练中的O头特征入口detach，eval不detach。end2end选择输出路径，不关闭双头训练。框decode的xywh/xyxy也随标志变化，本实验统一xyxy再后处理，防止格式混淆。
[已执行] 本地冻结协议后上传，单次隔离任务PID{start['pid']}/start{start['start_ticks']}执行100张旧开发集原图，每图一次前向取双头，第101次重放第1图核对完整O原生路径。4张先行和全部100张旧M高分预测一致性通过；最大框坐标差{max(x['max_box_abs_input_px'] for x in fid)}输入像素、最大分数差{max(x['max_score_abs'] for x in fid)}。O完整路径与缓存重构逐位一致，state_dict及源文件/权重哈希未变。
固定框头M/O×分数头M/O×后处理NMS/TOPK，共8组，所有组完整报告。GT仅用于评价与独立终态分配诊断，没有用于部署评分或修改网络。Mbox_Mscore_NMS与Obox_Oscore_TOPK是两个原生端点，其余是解释机制的反事实组合，不是训练出的新方法。
8组内部COCO结果：
{table}
两个原生端点的inputsmallAP差O−M={delta:.6f}pp，1000次成对图像bootstrap95%区间[{ci[0]:.6f},{ci[1]:.6f}]pp，区间跨0。这不证明两头完全等价，也不支持稳定的新算法收益。该区间仅条件于单checkpoint和当前100图，相关视频帧和训练seed不在区间中。
同误报预算：
{same}
必须区分同阈值与同误报：conf0.25时M为2411smallTP/1718smallFN/1200FP，O为2297smallTP/1832smallFN/920FP。直接按同阈值比较漏检会混入分数尺度与精度取舍；FP1200时则是2412对2415个smallTP。
M分数直接用于TOPK会因其稠密多预测语义造成大量重复；跨头换框/分数也没有获得可晋级的独立贡献。这符合已有双头职责差异，不能将反事实最高分一组挑出后当作新算法。

4. 原图小目标失败证据、推断与尚未回答的问题
[新观测] 4491个有效GT中，4129个为输入面积<32²小目标。检查NMS/TOPK之前的全部8400位置，M有{SM['small_with_raw_cover']}个小目标存在IoU>=0.5原始预测框，O有{SO['small_with_raw_cover']}个。原始支持是GT分析用的存在性统计，不是可恢复召回上界；同一框可以覆盖多个GT。
M的1718个同阈值漏检中，{SM['missed_with_raw_cover_score_below025']}个有位置合格但GT类分数<0.25的原始候选；O的1832个漏检中对应{SO['missed_with_raw_cover_score_below025']}个。这将此前“保存框里有低分定位证据”的观察扩展到原生O头，但没有证明分数不校准或语义缺失。
[新观测] 在终态模型重新调用原生TAL，M/O全部小目标分别有{SM['small_zero_effective_native_positives']}/{SO['small_zero_effective_native_positives']}个没有有效正目标。更关键的是，上述位置合格但低分的漏检中仅{SM['missed_low_cover_score_and_zero_effective_native_positives']}/{SO['missed_low_cover_score_and_zero_effective_native_positives']}个属于零有效正目标；O为23/1162。因此“当前没有正目标”不是该子群的占多数解释。
这里明确只检验最终checkpoint的匹配状态，不是训练轨迹；不能排除早期训练监督不足、过拟合、困难样本、遮挡或类别歧义。
[新观测] O的{SO['sum_native_effective_small_anchors']}个有效小目标正位置中，有{SO['sum_native_effective_small_anchors_outside_original_GT']}个位于原始GT框外，但有效位置都在扩展候选区域内。{SO['missed_best_cover_outside_original_GT']}个O漏检的最佳定位支持位置位于原始框外。它只提示检查局部语义读取位置的必要性，尚未控制大小、相位、类别和定位质量，不能称原因或新研究空白。
[我的推断] 当前更应区分“有定位证据”“有足够可分的类别证据”与“单次预测的去重/排他约束”。仅见好框低分，不能推出换新编码器、增加正样本或套用DeepSeek机制就有收益。
[待验证假设] 基于预测几何的局部语义重新读取是否能在保持实例排他性、相同框/候选/FP预算时改善小目标；该操作是否已被TOOD、质量评分或NMS-free方法覆盖。它是下一项诊断问题，尚未获准为候选算法或主线。

5. 算法增量、可行性与风险
本阶段没有新增被确认的新算法。仅换分支、交叉换分数/框、额外NMS、STAL尺寸调整或普通质量分支均不足以支持独立创新。P校准继续降级，O单向查询隔离不晋级，N未验证。
已证实宿主能运行：本轮101次FP32单图前向、200次原生TAL，优化器更新0；总分析约{result['elapsed_seconds']:.2f}秒，含归档约{read(R/'remote_complete.json')['elapsed_seconds']:.2f}秒。GPU allocated/reserved峰值约{result['GPU_peak_allocated_bytes']/2**30:.4f}/{result['GPU_peak_reserved_bytes']/2**30:.4f}GiB。这是推理资源，不能替代新架构的24GB训练预检查。更早D0的100轮训练可作为已存在对照，未重跑。
主要风险：已反复使用的100图是开发数据；单seed；内部COCOcrowd口径并非官方VisDrone忽略区口径；分数低可能符合质量软目标或普通泛化难度；反事实交换破坏原配对语义；终态TAL不能证明训练因果。上述风险限制本轮定题力度，不能用微小AP差或原始支持数量越过。

6. 最小下一步（草案，当前新增调用分配0）
先读最接近的TOOD空间对齐、NMS-free双分配等完整方法/源码，确认基于预测框中心重新读取分数或特征是否只是现成方法的等价实现。同步沿已核对的官方VisDrone忽略区规则规范后续指标，避免内部crowd结果被当成榜单或投稿对比。
若仍有具体算子差异，再冻结缓存控制：保持框和候选数不变，以原生零偏移与GT-free预测中心读取作对照，同时测重复匹配、固定FP召回和AP；若收益只来自阈值、NMS、更多候选或GT位置，则不晋级。只有存在可纠正、可复验的语义/实例交互缺陷后，才选择编码器、交互或解码器增量及新的训练协议。
详见next_semantic_readout_question_DRAFT.json。当前实验已正常完成且原PID缺失，无自动队列；不重启A22、旧DEIM或压缩/恢复分支。图像使用清单审计、独立确认集、完整验证和额外seed仍是后续条件。整体寻找可投稿主线的目标保持active。

7. 参考、链接与核验
原作者报告：{newrows[0]['url']}
官方当前文档：https://docs.ultralytics.com/models/yolo26/
固定源码：
{refs}
下一步需完整核查的TOOD（本阶段仅沿用此前官方摘要范围，不称完整方法已读）：
  https://openaccess.thecvf.com/content/ICCV2021/html/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.html
GFL/GFLv2已在既有核心矩阵，用于限定一般性质量评分的创新范围。

核验：144490777字节证据包SHA{av['archive_sha256']}，119个归档文件逐项匹配；冻结协议/runner字节一致，远端exit0且PID已缺失；独立NumPy实现重算8组AP与24条固定FP，最大AP误差{mv['AP_max_abs_error']}；核对1000次bootstrap抽样seed/量化区间并独立重算3次AP抽样；8982条GT×分支正目标计数与16条分层标量IoU支持记录另行核对。
架构分支累计整模型前向{phase['full_model_forwards']}，训练更新仍528，完整DETRdecoder-only仍48；更早D0/DA训练单列不混入。本轮的新预算101已用完并关闭，旧39092上限保持关闭，没有可转移的未用额度。
这份报告不是主线完成声明，没有把任何未验证假设写成已找到的创新。
'''
    REPORT.write_text(report,encoding='utf-8')
    logs=N.parent/'前期调研_码率受限航拍小目标_20261028_v1'/'远端验证_20261029_v1'/'ssh_operation_receipts.jsonl';ops=[json.loads(x) for x in logs.read_text(encoding='utf-8').splitlines() if 'A30_' in x];write(R/'ssh_operation_evidence.json',{'operations':ops})
    write(R/'search_strategy_and_log.json',{'phase':'A30','query_count':4,'query_batch_count':1,'queries':['"YOLO26" "STAL" "alignment"','"small object" "expanded" "assignment" "feature"','"tiny object" "classification" "anchor points" "alignment"','"YOLO26" "classification" "feature alignment"'],'raw':'literature_discovery_raw.json','primary_followup':'Openedofficialdocs and thirdpartyoverviewabstract;followedofficialdocs author-report link. Eight Gitblobs verified separately.','review_type':'Targetedsource/priorstudy,not systematicreview','technical_sources':'Pinnedauthorcode,runtimeevidence,originalauthorreport;secondaryforks andindexedunsupported descriptions not used.'})
    remap={str(back/Path(p).name) if Path(p).parent==N and Path(p).name in NAMES else p:h for p,h in old['files_sha256'].items()}
    for p,h in remap.items():assert sha(p)==h,p
    paths=[p for p in R.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths.extend(N/n for n in NAMES);paths.extend([REPORT,N/'68_A29状态校准候选降级与原始图像漏检筛查.txt',N/'69_A29最近方法与缓存漏检核验回执.json'])
    for row in newrows:paths.extend([Path(row['local_pdf']),Path(row['access_record']),Path(row['local_pdf']).parent/'_text'/(Path(row['local_pdf']).stem+'.txt')])
    files={str(p):sha(p) for p in sorted(set(paths),key=str)}
    receipt=dict(phase);receipt.update(archive_verification=av,independent_metric_verification=mv,raw_support_verification=rv,previous_A29_files_verified_with_canonical_backup=len(remap),previous_A29_evidence_remapped_sha256=remap,files_sha256=files)
    write(RECEIPT,receipt)
    for p,h in files.items():assert sha(p)==h,p
    assert read(N/NAMES[3])['model_forwards']==2466 and len(read(N/NAMES[2])['rows'])==98
    print(json.dumps({'phase':'A30','status':phase['status'],'verified_files':len(files),'prior_A29_files_verified':len(remap),'whole_forwards_new':101,'whole_forwards_cumulative':2466,'optimizer_updates_new':0,'core_matrix_rows':98,'mainline_validated':False,'goal_status':'active','report':str(REPORT)},ensure_ascii=False))
if __name__=='__main__':main()
