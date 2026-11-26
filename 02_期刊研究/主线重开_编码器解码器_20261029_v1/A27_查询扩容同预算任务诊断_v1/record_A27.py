"""Publish the terminal A27 evidence without changing historical frozen files."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parent
NEW=ROOT.parent
NAMES=['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json',
       '15_当前状态与结果校验.json','16_最近工作源码核查.json']
REPORT=NEW/'64_A27同预算查询扩容任务诊断与候选淘汰.txt'
RECEIPT=NEW/'65_A27任务指标与候选判定核验回执.json'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    p=Path(p);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(p)

def main():
    assert not REPORT.exists() and not RECEIPT.exists()
    verify=read(ROOT/'independent_verification.json')
    assert verify['status']=='PASS_ARCHIVE_COUNTS_POINT_AP_COVERAGE_BOOTSTRAP_AND_DECISION'
    e=ROOT/'verified_task_evidence';result=read(e/'task_result.json');protocol=read(e/'task_protocol_FROZEN.json')
    decoded=read(e/'decoder_result.json');export=read(ROOT/'export_receipt.json')
    assert sha(ROOT/'task_evidence.zip')==export['sha256']==verify['archive_sha256']
    terminal=read(ROOT/'terminal_process_snapshot.json')
    assert terminal['terminal'] is True and terminal['task_result_present'] is True
    prior=read(NEW/'63_A26缓存还原与三层解码器核验回执.json')
    for path,h in prior['files_sha256'].items():assert sha(path)==h,path
    backup=ROOT/'canonical_before_A27';backup.mkdir(exist_ok=False)
    for name in NAMES:shutil.copy2(NEW/name,backup/name)
    for name in NAMES:assert sha(NEW/name)==sha(backup/name)
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    p=result['point'];ci=result['paired_95pct_percentile_intervals'];m=result['metrics'];c=result['coverage'];d=result['decision']
    assert d['pilot_all_gates_pass'] is False
    next_text='当前没有已确认主线；不推进O当前的单向查询隔离增量。回到编码器/解码器候选的原文、具体失效及可区分算子，形成新的可证伪问题后再分配验证；N仍未验证，A22不重启。'
    phase={'phase':'A27','classification':'progress_negative_candidate_screen','recorded_utc':now,
      'status':'CURRENT_ONEWAY_QUERY_ISOLATION_INCREMENT_NOT_ADVANCED','goal_status':'active',
      'mainline_validated':False,'complete_mainline':False,'report':str(REPORT),'receipt':str(RECEIPT),
      'new_complete_three_layer_decoder_calls':45,'new_decoder_layer_calls':135,'A26_outputs_reused':3,
      'cumulative_complete_decoder_only_calls':48,'historical_last_layer_suffix_calls':72,
      'new_whole_detector_forwards':0,'new_optimizer_updates':0,'new_training_jobs':0,'new_GPU_model_calls':0,
      'new_AP_arm_evaluations':4,'paired_cached_image_contexts':12,'full_model_forwards':2365,
      'actual_cumulative_detector_optimizer_updates':528,'formal_detector_training_updates':526,
      'historical_capacity_updates':2,'cumulative_authorized_protocol_cap':39092,'full_forward_cap_closed':True,
      'counts_terminal':True,'CPU_decoder_component_cap':45,'CPU_decoder_component_cap_closed':True,
      'protocol_sha256':sha(ROOT/'task_protocol_FROZEN.json'),'O_task_harm_measured':True,
      'O_positive_APsmall_benefit_vs_same_budget_observed':False,'O_novelty_cleared':False,
      'point':p,'paired_95pct_intervals':ci,'coverage':c,'decision':d,'new_core_papers':0,
      'core_matrix_rows':82,'candidate_count':15,'A22_restarted':False,'N_scientific_design_changed':False,
      'next_task_decoder_allocation':0,'next':next_text}
    x=read(NEW/NAMES[0]);x['latest_update_utc']=now;x['latest_report']=str(REPORT)
    x['mainline_validated']=False;x['current_priority_question']=next_text;x['A27_evidence']=phase
    x['status']='SEARCHING_NO_VALIDATED_MAINLINE_A27'
    candidate=next(v for v in x['candidates'] if v['id']=='O_QUERY_BUDGET_EXPANSION_CONTEXT')
    candidate.update(status='CURRENT_ONEWAY_ISOLATION_INCREMENT_NOT_ADVANCED_AFTER_PRESPECIFIED_TASK_SCREEN',
      increment_approved=False,real_model_task_effect_measured=True,causal_evidence=False,
      real_model_probe_scope='12 reused cached image contexts,48 full3-layer decoder outputs (3A26 reused+45new), four fixed-budget/control arms; no image encoding or training.',
      A27_evidence=phase,next=next_text,
      rejection_scope='This frozen single-checkpoint/initializer/cohort one-way-isolation increment failed prespecified task screening. Not proof that every query-interaction method or trained architecture is ineffective.')
    write(NEW/NAMES[0],x)
    x=read(NEW/NAMES[1]);x['supplement_utc']=now;x['A27_task_budget_and_evaluation_source']={
      'source_manifest':str(ROOT/'official_toolkit_source_manifest.json'),
      'reading_scope':read(ROOT/'official_evaluation_source_audit.json'),
      'new_paper_searches':0,'new_paper_records':0,'task_screen':phase}
    write(NEW/NAMES[1],x)
    x=read(NEW/NAMES[2]);assert len(x['rows'])==82
    x.update(updated_utc=now,latest_report=str(REPORT),
      latest_empirical_boundary='A27: standard one-way query isolation preserves old outputs but does not pass same-budget AP/harm screening on12 reused cached contexts. Closest work still prevents claiming generic masks/grouping/KV as novel. No confirmed mainline.')
    write(NEW/NAMES[2],x)
    x=read(NEW/NAMES[3]);x['recorded_utc']=now;x['latest_report']=str(REPORT);x['latest_authoritative_phase']=phase
    x['authoritative_phase_history'].append(phase);x['previous_goal_turn']=x['current_goal_turn']
    x['current_goal_turn']='A27_same_budget_query_expansion_task_screen_negative'
    x['current_priority_question']=next_text;x['goal_status']='ACTIVE';x['mainline_validated']=False
    x.update(model_forwards=2365,training_updates=528,formal_detector_training_updates=526,
      detector_optimizer_updates_including_capacity=528,formal_training_epochs=0,training_counts_terminal=True,
      counters_observed_lower_bounds_while_job_live=False,full_forward_cap_closed=True,counts_terminal=True,
      cap_scope='Historical full-detector allocation39092 is CLOSED after terminal A22; unused budget not transferable. A26 complete-decoder3calls closed; A27 additional45calls closed. No next experiment allocated.',
      complete_decoder_only_replays_cumulative=48,complete_decoder_only_replays_A27=45,
      historical_last_layer_suffix_calls=72,A27_evidence=phase)
    write(NEW/NAMES[3],x)
    x=read(NEW/NAMES[4]);x['updated_utc']=now;x['A27_query_task_evaluation']={
      'official_source_audit':read(ROOT/'official_evaluation_source_audit.json'),
      'metrics_preflight':read(ROOT/'metric_preflight_result.json'),
      'task_screen':phase,'old_A26_decimal_correction':read(ROOT/'A26_decimal_transcription_correction.json')}
    x['O_query_budget_context']['A27_task_screen']=phase
    write(NEW/NAMES[4],x)
    def pct(x):return f'{x*100:.4f}'
    def interval(k):return '['+','.join(f'{v*100:.4f}' for v in ci[k])+']'
    metrics='\n'.join(f'{arm}: AP={pct(m[arm]["AP"])}%, APsmall={pct(m[arm]["APsmall"])}%, AP50={pct(m[arm]["AP50"])}%' for arm in protocol['arms'])
    diffs='\n'.join(f'{k}: {pct(p[k])}'+('%,95%区间' if k=='harm_rate' else 'pp,95%区间')+interval(k) for k in p)
    gates='\n'.join(f'{k}: '+('通过' if v else '未通过')+'；预设 '+protocol['screen_gates_all_required'][k] for k,v in d['gates'].items())
    report=f'''A27 同预算查询扩容任务诊断与候选淘汰
记录时间：{now}

当前结论：尚无可定题的已确认主线。最近候选是“密集航拍小目标的查询扩容是否产生未被新增覆盖补偿的损失，以及能否在同500候选预算下稳定旧查询并改善检测”。本次不推进“直接单向隔离”这个具体增量。它的输出稳定性质成立，但未达到事先冻结的任务收益/损失门槛。此结论不等于证明所有查询交互架构都无效，也不把未验证的N自动升为主线。

1. 本轮检索策略与记录
本轮没有新增论文检索或论文矩阵条目；矩阵仍82篇，候选15个。新增核对官方 VisDrone DET 工具箱，固定commit 005445782213e20cb91bc50a597db3dd949e749a，11个文件逐一核验Git blob与SHA256。完整读取与本轮评价相关的evalDET/saveAnnoRes/dropObjectsInIgr/createIntImg/evalRes/compOas/VOCap/calcAccuracy；没有执行MATLAB。检索与源码记录见A27/official_toolkit_source_manifest.json及official_evaluation_source_audit.json。原文方法矩阵仍以14_架构核心文献矩阵.json为准。

2. 最接近工作的逐项差异与阅读范围
Group DETR：训练时分组自注意力及逐组一对一监督，推理保留一组。已读官方版本选定3.1、算法、图与损失段落；未通读全部实验/源码。其分组/隔离先例排除“首次分组掩码”说法。
GroupEnsemble：使用Group DETR已训练的多组查询进行推理，组间对称隔离并BSAS聚类融合，目标为不确定性估计。已读III-B—D及部分III-A，图式视觉核对；未完整读实验或作者代码。本次双向隔离仅是一个标准分组控制，不是该方法复现。本次固定普通checkpoint的TopK池，不含其组训练或融合，差异本身不等于创新成立。
原生DN掩码：单向阻断已存在于代码。掩码邻接模式、查询排序和普通KV缓存不是本研究创新。
VRWKV-DETR：目前仅出版商摘要/引言及片段阅读，未取得完整方法或代码；片段已提及同一训练解码器的查询扩容。不能据片段宣称对方未做某一机制。BUTR已提供预算匹配的必要性，本轮据此保持500候选作主比较。

3. 预先冻结的最小验证
平台：VisDrone val中的12张历史复用缓存图像，12个文件前缀；1241个有效GT，56个class0忽略区域，class11为0。原图面积<=1024的smallGT共933个；它与历史800输入下size<8的263个tinyGT不是同一口径。
模型：Dome-DETR作者权重，源码commit {protocol['source_commit']}；固定完整权重和图像编码后的memory，使用原生TopK500及原生查询初始化头，250为严格前缀。本受控初始化绕过PAQI密度选择，不作为作者完整检测器AP复现。
四组：BASE250未掩码；NATIVE500未掩码；TWOWAY500新旧组双向隔离；ONEWAY500旧查询不读新增查询、新增查询可读全部。主比较O对N，O对T检查是否超过普通分组，B仅作扩容参考。
本轮新增45次完整三层CPU decoder调用/135层进入，复用A26的3个case00输出，共48个图像-组输出。新增整图/编码器forward=0、优化器更新=0、GPU调用=0。既有整检测器累计2365次、实际更新528次保持；历史72次末层suffix调用不与48次完整decoder调用混合。
任务协议SHA256：{sha(ROOT/'task_protocol_FROZEN.json')}，生成时真实任务AP尚未计算。CPU运行约{decoded['seconds']:.2f}秒仅是本环境组件耗时，不能作为GPU端到端速度或24GB训练可行性的证据。

4. 已测结果与预设判定
评价：固定所有组Top500扁平sigmoid类别分数后再过滤类别1..10，不补足；原图坐标与源码一致的union忽略区域过滤；COCO101 AP@500。下列不是官方VisDrone榜单成绩。另存的连续、类别出现次数加权AP仅供描述，未宣称与MATLAB官方分数逐值等价。
{metrics}

主要差值/覆盖损失与10000次配对前缀bootstrap百分位95%区间：
{diffs}
损失统计分母：BASE250在无分数阈值、逐query argmax类别、IoU>=0.5规则下匹配的{c['base_correct_small']}个smallGT；NATIVE500任意同类真实query均无法覆盖其中{c['native_uncompensated_small_losses']}个，ONEWAY500为{c['oneway_uncompensated_small_losses']}个。此为保守的几何/类别覆盖诊断，不是常用分数阈值下的漏检率。
{gates}
最终：所有门槛同时通过={d['pilot_all_gates_pass']}。当前增量不晋级；创新性仍未排除，与统计门槛是否通过是两回事。

5. 原文结论、推断及待验证假设
[原文/源码] 新增查询在未屏蔽的self-attention中会进入旧查询的上下文；已知分组及DN掩码可阻断相应边。本次官方评价源码的忽略规则已核对。
[本次观测] 12例单向/双向隔离旧前缀数值差均在冻结容差内；单向保护虽然保住旧输出，其APsmall点估计仍低于普通500查询。原始数据、匹配标记、重采样及六门槛已独立核验。
[我的推断] 不能将输出变化直接视为有害，小样本现有结果不足以支撑“单向隔离能提高航拍小目标AP”作为论文主线；先停止该具体增量是合理的筛选决定。没有声称负差异具备总体统计显著性。
[待验证] 独立场景、其他checkpoint/训练后的体系是否存在足够大的真实查询扩容损失，及是否有区别于既有分组/掩码的算法；本轮没有分配这些试验。N的共享状态路径机制同样没有取得训练后效应证据；A22保持终止，数值问题诊断不作为创新。

6. 可行性、风险与下一步
现有单机/远端条件能承担这类冻结缓存的CPU解码诊断；本次未训练改进算法，不能由此断言正式训练的显存、天数和论文效果。12例已被多轮分析复用，前缀未证明等于独立飞行场景，bootstrap仅为探索性稳定区间，类别支持与分数并列排序有明确条件。GT是原始坐标而非旧缓存裁剪框，所有arm采用同样规则。不得将组件CPU耗时混称端到端速度。
{next_text}
下一步不是继续给掩码加模块或调门槛；先重新核查一个具体架构失效及最接近算子差异。目标保持active，没有自动训练或下一组试验。

7. 可核验资料与完整性
官方评价工具箱：https://github.com/VisDrone/VisDrone2018-DET-toolkit/tree/005445782213e20cb91bc50a597db3dd949e749a
原生模型源码：https://github.com/RicePasteM/Dome-DETR/tree/{protocol['source_commit']}
Group DETR：10.1109/ICCV51070.2023.00610；https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Group_DETR_Fast_DETR_Training_with_Group-Wise_One-to-Many_Assignment_ICCV_2023_paper.pdf
GroupEnsemble：https://arxiv.org/abs/2603.01847v1（预印本与上述阅读范围）
VRWKV-DETR：10.1016/j.eswa.2026.134086；https://www.sciencedirect.com/science/article/abs/pii/S0957417426029921（仅预览）
BUTR：https://arxiv.org/html/2610.33430v1（预印本；选定方法/分析阅读，非完整代码复现）
核心矩阵：{NEW/NAMES[2]}
冻结方案：{ROOT/'task_protocol_FROZEN.json'}
原始回执与独立校验：{ROOT/'independent_verification.json'}
证据ZIP SHA256：{export['sha256']}。
合成评价预检查v1曾因整数gtIgnore位取反得到错误计数而失败，原版本及失败回执保留；修正为显式==0计数的v2通过标准COCO与显式重复样本核验后，才冻结协议并读取真实任务结果。没有调整真实结果以适配门槛。
A26可读报告62中一个长小数的手工转录有误；原始JSON与回执63正确。其NATIVE500前缀框分量最大绝对差的正确值为{read(ROOT/'A26_decimal_transcription_correction.json')['observed_native_prefix_box_max_abs']!r}，本次作纠正说明，旧冻结文件不改、模型不重跑。
'''
    REPORT.write_text(report,encoding='utf-8')
    prior_remapped={str(backup/Path(path).name) if Path(path).parent==NEW and Path(path).name in NAMES else path:h for path,h in prior['files_sha256'].items()}
    for path,h in prior_remapped.items():assert sha(path)==h,path
    files=[f for f in ROOT.rglob('*') if f.is_file() and '__pycache__' not in f.parts]
    files.extend([REPORT]+[NEW/n for n in NAMES])
    toolkit=read(ROOT/'official_toolkit_source_manifest.json')
    for row in toolkit['files']:
        f=Path(toolkit['source_root'])/row['path'];assert sha(f)==row['sha256'];files.append(f)
    receipt={**phase,'remote_result':result,'independent_verification':verify,
      'previous_A26_evidence_files_verified_with_canonical_backup':len(prior_remapped),
      'previous_A26_evidence_remapped_sha256':prior_remapped,
      'files_sha256':{str(f):sha(f) for f in sorted(set(files),key=str)},
      'terminal_snapshot':terminal,'source_toolkit_files_checked':len(toolkit['files'])}
    write(RECEIPT,receipt)
    for path,h in receipt['files_sha256'].items():assert sha(path)==h,path
    print(json.dumps({'status':phase['status'],'mainline_validated':False,
      'new_receipt_files_verified':len(receipt['files_sha256']),
      'previous_receipt_files_verified':len(prior_remapped),'report':str(REPORT),
      'point':p,'decision':d},ensure_ascii=False))

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
