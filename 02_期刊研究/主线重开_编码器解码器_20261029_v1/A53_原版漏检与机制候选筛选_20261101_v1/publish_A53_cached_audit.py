"""Publish the completed descriptive audit, preserving the A52 terminal chain."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
A52=BASE/'A52_不同机制候选有界筛选_20261101_v1'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    r=load(ROOT/'error_audit_result_v4.json');plan=load(ROOT/'protocol_FIXED_v4.json')
    assert r['status']=='DESCRIPTIVE_AUDIT_COMPLETE_NOT_A_METHOD_PASS'
    assert r['COCO_evaluate_calls']==1 and r['new_model_forwards']==r['new_optimizer_updates']==0
    assert sha(ROOT/'protocol_FIXED_v4.json')==r['protocol_sha256']
    assert sha(ROOT/'GT_error_rows_v4.csv')==r['GT_rows_csv_sha256']
    assert all(sha(Path(p))==h for p,h in plan['source_files_sha256'].items())
    assert max(abs(x) for x in r['metric_minus_reference_pp'].values())<=1e-8
    prior=A52/'terminal_A52_preservation_receipt.json';old=load(prior)
    assert sha(prior)=='b2bd6bca56e1b1a793ee86e3715d4e4c04ef5871789ed51b7feb1b55367eebfb'
    start=load(ROOT/'00_A53开始与历史保留.json')
    for p,h in old['current_files_sha256'].items():assert sha(Path(p))==h
    for p,v in start['preserved'].items():assert sha(Path(v['preserved_path']))==v['sha256']==sha(Path(p))
    assert load(ROOT/'v1_metric_extraction_failure.json')['COCO_evaluate_calls']==1
    assert load(ROOT/'v2_preexecution_path_defect.json')['CPU_evaluate_calls']==0
    assert load(ROOT/'v3_input_path_failure.json')['COCO_evaluate_calls']==0
    report=ROOT/'03_原版漏检证据与下一步筛选边界.txt';receipt=ROOT/'A53_缓存诊断保留回执.json'
    assert not report.exists() and not receipt.exists()
    s=r['small_GT'];miss=s['GT']-s['matched50'];c=s['category50_counts']
    lines=[
        'A53原版漏检证据与下一步筛选边界。总目标active；尚无新候选或已确认主线。',
        'Material Passport: academic-research-suite，定向前期调研与缓存分析；所有本轮文献阅读为助手阅读。未声明用户已读。',
        '',
        '1. 当前研究问题与检索记录',
        'A52固定12三门槛未全部通过，当前完整离散边际关系配方已淘汰。A53先问：原版固定12最终输出对小目标还有哪些几何、类别和覆盖缺口？此为描述性问题，不提前指定网络层根因。',
        '两次检索共8条查询，原始返回见01/02；范围为DETR小目标定位、粗细中心/尺度回归、连续表示、相位下采样及航拍定位。部分返回与任务无关，未作为证据。不是系统综述，也不以无命中作为新颖性证明。',
        '',
        '2. 数据与原始评价口径',
        'VisDrone val548；沿用A49核验的ECDet-S原版B第12轮EMA，38602个被计分GT、153796个已过滤预测。GT/预测已按原class0 union IoA口径处理；maxDets300，small为原图裁剪后GT面积≤1024，不是输入画布面积。非官方挑战分数。',
        '本地从precision/recall数组重算8项AP/AR指标，最大绝对差%.3gpp。原版AP%.9f，APs%.9f。没有重新运行模型。'%(max(abs(x) for x in r['metric_minus_reference_pp'].values()),r['metric_reproduction']['AP'],r['metric_reproduction']['APs']),
        '下列召回是IoU0.50或0.75下的类别感知、一对一匹配数量/GT数量；不等同于跨IoU平均的ARs，也不是AP。使用全部已保留预测，无新增置信度截断。',
        '',
        '3. 已证实的缓存事实与数据计算',
        '小目标%d个；IoU0.50匹配%d，召回%.3f%%；IoU0.75匹配%d，召回%.3f%%。'%(s['GT'],s['matched50'],s['recall50_percent'],s['matched75'],s['recall75_percent']),
        'IoU0.50未匹配的小目标%d个，按预先固定规则分为：'%miss]
    labels={
        'partial_geometric_coverage_0_1_to_0_5':'最终输出的最大任意类别IoU在[0.1,0.5)',
        'no_final_geometric_coverage_ge0_1':'没有任意类别IoU≥0.1的最终输出',
        'other_class_geometric_coverage50':'存在IoU≥0.5的其他类别框，但没有合格同类框',
        'correct_class_coverage_unmatched50':'存在IoU≥0.5的同类框，但未获COCO一对一匹配'}
    for key,text in labels.items():lines.append('%s：%d，占未匹配小目标%.3f%%。'%(text,c.get(key,0),c.get(key,0)*100/miss))
    lines+=['这些分类描述最终输出形态。第一项不排除同时错类；不能直接写成“58%的漏检由回归头造成”。最后一项也不等于全部query交互影响，仅描述已保留合格框的匹配竞争。',
        '按原图640stretch后的短边像素分组，仍只统计原图面积≤1024的小目标：']
    for key,label in [('lt2','<2像素'),('2to4','2–4像素'),('4to8','4–8像素'),('ge8','≥8像素')]:
        z=r['small_by_input_min_side'][key]
        lines.append('%s：GT%d，IoU0.50召回%.3f%%，IoU0.75召回%.3f%%。'%(label,z['GT'],z['recall50_percent'],z['recall75_percent']))
    lines+=['分组是描述性关联，类别、图像、遮挡等构成不同，不能把尺寸差异单独解释为下采样因果效应。',
        '',
        '4. 初筛核心文献与阅读范围',
        'SD-DETR，ICCV2023：当前只读CVF原文摘要。任务感知query生成、分类/定位空间解耦已有，不能把“定位专用分支”作为新的宽泛创新。https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Decoupled_DETR_Spatially_Disentangling_Localization_and_Classification_for_Improved_End-to-End_ICCV_2023_paper.html',
        'CFINet，ICCV2023：本轮仅检索页摘要片段/标题，未读方法全文。粗到细proposal与imitation已有，具体算子差异待原文核查。https://openaccess.thecvf.com/content/ICCV2023/html/Yuan_Small_Object_Detection_via_Coarse-to-fine_Proposal_Generation_and_Imitation_Learning_ICCV_2023_paper.html',
        'Feature Information Driven Position Gaussian Distribution Estimation for Tiny Object Detection，CVPR2025：当前原文摘要。信息图、位置高斯分布图监督和弱激活区域增强已有；不能把GT高斯辅助定位图当作新概念。https://openaccess.thecvf.com/content/CVPR2025/html/Bian_Feature_Information_Driven_Position_Gaussian_Distribution_Estimation_for_Tiny_Object_CVPR_2025_paper.html',
        'InvDet，CVPR2026：当前原文摘要。目标感知可逆编码和重建约束已有，但任务为红外小目标；没有由此证明RGB航拍收益。https://openaccess.thecvf.com/content/CVPR2026/html/Yan_Target-Aware_Invertible_Encoder_with_Reconstruction_Guidance_for_Infrared_Small_Target_CVPR_2026_paper.html',
        'HEDS-DETR，arXiv2507.00825：当前原文摘要。高频语义、小目标金字塔及几何位置编码组合已有；未读其完整算子或复现结果。https://arxiv.org/abs/2507.00825',
        '本次初筛尚不能提供相对最近方法的逐式完整比较；不得将摘要级查重包装为已确认研究空白。',
        '',
        '5. 历史排重与候选边界',
        'D007细尺度残差桥和D048目标感知Mosaic有旧阳性，保留；不当作被淘汰方向，也不重新包装成新模块。D011裁块有作用但约6.18视图，非等预算新算法。',
        'D024与A42已涉及几何/语义分离和定位专用读出；D025及撤回的A44涉及尺度归一化/读取分配；D069有网格相对中心与二分量几何备选。不能因本次看到定位缺口便把这些旧方案重新命名推荐。',
        'D061重复分数分摊不获旧缓存支持；D067特定两对错类解释范围有限；D071类别限额预检未过；D072标准NMS+重选有旧AP阳性但非创新、tiny TP不稳定。本轮未重跑这些方法。',
        'A52裁决只约束其精确关系配方；A53没有把它扩展为所有关系/分布表示均无效。',
        '',
        '6. 我们的推断、未证实部分与下一步',
        '[推断] 目前更有依据优先调查几何精度和最终输出覆盖缺口，而不是再凭信息接口空缺选择新的关系偏置。输入4–8像素组有明确的严格定位差距，值得作为机制设计的观察对象。',
        '[尚未证实] 最终覆盖不足发生在输入、encoder、proposal筛选、decoder、监督还是输出排序。现有缓存缺少query编号、全池、各层特征和FDR bins。当前只有有数据支持的问题，没有已成立的文献空白或新算法。',
        '[下一步] 从少数机制中给出具体操作，并完成旧编号/裁决、本次实质差异、新依据及强简单对照的对照；最近方法查重允许淘汰候选。合格后再冻结最小验证，单卡24GB、每组≤12轮；本轮未新分配GPU或训练。',
        '不要求先证明所有旧方法无效或完成全部因果链；当前诊断也不自动构成方法有效性或主线成立。',
        '',
        '7. 工程与计数的真实范围',
        'CPU COCO评价共2次：v1指标提取保护失败，v4完成。v1源码/协议/输出保留。v2准备脚本错误地替换了归档目录后缀，未执行；v3因嵌套输入路径错误在评价前退出。v4从原v1重建，仅替换输出名和原生precision/recall提取方式；数据、阈值、组别和1e−8容差未变。',
        '本阶段新增模型前向0、更新0、正式轮0。累计仍为185612整模型前向、155854正式更新、49容量更新、含容量155903更新、192正式轮；没有新训练队列。',
        '原始完整统计见error_audit_result_v4.json；逐GT描述见GT_error_rows_v4.csv；冻结协议及失败/修订记录均保留。']
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    board=load(BASE/'01_候选判别表.json');state=load(BASE/'15_当前状态与结果校验.json');now=datetime.now(timezone.utc).isoformat()
    entry={'state':'CACHED_BASELINE_AUDIT_COMPLETE_NO_CANDIDATE','result':str(ROOT/'error_audit_result_v4.json'),
        'report':str(report),'protocol':str(ROOT/'protocol_FIXED_v4.json'),'CPU_COCO_evaluate_calls_total':2,
        'new_model_forwards':0,'new_optimizer_updates':0,'new_formal_epochs':0,'additional_training_queued':False,
        'small_GT':s,'candidate_selected':False}
    status='A53_BASELINE_ERROR_EVIDENCE_COMPLETE_CANDIDATE_PENDING'
    question='从原版小目标几何精度与最终输出覆盖缺口筛选实质不同机制；尚无新候选。'
    shared={'status':status,'goal_status':'active','mainline_validated':False,'mainline_selected_for_pilot':False,
        'candidate_selected_for_pilot':False,'selection_stage_complete':False,'current_priority_candidate':None,
        'current_priority_question':question,'latest_report':str(report),'A53_execution':entry}
    board.update(**shared,latest_update_utc=now)
    state.update(**shared,recorded_utc=now,current_phase='A53_CACHED_ERROR_EVIDENCE',latest_authoritative_phase='A53_CPU_AUDIT_COMPLETE',
        model_forwards=185612,formal_detector_training_updates=155854,capacity_only_optimizer_updates=49,
        training_updates=155903,detector_optimizer_updates_including_capacity=155903,formal_training_epochs=192,
        counts_terminal=True,training_counts_terminal=True,counters_observed_lower_bounds_while_job_live=False,current_native_job_terminal=True)
    save(BASE/'01_候选判别表.json',board);save(BASE/'15_当前状态与结果校验.json',state)
    signed=[BASE/'01_候选判别表.json',BASE/'15_当前状态与结果校验.json',report,ROOT/'error_audit_result_v4.json',
        ROOT/'protocol_FIXED_v4.json',ROOT/'01_初次主来源检索原始记录.json',ROOT/'02_定位与编码机制增补检索原始记录.json']
    save(receipt,{'status':'PASS','phase':'A53_CACHED_BASELINE_AUDIT','goal_status':'active','mainline_confirmed':False,
        'previous_receipt':str(prior),'previous_receipt_sha256':sha(prior),'previous_signed_current_inputs_resolved':start['preserved'],
        'current_files_sha256':{str(p):sha(p) for p in signed},'model_forward_operations_added':0,'optimizer_operations_added':0,
        'CPU_COCO_evaluate_calls_total':2,'training_queued':False,'cumulative_counts':plan['cumulative_counts_unchanged']})
    save(ROOT/'CURRENT_A53.json',{'status':status,'goal_status':'active','mainline_confirmed':False,'current_priority_candidate':None,
        'receipt':str(receipt),'receipt_sha256':sha(receipt),'latest_report':str(report),'owned_jobs_live':False,
        'bridge_session_if_alive':None,'next':'Bounded operator-level old/new-method comparison, then select a concrete candidate before allocating any GPU validation.'})
    print(json.dumps({'status':status,'small_GT':s,'CPU_evaluate_calls':2,'new_model_operations':0,'receipt_sha256':sha(receipt)},ensure_ascii=False))
if __name__=='__main__':main()
