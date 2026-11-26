from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/assignment_mechanism_diagnostic_20261010_v1'
ANALYSIS=OUT/'local_analysis_verified_v2'
STATE=ROOT/'outputs/sod_roi12_20261009/execution_state.json'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
verification=read(ANALYSIS/'verification.json');assert verification['status'].startswith('PASS_')
summaries=read(ANALYSIS/'summaries.json');decision=read(ANALYSIS/'decision.json')
download=read(OUT/'download_verification.json')
now=datetime.now(timezone.utc).isoformat()
fallacies={
 '1_Simpson':'Both seeds, both frozen endpoints and all stage/pair rows retained separately. Mature and early averages are not combined into an effect claim.',
 '2_Ecological':'Exact query/GT identity and per-query leaf gradients checked against original annotations; group summaries are not generalized to network parameter updates.',
 '3_Berkson':'Protected correct-class IoU75 subset is explicitly conditioned; its frequencies cannot describe the whole miss population.',
 '4_Collider':'Conditioning on prediction quality/selection may couple errors and matching. No conditional association is claimed to prove historical training damage.',
 '5_Base_rates':'All2712 GT, small1910 GT per panel, own small pairs and protected487/515 mature small matches are distinguished from131/141 assignment conflicts and1/2 protected opposed cases.',
 '6_Regression_to_mean':'Frozen same-array target counterfactual; no pre/post learning gain, noAP improvement or independent generalization claim.',
 '7_Survivorship':'Both complete zero-retry remote runs, all128 saved samples and positive/negative directions retained. Transport/preparation/analytic verifier failures disclosed; no model rerun.',
 '8_Look_elsewhere':'Prior exploratory validation and many earlier closed routes disclosed. No significance tests, no fresh-holdout claim and no selected favorable-only endpoint.',
 '9_Forking_paths':'Cohort and decision boundary frozen before execution. Evidence-completeness revision done before first run; analytic precision branch repair documented without changing cohort, data, thresholds or tolerance.',
 '10_Correlation_causation':'Leaf-box loss directional conflict verified, not model-parameter conflict, leading root cause or detector AP causality. Native union intent respected.',
 '11_Reverse_causality':'Final frozen EMA in train mode with direct-resized validation targets is not actual augmented optimization history; no retrospective learning-cause inference.'
}
validation=dict(material_passport={'skill':'academic-research-suite','workflow':'experiment-agent','mode':'run+validate','verification_status':'ANALYZED_SCIENTIFIC_SCOPE_WITH_VERIFIED_CAPTURE_IDENTITY_AND_SAVED_ARRAY_RECOMPUTATION'},observed_utc=now,statistics='Descriptive counts and local counterfactual loss derivatives; no inferential statistical tests.',fallacy_coverage={'checked':11,'total':11,'items':fallacies},engineering_verification=verification,scientific_limits=decision)
write(OUT/'validation_report.json',validation)
delivery=dict(status='COMPLETED_DUAL_SEED_FROZEN_NATIVE_SUPERVISION_DIAGNOSTIC',observed_utc=now,material_passport=validation['material_passport'],protocol=str(OUT/'protocol_execution_v2.json'),code_root=str(Path(__file__).parent),capture_code_sha256='e5e55e1d2282e5b8b68ee7315aba063d38ed6b1d5c3fcfc7d40e7ff251918b64',collection=download,analysis=str(ANALYSIS/'verification.json'),summaries=summaries,decision=decision,model_forward_calls_total=132,model_backward_calls_total=0,native_leaf_box_gradient_evaluations_total=384,training_epochs_added=0,optimizer_steps=0,root_cause_identified=False,method_AP_evidence=False,training_launched=False,failures_preserved=[{'phase':'local preparation','error':'Payload builder initially lacked a local copy of training_evidence.py; resolved dependency before remote execution, no model run.'},{'phase':'SSH download','error':'Strict first-chunk parser rejected terminal-added whitespace. No chunk accepted until normalized encoded length matched; assembled archive exact SHA verified. No scientific rerun.'},{'phase':'independent NumPy verifier','error':'FP64 primal branch separated native FP32 equal edges at q64. Preserved original attempt and failure; native FP32 primal values with FP64 derivative propagation pass unchanged tolerances; maxgradient error2.9933e-6.'}],query_capacity_precheck=str(OUT/'query_capacity_precheck.json'),nearest_prior_assessment={'D_FINE':'Existing local publisher105 PDF fulltext + official OpenReview corroborates localization union vs classification O2O as intended. Majority source and snapshots inspected. The conflict is a consequence requiring benefit testing, not a demonstrated implementation bug.','DEIM':'Existing local CVPR2025 PDF14 and official CVF supports matching/MAL already central to baseline. Replacing assignments alone is not proof of a substantive novel mechanism.','excluded_search_hits':['Sim-DETR ICCV2025 is temporal sentence grounding, not object detection; not used as applicable evidence.','SA-Matching CVPR2026 Findings is not a CVPR main-conference paper; not promoted into the main-conference evidence corpus.','Mr. DETR CVPR2025 publisher page request403; fulltext unavailable through this route, not pursued or claimed read.']},next_action='Observe actual native post-Mosaic/MixUp target coverage and small-object losses on a predeclared bounded panel. Existing trace shows training labels can exceed300 while final O2O query capacity300; establish who loses supervision before considering a training-query mechanism. No new detector training screen launched; closed routes remain closed.',overall_goal_status='active')
write(OUT/'delivery.json',delivery)
lines=['原生分类/回归配对冲突：双seed核验结果','', '本轮采集和本地核验已完成；新增训练0轮、优化器更新0次。','固定32张验证图，两个种子、BASE12与BASE132四套权重；每套2712个GT，其中COCO-small1910。','每个种子66次前向、192次独立叶子框梯度计算，均正常退出0，无自动重试。','记录的是冻结EMA在train模式下的原生前向与监督，不是实际数据增强学习历史，也不是AP评价。','', '数据：seed / 端点 / 配对冲突 / 跨类别冲突 / 已正确小目标框 / 其中反向牵引框 / 比例']
for s in summaries:
    lines.append(f"{s['seed']} / {s['endpoint']} / {s['GT_conflicts']} / {s['cross_class_conflicts']} / {s['strict_correct_small_denominator']} / {s['gradient_opposed_strict_correct_small_conflicts']} / {100*s['protected_small_opposed_fraction']:.4f}%")
lines.extend(['','“已正确”指实际分类匹配目标类别为argmax且框IoU≥0.75；不等同于标准验证TP/AP。','同一查询被最终分类匹配和全局回归union指向不同GT确实存在。D-FINE原本就设计了回归union、分类一对一，不能称作程序错误。','成熟两seed中保护框的反向牵引只有1/487与2/515。预先约定的“成熟双seed均出现”边界达到，但现有比例不支持主要瓶颈或AP收益结论。','', '把所有冲突直接改为最终层GT的风险：'] )
for s in summaries:
    if s['endpoint']==132:
        lines.append(f"seed{s['seed']}：最终层own目标比union目标平均IoU差 {s['mean_own_minus_union_IoU']:+.6f}；own目标IoU更好{s['own_target_higher_IoU_conflicts']}例、更差{s['own_target_lower_IoU_conflicts']}例；改目标后的加权L1+GIoU均值变化{s['anchored_minus_union_bbox_loss_mean']:+.6f}。")
lines.extend(['改标注后的不同目标损失不能作为方法优劣或收益估计；上述混合方向说明最终层匹配并非普遍更可靠。','暂不将“全部跟随最终层”作为已成立的解决方案；需要真实训练收益/损伤验证。','', '独立核验：128份样本、14848个保存数组，384组叶子梯度解析复算；最大损失误差1.2301e-6，最大梯度误差2.9933e-6。','原生union的每个目标均为该查询跨层配对的最高票目标；所有未换目标的查询梯度完全相等。','两份归档分别6049027/6049963字节；每份328个不可变文件及3个原生源码快照SHA通过。','', '复盘中的另一个线索：现有原生增强64图里，MixUp之前已有4图超过300个GT，最大706；这4图的数量溢出合计504。原生正常查询数为300。','全验证548图仅3图超过300GT，数量溢出34。因此不能直接用验证集查询数量上限解释整体小目标AP；更应检查训练增强后实际匹配覆盖。','64图只是既有单seed固定几何面板，且此统计在batch MixUp之前；不能把它当全训练分布，也不能认为DN/辅助监督完全缺失。','', '下一步：用记录器核对Mosaic/MixUp之后哪些小目标在最终一对一、回归union与DN里得到/丢失监督，确认是否值得改变训练查询机制。','', '重要工程修复保留：本地依赖缺文件在运行前修复；SSH终端空白导致的编码校验拒收后最终归档SHA正确；NumPy初次验证误用FP64前向分支，现已匹配原生FP32边界并保留失败记录。没有重跑模型或改宽验证容差。','', '完整索引：delivery.json；validation_report.json；download_verification.json；local_analysis_verified_v2/；query_capacity_precheck.json。'])
(OUT/'结论与证据索引.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
state=read(STATE)
state['status']='PROBLEM_DRIVEN_RESEARCH_ACTIVE_NATIVE_ASSIGNMENT_DIAGNOSTIC_VERIFIED'
state['goal_status']='active'
state['previous_goal_turn_classification']='PROGRESS: HTML template entry fixed and browser-verified; full native graph artifact delivered.'
state['current_goal_turn_classification']='PROGRESS: actual dualseed frozen supervision captures completed, downloaded and independently analyzed; rare protected conflicts and mixed all-conflict geometry change the next action.'
state['assignment_mechanism_diagnostic']=delivery
state['research_iteration_history'].append(dict(utc=now,status=delivery['status'],evidence=str(OUT/'delivery.json'),detector_epochs_added=0,optimizer_steps=0,root_cause_identified=False,next_action=delivery['next_action']))
write(STATE,state)
with STATE.with_name('continuation_state.txt').open('a',encoding='utf-8') as f:
    f.write(f'\n\n{now} PROGRESS: Dualseed actual native assignment panel completed exit0/no retry,32fixed images each BASE12/132,2712GT/1910small;132total forwards384leaf gradients0model backward/optimizer/epochs. Both archives/all656immutablefiles verified,128samples/14848arrays/384analyticgradientsets verified, original NumPy FP64 tie-branch failure preserved and repaired with native FP32 primal, unchanged tolerance, maxgradient error2.9933e-6. Mature conflicts141/131; protected correctclassIoU75small487/515, opposed1/2 (0.2053%/0.3883%); no leading-root/AP claim. Full own-anchor conflict target meanIoU relativeunion -0.037292/-0.036656, directions mixed, no candidate training launched. Existing native64 augmented samples preMixUp count8011 GT,4images>300 max706 excess504; fullval548 only3images>300 excess34, no inference AP conclusion. Next inspect actual postMixUp normal/union/DN supervision coverage before any query mechanism training. Code and reports outputs/assignment_mechanism_diagnostic_20261010_v1. All recorded diagnostic children/controllers terminal; shell50937/75320 retained; no old-route restart. Goalactive.\n')
print(json.dumps({'status':delivery['status'],'goal_status':'active','fallacy_coverage':'11/11','report':str(OUT/'结论与证据索引.txt')},ensure_ascii=False))
