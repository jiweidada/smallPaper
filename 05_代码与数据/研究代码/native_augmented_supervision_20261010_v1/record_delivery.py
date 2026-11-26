from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json

root=Path(__file__).resolve().parents[3]
out=root/'outputs/native_augmented_supervision_20261010_v1'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
now=datetime.now(timezone.utc).isoformat()
download=read(out/'download_verification.json');analysis=read(out/'local_analysis_verified/verification.json');summary=read(out/'local_analysis_verified/summary.json');decision=read(out/'local_analysis_verified/decision.json')
assert download['status']=='PASS_DUAL_SEED_ARCHIVES_AND_ALL_CAPTURE_FILES'
assert analysis['samples']==64 and analysis['composite_images']==256 and analysis['normal_batch_head_losses_verified']==320
assert decision['DN_covers_every_saved_GT'] and decision['predeclared_mature_both_seed_small_no_normal_ge10percent'] is False
next_action='Use already archived native normal outputs/loss assignments to examine what supervision does to matched <4px targets in unsaturated composites (GT<=300), separating classification-quality calibration from localization and matched index coverage. Freeze the comparison and verify native loss values/derivatives before any candidate. Query capacity remains lower priority under the unchanged both-seed gate; do not enlarge this cohort or redefine the denominator to manufacture a pass. No new detector training launched.'
fallacies={
 '1_Simpson':'Both endpoints/seeds retained separately; MixUp, density, input-size and class strata reported. Seed1 and seed2 augmentation draws are not pooled into an improvement claim.',
 '2_Ecological':'Unit is an augmented target-row exposure, source ID retained. Distinct source annotations counted separately; no network-parameter or population inference from row statistics.',
 '3_Berkson':'Primary positions outcome-independent; post-augmentation survival and original-small conditioning are explicit. Surviving rows cannot describe cropped-away objects or whole training prevalence.',
 '4_Collider':'Dense/selected/matched subsets can couple size, augmentation and prediction quality. Stratum associations do not prove that size or MixUp caused the misses.',
 '5_Base_rates':'All target exposures, original area<1024, GT>300 images, any-normal and DN coverage separately shown. Actual stage-index inclusion is not effective positive gradient or AP TP.',
 '6_Regression_to_mean':'Frozen endpoints on exact same within-seed augmented data; no learned intervention, improvement or AP effect claimed.',
 '7_Survivorship':'Both one-shot runs completed exit0; all64 recorded batches and134 immutable files perseed verified. No retry, sample removal or replacement.',
 '8_Look_elsewhere':'Earlier explored routes disclosed by authoritative state. All original bin/class/seed strata retained; no inferential tests or favorable subgroup promoted into the primary decision.',
 '9_Forking_paths':'Both-seed10percent boundary and64primary cohort frozen before execution. Boundary not met, remains false. Exactly1024 area rows=0; strict area<1024 follows explicit protocol. No extra sampling to change outcome.',
 '10_Correlation_causation':'Query capacity establishes final O2O count ceiling, not an optimizer-history cause or AP bottleneck. DN covers all rows; auxiliary and DN paths are not equivalent to ordinary deployed queries.',
 '11_Reverse_causality':'Late frozen EMA outputs on reconstructed epoch4 augmentations are not historical epoch4 learning dynamics. Prediction-dependent matching may reflect model weakness as well as supervision selection.'}
validation={'material_passport':{'skill':'academic-research-suite','workflow':'experiment-agent','mode':'run+validate','verification_status':'ANALYZED_SCIENTIFIC_SCOPE_WITH_EXACT_CAPTURE_PARITY_AND_INDEPENDENT_SAVED_ARRAY_CHECKS'},'observed_utc':now,'statistics':'Descriptive augmented-row counts; no p-values, confidence intervals or AP inference. Native Hungarian optimality not independently solved; actual indices, own-loss equality and majority-union/DN construction checked.','fallacy_coverage':{'checked':11,'total':11,'items':fallacies},'engineering_verification':analysis,'decision':decision,'limits':['RNG fixtures are single-process and outcome-independent, not the actual four-worker historical stream.','Mosaic/MixUp exposure counts and augmentation draws differ substantially byseed; no paired augmentation effect estimate.','Direct losses are evaluated in CUDA FP32 without AMP/backward/optimizer; no parameter-update effect.','A matched loss index is not sufficient proof of a gradient pulling the GT class score upward.','DN supervised all GT but uses noisy/isolated queries; complete DN coverage does not disprove a possible ordinary-query training limitation.']}
write(out/'validation_report.json',validation)
delivery={'status':'COMPLETE_NATIVE_AUGMENTED_SUPERVISION_DIAGNOSTIC_VERIFIED','observed_utc':now,'code':str(root/'英文小论文/code/native_augmented_supervision_20261010_v1'),'protocol':str(out/'protocol.json'),'download_verification':download,'local_analysis':analysis,'endpoint_totals':summary['endpoint_totals'],'augmentation':summary['augmentation'],'decision':decision,'forward_calls_total':68,'model_backward_calls':0,'optimizer_steps':0,'training_epochs_added':0,'historical_training_reproduced':False,'actual_matcher_optimality_independently_solved':False,'root_cause_established':False,'AP_effect':False,'new_detector_training_launched':False,'next_action':next_action,'overall_goal_status':'active','last_remote_check':{'observed_utc':'2026-10-10T08:46:14+00:00','both_gpus_idle_2MiB_zero_percent':True,'both_capture_and_controller_handles_absent':True},'scope_note':'GT supervision coverage means actual target inclusion in native loss calls, not positive gradient quality or detection AP.'}
write(out/'delivery.json',delivery)
state_path=root/'outputs/sod_roi12_20261009/execution_state.json';state=read(state_path)
state['status']='PROBLEM_DRIVEN_RESEARCH_ACTIVE_NATIVE_AUGMENTED_SUPERVISION_VERIFIED'
state['augmented_supervision_diagnostic']=delivery
state['current_research_iteration']={'status':delivery['status'],'evidence':str(out/'delivery.json'),'mechanism_assessment':decision,'training_epochs_added':0,'optimizer_steps':0,'next_action':next_action}
state['last_goal_turn_classification']='PROGRESS: completed both one-shot native augmented-data/model supervision runs;68batch forwards, all268 capture files SHAverified; independently recomputed256composite images,21056arrays,320normal box losses. DN covers every target; both-seed10percent normal-gap boundary false, so capacity not promoted to training.'
state['next']=next_action;state['updated_utc']=now
state['research_iteration_history'].append({'utc':now,'status':delivery['status'],'evidence':str(out/'delivery.json'),'detector_epochs_added':0,'optimizer_steps':0})
write(state_path,state)
with state_path.with_name('continuation_state.txt').open('a',encoding='utf8') as f:
 f.write(f'\n{now} NATIVE AUGMENTED SUPERVISION COMPLETE VERIFIED: Both servers idle/terminal. 64primary images and16native batches perseed at augmentation epoch4, BASE12/132 frozen EMA,68batch forwards total/0backward/0epochs/0optimizer. Mature normal-small supervision gaps1854/7420=24.9865% vs177/4246=4.1686%; DN coversall12162/7507 exposures, no-any-supervision0. Predeclared BOTHseed10% gatefalse, no query-capacity training promotion, no threshold/cohort revision. Local saved arrays/source IDs/majority union/DN index construction/320normal box losses verified; maximum loss error2.5253e-7. Prior UI and scientific histories preserved. Overall goalactive. Next: {next_action}\n')
lines=['原生增强后的监督覆盖诊断：结果与证据','', '每个seed固定64张训练主图、16个batch4；增强epoch索引4。','原生Mosaic/MixUp/多尺度处理后，用BASE12与BASE132冻结EMA执行训练分支前向和原生损失。','这不是实际优化器历史，没有新的检测器训练或AP结果。','']
for r in summary['endpoint_totals']:
 lines.append(f"seed{r['seed']} BASE{r['endpoint']}：目标曝光{r['GT_rows']}，不同原始标注{r['distinct_source_annotations']}；原始面积<1024曝光{r['original_COCO_small_rows']}；小目标无任何普通分支监督{r['small_no_normal_supervision']}（{100*r['small_no_normal_fraction']:.4f}%）；去噪分类/回归覆盖全部曝光。")
lines+=['','监督指实际损失调用包含该GT行，不等于有效正向梯度；最终普通查询、所有普通辅助头、回归联合匹配、DN分开计算。','超过300个GT的组合图：seed1为16/64，seed2为5/64；最大GT866/389。MixUp批次数6/16与2/16。','“小目标”在本诊断中按原始标注面积<1024，恰好1024行数0；另有真实输入短边尺寸分层。','预先规定两组成熟权重的小目标普通监督缺口均≥10%才提升查询容量方向优先级；结果不满足，未启动该配置。','全部GT都进入DN监督，否定“超过300就完全没有任何监督”的强断言；DN覆盖不能证明与普通查询学习等效。','','核验：两份归档与268个不可变采集文件SHA通过，256个组合图、21056个数组、320个普通头边框损失重新计算通过，最大误差2.5253e-7。','记录器与普通执行的输入、输出、损失、私有BN更新、随机状态完全一致；源码和参数未变。','总计68次batch前向，0次model backward，0次optimizer，新增训练0轮。','','证据：protocol.json；download_verification.json；local_analysis_verified/summary.json、decision.json、verification.json；per_augmented_GT_coverage.csv；validation_report.json；delivery.json。','科研统计11/11误区扫描已完成，结果为描述性观察，不作统计显著性/根因/AP因果结论。','下一步使用既有数组分析GT≤300图中已匹配极小目标的分类质量目标和定位信号；先核实实际损失含义，再形成候选。研究目标保持进行中。']
(out/'结论与证据索引.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps({'status':delivery['status'],'mature_small_normal_gap_fractions':decision['measured_fractions'],'capacity_gate':False,'forward_calls':68,'epochs_added':0,'goal_status':state['goal_status']}))
