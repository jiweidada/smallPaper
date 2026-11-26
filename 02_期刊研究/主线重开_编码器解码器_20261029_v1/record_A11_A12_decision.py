"""Reconcile the completed source/math audit and bounded native validation screen."""
from pathlib import Path
from datetime import datetime,timezone
import copy
import hashlib
import json
import shutil
import tarfile

ROOT=Path(__file__).resolve().parent
A11=ROOT/'A11_坐标插值相位与分布扩散审计_v1'
A12=ROOT/'A12_原生边界分布诊断_v1'
NOW=datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
math=read(A11/'summary.json');result=read(A12/'analysis_summary.json');run=read(A12/'summary.json');verify=read(A12/'independent_verification.json')
assert math['status']=='PASS_SOURCE_INTERPOLATION_AND_ANALYTICAL_GRADIENT_AUDIT'
assert run['new_full_model_forwards']==13 and run['cumulative_full_model_forwards']==213
assert result['decision']=='DEPRIORITIZE_PHASE_CORRECTION_AS_CURRENT_MAINLINE_NO_MODULE_TRAINING'
assert result['enough_comparable_data'] and verify['different_raw_vs_corrected_rankings']==0
report=ROOT/'34_A11A12分布解码候选与原生验证结论.txt'
report.write_text('''当前结论：插值相位校正未提供优于普通方差的实际误差信号，本候选降级。尚未找到通过机制与新颖性核查的小论文主线。

1. 本轮研究问题与检索
检查“坐标分布的扩散是否混合了插值编码本身与定位误差，进而影响微小目标的解码反馈”。这是新候选假设，不是已确认的原生Dome故障。检索为定向非系统调研；原样检索输入、返回片段、访问失败和阅读范围见13记录及source_audit_v1/A11_*.json。
初始还检查了联合/多峰框回归和均值转众数。2019/2020年的Deep Mixture Density Network已有完整协方差、多峰框与条件most-probable推理，因此这些一般思路不能当新创新。该工作旧标题与新版属于同一arXiv1911.10614家族，不重复计为两篇独立方法。OpenReview访问要求浏览器验证，改读作者arXiv原文，没有绕过访问检查。检索中出现的第三方AI摘要不作为方法证据。

2. 与最近工作的逐项对比及阅读范围
GFL（NeurIPS2020）：相邻两bin软标签加权CE，以分布期望还原连续坐标。已读原始DFL方法片段及前述期望公式；未整篇/附录通读。
GFLV2（CVPR2021）：DGQP用Top-k概率及这些Top-k数值的均值估计定位质量；已读原文3.2完整DGQP方法、3.1期望读出公式和4.1统计消融片段。这里的mean是Top-k概率均值，不是bin坐标期望。未执行GFLV2作者代码。
D-FINE（ICLR2025）：非均匀有符号坐标bin、FGL插值监督和逐层分布细化已有先例。已读arXiv v1的FDR/FGL方法及出版商相应公式片段，源码对照使用固定版本Dome中的D-FINE派生实现；没有执行原版D-FINE权重。
UGS（ICCV2025）：原始熵最小化与熵梯度引导特征扰动已有直接近邻。完整3.2方法已读，未获得并复现作者训练源码。本轮独立用softmax链式求导及自动微分核查数学性质，不采用页面梯度公式作为证明，也不据此否定整个UGS框架的实验结论。
Deep Mixture Density Network（arXiv v2，作者注明IROS2020oral）：完整III-A学习/推理及III-B部分已读；完整协方差、多模态和条件most-probable读出均为已有方法。会议DOI未独立核实，引用可核验arXiv版本。

3. A11源码和数学核查：成立的仅是表示性质
固定Dome源码2dde3bc1946a3e9fad9abd0612b59fc39bd6b861及作者权重。实际checkpoint的up=0.5、reg_scale=4，33个非均匀bin。Integral对每条边作softmax及坐标加权期望，LQE使用Top-4概率及其均值；原生Dome没有UGS的UM/UR。
令相邻坐标a<b，目标y=(1-t)a+tb，则理想两点分布p=(1-t,t)能精确还原y，但熵为-t log(t)-(1-t)log(1-t)，方差为t(1-t)(b-a)^2，均随插值相位t变化。对任意相同离散网格上的分布，若均值mu位于[a,b]，则Var(p)>=(mu-a)(b-mu)。证明来自网格点满足(x-a)(x-b)>=0后的期望不等式。这是基础表示/凸性性质，不宣称新定理。
候选量delta=Var(p)-(mu-a)(b-mu)只表示超出相邻插值所需的额外扩散，不能直接称为校准后的预测不确定性。直接做“熵减理想熵”甚至不保证非负，已保存反例。
作者目标生成函数的32区间x3相位共96个案例通过，连续相对边界偏移还原最大误差2.38e-8。64个非中点案例中，单独沿原始熵对独立logits的下降方向会偏离理想坐标；这不是完整网络的实际训练轨迹。128个固定合成分布通过下界数值检查，无模型前向或拟合。

4. A12最小验证协议：先冻结，再执行
数据为本地官方VisDrone2019-DET-val，548图中按标注筛选，取12个不同前缀的12图，共263个sqrt面积<8像素（800方形输入）的微小GT。没有按预测结果挑图。此前D0的dev300来自官方train拆分，不能称为官方val。本轮官方val仍不称盲测，也不声称作者选权重时从未用过它。
按原生Dome-DETR-S配方做1次无钩子身份对照和12次观测，总13次完整前向，无warmup、结构改动或训练。此为新问题的独立有界观察阶段，之前200次阶段保持结束，没有重跑旧DEIM或压缩任务。
以作者固定Hungarian成本做诊断配对：class2、bbox5、GIoU2，focal=True、alpha0.25、gamma2；非正宽高query排除并记录。只用最终层微小GT边界，排除目标超出原生bin支持的边。比较同图、同GT类别、同边、同预测bin区间且物理bin宽度相差不超过2倍的好边（误差<=0.5像素）与坏边（>=1像素）。
主判据是额外扩散相对普通物理方差的成对区分能力增量，按图平均、按前缀bootstrap。预设至少8图/8前缀、120微小GT、200对；平均增量>=0.10且区间下界>0，并且不劣于熵的次要比较，才保留候选。门槛为资源筛选规则，不是论文显著性或AP标准。

5. A12实际结果及可核验范围
身份对照raw boxes/logits差0；模型state digest前后相同。作者distance2bbox重建门槛通过。263个微小GT的1052条边中，29条目标偏移超出原生支持，主分析保留1023条边；12图均有可比对，共270对，满足预设数据量要求。
额外扩散与普通方差在270对上的排序逐项相同。主增量0.000，图级bootstrap区间[0.000,0.000]，未过门槛。相对熵的次要图均差为+0.0501，但普通方差已达到同一结果，不能把它算作相位校正的新增收益。
29/1052条边越界是观察到的次要描述，不证明越界是主要漏检原因，不据此重启旧范围/残差候选。没有更换阈值、挑选子集或追加训练寻找正结果。
原始12份预测及29个归档文件已下载核验。独立以二阶矩/凸包插值公式检查2046个物理方差数值、1023条边和全部270对的资格与排序；原始logit重建source均值最大差3.19e-7。此核验不代表独立数据复现或因果验证。

6. 研究空白、可能增量与可行性判断
能确认的是表示的理想扩散与定位误差不能简单等同；但本轮实际观察不支持将相位修正作为优先算法增量。当前没有有证据支撑的“独有新模块”或可立项主线。通用方差head、联合高斯/多峰、均值改众数、置信度校准、熵正则和更密坐标bin均有直接先例，不能重新包装。
本候选不进入训练。后续使用已有逐层轨迹继续检查坐标更新与特征读取机制，并保持新视觉编码器方向开放；新的结构干预需要独立失败证据、最近算子对照和有限协议。单卡24GB已支持当前冻结推理；A12自身allocated峰值0.723GiB，不是训练显存、原生速度或多日训练可行性的证明。
架构分支累计213次完整前向（TinyFormer185、Dome28），主模型更新0；仍为64个诊断线性模型/96个标量响应头。合成自动微分与参数无关匹配不计为拟合。A12已终止exit0，无在跑的新训练，目标保持ACTIVE。
服务器与本地存在时钟偏移，保留各自产生的时间；先后顺序以冻结清单、传输回执和单次运行锁核验，不按跨机时间戳推断。

7. 可核验原始参考
GFL：https://proceedings.neurips.cc/paper/2020/file/f0bda020d2470f2e74990a07a607ebd9-Paper.pdf
GFLV2：https://arxiv.org/html/2011.12885
D-FINE：https://arxiv.org/html/2410.13842
UGS：https://arxiv.org/html/2303.01803v2
Deep Mixture Density Network：https://arxiv.org/html/1911.10614v2
An Uncertainty-aware DETR Enhancement Framework：https://arxiv.org/html/2507.14855v1
作者Dome代码：https://github.com/RicePasteM/Dome-DETR/tree/2dde3bc1946a3e9fad9abd0612b59fc39bd6b861
''',encoding='utf-8')

# Compact self-contained A11 evidence archive; A12 archive is independently downloaded/verified.
backup11=A11/'backup';assert not backup11.exists();backup11.mkdir()
files11=[p for p in A11.iterdir() if p.is_file()]+[ROOT/'a11_coordinate_phase_math.py',ROOT/'prepare_A11_math.py']
archive11=backup11/'A11_source_math_audit.tar.gz'
with tarfile.open(archive11,'w:gz') as tar:
    for path in files11:tar.add(path,arcname=('code/'+path.name) if path.parent==ROOT else path.name)
receipt11={'archive':str(archive11),'sha256':sha(archive11),'bytes':archive11.stat().st_size,'files':len(files11)}
write(backup11/'receipt.json',receipt11)
receipt12=read(A12/'backup'/'receipt.json')
rawfiles=sorted((ROOT/'source_audit_v1').glob('A11_*.json'))
source_audit={'utc':NOW,'source_commit':'2dde3bc1946a3e9fad9abd0612b59fc39bd6b861','A11_source_functions':['weighting_function','translate_gt'],'A11_math':str(A11/'summary.json'),'A12_native_capture':str(A12/'summary.json'),'A12_protocol':str(A12/'protocol.json'),'A12_verification':str(A12/'independent_verification.json'),'source_observations':['Integral independently normalizes four33-bin vectors and returns their coordinate expectations.','LQE takesTop4 probabilities andtheir mean,without bin-coordinate inputs.','NativeDome doesnot containUGS entropyUM/UR;syntheticentropytradeoff isnotanobservednativebug.','FDR uses fixed first-layerpre_boxes as regression reference and updated boxes as next-layer sampling reference;intentional source design,not automatically an error.'],'decision':result['decision'],'no_algorithm_approved':True,'no_new_training':True}
write(A11/'source_operator_and_decision.json',source_audit)

names=['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json']
history=ROOT/'history_before_A11_A12_update_v1';assert not history.exists();history.mkdir()
for name in names:shutil.copy2(ROOT/name,history/name)
records={name:read(ROOT/name) for name in names}
board=records[names[0]]
board.update({'latest_update_utc':NOW,'latest_report':str(report),'status':'NO_CANDIDATE_PASSED_MECHANISM_AND_NOVELTY_GATES_AFTER_A12','current_priority_question':'实际特征读取与几何坐标更新的哪项机制限制微小框定位？A12未支持插值相位修正优于普通方差，继续检查已有逐层轨迹及新的视觉编码器证据。','model_candidates':['TinyFormer-S-PBM:185 historical frozen diagnostic forwards','Focus-DETR:source-only label geometry;no full author model run','Dome-DETR-S:strict author state;28 frozen forwards acrossA8/A12;final paper baseline unselected'],'new_empirical_evidence':'213 complete detector forwards;A11 parameter-free math;A12=13 native passes on12 official-val images263tinyGT. Corrected-versus-raw-variance rankings identical on270 pairs. Main model updates0;64 fitted diagnostic models/96 scalar heads.','remaining_deliverable':'A specific architecture-dependent failure and a distinct algorithm increment with a defensible bounded validation route;independent generalization and training resources unresolved.','next_decision_protocol':'A12 phase13/13 complete. No phase-correction module/training or automatic additional forwards. Use cached trajectories and closest-source comparison before selecting another structural intervention.'})
board['candidates'].append({'id':'H_INTERPOLATION_PHASE_AND_DISTRIBUTION_SPREAD','question':'Does intrinsic coordinate interpolation confound useful localization feedback from distribution spread?','status':result['decision'],'theory_status':'Source/math property verified;basic identity,notnewtheorem.','real_data_status':'12 official-val images263tinyGT;1023 eligibleedges270 comparablepairs;corrected andrawvariance orderingidentical.','increment_approved':False,'causal_evidence':False,'nearest':['GFL','GFLV2','D-FINE','UGS','Deep Mixture Density Network','Uncertainty-aware DETR'],'next':'No new uncertainty/refinement module or training fromthisscreen;no threshold/subsetsearch.'})
board['A11_A12_evidence']={'A11':receipt11,'A12_archive':receipt12,'verification':verify,'cumulative_forwards':213,'bounded_phase_forwards':13,'decision':result['decision']}
board['rejected_generic_novelty_claims'].extend(['Generic full covariance/multimodal box head ormean-to-probable inference ispriorart inDeepMDN.','Discrete coordinatebins alone do not impose an idealinterpolation quantization ceiling.','The phase-corrected spread identity isnot a newtheorem,andA12did not show addedrankingutility.'])

log=records[names[1]]
log['supplement_utc']=NOW
log['A11_A12_source_math_and_native_screen']={'raw_primary_requests':[{'file':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in rawfiles],'exact_inputs_archived':True,'literature_scope':'Targeted nonsystematic. SecondaryAIgeneratedsummariesnotmethod evidence. MMG titleversionresolvedtoarXiv1911.10614v2.','limited_access':['OpenReview human verification:didnotbypass;authorarXivalternativeused.','QCAR/Look-and-Confirm fullmethodsremainunavailable;readscopeunchanged.','DSDL locatedasclosegenericregressionmetadataonly,notfullmethodcleared.'],'source_audit':source_audit,'A12_protocol':str(A12/'protocol.json'),'A12_archive':receipt12,'report':str(report)}
log['query_groups'].append({'name':'A11 coordinate representation anddistributionuncertainty','requests':[{'file':str(p.relative_to(ROOT)),'request':read(p)['request']} for p in rawfiles]})

matrix=records[names[2]]
additions=[
{'id':'GFL_V1','title':'Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection','url':'https://proceedings.neurips.cc/paper/2020/file/f0bda020d2470f2e74990a07a607ebd9-Paper.pdf','tier':'NeurIPS2020','read_scope':'Primaryabstract/introduction andDFL method/expectation excerpts;notentirepaperorappendix.','author_method':'Two-adjacent-bin soft-label CE andcontinuous expectation readout,withquality-aware classification.','our_consequence':'Idealadjacentinterpolation canencodecontinuouspositions;genericDFL/morebins isnotnew.','unresolved':'No author modelorcompletecode executed;not a guaranteeoflearnedsubpixelprecision.'},
{'id':'GFL_V2','title':'Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection','url':'https://arxiv.org/html/2011.12885','tier':'CVPR2021;inspectedarXivv1 Nov2020','read_scope':'Complete3.2DGQP,3.1expectationformula,selected4.1statsablation.','author_method':'Top-k probabilities andtheirmean feed a learned IoU-quality predictor.','our_consequence':'Distributionflatness/qualityfeedback ispriorart. Top-k probabilitymean mustnotbe confusedwithcoordinateexpectation.','unresolved':'Authorcode/modelnotexecuted;inspectedDomeLQE is aderived implementation,notfullGFLV2reproduction.'},
{'id':'D_FINE','title':'D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement','url':'https://arxiv.org/html/2410.13842','tier':'ICLR2025;inspectedHTMLservesarXivv1 Oct2024','read_scope':'FDR/FGL4.1 andpublishedcorrespondingformula excerpts;selectedGO-LSD/conclusion;D-FINE-derived Dome sourcechainandauthorweightstate audited. Notfullpublishedpaper/codeclosure.','author_method':'Iterative nonuniformbin distributionrefinement andglobaloptimallocalization self-distillation.','our_consequence':'FDR,nonuniformbins andadjacentCE interpolationareexistingbaselineoperators;A11 tests theDome-derivedoperator.','unresolved':'OriginalD-FINEauthorweightsnotrun. DomeobservationsdonotconstituteD-FINEbenchmarkreproduction.'},
{'id':'Deep_MDN','title':'Deep Mixture Density Network for Probabilistic Object Detection','url':'https://arxiv.org/html/1911.10614v2','tier':'arXivv2 July2020;authorcommentsIROS2020oral,conferenceDOInotindependentlyverified','read_scope':'CompleteIII-A formulation/learning/inference;III-Bselectedcases andaveraging-versus-probableablation.','author_method':'Cholesky full-covariance mixtureboxhead withmean orconditionalmost-probable inference.','our_consequence':'Genericjoint/correlated/multimodalbboxandmean-to-modeclaimsalreadyhavepriorart.','unresolved':'Noauthorcodeexecuted. Old2019OpenReviewtitleis thesamearXivfamily,notanotherindependentmethod.'}
]
ids={r['id'] for r in matrix['rows']};assert not any(r['id'] in ids for r in additions)
matrix['rows'].extend(additions)
ugs=next(r for r in matrix['rows'] if r['id']=='UGS')
ugs['our_consequence']='RawentropyUM/URisexistingwork. A11independentlychecks anidealsoft-targetgradienttradeoff,notcompleteUGS. A12correctedspreadaddsnoorderingutilityoverrawvariance.'
u=next(r for r in matrix['rows'] if r['id']=='UncertaintyDETR')
u['author_method']='ImplementedpaperformassumesdiagonalGaussianboxcovariance;L2Bayesrisktraceadjustsoutputembeddings,classificationlossandmatching.'
u['read_scope']='Complete3.1diagonalindependenceassumption,3.3L2Bayesriskand4.1-4.3;selectedablations;notwholeappendix/proofaudit.'
matrix.update({'updated_utc':NOW,'latest_report':str(report),'latest_empirical_boundary':'A11representationidentityholds,butA12corrected-versus-rawvariance ordering identicalon270prespecifiedpairs;no algorithmorcausaladvancevalidated.'})

status=records[names[3]]
assert status['model_forwards']==200 and not status['mainline_validated']
status['authoritative_phase_history'].append(copy.deepcopy(status['latest_authoritative_phase']))
status['latest_authoritative_phase']={'utc':NOW,'classification':'progress','goal_status':'active','complete_mainline':False,'full_model_forwards':213,'cumulative_authorized_protocol_cap':213,'previous_closed_phase_forwards':200,'A12_phase_cap':13,'A12_phase_forwards':13,'bounded_phase_complete':True,'main_model_training_updates':0,'CPU_diagnostic_linear_models':64,'scalar_response_heads':96,'A11_full_model_forwards':0,'A11_synthetic_adjacent_cases':96,'A12_decision':result['decision'],'A12_verified_primary_pairs':270,'A12_source_moment_values_verified':2046,'A12_archive':receipt12,'latest_report':str(report),'next':'Inspectcachedlayertrajectoriesandclosestarchitectureoperators;noautomaticmoreforwardsortrainingfromphasecorrection.','remaining_uncertainty':['Specificarchitecture-dependent localizationmechanism','Distinctalgorithmoperator','Causalefficacy','Independentgeneralization','Trainingfeasibility']}
status.update({'recorded_utc':NOW,'previous_goal_turn':status['current_goal_turn'],'current_goal_turn':'progress_A11_source_math_and_A12_frozen_native_val_screen','model_forwards':213,'cap':213,'latest_report':str(report),'current_priority_question':board['current_priority_question']})
status['cap_scope']='200 historicalA0-A8 plusindependentA12protocol13;bothboundedphasescompleted. Thisdoesnotauthorizeautomaticextension.'
audit=records[names[4]]
audit.update({'updated_utc':NOW,'A11_A12_distribution_source_and_math':source_audit,'A12_native_execution':{'model':'Dome-DETR-S','source_commit':source_audit['source_commit'],'full_forwards':13,'cumulative_Dome_forwards':28,'state_digest_unchanged':True,'identity_exact':True,'verification':verify},'current_novelty_boundary':'Noapprovedalgorithm. GenericjointGaussian/multimodal/mean-to-modehaspriorart;A12phasecorrectionfailsitsincrementalsignalgate.'})
for name,value in records.items():write(ROOT/name,value)
receipt={'utc':NOW,'status':'PASS_A11_MATH_AND_A12_NATIVE_SCREEN_RECONCILED','goal_status':'ACTIVE','previous_goal_turn_classification':'progress_A10_source_methods_and_negativegeometryscreen','current_goal_turn_classification':'progress_A11_source_math_plus_A12_prespecifiednativevalidationnegative','mainline_validated':False,'full_model_forwards':213,'new_full_model_forwards':13,'new_fits':0,'diagnostic_fitted_models':64,'scalar_response_heads':96,'main_model_updates':0,'A11_archive':receipt11,'A12_archive':receipt12,'A12_verification':verify,'A12_decision':result['decision'],'matrix_rows':len(matrix['rows']),'files':[{'file':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in [report]+[ROOT/n for n in names]],'not_proven':['Novelmainline','Causalarchitecturemechanism','Algorithmgain','Generalization','Trainingtime/memory']}
write(ROOT/'35_A11A12完成核验与候选降级回执.json',receipt)
print(json.dumps({'status':receipt['status'],'goal_status':'ACTIVE','full_model_forwards':213,'new_full_model_forwards':13,'matrix_rows':len(matrix['rows']),'mainline_validated':False,'A12_decision':result['decision']},ensure_ascii=False))
