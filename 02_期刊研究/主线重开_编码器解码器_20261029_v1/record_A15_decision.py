"""Reconcile the terminal bounded native intervention and its independent arithmetic audit."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json,shutil

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'A15_固定query的采样中心干预_v1'
NOW=datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

protocol=read(RUN/'protocol.json');run=read(RUN/'summary.json');result=read(RUN/'analysis_summary.json')
verified=read(RUN/'independent_verification.json');archive=read(RUN/'backup'/'receipt.json')
assert read(RUN/'process'/'a15_sampling_center_v1_exit.json')['exit_code']==0
assert run['new_full_model_forwards']==13 and run['last_layer_suffix_forwards']==72
assert run['cumulative_full_model_forwards']==226 and result['gate_passed'] is False
assert verified['independent_scalar_ious']==1578 and verified['tiny_iou75_counts']==[40,23,40,23,34,33]
assert run['state_digest_before']==run['state_digest_after']==protocol['state_digest']

report=ROOT/'40_A15采样位置干预结果与主线边界.txt'
report.write_text('''当前结论：原生Dome最后一层的采样reference中心重定向没有改善微小框，候选不进入对应细化模块训练。仍未找到通过问题机制、方法差异与验证路线核查的论文主线。本轮完成13次整图前向和72次最后一层后缀重放，主检测器未更新。

1. 研究问题、检索策略与阅读范围
在“用局部图像特征的空间变化引导微小框位置更新”假设下，先查是否需要已知目标模板，以及梯度框细化是否已有先例。定向查询IoU-Net、ATOM、DiMP、feature-metric query等；精确请求与原始返回见source_audit_v1/A15_*.json。非系统综述，不以无搜索命中证明空白，第三方笔记不作为方法证据。
IoU-Net（Acquisition of Localization Confidence for Accurate Object Detection）：已读3.1—3.3主要算法及PrRoIPooling公式片段，未整篇/代码复现。它学习预测框与真实框的IoU，再用预测IoU对框坐标作梯度优化；不需要跟踪参考模板。因此，不能把“无需参考模板的梯度框细化”当新问题。
ATOM（CVPR2019）：完整3.1及3.2部分已读，未执行作者代码。利用参考图像和首帧目标框生成调制信息，再优化测试帧的预测IoU。
DiMP（ICCV2019，所读arXiv v2）：完整3.1—3.3和3.5，3.4/3.6/3.7部分已读。Gauss-Newton式曲率用于分类滤波器优化；框定位采用ATOM式IoU最大化。不能把它写成直接对框位置做Gauss-Newton求解。
PixLoc、DEQDet、TAPTRv2的既有阅读范围保持不变。通用特征度量、阻尼优化、细化感知训练和注意力位置更新已有先例；本轮没有声称已排除所有更接近方法。

2. 已读近邻的逐项比较
IoU-Net：优化目标是学习到的定位质量；优点是目标可在检测推理时得到，代价是额外质量学习与优化。与待议query特征目标不同，但足以否定通用梯度框优化的独有新颖性。
ATOM/DiMP：有参考帧/首帧目标信息，适用于已给定目标的跟踪。未知目标检测query怎样形成合法对齐目标仍待定义，换成无人机检测任务本身不能构成算法增量。
TAPTRv2：已有key-aware attention及位置更新/后续回归感知。新方案须在同等读取预算下与它比较，不能只比固定点MLP。
DEQDet：共享隐式decoder、细化感知梯度和扰动已有先例，latent-state Jacobian与image-coordinate feature Jacobian应区分。
本轮实际执行的是原生Dome算子的诊断性干预，未实现以上方法的作者复现或某个新优化器。

3. A15最小验证协议：先冻结再执行
复用A12的12张官方VisDrone val图和263个微小GT（800方形输入后sqrt面积<8）。图像最初按标注选取，A12/A13的预测与定位残差本轮已知；因此A15是对已有案例的探索性受控干预，不是独立盲测。GT-query配对固定为A12原生Hungarian结果，不重匹配。
版本仍为Dome-DETR提交2dde3bc1946a3e9fad9abd0612b59fc39bd6b861及既有作者权重。当前decoder实际读取200×200、100×100、50×50、25×25四级256通道特征，步长4/8/16/32，每级每头4个采样点，8头。已有P2层，不能把补P2当作此模型尚未具备的增量。
只干预最后一层。固定进入该层的query、之前的FDR角点logit、第一层回归reference和所有权重。对已配对微小GT对应的query，将采样reference中心换成真实中心，宽高不变；这不等于把每一个采样点都放到目标中心，原有偏移仍然保留。
6臂固定为：原生后缀重放；仅采样reference中心改变；仅query位置编码改变；两条路径都改变；原生读取打乱的value；改变采样中心且读取同一打乱value。后两臂用同一空间roll置换保留值分布，比较最后一次读取的真实内容作用。进入最后一层的query已经包含前面读取的图像信息，不能把这个对照称为移除了全部图像信息。
总预算13次完整前向（首图1次无hook身份对照+12次原生捕获）和72次最后一层后缀重放，无warmup、训练或优化器。此前213次阶段保持结束，本次是新机制问题的独立有界阶段。首次原生与全部A12缓存、原生后缀重放均要求raw boxes/logits精确一致；仅改变采样时cross-attention query必须精确不变，未干预query的框/分数也必须精确不变。
主指标是所有263个微小GT的固定配对几何IoU>=0.75覆盖率，逐图等权，按12个前缀作2000次bootstrap。主效应为真实value下“采样中心改变减原生”；内容交互效应为该差值再减去打乱value下同一差值。预设资源筛选要求主效应>=5个百分点、内容交互>=3个百分点，且各自区间下界>0；二者都满足才保留。不是AP或论文确认性显著性标准。

4. 实际结果与独立核验
263个微小GT达到IoU>=0.75的数量，按上述6臂顺序为40、23、40、23、34、33。这是固定配对几何数量，不是带分类置信度的检测AP/召回。
主效应逐图均值为−5.6853个百分点，bootstrap区间[−9.3455,−2.6626]；内容交互为−4.8463个百分点，区间[−7.5444,−2.4177]。平均IoU的采样干预差为−0.03012，区间[−0.03580,−0.02428]。主判据明确未通过，未改用次要指标或另挑子集挽救候选。
仅位置编码改变的IoU75数量未变，不能据此说它的全部框坐标、分数或其他场景效果都不变。采样中心改到GT反而变差，也不能推论GT位置没有信息：原生偏移、query内容与上游输出是联合学习的，reference中心不必等于最佳读取区域中心。本轮未测得哪一种解释是真实原因。
所有源身份门槛在执行中通过；模型state digest前后均为82c3ba49d69370a3d0105e47fc0ab5d715a9e7b1ad15feb275b1b497b5b0ccce。执行回执exit0，13/13整图与72/72后缀预算均结束。
原始记录已下载：615751926字节归档，45个成员；SHA256为9436bbed42366c8cf5014408cf4fca2377b2f9e12a987444f3ec04894efd7e069。独立核对41项完成清单、1578个逐GT逐臂标量IoU、bootstrap聚合及4313项未干预query的输出不变。独立核验没有再执行检测器或后缀；它验证算术与证据完整性，不冒充新的数据复现。
保存的FP32多尺度value、原生query与reference可供后续机制审计；它们不能自动证明这些特征可被某个新方法有效读取。

5. 研究空白与候选增量判断
原文结论：梯度框细化、参考目标调制、曲率辅助优化和细化感知训练都有先例。
本次测量：固定query时，将最后一层采样reference中心改到配对GT中心，没有带来定位改善，且打乱value对照也未支持预设的真实内容增益。
我们的推断：当前没有依据把“最后一层读点中心偏了，移正即可改善”作为主线；不据此训练相应的特征度量/梯度细化模块。
待验证问题：原生多尺度特征对相邻位置的变化是否足够可分，以及query聚合/框读出是否保留这些变化。A0测的插值系数秩不等于实际feature Jacobian；A4/A5的负结果仍保留。这里尚未诊断出新的编码器或解码器故障，不是已确认研究空白。

6. 可行性、风险和下一步
本轮自身GPU allocated峰值0.827GiB，只证明该冻结诊断可运行，不能推导端到端训练预算。单卡约24GB、数天训练的用户约束保持不变。架构分支累计226次完整前向（TinyFormer185、Dome41）；本轮额外72次最后一层后缀单列，不折算或隐藏成整图前向。主模型更新0，既有64个诊断拟合/96个标量响应头不变。
下一步优先使用本轮已保存的实际多尺度value和query核查局部空间变化及汇聚，不先追加新模型前向或训练。若提出新算子，须给出与最近工作的实质差异和新的有限验证协议。不得把本轮oracle干预当成可部署模块，不因负结果自动断言编码器缺信息，也不在同一阶段追加更多中心/阈值搜索。
目前依然没有可宣称已经找到的小论文主线；目标保持ACTIVE。旧DEIM、压缩和恢复分支没有重启。时间先后以冻结文件hash、传输和单次启动回执证明，保留本地/服务器时钟偏移。

7. 可核验参考
IoU-Net：https://arxiv.org/html/1807.11590
ATOM：https://arxiv.org/html/1811.07628
DiMP：https://arxiv.org/html/1904.07220
PixLoc：https://arxiv.org/html/2103.09213
DEQDet：https://arxiv.org/html/2308.09564
Dome固定源码：https://github.com/RicePasteM/Dome-DETR/tree/2dde3bc1946a3e9fad9abd0612b59fc39bd6b861
协议、原始数据、终态、独立核验：A15_固定query的采样中心干预_v1/protocol.json、summary.json、analysis_summary.json、independent_verification.json。
''',encoding='utf-8')

names=['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json']
history=ROOT/'history_before_A15_update_v1';assert not history.exists();history.mkdir()
for name in names:shutil.copy2(ROOT/name,history/name)
write(history/'manifest.json',{'utc':NOW,'files':[{'file':n,'sha256':sha(history/n)} for n in names]})
records={name:read(ROOT/name) for name in names}
question='微小目标精确定位：用已缓存的实际多尺度value/query查证局部空间变化在特征汇聚和框读出中的保留；A15不支持最后一层直接重定采样中心。'
board=records[names[0]]
board['constraints']['new_experiments_this_turn']='A15 completed frozen-model sampling interventions:13 full passes and72 final-layer suffix replays;zero training or new architectures.'
board.update({'latest_update_utc':NOW,'latest_report':str(report),'status':'NO_CANDIDATE_PASSED_MECHANISM_AND_NOVELTY_GATES_AFTER_A15',
    'current_priority_question':question,
    'new_empirical_evidence':'A15:13newnativefullpasses+72finalsuffixreplays;native40/263vsGTsamplingreference23/263IoU75. Primaryimageeffect-5.6853pp;contentinteraction-4.8463pp. 0training;failuremechanismunresolved.',
    'next_decision_protocol':'A15phase13/13and72/72terminalexit0. Usecachedrealfeature/querygeometry;noadditionalcentervariantsorgradientrefinertrainingfromthisnegativegate.'})
board['candidates'].append({'id':'J_QUERY_SPATIAL_METRIC_AND_LAST_READ_LOCATION',
    'question':'Canactualquery-conditionedspatialreadingsupportfinercentrelocationwithoutGTatdeploytime?',
    'status':result['decision'],'increment_approved':False,'failure_cause_identified':False,
    'controlled_intervention':True,'causal_improvement_evidence':False,
    'scope':'Negativefornativefinal-layerGT-reference-centreheuristic;notanempiricaltestofeverypossiblelearnedfeaturemetricorencoder.',
    'nearest':['IoU-Net','ATOM','DiMP','PixLoc','DEQDet','TAPTRv2'],
    'next':'No correspondingmetric-refinertrainingfromthisgate;preservetheopenfeatureinformationquestion.'})
board['A15_evidence']={'archive':archive,'verification':verified,'protocol_sha256':sha(RUN/'protocol.json'),
    'full_forwards':13,'suffix_forwards':72,'decision':result['decision']}
board['model_candidates']=['TinyFormer-S-PBM:185historicalfullfrozenpasses',
    'Dome-DETR-S:41fullfrozenpassesand72last-layerreplays;4decoderlevels/P2present;finalpaperbaselineunselected',
    'Focus-DETR:source-onlytargetgeometry,noauthormodelrun']
board['next_exploratory_question']={'status':'NOT_A_CONFIRMED_GAP_OR_APPROVED_MODULE',
    'question':question,'no_new_forward_phase_from_this_record':True,
    'distinctions':['Bilinearcoefficientrankisnotactualimage-featureJacobian.','StronglocalgradientsdonotproveatargetdirectioncanbeestimatedwithoutGT.',
        'Weakaggregatedgradientsalonearenotcausalproof;controlsize,initialgeometryandknownoperatorsbeforeclaimingfailure.']}

log=records[names[1]];log['supplement_utc']=NOW
raw=sorted((ROOT/'source_audit_v1').glob('A15_*.json'))
rawrows=[{'file':str(p.relative_to(ROOT)),'sha256':sha(p),'request':read(p)['request']} for p in raw]
log['query_groups'].append({'name':'A15gradientboxpriorsandnativecontrolledsampling','requests':rawrows})
log['A15_controlled_sampling']={'source_audit':str(RUN/'source_operator_audit.json'),'protocol':str(RUN/'protocol.json'),
    'archive':archive,'verification':verified,'report':str(report),'no_full_author_optimizer_reproduction':True}
matrix=records[names[2]]
newrows=[
 {'id':'IoU_Net','title':'Acquisition of Localization Confidence for Accurate Object Detection',
  'authors':['Borui Jiang','Ruixuan Luo','Jiayuan Mao','Tete Xiao','Yuning Jiang'],
  'url':'https://arxiv.org/html/1807.11590','tier':'InspectedarXivv1Jul2018',
  'read_scope':'3.1-3.3corealgorithmsincludingfullAlg2,PrRoIPoolformulaexcerpts;notwholepaper/code.',
  'author_method':'Learnclass-awareIoUqualityandrefineboxesbygradientascentthroughPreciseRoIPooling.',
  'our_consequence':'Generictemplate-freegradientboxrefinementisexistingdetectionwork.',
  'unresolved':'Noauthorcode/modelsrunorqualityobjectivecomparisoninDome.'},
 {'id':'ATOM','title':'ATOM: Accurate Tracking by Overlap Maximization',
  'authors':['Martin Danelljan','Goutam Bhat','Fahad Shahbaz Khan','Michael Felsberg'],
  'url':'https://arxiv.org/html/1811.07628','tier':'CVPR2019;inspectedarXivv2Apr2019',
  'read_scope':'Complete3.1,targetestimation;selected3.2onlineclassifier;notwholepaper/code.',
  'author_method':'Referenceimage/targetboxmodulatesIoUpredictor,thenboxgradientascent;onlineclassifierusesGN/CG.',
  'our_consequence':'Reference-conditionedtargetqualityandgradientboxoptimizationareprior.',
  'unresolved':'Unknownobjectquerieslackthegivenfirst-framereference;tasktransferaloneisnotnew.'},
 {'id':'DiMP','title':'Learning Discriminative Model Prediction for Tracking',
  'authors':['Goutam Bhat','Martin Danelljan','Luc Van Gool','Radu Timofte'],
  'url':'https://arxiv.org/html/1904.07220','tier':'ICCV2019;inspectedarXivv2Jun2020',
  'read_scope':'Complete3.1-3.3and3.5;selected3.4/3.6/3.7;notwholepaper/appendixorcode.',
  'author_method':'LearneddiscriminativelossandGN-stylecurvaturestepforclassificationfilter;ATOM-styleoverlapmaximizationforboxes.',
  'our_consequence':'Donotconflatefilter-parameterJacobianwithimage-positionJacobian;genericlearnedcurvature/refinementnotnew.',
  'unresolved':'Noauthormodelrunorlegitimateinference-timequerytemplateestablished.'}]
assert not ({r['id'] for r in newrows}&{r['id'] for r in matrix['rows']})
matrix['rows'].extend(newrows);matrix.update({'updated_utc':NOW,'latest_report':str(report),
    'latest_empirical_boundary':'A15nativeGTsampling-centreinterventionnegative;noactualfeature-gradientcauseornewalgorithmvalidated.'})

status=records[names[3]];assert status['model_forwards']==213 and not status['mainline_validated']
status['authoritative_phase_history'].append(copy.deepcopy(status['latest_authoritative_phase']))
status['latest_authoritative_phase']={'utc':NOW,'classification':'progress','goal_status':'active','complete_mainline':False,
    'full_model_forwards':226,'cumulative_authorized_protocol_cap':226,'previous_closed_phase_forwards':213,
    'A15_phase_full_cap':13,'A15_phase_full_forwards':13,'A15_suffix_cap':72,'A15_suffix_forwards':72,
    'bounded_phase_complete':True,'actual_process_exit_code':0,'main_model_training_updates':0,
    'CPU_diagnostic_linear_models':64,'scalar_response_heads':96,'A15_new_fits':0,
    'A15_decision':result['decision'],'A15_archive':archive,'A15_verified_scalar_ious':1578,
    'A15_verified_unchanged_query_instances':4313,'latest_report':str(report),
    'next':'Usecachednativefeaturefieldsandqueryaggregationtoinspectactualspatialvariation;noautomaticnewmodelsortraining.',
    'remaining_uncertainty':['Specificnaturalfailuremechanism','Distinctalgorithm','Deployablequery-targetdefinition','Independentgeneralization','Trainingfeasibility']}
status.update({'recorded_utc':NOW,'previous_goal_turn':status['current_goal_turn'],
    'current_goal_turn':'progress_A15_primary_nearest_methods_and_bounded_native_causal_sampling_intervention',
    'model_forwards':226,'cap':226,'last_layer_suffix_forwards':72,'latest_report':str(report),'current_priority_question':question})
status['cap_scope']='200historical+A12separate13+A15separate13;allthreefull-forwardphasescomplete. A15additional72last-layerreplaysseparatelyclosed;noautomaticextension.'
audit=records[names[4]]
audit.update({'updated_utc':NOW,'A15_sampling_intervention':{'source_audit':read(RUN/'source_operator_audit.json'),
    'execution':run,'verification':verified,'decision':result['decision'],
    'source_interpretation':'GTreferencecentreisnotthesamethingaseverysamplingpointatGT;nativeoffsetpatternsandincomingquerywerepreserved. Negativecannottestallpossibletrainedmetricoperators.'},
    'current_novelty_boundary':'GradientboxrefinementhasIoU-Net/ATOM/DiMPprior;A15doesnotsupportnativeGTsampling-centreheuristic. Noapprovednewencoder/decoder.'})
for name,value in records.items():write(ROOT/name,value)
receipt={'utc':NOW,'status':'PASS_A15_TERMINAL_EXECUTION_AND_INDEPENDENT_METRICS_RECONCILED','goal_status':'ACTIVE',
    'previous_goal_turn_classification':'progress_A14_primary_sources_and_operator_algebra',
    'current_goal_turn_classification':'progress_A15_bounded_native_sampling_intervention_negative',
    'mainline_validated':False,'full_model_forwards':226,'new_full_model_forwards':13,'last_layer_suffix_forwards':72,
    'new_fits':0,'main_model_updates':0,'actual_exit_code':0,'matrix_rows':len(matrix['rows']),
    'archive':archive,'verification':verified,'A15_decision':result['decision'],
    'files':[{'file':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in [report]+[ROOT/n for n in names]]}
write(ROOT/'41_A15完成核验与主线状态回执.json',receipt)
print(json.dumps({'status':receipt['status'],'goal_status':'ACTIVE','mainline_validated':False,
    'matrix_rows':len(matrix['rows']),'full_model_forwards':226,'suffix_forwards':72,
    'decision':result['decision']}))
