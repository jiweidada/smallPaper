"""Record source evidence and a bounded diagnostic draft. No detector execution."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

R=Path(__file__).resolve().parent
UTC=datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

files=['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json']
h=R/'history_before_A7_clearance_v1';h.mkdir(exist_ok=True)
for name in files:
 if not (h/name).exists():shutil.copy2(R/name,h/name)

sv=read(R/'comparison_sources_v1/SViT_operators_A7_v1/source_verification.json')
sa=read(R/'comparison_sources_v1/Salience_operators_A7_v1/source_verification.json')
assets=R/'official_assets_v1/Dome_DETR_2dde3bc_v1'
av=read(assets/'asset_verification.json')
remote=read(assets/'remote_asset_preflight.json')
assert len(sv['files'])==6 and len(sa['files'])==3
assert av['weight']['status']=='PASS_PUBLISHED_LFS_SHA256_SIZE_ZIP_CRC'
assert remote['status']=='PASS_REMOTE_ASSET_HASH_SOURCE_ARCHIVE_AND_CPU_WEIGHT_ONLY_LOAD'
assert remote['source_files']==len(av['source_files'])==293
assert remote['weight_sha256']==av['weight']['sha256']==sha(assets/'dome-s-visdrone_converted.pth')
assert remote['new_full_detector_forwards']==0 and not remote['strict_model_load_verified']

audit={'utc':UTC,'status':'SOURCE_OPERATOR_OVERLAP_CONFIRMED_DOME_COUPLED_PATH_IDENTIFIED',
 'SViT':sv,'Salience_DETR':sa,
 'findings':[
  {'source':'SViT','file':'mmdet_custom/models/backbones/base/selective_vit.py','line':43,
   'conclusion':'LayerNorm + 2-layer MLP + hard Gumbel gate; raw differentiable selector is returned separately from boolean attention mask.'},
  {'source':'SViT','file':'mmdet_custom/models/backbones/adapter_modules.py','line':430,
   'conclusion':'Detector gradients pass through selected-update/skip interpolation. Inference gathers active tokens and scatters updated features back; no inherited mask in selected tiny config, allowing reactivation.'},
  {'source':'SViT','file':'mmdet_custom/models/detectors/gumbel_two_stage.py','line':69,
   'conclusion':'RPN/ROI task losses and weighted backbone ratio loss train the same network. Generic task-driven update gating is already prior art.'},
  {'source':'Salience_DETR','file':'models/detectors/salience_detr.py','line':72,
   'conclusion':'Strict center-inside and scale-range positive support remains; noise_scale=0 in inspected config. No per-GT fallback in this function.'},
  {'source':'Salience_DETR','file':'models/bricks/salience_transformer.py','line':149,
   'conclusion':'Level top-k then nested layer filtering; full-memory values remain available. Do not equate skipped updates with feature deletion.'},
  {'source':'Dome_DETR','file':'src/zoo/dome/get_roi_features.py','line':197,
   'conclusion':'WindowProcessor clones the full feature map, transforms selected equal-sized windows, then reconstructs them in place. Window-level—not point-label—coverage must be audited.'},
  {'source':'Dome_DETR','file':'src/zoo/dome/dome_decoder.py','line':800,
   'conclusion':'The same encoder window mask filters optional decoder anchors, and the density map also affects query NMS thresholds. A raw mask swap changes two mechanisms.'},
  {'source':'Dome_DETR','file':'src/zoo/dome/dome_decoder.py','line':909,
   'conclusion':'The complete query-seed return tuple can be cached for a controlled memory-path intervention; this excludes effects mediated through query selection/initialization.'}],
 'not_proven':['New algorithm novelty','Actual missed-update prevalence','Causal detection gain','Generalization','24GB training cost'],
 'new_full_detector_forwards':0,'new_fits':0,'main_detector_updates':0}
outdir=R/'A7_源码去重与资产准备_v1';outdir.mkdir(exist_ok=True)
save(outdir/'source_operator_audit.json',audit)

# Case selection uses annotation size/count only, before any Dome prediction.
gt_file=R/'A6_稀疏编码器前景监督覆盖审计_v1/per_gt.jsonl'
groups={};gt_by_image={}
with gt_file.open(encoding='utf-8') as f:
 for line in f:
  x=json.loads(line)
  if x['scenario']!='square640_source':continue
  n=x['file_name'];g=groups.setdefault(n,{'file_name':n,'group':x['group'],'valid_gt':0,'lt8_at_800':0})
  g['valid_gt']+=1;g['lt8_at_800']+=x['size640']*1.25<8
  gt_by_image.setdefault(n,[]).append({'uid':x['uid'],'class_id':x['class_id'],
   'box800_xyxy':[v*1.25 for v in x['source_box']],'size800_from640':x['size640']*1.25})
metadata=read(R/'A6_稀疏编码器前景监督覆盖审计_v1/dataset_geometry_metadata.json')
md={x['file_name']:x for x in metadata['rows']}
cases=[];seen=set()
for n,g in sorted(groups.items()):
 if g['lt8_at_800']>=10 and g['group'] not in seen:
  cases.append(dict(g,image_metadata=md[n],analysis_gt=gt_by_image[n]));seen.add(g['group'])
  if len(cases)==3:break
assert len(cases)==3 and sum(x['lt8_at_800'] for x in cases)==69
case_path=outdir/'A8_draft_cases.json'
save(case_path,{'selection':'First3 lexicographic distinct-prefix images with >=10 GT sqrt(area)<8 at square800 among fixed100 train images; labels only, no Dome outcomes.',
 'scope':'Exploratory in-sample technical/mechanistic preflight, not independent performance evidence.',
 'source_gt_sha256':sha(gt_file),'cases':cases})
draft={
 'material_passport':{'user_scope':'UAV tiny-object algorithm mainline; single~24GB GPU; no historical jobs resumed.',
  'workflow':'ARS targeted non-systematic research and bounded diagnostics; single assistant.',
  'author_reads':'Paper and selected-source scopes are enumerated in the literature matrix; no human read attestation.'},
 'status':'DRAFT_NOT_EXECUTION_READY_STRICT_LOAD_AND_INTERVENTION_IMPLEMENTATION_PENDING',
 'question':'At a fixed encoder window budget and native query seeds, can changing window updates improve tiny-object localization through the memory-read path?',
 'model':'Dome-DETR-S official VisDrone converted checkpoint',
 'source_commit':av['source_commit'],'weight_sha256':av['weight']['sha256'],
 'case_file':str(case_path),'case_sha256':sha(case_path),
 'input':'Author square800 Resize + ConvertPILImage float32 scale=True; exact active resize implementation to be verified/frozen before running. No ImageNet normalization added.',
 'runtime':'FP32 eval,no AMP,no deploy fusion,no gradients,TF32 disabled,batch1,fixed RNG20261030. Compatibility adapter must preserve every active arithmetic operation and strictly load all model state keys.',
 'counting':{'prior_full_detector_forwards':185,'maximum_new_full_detector_forwards':15,'maximum_global_total':200,
  'fits':0,'detector_updates':0,'warmups':0,'rule':'Every entered full-model evaluation counts, including failures. Do not replace failed cases or add attempts; partial-network repeats may not bypass the cap.'},
 'stages':[
  'Strict CPU model-state agreement, active config/source closure and original image hashes must pass before the first full forward.',
  'Run native baseline on3 fixed cases; cache projected P3 inputs, native density map/window mask and the complete decoder _get_decoder_input return tuple.',
  'If native masks are all full, report degeneracy and do not manufacture a lower-budget problem by changing thresholds.',
  'Verify identity replay with native window mask and cached query tuple. Require raw normalized boxes/scores maxabs<=1e-6 and identical query counts/native postprocess identities before interpreting intervention.',
  'Change only the WindowProcessor update mask. Retain all unselected feature values and native positional encoding; keep cached native decoder seed tuple for every replay arm.'
 ],
 'arms':[
  {'id':'NATIVE','scope':'Unmodified native model; establishes source outputs and query tuple.'},
  {'id':'REPLAY_NATIVE','scope':'Identical window choice with cached native query tuple; numerical identity control.'},
  {'id':'RANDOM_SAME_K','scope':'Uniform seeded subset of exactly the native number of fixed10x10 P3 windows.'},
  {'id':'GT_COVERAGE_SAME_K','scope':'Known area-normalized GT-coverage greedy control restricted to <8px targets; same K, ties by flat window index. This is GT-assisted and not a new/deployable algorithm.'},
  {'id':'ALL_WINDOWS','scope':'All native fixed windows updated; unequal compute diagnostic reference, not a mathematical upper bound or same-budget result.'}],
 'scope_of_query_freeze':'Controlled direct effect through decoder memory with initial content/reference boxes/query counts fixed; excludes total effects mediated by query selection/initialization. A negative result does not disprove all encoder improvements.',
 'endpoints':['Actual updated-window count and native tiny-GT center/area coverage',
  'Raw per-query box/score changes and class-aware IoU50/75 one-to-one physical-query coverage under native Top300 rule',
  'State digests before/after; observed inference peak memory and latency only as runtime evidence, not training feasibility'],
 'decisions':[
  'Identity failure, missing keys, changed image/config/state or cap exhaustion ends this preflight with no efficacy claim.',
  'Three preselected train cases can establish implementation applicability and suggest a mechanism; they cannot establish AP gain, statistical generalization or a publishable mainline.',
  'Without a nondegenerate actual coverage gap and controlled effect, do not train a new routing module. Continue only with a justified separate protocol; no automatic seed, threshold, resolution or module search.'
 ],
 'novelty':'Generic task-driven gating, counterfactual utility, Gaussian density and coverage greedy are existing ideas. No new operator approved.',
 'training_memory_verified':False,'executed':False}
save(outdir/'A8_minimal_memory_path_draft.json',draft)

matrix=read(R/files[2]);byid={x['id']:x for x in matrix['rows']}
byid['SViT'].update(read_scope='Primary arXiv2306.07050v3 complete3.1–3.3 and selected4 setup;6 pinned source/config blobs and active gating/adapter/detector-loss chain. No author code executed.',
 author_method='Differentiable per-layer task-trained update gates, preserved/re-enabled features and batch-averaged ratio loss.',
 our_consequence='Generic learned update utility, token reactivation and feature preservation overlap existing operators. No new gating claim approved.',
 source_commit=sv['commit'],source_verification=str(R/'comparison_sources_v1/SViT_operators_A7_v1/source_verification.json'))
byid['Salience_DETR'].update(read_scope='Primary arXiv2403.16131v1 method3.1–3.4 and selected4.1;3 pinned operator/config blobs. No full model executed.',
 our_consequence='Salience hierarchy already addresses scale bias. Inspected support retains strict inside-box/range conditions; memory persists. A6 rates are not asserted as actual Salience prediction/update statistics.',
 source_commit=sa['commit'],source_verification=str(R/'comparison_sources_v1/Salience_operators_A7_v1/source_verification.json'))
byid['Dome_DETR'].update(read_scope='Selected primaryv2 density/mask method; full293-file source archive acquired and hash-verified, not all read. Active WindowProcessor/encoder-mask/decoder-selection path read. Official VisDrone-S weights CPU-loaded; no detector instantiated/forwarded.',
 our_consequence='Encoder window choice and decoder optional-query selection share the same mask; raw mask interventions are confounded. Actual window coverage still unmeasured.',
 source_verification=str(assets/'asset_verification.json'))
for row in [
 {'id':'CoRePrune','title':'From Token Importance to Conditional Removability: Rethinking Visual Token Pruning in Multimodal Large Language Models','url':'https://arxiv.org/abs/2609.26484','tier':'2026-09-22 arXiv preprint',
  'read_scope':'Primary HTMLv1 introduction,formulationIII,selectedIV diagnostics,completeV method; selectedVI passages. No author source acquired or executed.',
  'author_method':'Depth-refreshed deletion perturbations and set-conditioned reverse-greedy rescue under fixed token budgets in MLLMs.',
  'our_consequence':'Counterfactual utility and set-aware rescue are prior ideas. Deleting tokens differs from skipping encoder updates, but that difference alone is not new algorithm evidence.'},
 {'id':'QA_MBG_visual_pruning_limits','title':'How Much Can We Prune? Exploring the Performance Limits of Visual Token Pruning','url':'https://arxiv.org/abs/2610.23974','tier':'2026-10-13 arXiv preprint',
  'read_scope':'ABSTRACT_ONLY; full method/source not read.',
  'author_method':'Offline frozen-VLM fixed-budget visual subset search with QA minibatch greedy, as described by authors in abstract.',
  'our_consequence':'Search/utility language is not by itself novel; different-task evidence only, not a verified detector method comparator.'}]:
 if row['id'] in byid:byid[row['id']].update(row)
 else:matrix['rows'].append(row)
matrix['updated_utc']=UTC
matrix['latest_empirical_boundary']='A7 source/asset evidence only; architecture full-forward count185, detector updates0. A8 direct-memory-path draft unexecuted.'
report=R/'26_稀疏编码近邻源码对照与最小因果验证.txt'
matrix['latest_report']=str(report);save(R/files[2],matrix)

log=read(R/files[1])
raws=sorted((R/'source_audit_v1/sparse_gate_A7').glob('A7_*.json'))
log['A7_source_clearance']={'utc':UTC,'review_type':'targeted non-systematic',
 'queries':[
  '"Revisiting Token Pruning for Object Detection and Instance Segmentation" github',
  '"token" "counterfactual" "pruning" detection','"token pruning" "utility" detection',
  '"DETR" "routing" "benefit" encoder',
  '"object detection" "token pruning" "loss" "importance" -site:researchgate.net -site:alphaxiv.org',
  '"tiny object" "token" "coverage" encoder','"token" "instance-balanced" "detection"'],
 'primary_source_followup':'arXiv/CVF papers and paper-linked author repositories only support conclusions. Other search hits were discovery material, not evidence.',
 'raw_receipts':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in raws],
 'limits':['Final CVF PDF fetches403; primary arXiv versions were used and version-scoped.',
  'Salience arXivv2 HTML404; abs/PDF confirm onlyv1.',
  'Initial weight transfer and first resume each timed out terminally; same immutable file resumed twice and final published SHA matched. No TLS verification disabled.'],
 'decision':'Generic learned update gating/conditional utility already have close prior work; precise encoder-memory effect still unverified.'}
save(R/files[1],log)
source=read(R/files[4]);source['A7_sparse_gate_audit']=audit
source['Dome_full_asset_receipt']=str(assets/'asset_verification.json')
source['Dome_remote_CPU_asset_preflight']=remote
source['updated_utc']=UTC;source['scope_this_update']='Nine new pinned operator/config blobs;fullDome archive and published weight prepared/CPU-loaded. No author model execution or training.'
save(R/files[4],source)

body=f'''稀疏编码最近工作核查与最小验证路线
记录时间：{UTC}（本地记录钟；远端回执保留其原始报告时间，不据两台机器时间差推断事件顺序）

当前判断
继续保留“微小目标的稀疏编码更新分配”作为待证问题。它尚未成为可投稿的新算法主线。
本次完成的是最近工作算子去重与可运行资产准备，不把“学更新收益”或“反事实效用”本身作为创新。

1. 检索策略与记录
沿既有非系统性定向调研，补读SViT和Salience-DETR方法，追踪官方调用链，再检索任务损失、更新效用、反事实选择及实例覆盖的近邻。
13_架构检索增补记录.json记录本次7条查询及{len(raws)}份原始响应SHA；矩阵目前{len(matrix['rows'])}项。检索不是穷尽，空结果不证明无人做过。

2. 核心文献与阅读范围
SViT：主arXivv3的3.1–3.3与实验设置；官方固定版本6个相关源文件。Salience：主arXivv1的3.1–3.4与设置；官方固定版本3个相关源文件。都未运行完整作者模型。
CoRePrune：读到主文V完整方法和III公式、部分IV/VI；未核查代码。QA-MBG论文仅摘要，不能称为全文阅读。
Dome：完整293个Git文件现已取得并校验；本次只阅读了窗口更新、编码掩码与decoder选择相关路径，未通读全部文件。

3. 与最接近工作的逐项比较
【原文+源码】SViT的门控已由检测任务损失和预算约束共同训练，支持跳过更新、保留特征与后层重新激活。因此仅加MLP门控或让任务损失学更新优先级没有充分新意。
【原文+源码】Salience已有分尺度/分层显著性选择。它的目标函数仍使用严格中心在框内和尺度范围；未选位置仍保留在memory。不能把其未更新位置描述为信息被删除。
【原文】CoRePrune已经按删除造成的扰动与当前集合中的边际恢复收益选择token。跳过编码更新与删除token有差别，但任务/算子差别本身还不足以证明创新。
【源码新核查】Dome的同一窗口mask同时控制MWAS更新和可选decoder锚点筛选；密度图也参与query的NMS阈值。直接改mask观察提升，会混合编码和query初始化两条路径。
【我们的推断】应先固定原生query内容、参考框及数量，检验窗口更新经memory读取产生的受控直接效应。该诊断不估计encoder的全部效果，也不包含query选择这一中介路径。

4. 有依据的问题与尚不能声称的研究空白
A6已验证的是Focus某份源码的正标签几何缺口；并未证明Dome实际漏更新或漏检。Dome按完整窗口更新，其覆盖必须另行实测。
当前待证问题缩写为：在相同窗口更新预算下，原生选择是否错过对微小目标定位有用的更新？
尚不能声称这种计算错配是新发现，也不能称已有新encoder。此前通用中心精修的负结果仍保留，不恢复旧压缩或旧DEIM训练。

5. 候选增量、可行性与风险
尚未批准新模块。若后续因果信号成立，增量必须对具体的更新错配提出与SViT/Salience/Dome/UHR/CoRePrune不同的机制，并做同预算已知方法控制。
风险包括：原生窗口本就全选；未更新区域仍足以支撑检测；替换mask改变query造成伪归因；仅3个训练案例无法泛化；单卡训练显存与数天预算仍未核实。

6. 最小验证协议（草案，未执行）
基线候选定为作者Dome-DETR-S VisDrone检查点，原生800方形输入。固定3个不同图像前缀的训练案例，仅依据标注尺寸/数量挑选，共69个<8像素分析目标。这是实现/机制预检，不是独立AP试验。
最多3图×5臂=15次完整前向：原生、原生mask及query种子重放、等窗口数随机、等窗口数GT覆盖已知控制、全窗口更新参考。总数上限185+15=200，无额外warmup或失败后扩展。
所有重放固定原生decoder初始化整个返回元组，只改变MWAS接受更新的窗口。未选特征保留；GT控制不能部署，全窗口臂计算量不同，均不能作为新算法同算力效果。
先严格加载模型所有state键，再做原生重放数值一致检查。原生mask若全满，记录不可辨识并停止这项干预，不改阈值造问题。
主要记录实际窗口覆盖、逐query框/分数变化与有类别约束的一对一IoU50/75覆盖，均不冒充正式AP。小案例阳性也只能决定是否值得更大诊断，不能证明投稿价值。
执行前仍需完成兼容加载器和干预实现，核验后冻结源码/协议。具体草案及案例SHA见A7目录。当前没有启动A8。

7. 可核验文献与资产
SViT：https://arxiv.org/html/2306.07050v3
官方源码：https://github.com/uzh-rpg/svit/tree/{sv['commit']}
Salience-DETR：https://arxiv.org/abs/2403.16131
官方源码：https://github.com/xiuqhou/Salience-DETR/tree/{sa['commit']}
CoRePrune：https://arxiv.org/html/2609.26484v1
QA-MBG（仅摘要）：https://arxiv.org/abs/2610.23974
Dome-DETR：https://arxiv.org/abs/2505.05741
作者仓库：https://github.com/RicePasteM/Dome-DETR/tree/{av['source_commit']}
作者权重：https://huggingface.co/RicePasteM/Dome-DETR/blob/{av['weight']['repo_commit']}/best_ckpts_dome_2026/dome-s-visdrone_converted.pth

本地下载、校验后已上传隔离远端目录。源码293文件Git内容哈希全匹配；权重53,509,575字节，发布SHA256={av['weight']['sha256']}。
远端CPU weights_only加载通过，含1033个state张量、13,278,573个张量元素（含buffer，不等于trainable参数量）。完整模型尚未构建或严格加载；不能据此宣称可训练或复现AP。
本轮新增完整检测器前向0、诊断拟合0、主检测器更新0；架构分支累计185前向、64个诊断模型/96个标量头。整个研究目标保持进行中。
'''
report.write_text(body,encoding='utf-8')
board=read(R/files[0]);board.update(latest_update_utc=UTC,latest_report=str(report),mainline_validated=False)
for c in board['candidates']:
 if c['id']=='F_SPARSE_ENCODER_UPDATE_ALLOCATION':
  c.update(status='GENERIC_TASK_GATING_AND_CONDITIONAL_UTILITY_ARE_PRIOR_ACTUAL_WINDOW_EFFECT_PENDING',
   nearest=['Focus-DETR','RFLA','Dome-DETR','UHR-DETR','Salience-DETR','SViT','CoRePrune'],
   increment_approved=False,causal_evidence=False,
   possible_increment='No specific operator approved. First isolate an actual target-specific same-budget window-update effect from query-seed changes.',
   next='Finish strict Dome model compatibility load and instrumented identity replay; then execute only a frozen A8 bounded protocol, no training.')
board['A7_source_audit']=str(outdir/'source_operator_audit.json')
board['next_decision_protocol']=str(outdir/'A8_minimal_memory_path_draft.json')
board['asset_gate']={'local_and_remote_integrity':'PASS','CPU_weight_load':'PASS','strict_detector_load':'NOT_DONE','detector_runtime':'NOT_DONE','training_memory':'NOT_DONE'}
save(R/files[0],board)
state=read(R/files[3]);old=state['latest_authoritative_phase']
state.setdefault('authoritative_phase_history',[]).append(old)
state.update(recorded_utc=UTC,goal_status='ACTIVE',previous_goal_turn='progress_reconciled_A5_A6_and_saved18_retrievals',
 current_goal_turn='progress_A7_operator_clearance_full_Dome_assets_verified_remote_CPU_load',mainline_validated=False,latest_report=str(report))
state['latest_authoritative_phase']={'utc':UTC,'classification':'progress','goal_status':'active','complete_mainline':False,
 'full_model_forwards':185,'global_cap':200,'main_model_training_updates':0,'CPU_diagnostic_linear_models':64,'scalar_response_heads':96,
 'A7_audit':str(outdir/'source_operator_audit.json'),'Dome_asset_verification':str(assets/'asset_verification.json'),
 'Dome_remote_CPU_preflight':str(assets/'remote_asset_preflight.json'),'strict_model_load_verified':False,
 'A8_status':draft['status'],'new_forwards_this_turn':0,'new_fits_this_turn':0,'latest_report':str(report),
 'next':'Implement source-faithful Dome load, verify strict keys/config/state and identity control before any bounded A8 intervention.'}
save(R/files[3],state)
receipt={'utc':UTC,'status':'A7_SOURCE_CLEARANCE_AND_REMOTE_ASSETS_COMPLETE_A8_DRAFT_UNEXECUTED','goal_status':'active',
 'report':str(report),'report_sha256':sha(report),'core_matrix_records':len(matrix['rows']),'new_operator_blobs':9,'Dome_full_source_blobs':293,
 'Dome_weight_sha256':av['weight']['sha256'],'remote_CPU_state_tensors':remote['state_dict_tensors'],
 'new_full_detector_forwards':0,'full_forwards_total':185,'main_detector_updates':0,
 'files':[{'path':str(R/n),'sha256':sha(R/n)} for n in files],
 'audit_sha256':sha(outdir/'source_operator_audit.json'),'draft_sha256':sha(outdir/'A8_minimal_memory_path_draft.json')}
save(R/'27_A7源码去重与资产就绪回执.json',receipt)
print(json.dumps({k:v for k,v in receipt.items() if k!='files'},ensure_ascii=False,indent=2))
