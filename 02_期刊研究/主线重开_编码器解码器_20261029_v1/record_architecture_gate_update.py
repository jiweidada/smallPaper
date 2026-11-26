"""Record source-reading progress and a non-executable next-stage decision plan."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parent
NOW = datetime.now(timezone.utc).isoformat()

def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))

def write(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

history = ROOT / 'history_before_architecture_gate_update_v1'
history.mkdir(exist_ok=True)
for name in ['01_候选判别表.json', '13_架构检索增补记录.json', '14_架构核心文献矩阵.json',
             '15_当前状态与结果校验.json', '16_最近工作源码核查.json']:
    if not (history / name).exists():
        (history / name).write_bytes((ROOT / name).read_bytes())

matrix = read('14_架构核心文献矩阵.json')
byid = {r['id']: r for r in matrix['rows']}
byid['EASE_DETR'].update(
    read_scope='Official local PDF: complete method3.1-3.4 across PDF pages3-6; formulas/Figure2 on pages3-5 visually checked; selected results only; not all10 pages',
    author_method='Previous-layer score ordering (+1/-1) times point-box IoU; sigmoid-MLP decay changes self-attention and scales cross-attention output',
    our_consequence='Pairwise confidence/geometry modulation of both attention types is prior art; the inspected method does not marginalize localization distributions',
    unresolved='Eq4 is D*exp(W)/sum(D*exp(W)); section3.4 Eq5-6 uses max signed IoU, whereas following prose describes summed attention. No author implementation was verified to resolve that prose/formula discrepancy',
    local_pdf='source_audit_v1/EASE-DETR_CVPR2024_official.pdf',
    pdf_sha256='b4cffd5caa541bf4346f815e738cc66800f1ba487a24eed1dea9ebcf3902dba6',
    structural_preflight='PASS,10 pages; a structural result, not scientific verification')
byid['FSDC_DETR'].update(
    read_scope='abstract,introduction,section2.2 and complete methods3.1-3.4; implementation details4.2; selected results passages, not all experiments/supplement; entire current4-file default Git tree inspected',
    tier='arXiv preprint read; author repository states ECCV2026 acceptance, conference record not independently checked',
    author_method='FDConv plus partial spatial/frequency refinement in DBFSAF; split FFT/spatial/residual fusion in SFS-FF; DWT plus stride2 grouped convolution and channel gating in FSD-Down',
    our_consequence='Generic CNN/ViT spectral alignment, wavelet detail retention and frequency-aware encoder downsampling are already covered',
    unresolved='Current author default commit has4 tracked files: README,LICENSE,.gitignore,architecture image; no runnable method/config/weights. Reported800-square,100epoch,batch64,8xH200 recipe is not a measured24GB reproduction')
new_rows = [
    dict(id='Dual_R_DETR', title='Dual-R-DETR: Resolving Query Competition with Pairwise Routing in Transformer Decoders',
         url='https://arxiv.org/html/2512.13876v2', tier='primary preprint read; no independently verified venue claim',
         read_scope='abstract,introduction,complete methodsIII-A through III-D,implementation detail passages; exact query-router/feature helper,decoder integration and dual-branch detector source read; no training reproduction',
         author_method='Appearance/confidence/point-box area descriptors gate two low-rank pairwise biases; routed auxiliary and standard main decoder share training weights',
         our_consequence='Generic pairwise competition/cooperation routing is prior art; checked geometry input is scalar log area, not a localization distribution',
         unresolved='Paper motivation says standard attention is symmetric; that is not a general mathematical property of separately projected Q/K and row softmax. Source gamma signs also do not fix each unconstrained U*V pairwise bias sign. No AP claim replicated'),
    dict(id='SET', title='SET: Spectral Enhancement for Tiny Object Detection',
         url='https://openaccess.thecvf.com/content/CVPR2025/html/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.html',
         tier='CVPR2025,pp4713-4723',
         read_scope='primary abstract,introduction and Figure1 attribution description; full HBS/API method and source not read',
         author_method='Hierarchical background smoothing and adversarial feature perturbation; reported tiny-object benefit from removing some high-frequency background responses',
         our_consequence='Do not equate all high-frequency energy with useful tiny-object information; foreground/background discrimination is prior art',
         unresolved='Must read full method before proposing a related spectral gate; its FCOS/AI-TOD observation does not establish a TinyFormer/VisDrone failure cause'),
    dict(id='BS_O2G', title='Match One, Learn with Graph: One-to-Graph Query Collaboration with Backward Sharing for Object Detection',
         url='https://arxiv.org/html/2609.26092v1', tier='preprint',
         read_scope='abstract,introduction,related work and beginning of Method through shared-graph affinity Eq2; remaining method/appendix/proofs not read',
         author_method='Prediction-aware sparse post-decoder query collaboration and backward sharing while retaining one-to-one assignment',
         our_consequence='Generic feature/box/class query graph cooperation and sharing complementary evidence are already proposed',
         unresolved='Only partial method scope; do not claim this work lacks every form of uncertainty reasoning without further reading')
]
for row in new_rows:
    if row['id'] in byid:
        byid[row['id']].update(row)
    else:
        matrix['rows'].append(row)
matrix['updated_utc'] = NOW
matrix['scope_note'] = 'Non-systematic targeted review. Returned search excerpts do not imply complete reading; no human read attestation inferred.'
write('14_架构核心文献矩阵.json', matrix)

sources = read('16_最近工作源码核查.json')
receipts = read('comparison_sources_v1/supplement_acquisition_receipts.json')
for receipt in receipts:
    assert receipt['status'] == 'ALL_TRACKED_BLOBS_VERIFIED'
    name = Path(receipt['local']).name
    files = receipt['files']
    source_row = dict(commit=receipt['commit'], verified_files=len(files),
                      repo=receipt['repo'], acquisition_receipt='comparison_sources_v1/supplement_acquisition_receipts.json')
    if name == 'FSDC-DETR':
        source_row.update(status='NO_RUNNABLE_IMPLEMENTATION_IN_VERIFIED_DEFAULT_TREE',
                          tracked_paths=[f['path'] for f in files],
                          note='README quick-start references requirements.txt, which is not present in this commit. No package install or model execution attempted.')
    else:
        source_row.update(status='ROUTER_AND_DUAL_BRANCH_INTEGRATION_READ_NOT_EXECUTED', symbols=[])
        rels = ['models/bricks/query_router.py', 'models/bricks/deformable_transformer.py',
                'models/detectors/deformable_detr.py']
        for rel in rels:
            path = Path(receipt['local']) / rel
            data = path.read_bytes()
            file_sha = hashlib.sha256(data).hexdigest()
            file_row = next(x for x in files if x['path'] == rel)
            assert file_sha == file_row['sha256']
            tree = ast.parse(data.decode('utf-8-sig'))
            wanted = {'QueryRouter', 'compute_query_features', 'DeformableTransformerDecoder',
                      'DeformableTransformerDecoderLayer', 'DeformableDETR'}
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
                    source_row['symbols'].append(dict(symbol=node.name, file=rel, line=node.lineno,
                                                      end_line=node.end_lineno, file_sha256=file_sha))
        source_row['observations'] = [
            'Geometry helper uses log(w*h), class helper uses softmax previous logits, similarity uses mean positive cosine with diagonal removed; these are source behaviors, not a correctness endorsement.',
            'Pairwise bilinear gating and two unconstrained low-rank products create a clamped additive attention bias; no bbox covariance or FDR probability input in the read operators.',
            'Main/aux training paths share one encoder and run the decoder twice. Inference detector forwards the gating state as-is; exact evaluation disabling must be checked if reproducing.',
            'No pretrained weights downloaded, no dependencies installed, no source from these projects imported into the research environment.'
        ]
    sources[name.replace('-', '_')] = source_row
sources['EASE_DETR'] = dict(status='OFFICIAL_PDF_METHOD_READ_NO_VERIFIED_AUTHOR_CODE',
                            pdf='source_audit_v1/EASE-DETR_CVPR2024_official.pdf',
                            sha256=byid['EASE_DETR']['pdf_sha256'],
                            preflight='source_audit_v1/EASE-DETR_CVPR2024_pdf_preflight.json',
                            visual_pages=[3,4,5], read_method_pages=[3,4,5,6])
sources['updated_utc'] = NOW
write('16_最近工作源码核查.json', sources)

log = read('13_架构检索增补记录.json')
log['supplement_utc'] = NOW
log['query_groups'].extend([
    dict(name='source and query-routing clearance supplement', queries=[
        '"EASE-DETR" github', '"Dual-R-DETR" detection relation', '"Route-DETR" uncertainty query relation']),
    dict(name='encoder aliasing and separability supplement', queries=[
        '"DETR" "small objects" "aliasing"', '"tiny object detection" "shift" "encoder"',
        '"small object" "polyphase" detection', '"DETR" "localization uncertainty" "relation"',
        '"SET" "tiny object detection" "spectral"', '"DETR" "instance separability"',
        '"tiny object" "feature smearing"', '"DETR" "query generation" "sub-pixel"']),
    dict(name='memory and relation uncertainty supplement', queries=[
        'DETR small object "dual memory" localization', 'DETR "detail memory" query',
        '"small object detection" "decoupled" "memory"', 'DETR "uncertainty" "pairwise" "box" attention',
        '"DETR" "distribution-aware" "relation"', '"DETR" "uncertainty-aware" "pairwise"',
        '"DETR" "expected IoU" attention', '"probabilistic" "relative position" "object detection" transformer'])
])
raw = []
for path in sorted((ROOT / 'source_audit_v1').glob('arch_*.json')):
    raw.append(dict(file=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
log['raw_architecture_excerpt_inventory'] = raw
log['new_scope_outcomes'] = [
    'EASE official PDF obtained,structural preflight PASS,complete selected method read; prior403 access statement superseded for current status.',
    'FSDC entire3.1-3.4 method and4.2 recipe now read;4-file author default tree lacks implementation.',
    'Dual-R primary full method and96-file author Git tree acquired; selected source verified/read.',
    'SET contradicts the simplistic assumption that more high-frequency energy is necessarily better; only primary abstract,introduction,Figure1 read.',
    'No exhaustive novelty conclusion. Absence of a result for an exact search string is not proof of a gap.'
]
write('13_架构检索增补记录.json', log)

board = read('01_候选判别表.json')
board['latest_update_utc'] = NOW
board['latest_report'] = str(ROOT / '17_架构方向收敛与最近工作对比.txt')
board['status'] = 'ARCHITECTURE_SOURCE_GATE_ADVANCED_MAINLINE_NOT_VALIDATED'
board['current_priority_question'] = '在相同输入与候选预算下，微小目标的定位线索在哪个编码/融合/读出环节失去可用性？先区分表示不足、读出不足和输入缩放瓶颈。'
for candidate in board['candidates']:
    if candidate['id'] == 'A_ENCODER_INSTANCE_SEPARABILITY':
        candidate['nearest'] = ['TinyFormer SSA/PBM', 'DEIMv2 STA', 'FSDC-DETR complete3.1-3.4 method',
                                'UHR-DETR', 'SET(partial scope)', 'Anchor DETR for same-location multiplicity controls']
        candidate['next'] = 'Design a matched spatial-information/readout probe; ordinary P2,larger input,denser candidates and readout supervision are required explanations/controls, not innovations.'
    if candidate['id'] == 'D_DISTRIBUTIONAL_QUERY_RELATIONS':
        candidate['status'] = 'SECONDARY_HYPOTHESIS_NO_DEMONSTRATED_FAILURE_CAUSE'
        candidate['nearest'] += ['Dual-R-DETR primary method and exact router source', 'BS-O2G(partial method)']
        candidate['next'] = 'If pursued, first validate uncertainty relevance within size and initial-quality strata and identify a relation-dependent causal effect. Point-box operator sensitivity alone is insufficient; no training chosen.'
board['rejected_generic_novelty_claims'] += [
    'frequency-aware CNN/ViT fusion or wavelet downsampling alone',
    'generic query competition/cooperation routing',
    'multiple slots per spatial location without a stronger distinction',
    'high-frequency energy increase as a sufficient explanation of tiny-object benefit'
]
board['next_decision_protocol'] = str(ROOT / '18_下一阶段最小判别方案_未执行.json')
board['next_decision_is_not_algorithm_approval'] = True
write('01_候选判别表.json', board)

protocol = dict(
    material_passport=dict(origin_skill='academic-research-suite', origin_mode='targeted research and experiment design',
                           origin_date=NOW, verification_status='DESIGN_ONLY_NOT_EXECUTION_READY_OR_RUN', version='architecture_decision_v1'),
    status='DRAFT_NOT_EXECUTED',
    question=board['current_priority_question'],
    current_measurement=dict(full_model_forwards=169, current_hard_cap=200, new_training_updates=0,
                             new_architecture_interventions=0, note='This document neither raises the existing cap nor starts a new queue.'),
    ordered_gates=[
        dict(id='G0_source_to_signal', goal='Select an exact tensor boundary and a falsifiable failure mechanism',
             required=['Map native P2 detail tensor,PBM fusion,encoder proposal readout and resize coordinates from immutable source.',
                       'Specify what would count against input shrinkage,readout-supervision and proposal-budget explanations.',
                       'Finish full closest method reading for any mechanism promoted; a broad detail-preserving or uncertainty label is insufficient.'],
             stop='If the increment reduces to existing SSA/PBM/P2,spectral fusion,multi-pattern anchors or query routing, reject that increment before training.'),
        dict(id='G1_matched_representation_probe', goal='Distinguish accessible local information from a fixed-head readout gap',
             possible_design='Frozen features,identical small diagnostic readouts with source-image/sequence-blocked fit/evaluation,matched ROI geometry/channels/parameter budget. GT-assisted ROI probes measure representation only; they are not deployable recall/AP.',
             controls=['Shallow detail versus post-fusion feature with identical feature normalization and physical ROI support.',
                       'Re-fit the same readout on native fused features to expose a supervision/readout explanation.',
                       'Randomized shallow features and spatially displaced ROIs as negative controls.',
                       'Stratify by input sqrt-area,initial candidate quality,clutter and truncation; inspect class and sequence imbalance.'],
             limitations=['Equal parameter counts do not guarantee equal capacity or spatial information; specify actual tensors before freezing.',
                          'Train-seen author checkpoint does not provide independent generalization evidence.',
                          'Small positive probes do not prove irreversible information loss or architecture benefit.'],
             execution_gate='Exact sample IDs,split,probe capacity,optimization budget,primary endpoint and code are not frozen here. No extraction or probe fitting launched.'),
        dict(id='G2_minimal_architecture_screen', goal='Only after G0/G1 support a distinct mechanism, screen one actual replacement',
             proposed_arms=['Native small detector baseline', 'Simplest existing explanation/control selected by G1,e.g.direct P2 readout or denser candidates',
                            'One precisely specified architecture replacement'],
             fairness=['Same source initialization,training input stream,schedule,effective batch,query count and optimizer updates.',
                       'Account separately for candidate token count,parameters,MACs,peak training memory and end-to-end latency.',
                       'No paired extra restoration,codec,data augmentation or loss change unless the single researched mechanism requires it.'],
             budget='Proposed short screen: at most12epochs per arm,one prespecified seed; not paper-level evidence. Benchmark real forward/backward capacity and throughput before deciding whether the full recipe fits24GB and days.',
             metrics=['Standard COCO-style AP/AP50/AP75 on a frozen VisDrone validation protocol; document maxDets and ignore-region handling.',
                      'Supplementary input-scale <8,8-16,16-32 pixel metrics and candidate-stage coverage; do not call these COCO APs.',
                      'Matched-size/initial-quality diagnostic endpoints plus recovery/loss pairs,not cherry-picked images.'],
             stop='No mechanism-specific advantage beyond the simple control,unacceptable cost,or unstable implementation means stop this candidate; do not add extra modules to rescue it.'),
        dict(id='G3_paper_evidence_if_screen_passes', goal='Decide publication value only from completed matched evidence',
             required=['Train a complete same-recipe baseline/control/candidate with at least two paired seeds,report all runs.',
                       'Validate the claimed mechanism and a relevant second dataset such as AI-TOD-v2 if resources permit; report a single-dataset limitation otherwise.',
                       'Compare verified available closest implementations; do not substitute paper-table numbers at different sizes/schedules as controlled results.',
                       'Test a transferable second architecture if claiming a general encoder/decoder mechanism.'],
             execution_gate='No full-training or additional-data queue is scheduled by this draft.')
    ],
    secondary_decoder_question=dict(question='Can localization-distribution information improve pairwise geometric relations beyond point-box priors?',
                                    necessary_controls=['Point relation baseline','Scale-only or class-confidence reliability gate','Constant noise/jitter marginalization','Distribution-derived relation with matched compute'],
                                    primary_risks=['FDR bins are not automatically calibrated bbox posteriors.','Marginalized relation may only smooth geometry or erase useful resolution.','A0 has no Relation-DETR prior,so it cannot establish a failure of that operator.'],
                                    status='SECONDARY_NOT_SELECTED_FOR_TRAINING')
)
write('18_下一阶段最小判别方案_未执行.json', protocol)

report = '''Material Passport
Origin Skill: academic-research-suite
Origin Mode: targeted non-systematic architecture research
Origin Date: {now}
Verification Status: source/file checks passed as stated; scientific gaps and proposed increments remain UNVERIFIED
Version Label: architecture_mainline_gate_v1

当前判断：把开放选题长期锁在压缩/恢复上，确实收窄过头。该分支保留历史记录，当前改查无人机小目标检测自身的编码、特征交互和候选读出问题。新架构值得查，但目前没有可诚实称为“已验证论文主线”的算法。

最值得继续的问题：相同输入与候选预算下，小目标的定位线索在编码/融合时变弱了，还是仍然存在、但现有候选预测头没有读出来？输入缩放造成的线索不足也必须作为竞争解释。
这比“换一个新encoder/decoder”更具体；它还不是新模块的名称。

1. 检索策略与记录
沿用定向、非系统调研：先找编码细节通路/异构融合/候选生成及查询交互，再检索同位置多候选、频域保留、分布或不确定性关系，最后核对作者源码。
本次补充19条查询，覆盖EASE、Dual-R/Route、aliasing/polyphase、detail/dual memory、instance separability与distribution-aware pair relations。准确查询串、返回片段文件与SHA见13_架构检索增补记录.json。
只把可核验的论文或作者仓库作为技术依据；第三方标题与解读仅用于发现，不当作已读原文。没有系统综述或“无人做过”的结论。

2. 核心文献矩阵与实际阅读范围
15条核心记录见14_架构核心文献矩阵.json，逐条分原文方法、阅读范围、我们的推断、未解决项。本次更新如下：
TinyFormer：原有SSA/PBM已有浅层细节通路；完整对应源码、作者EMA和诊断已核对。
FSDC-DETR：现已读完3.1-3.4方法及4.2训练细节。DBFSAF/FFT-空间分路融合/DWT与分组卷积下采样均已有，不能再以“加频域融合”申报空白。作者默认仓库4文件，没有可运行实现；原文800输入、100轮、batch64、8张H200，不等于本机可复现配方。
SET：仅原文摘要、引言和图1归因描述。其结果提示某些高频背景会干扰微小目标；不能用高频能量增加代替有用定位信息增加。
EASE-DETR：官方PDF已取得，结构预检10页PASS；完整方法3.1-3.4已读，图2及主要公式页已目视检查。并非全10页或全文实验读完。
Relation DETR：位置关系算子和解码器源码已读；归一化相对点框几何经sin/cos及可学习投影参与注意力。
Dual-R-DETR：完整III-A至III-D和路由器/双分支源码已读；相似度、置信度、点框log面积参与成对路由，没有在所读算子中使用定位分布。
DETRNN：完整3.1-3.4已读；当前作者默认树只有README，不是现成可运行对照。
DS-Det、不确定性DETR、UHR-DETR及BS-O2G的阅读边界均见矩阵；其中BS-O2G只读到方法开头，不声称已排除它的所有细节。

3. 与最接近工作的逐项对比
比较项 | 已有工作覆盖 | 当前可能追问 | 判断
浅层细节进入编码器 | TinyFormer SSA/PBM、DEIMv2 STA | 混合后是否仍可读出实例定位线索 | “再加浅层支路”不足以成创新
频率/空间协同与保留 | FSDC-DETR、SET(部分阅读) | 有用定位线索与背景高频如何区分 | 不能以FFT/小波或高频增强本身为增量
全局语义＋局部细节读取 | UHR-DETR的稀疏高分辨率编码/全局局部解码 | 普通VisDrone与固定预算下的具体读出缺陷 | 大图裁块或双分支概念已有
同位置多对象 | Anchor DETR多pattern | 真实局部实例可分性，而非增加slot | 多slot本身已有，必须有更强机制差异
查询间点框关系 | Relation DETR、EASE-DETR | 定位分布是否使关系更可靠 | 这是备选假设，A0不构成其失败证据
查询竞争/协作路由 | Dual-R-DETR、DETRNN、BS-O2G | 新方法是否超出普通路由与证据共享 | 不能把query routing当空白
不确定性使用位置 | 已有Gaussian/Bayes-risk输出、损失和匹配调整 | 分布参与成对关系而非每query标量重权 | 有算子位置差异，尚无效用或新颖性最终结论

EASE方法核对要点：上一层score确定二元领先/落后关系，与上一层点框IoU相乘后经MLP/sigmoid。自注意力原文Eq4是D*exp(W)再逐行归一化，等价于在W上加log(D)，不能写成简单乘logits。交叉注意力输出另乘逐query/逐head衰减。Eq5-6写max signed IoU，随后文字另描述attention求和，存在表述不一致；未取得作者实现，不能擅自选择一种当源码事实。
Dual-R方法的“普通注意力必为对称”不是一般数学事实；Q/K可不同，逐行softmax也不保证对称。引用其路由方法，不照搬这个动机。源中低秩乘积无符号约束，系数正负也不保证每条边一定抑制/促进。这些是我们的源码/数学核查，不是作者自述结论。

4. 有依据的未解决问题；区分事实、推断、假设
[本轮已有实测] 100张训练集探索图、4491个GT，最终IoU0.5几何未覆盖586个，其中475个在全部编码候选的追加回归读出中也未覆盖；而decoder从第一层到末层的总体覆盖增加。
[限制] 追加回归读出不是编码信息上界，未选位置没有同等监督；该权重见过原训练集，覆盖不是AP。不能由此证明encoder不可逆丢失信息。
[推断] 优先分辨候选产生之前的表示/融合/读出瓶颈，比先加query或防止整个decoder漂移更贴近当前证据。
[待验证H1] 浅层有可用于定位的线索，统一融合/读出后其可用性下降；相同容量、物理区域与监督的诊断读出应能区分这个解释。
[竞争解释H0] 缩放后本来就缺乏像素，或现有预测头监督/容量不足。若重拟合同一读出已解决，不能归因新的encoder；若升输入分辨率解释全部恢复，也不能宣称新细节机制成立。
[待验证H2，次选] 点框几何关系在不确定定位下可能不稳，分布参与关系可否改善。FDR离散回归分布不自动等于校准后验，整体decoder改善也不支持把它当主要损害来源。
以上是限定模型/条件下可证伪的问题，既不等于整个领域无人研究，也没有“首次”结论。

5. 候选算法增量、可行性与风险
优先保留的设计空间是：在候选形成阶段，改变细节定位线索的表示与读取，使它在被混合进粗网格之前仍能被有效利用。只有H1及最近工作对照支持时，才把“内容语义与局部定位线索分别编码、在候选形成时交互”收敛成一个具体替换算子。目前不批准把加P2、space-to-depth、多pattern或普通残差支路单独当论文创新。
这个空间最主要的风险是已有TinyFormer/PBM、FSDC及其他局部读取设计已经解释增益；输入/候选预算或读出监督也可能解释全部变化。因此最近工作差异和简单控制必须先成立，不能靠多模块叠加。
备选是定位分布参与成对几何关系：与“点框sin/cos/IoU”或“每query风险重权”有作用位置差别。它需击败尺度门控、类别置信度门控与固定抖动平滑；还需证明分布可靠且有关系相关的失败。当前不优先训练它。
普通共享encoder value、直接换Mamba/最新backbone、加频域或增加query已降级为现有设计或控制项。

6. 最小验证路线；本次没有执行新实验
具体分门方案见18_下一阶段最小判别方案_未执行.json，状态DRAFT，不是已冻结或已开训。
先定精确张量边界和匹配的表示/读出诊断；同物理范围、归一化、容量和监督比较浅层/融合后信息，并用native读出重拟合、错位区域及随机特征控制。GT辅助ROI仅用于表示诊断，不能报告为可部署召回率。
若能区分H1与H0，再只实现一个替换：原生baseline、最简单的已有解释控制、候选三个arm；同初始化、输入流、query数、更新步数与预算。短筛可先固定各不超过12轮、一个seed；不靠短筛写最终论文。
正式证据再考虑完整配方、至少两组配对seed及合适的第二数据集。是否能在24GB和数天内完成，以真实前反向容量和吞吐测量决定。
主数据仍用VisDrone原图；当前可追踪对象是TinyFormer-S-PBM作者VisDrone EMA，完整源码187文件。它是机制诊断对象，未锁定为最终基线；YOLO26s可以保留参考，但不能与不同输入/训练/后处理的数值直接排名。
作者EMA推理已在3090/24GB成功，不代表训练显存通过。当前总计169/200次完整前向、0新训练更新；本次文献/源码更新未新增前向，旧200次上限未改变，也未启动任何自动训练队列。

7. 可核验参考链接
DeepInfer（若用户所指为该推理框架）：https://github.com/deepseek-ai/DeepInfer
DeepSeek-V4.1-Flash（架构灵感，非检测效用证据）：https://arxiv.org/html/2609.19969v1
VisDrone作者数据页：https://github.com/VisDrone/VisDrone-Dataset
TinyFormer：https://arxiv.org/html/2605.25046v1 ；https://github.com/mmpmmpmmpjosh/TinyFormer
FSDC-DETR：https://arxiv.org/html/2607.05176v1 ；https://github.com/nevereverinsomnia/FSDC-DETR
SET：https://openaccess.thecvf.com/content/CVPR2025/html/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.html
UHR-DETR：https://arxiv.org/html/2604.21435v1
Relation DETR：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06646.pdf ；https://github.com/xiuqhou/Relation-DETR
EASE-DETR：https://openaccess.thecvf.com/content/CVPR2024/papers/Gao_EASE-DETR_Easing_the_Competition_among_Object_Queries_CVPR_2024_paper.pdf
Dual-R-DETR：https://arxiv.org/html/2512.13876v2 ；https://github.com/YZk67/Dual-R-DETR
DETRNN：https://arxiv.org/pdf/2610.05191 ；https://github.com/nigelwoo2008-coder/DETRNN
DS-Det：https://doi.org/10.1145/3746027.3755045
An Uncertainty-aware DETR Enhancement Framework：https://arxiv.org/pdf/2507.14855
BS-O2G：https://arxiv.org/html/2609.26092v1
Anchor DETR：https://arxiv.org/pdf/2109.07107

研究判断：架构方向值得继续做一次能区分原因的诊断；当前不值得直接投入完整新模块训练，更不能许诺期刊录用。先前压缩/恢复资产保留，不追加该分支，不重启旧DEIM实验。
本次属于progress：补齐两个关键完整方法，取得并核对两个作者Git树，排除宽泛创新表述，形成了下一次判别的门槛。主目标保持active，尚未完成“找到可支持论文的具体算法主线”。
AI披露：单助手辅助检索、源码核查和报告整理；未推定用户已阅读任何文献，未进行人工同行复核。
'''.format(now=NOW)
# The query log above contains3+8+8=19 exactly; preserve that count as a record, not a search-exhaustiveness claim.
(ROOT / '17_架构方向收敛与最近工作对比.txt').write_text(report, encoding='utf-8')

state = read('15_当前状态与结果校验.json')
state['architecture_source_update'] = dict(utc=NOW, full_model_forwards_unchanged=169,
    new_training_updates_this_update=0, core_records=len(matrix['rows']),
    verified_supplemental_trees=[dict(repo=x['repo'],commit=x['commit'],files=len(x['files'])) for x in receipts],
    latest_report=str(ROOT / '17_架构方向收敛与最近工作对比.txt'),
    next_protocol_status='DRAFT_NOT_EXECUTED', scientific_mainline='NOT_VALIDATED')
write('15_当前状态与结果校验.json', state)
print(json.dumps(dict(status='RECORDED_NO_MODEL_EXECUTION', core_records=len(matrix['rows']),
                      verified_new_source_files=sum(len(x['files']) for x in receipts), full_model_forwards=169,
                      new_training_updates=0), ensure_ascii=False))
