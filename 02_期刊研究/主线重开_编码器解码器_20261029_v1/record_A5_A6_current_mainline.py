"""Reconcile completed bounded diagnostics; no model execution or new fitting."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent
NOW = datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

names = ['01_候选判别表.json', '13_架构检索增补记录.json',
         '14_架构核心文献矩阵.json', '15_当前状态与结果校验.json', '16_最近工作源码核查.json']
history = ROOT / 'history_before_A5_A6_update_v1'
history.mkdir(exist_ok=True)
for name in names:
    target = history / name
    if not target.exists():
        shutil.copy2(ROOT / name, target)

evidence = {}
for label, folder in [('A5', 'A5_既有中心场局部Taylor控制_v1'),
                      ('A6', 'A6_稀疏编码器前景监督覆盖审计_v1')]:
    p = ROOT / folder
    done, result = read(p/'completed.json'), read(p/'summary.json')
    assert done['summary_sha256'] == sha(p/'summary.json')
    receipt = read(p/'backup'/'receipt.json')
    assert receipt['status'].startswith('PASS_')
    assert receipt['archive_sha256'] == sha(receipt['archive'])
    evidence[label] = {'directory': str(p), 'decision': result['decision'],
                       'summary_sha256': done['summary_sha256'], 'receipt': receipt}

a5 = read(Path(evidence['A5']['directory'])/'summary.json')
a6 = read(Path(evidence['A6']['directory'])/'summary.json')
assert a5['scientific_mainline_validated'] is False
assert a6['scientific_mainline_validated'] is False
question = '固定编码预算下，前景置信度或密度引导的稀疏编码，是否错过了对微小目标定位最有价值的编码更新？'
status = 'CANDIDATE_ENCODER_ALLOCATION_QUESTION_ONLY_NO_NOVEL_OPERATOR_OR_CAUSAL_EFFICACY_VALIDATED'
report = ROOT / '24_当前候选主线与A5A6证据边界.txt'

board = read(ROOT/names[0])
board['latest_update_utc'] = NOW
board['status'] = status
board['mainline_validated'] = False
board['current_priority_question'] = question
board['model_candidates'] = ['TinyFormer-S-PBM: historical frozen diagnostic baseline',
                             'Focus-DETR: source-bound foreground-target audit; no full model executed',
                             'Dome-DETR-S: potential compact sparse-encoder comparator; not installed or measured']
board['latest_report'] = str(report)
board['A5_A6_evidence'] = evidence
board['remaining_deliverable'] = 'Establish actual update-selection failures and a causal benefit at fixed compute; clear SViT/Salience/Dome/UHR closest operators before choosing a new algorithm. Training memory feasibility remains unverified.'
for c in board.get('candidates', []):
    if c.get('id') == 'E_CENTER_TO_BOX_SPATIAL_CORRESPONDENCE':
        c['status'] = 'DEPRIORITIZED_A4_NEGATIVE_A5_KNOWN_LOCAL_TAYLOR_CONTROL_NEGATIVE'
        c['next'] = 'No generic center-refinement architecture training or additional head search justified.'
        c['read_scope_correction'] = 'DARK primary arXiv methodology and SD-DETR primary method sections were subsequently read; prior abstract-only labels are historical.'
new_candidate = {
 'id': 'F_SPARSE_ENCODER_UPDATE_ALLOCATION', 'status': status, 'question': question,
 'increment_approved': False, 'causal_evidence': False,
 'evidence': 'A6 source-exact foreground-target geometry; no actual Focus predictions, selected updates or AP measured.',
 'nearest': ['Focus-DETR', 'RFLA', 'Dome-DETR', 'UHR-DETR', 'Salience-DETR', 'SViT'],
 'not_novel_by_itself': ['fallback positive labels', 'Gaussian density supervision',
                       'area-normalized target coverage or greedy patch selection',
                       'task-aware token gates or preservation of unselected features'],
 'possible_increment': 'An update-benefit criterion that differs materially from existing task-aware gates; not designed or novelty-cleared.',
 'next': 'Complete SViT/Salience operator comparison and define a fixed-budget causal control before model training.'}
board['candidates'] = [c for c in board.get('candidates', []) if c.get('id') != new_candidate['id']] + [new_candidate]
write(ROOT/names[0], board)

log = read(ROOT/names[1])
raw_names = [
 'precision_DARK_SDDETR_method_locators', 'precision_DARK_SDDETR_primary_alternatives',
 'precision_DARK_SDDETR_fulltext_retrieval', 'precision_DARK_SDDETR_method_read_1',
 'precision_DARK_SDDETR_method_read_2', 'architecture_open_tokenization_and_position_search',
 'architecture_null_evidence_and_differential_attention_clearance',
 'architecture_sparse_encoder_foreground_label_prior_methods',
 'architecture_Focus_Dome_DQ_foreground_method_read', 'architecture_foreground_label_full_sections',
 'architecture_Dome_DQ_density_target_PDF_locators',
 'architecture_sparse_encoder_bibliography_and_RFLA_primary',
 'architecture_Dome_v2_RFLA_target_methods', 'architecture_Focus_author_resize_and_protocol',
 'architecture_RFLA_HLA_and_encoder_coverage_closest', 'architecture_Dome_author_assets_current',
 'architecture_instance_coverage_loss_and_UHR_clearance',
 'architecture_update_utility_salience_gradient_prior_search']
inventory = []
for name in raw_names:
    p = ROOT/'source_audit_v1'/(name+'.json')
    assert p.is_file()
    inventory.append({'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)})
log['A5_A6_evidence_inventory'] = {'reconciled_utc': NOW, 'raw_result_files': inventory,
 'review_type': 'Targeted non-systematic architecture pre-study; not PRISMA or exhaustive coverage.',
 'query_record_limit': 'Saved raw tool responses retain source/retrieval evidence. Exact arguments not present in these response-only receipts are not reconstructed as if contemporaneously logged.',
 'outcome': 'Generic coordinate refinement deprioritized; foreground-target geometry verified; simple positive fallback, Gaussian density and coverage-greedy remedies already have close prior work.',
 'unresolved': 'Full SViT and Salience implementation comparison; actual sparse-update causal effects; novel operator; training resource feasibility.'}
write(ROOT/names[1], log)

matrix = read(ROOT/names[2])
rows = matrix['rows']
by_id = {r['id']: r for r in rows}
for key, scope in {
 'DARK': 'Primary arXiv1910.06278v1 methodology including log-Gaussian Taylor decoding and modulation; publisher record checked, final CVF PDF fetch403. A5 is an incomplete local control, not full DARK reproduction.',
 'SDDETR': 'Primary arXiv2310.15955 sections3.1,3.2.1–3.2.3,3.3 and selected setup; CVF PDF fetch403. Author implementation not acquired.',
}.items():
    if key in by_id:
        by_id[key]['read_scope'] = scope
        by_id[key]['updated_utc'] = NOW
if 'UHR_DETR' in by_id:
    by_id['UHR_DETR']['additional_overlap_audit'] = 'III-B equations4–8 already define per-instance area-normalized overlap gain, maximum coverage and iterative soft-subtraction greedy patch selection. Token granularity alone is not a sufficient novel distinction.'
new_rows = [
 {'id':'Focus_DETR','title':'Less is More: Focus Attention for Efficient DETR','url':'https://arxiv.org/abs/2307.12612','tier':'ICCV2023',
  'read_scope':'Primary paper3.2,selected3.1/3.3,appendixA.1.1–A.1.2; seven author Git blobs and relevant target/criterion/encoder call chain. A6 executes only two parameter-free AST definitions on CPU.',
  'author_method':'Foreground-supervised cascaded token selection with separate detection losses; unselected tokens remain in memory.',
  'our_consequence':'A6 verifies an exact foreground-label geometric gap, not deletion of features, lack of detection supervision or measured missed detections.',
  'source_commit':'5a088143f01ce106af2ba0c58bf9951146b60e5a'},
 {'id':'RFLA','title':'RFLA: Gaussian Receptive Field based Label Assignment for Tiny Object Detection','url':'https://arxiv.org/abs/2208.08738','tier':'ECCV2022',
  'read_scope':'Primary abstract/introduction and3.1–3.2; complete3.3 HLA not yet confirmed in saved read scope; source not acquired.',
  'author_method':'Gaussian receptive-field distances and hierarchical label assignment address tiny-GT outliers under box/point priors.',
  'our_consequence':'Missing positive samples and generic fallback assignment are existing issues; transferring a known fix to an encoder is not by itself a defensible innovation.'},
 {'id':'Dome_DETR','title':'Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection','url':'https://arxiv.org/abs/2505.05741','tier':'arXivv2; authors report ACM MM2025 acceptance',
  'read_scope':'Primary latest-v2 PDF selected3.2/3.3 and implementation appendix; four official source/config blobs. No complete project, imports, weights or execution.',
  'author_method':'Gaussian density supervision, density-focal extraction, masked window sparsification and progressive adaptive query initialization.',
  'our_consequence':'Gaussian coverage, density-guided sparse encoding and variable query budgets are close prior art. Source fallback guarantees a foreground region per image, not a distinct selected region per GT.',
  'source_commit':'2dde3bc1946a3e9fad9abd0612b59fc39bd6b861'},
 {'id':'Salience_DETR','title':'Salience DETR: Enhancing Detection Transformer with Hierarchical Salience Filtering Refinement','url':'https://openaccess.thecvf.com/content/CVPR2024/html/Hou_Salience_DETR_Enhancing_Detection_Transformer_with_Hierarchical_Salience_Filtering_Refinement_CVPR_2024_paper.html','tier':'CVPR2024',
  'read_scope':'Primary section3.1 and opening3.2; author README; source operators not acquired.',
  'author_method':'Scale-independent soft salience supervision and hierarchical filtering/refinement.',
  'our_consequence':'Scale normalization or replacing binary foreground targets with salience is not a new idea; A6 rates must not be generalized to this implementation.'},
 {'id':'SViT','title':'Revisiting Token Pruning for Object Detection and Instance Segmentation','url':'https://openaccess.thecvf.com/content/WACV2024/papers/Liu_Revisiting_Token_Pruning_for_Object_Detection_and_Instance_Segmentation_WACV_2024_paper.pdf','tier':'WACV2024',
  'read_scope':'Search-visible method/setup passages; complete section3 and exact implementation not yet audited. Partial reading, not full paper.',
  'author_method':'Task-trained token gating for dense prediction with preserved token features; exact gating/reactivation details still require audit.',
  'our_consequence':'Generic learned/task-aware token gates, feature preservation and compute-adaptive allocation cannot be treated as new. Update-benefit routing remains unverified.'},
]
for row in new_rows:
    row['updated_utc'] = NOW
    if row['id'] in by_id:
        by_id[row['id']].update(row)
    else:
        rows.append(row)
matrix['updated_utc'] = NOW
matrix['latest_report'] = str(report)
matrix['latest_empirical_boundary'] = 'A5 negative known local readout control; A6 geometric foreground-label audit only. No new architecture efficacy, full detector AP or independent generalization demonstrated.'
write(ROOT/names[2], matrix)

source = read(ROOT/names[4])
source['updated_utc'] = NOW
for key, folder in [('Focus_DETR','Focus_DETR_operators_v1'),('Dome_DETR','Dome_DETR_operators_v1')]:
    p = ROOT/'comparison_sources_v1'/folder/'source_verification.json'
    source[key] = {'verification_file': str(p), 'verification': read(p)}
source['Focus_DETR']['isolated_CPU_execution'] = read(Path(evidence['A6']['directory'])/'author_source_equivalence.json')
source['scope_this_update'] = 'Retrospective reconciliation of completed A5/A6 and nearest sources; no additional models run or fit during this record update.'
source['current_novelty_boundary'] = 'Simple label fallback/Gaussian density/coverage greedy and generic learned token gating have close prior art; no novel encoder operator validated.'
write(ROOT/names[4], source)

e = a6['estimates']['square640_source']
r = a6['estimates']['aspect800_reference_source']
def pct(x):
    return f"{100*x['mean']:.2f}% [95%CI {100*x['low']:.2f}%,{100*x['high']:.2f}%]"

text = f'''当前候选主线与A5/A6证据边界
记录时间：{NOW}

当前回答
最值得保留的候选问题：{question}
它位于视觉编码器及特征交互方向。研究目标仍是无人机航拍小目标检测，候选数据为VisDrone。
当前有可核验的监督覆盖问题，尚没有经过最近工作去重和因果验证的新算法，不能称为已找到可投稿的主线。
TinyFormer-S是已有冻结诊断基线；Focus-DETR是本次监督源码审计对象；Dome-DETR-S是可能采用的紧凑稀疏编码比较对象，均不等于新算法已经选定。

1. 检索策略与记录
本次为非系统性的定向前期调研。围绕精细坐标读出、稀疏编码前景监督、密度/覆盖计算分配，以及任务驱动token门控逐层查证。
先核对主要论文，再追溯官方仓库的实际调用链；负结果用于淘汰候选，不据空检索声称首次。
13_架构检索增补记录.json保留既有查询组和本次18份原始响应的SHA清单。响应中未保留的确切检索参数不事后伪造。检索非穷尽。

2. 核心文献与阅读范围
完整动态矩阵见14_架构核心文献矩阵.json，目前{len(rows)}项。原文结论、源码核查、我方测量和推断分开记。
Focus-DETR：论文3.2及相关附录、7个固定Git源文件；仅两段无参数标签函数在CPU隔离执行，未运行完整检测器。
Dome-DETR：原始v2论文的密度/掩码相关部分与实现附录、4个官方源码/配置文件；未下载权重、未运行。
RFLA：摘要/引言与3.1–3.2；完整HLA段落尚未确认，未读官方源码。
Salience-DETR：3.1及3.2开头、官方README；未做源码算子核查。
SViT：检索可见的方法/设置段落；完整方法与实现尚未核查。以上部分阅读均不得表述为通读全文。
DARK：读到主arXiv版本的完整方法；A5不包含完整高斯调制或稠密热图，不是完整DARK复现。

3. 最接近工作的逐项对比
Focus-DETR按前景评分筛选编码token，独立Hungarian检测监督仍存在；未选token保留在memory，不等于特征被删除。源码尺度区间与论文附录的重叠区间不同，A6分别记录。
RFLA已经讨论点/框先验造成的微小GT正样本缺口。最近点补正样本或换距离规则，不能作为本项目的新问题或独立创新。
Dome-DETR已有归一化高斯密度监督、密度掩码稀疏窗口和动态query。简单加高斯密度图、保障每图一个前景区不足以构成区别。
UHR-DETR已有按目标面积归一化的交叠覆盖收益及贪心区域选择。把同类目标从patch缩小为token，尚不是充分差别。
Salience-DETR已有尺度独立显著性筛选；SViT已有任务驱动门控。按任务学习更新分配不能单独称为新机制，需完成算子级对照。

4. 已验证的问题与待证研究空白
【我方测量】A6对100张固定VisDrone训练图的4491个有效GT进行几何审计。640输入下sqrt(area)<8像素的固定物理目标共901个，分布在62张图、33个文件前缀组。
按所核查Focus源码的严格中心在框内及尺度规则，该组缺少自身前景正token标签的比例，按图平均为{pct(e['fixed640_lt8_no_positive'])}；合并GT计数为499/901，二者权重口径不同。
同时缺少邻近3×3 P3网格的任意GT前景标签支持，按图平均为{pct(e['fixed640_lt8_no_positive_no_nearby'])}；合并计数190/901。邻近支持是二值并集，不表示某实例独占token。
保持长宽比的800短边/1333长边上限参考输入，对同一物理目标组缺少正标签的按图平均为{pct(r['fixed640_lt8_no_positive'])}。此为常规敏感性参考，尚未验证是Focus的完整原生输入配方。
独立格点枚举复核全部13473个GT情境；16个合成情境与作者CPU目标函数逐元素一致。原始图像尺寸与SHA已核验。
【边界】这些结果没有测量Focus的实际漏检、前景得分、被选更新或AP，也不能把TinyFormer的错误归因给Focus。监督缺口不等于更新缺口，不等于漏检原因。
【我们的推断】值得先查“有限预算的更新机会是否错配”；现有证据不足以声称这一问题尚无人研究。
【待验证假设】对微小目标定位收益较大的更新，可能与前景置信度/密度排序不一致；改用具有非平凡区别的更新收益标准，或可在同算力下提升微小目标检测。该假设尚未被因果实验支持。

5. 候选增量、可行性与风险
候选仅保留“更新分配标准与实际微小目标定位收益之间的差距”；尚不命名新模块，不锁定新encoder。
必须与RFLA式已知标签修复、Dome密度选择、UHR覆盖及SViT任务门控区分。简单拼装、换骨干或改名不算创新。
主要风险：几何标签缺口被其他监督/邻近特征弥补；现有门控方法已覆盖拟议增量；训练成本超过单卡24GB；小样本诊断不泛化。
Dome-S官方报告13.2M参数，但未在本机/当前远端测量训练显存与耗时；Focus依赖未安装。不能据参数量保证单卡训练可行。

6. 最小验证路线（下一协议草案，未执行）
先补齐SViT与Salience算子核查，再固定一个可运行的稀疏编码基线、原生预处理、数据拆分与预算；只选择一个候选机制。
先在等更新token/窗口预算下审计实际选择，按目标尺度、拥挤度、原始得分分层；不要把标签几何直接替代实际选择。
因果控制须对比原生选择、随机等预算、已知标签修复/覆盖选择及GT引导的诊断上界，保持总更新数一致。GT控制仅定位潜力，不能作为可部署方法的结果。
若等预算下微小目标定位/检测收益没有稳定改善，停止该候选；若只有更高分辨率或更多计算带来收益，也不据此主张新算法有效。
只有因果信号和新机制区别成立后，才冻结正式VisDrone训练/验证协议，测量单卡24GB峰值显存和耗时。正式报告AP/APs及<8、8–16像素分层；额外像素分箱应明确是分析指标，不冒充官方APs。
本次没有追加完整模型推理或训练。编码器诊断分支累计185次完整前向；A1/A4共64个诊断线性模型、96个标量响应头；主检测器参数更新0。旧压缩分支训练与策略拟合属于历史记录，不包含在此0中。

7. 可核验参考链接
Focus-DETR：https://arxiv.org/abs/2307.12612
官方源码：https://github.com/huawei-noah/noah-research/tree/5a088143f01ce106af2ba0c58bf9951146b60e5a/Focus-DETR
RFLA：https://arxiv.org/abs/2208.08738
Dome-DETR：https://arxiv.org/abs/2505.05741
官方源码：https://github.com/RicePasteM/Dome-DETR/tree/2dde3bc1946a3e9fad9abd0612b59fc39bd6b861
UHR-DETR：https://arxiv.org/html/2604.21435v1
Salience-DETR：https://openaccess.thecvf.com/content/CVPR2024/html/Hou_Salience_DETR_Enhancing_Detection_Transformer_with_Hierarchical_Salience_Filtering_Refinement_CVPR_2024_paper.html
SViT：https://openaccess.thecvf.com/content/WACV2024/papers/Liu_Revisiting_Token_Pruning_for_Object_Detection_and_Instance_Segmentation_WACV_2024_paper.pdf
DARK：https://arxiv.org/abs/1910.06278
SD-DETR：https://arxiv.org/abs/2310.15955

为何放下前一候选
A4精细偏移线性读出未超过常量控制；A5对461个有完整局部邻域的GT，既有OOF中心场局部Taylor控制比零偏移的径向误差按图平均多0.1208像素（95%CI 0.0437–0.1960）。A5是负控制结果，不否定完整DARK或一切非线性读出。
所以通用中心精修暂不具备开训依据。本次改变的是候选优先级，尚未找到被证实的新论文方法。
'''
report.write_text(text, encoding='utf-8')

state = read(ROOT/names[3])
state.update({'recorded_utc': NOW, 'goal_status': 'ACTIVE', 'previous_goal_turn': 'progress_A5_A6_completed_nearest_work_checked',
 'current_goal_turn': 'progress_reconciled_verified_A5_A6_status_answer_no_new_experiments',
 'mainline_validated': False, 'latest_report': str(report), 'model_forwards': 185,
 'training_updates': 0, 'CPU_diagnostic_readout_fits': 64,
 'CPU_diagnostic_scalar_response_heads': 96,
 'current_priority_question': question,
 'training_update_scope': 'Architecture branch main detector updates0;64 fitted diagnostic linear models/96 scalar response heads. Historical compression branch is outside this accounting.'})
old_phase = state.get('latest_authoritative_phase', {})
phase_history = state.setdefault('authoritative_phase_history', [])
if old_phase and not any(x.get('utc') == old_phase.get('utc') for x in phase_history):
    phase_history.append(old_phase)
state['latest_authoritative_phase'] = {
 'utc': NOW, 'classification': 'progress', 'goal_status': 'active', 'complete_mainline': False,
 'full_model_forwards': 185, 'global_cap': 200, 'main_model_training_updates': 0,
 'new_architecture_training_runs': 0, 'CPU_diagnostic_linear_models': 64, 'scalar_response_heads': 96,
 'A5_A6': evidence, 'new_full_detector_forwards_this_reconciliation': 0, 'new_fits_this_reconciliation': 0,
 'latest_report': str(report), 'next': new_candidate['next'],
 'remaining_uncertainty': ['novel operator', 'actual update-selection failures', 'causal detection gain',
                           'generalization', 'single-GPU training memory/time']}
write(ROOT/names[3], state)

receipt = {'utc': NOW, 'status': 'A5_A6_COMPLETED_EVIDENCE_RECONCILED_MAINLINE_UNVALIDATED',
 'goal_status': 'active', 'report': str(report), 'report_sha256': sha(report),
 'raw_search_response_files': len(inventory), 'core_matrix_records': len(rows),
 'accounting': {'full_forwards':185,'main_detector_updates':0,'diagnostic_models':64,'scalar_response_heads':96},
 'files': [{'path': str(ROOT/name), 'sha256': sha(ROOT/name)} for name in names],
 'A5_A6': evidence}
write(ROOT/'25_A5A6核验与主线状态回执.json', receipt)
print(json.dumps({k:v for k,v in receipt.items() if k not in ['files','A5_A6']}, ensure_ascii=False, indent=2))
