"""Reconcile completed cached analyses and read-scope records; never runs a model."""
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
NOW = datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


A9 = ROOT / 'A9_已有预测的几何分类筛选容量分解_v1'
summary = read(A9 / 'summary.json')
completed = read(A9 / 'completed.json')
assert digest(A9 / 'summary.json') == completed['summary_sha256']
assert digest(A9 / 'protocol.json') == completed['protocol_sha256']
assert summary['global_forwards_unchanged'] == 200
assert summary['new_detector_forwards'] == summary['new_fits'] == 0
assert summary['independent_flow_counts'] == 90
tiny = next(r for r in summary['aggregate'] if r['cohort'] == 'lt8' and r['iou'] == 0.75)
assert [tiny[k] for k in ('gt', 'all_query_geometry', 'retained_query_geometry', 'native_class_TopQ')] == [69, 14, 13, 12]

names = ['01_候选判别表.json', '13_架构检索增补记录.json', '14_架构核心文献矩阵.json', '15_当前状态与结果校验.json', '16_最近工作源码核查.json']
snapshot = ROOT / 'history_before_A9_update_v1'
assert not snapshot.exists(), 'One-time reconciliation already started; inspect receipts instead of overwriting history.'
snapshot.mkdir()
for name in names:
    shutil.copy2(ROOT / name, snapshot / name)
records = {name: read(ROOT / name) for name in names}
status = records[names[3]]
assert status['model_forwards'] == 200 and status['mainline_validated'] is False
assert status['latest_authoritative_phase']['full_model_forwards'] == 200

source_root = ROOT / 'official_assets_v1' / 'Dome_DETR_2dde3bc_v1' / 'source'
source_audit = {
    'utc': NOW,
    'repo': 'https://github.com/RicePasteM/Dome-DETR',
    'commit': '2dde3bc1946a3e9fad9abd0612b59fc39bd6b861',
    'scope': 'Static source inspection only; no denoising draws, fitted model or training call in A9.',
    'files': [],
    'active_config': {'num_denoising': 100, 'label_noise_ratio': 0.5, 'box_noise_scale': 1.0, 'losses': ['vfl', 'boxes', 'local']},
    'observed_source_facts': [
        'Denoising generator perturbs each ground-truth box without an all-other-GT collision check in this function.',
        'get_cdn_matched_indices uses dn_positive_idx only; the active VFL function assigns zero target to the remaining query classes.',
        'The decoder calls this generator only during training when num_denoising>0.',
    ],
    'unmeasured_hypothesis': 'A negative denoising reference may coincide with another real object. Frequency, input-label consistency, training harm and novelty are unestablished.',
    'not_inferred': ['An overlapping initial reference is an erroneous final prediction.', 'A9 geometric shortfall was caused by denoising.', 'Ignoring or reassigning such samples is a new method.'],
    'closest_work_pending': ['Shadow-DETR full method/source', 'CCDN-DETR complete constraints/source', 'RHINO adaptive-denoising method/source', 'Other cross-instance denoising collision controls'],
}
for rel, scope in [
    ('src/zoo/dome/denoising.py', 'Complete generator read; lines15-156.'),
    ('src/zoo/dome/dome_criterion.py', 'Selected VFL lines100-136, loss map335ff, DN branch465ff and CDN matched-indices565ff.'),
    ('src/zoo/dome/dome_decoder.py', 'Selected training generator call969ff and denoising initialization.'),
]:
    path = source_root / rel
    source_audit['files'].append({'relative_path': rel, 'sha256': digest(path), 'read_scope': scope})
write(A9 / 'denoising_source_scope.json', source_audit)

report_name = '30_A9定位容量诊断与当前主线边界.txt'
report_path = ROOT / report_name
report = '''当前方向：密集无人机航拍微小目标的精确定位；尚未确认可立项的算法主线。

一、问题、数据与算法起点
数据候选仍为VisDrone。Dome-DETR-S已完成作者源码、权重和严格加载核验，并执行了冻结诊断；TinyFormer-S为此前的冻结对照。它们是诊断起点，不代表已确定论文最终基线。YOLO26可保留为后续外部对照候选，目前尚未证明新机制或训练收益，不能先替论文定结构。

二、本轮新证据：A9仅复用保存预测
对A8原有3张训练集图像的69个微小目标（800输入下sqrt面积<8像素）进行一对一最大匹配容量分析。IoU>=0.75时：
全部原始query、忽略类别：14/69；
原生筛选保留的query、忽略类别：13/69；
原生类别与筛选结果：12/69。
IoU>=0.50时相应为54、52、51/69。
主指标为(14-12)/(69-12)=2/57，约3.51%。因此在这些案例中，仅改变筛选或类别可用性所释放的高IoU匹配容量较小，现有框的几何精度值得优先追查。
这是最大匹配容量的差异，不是逐GT因果错误分解，不是AP或官方AR，也不证明故障发生在编码器、解码器或某项训练损失。样本来自作者训练集，不能外推到验证集。A9方案在已知A8结果后冻结，属于探索性后验分析。
本轮没有新增完整模型前向、拟合或主模型更新；累计仍为200次完整前向，已结束当前有界诊断阶段。A9的90项计数经过另一套网络流计数交叉核验，float64转换敏感性检查未改变计数。

三、已降级的解释
压缩和图像恢复保留为历史结果，当前不追加实验。A8原生窗口已覆盖全部69个微小GT；固定query的同预算GT选窗、全窗口参考均未改善微小目标IoU75计数，因而“补救稀疏编码遗漏更新”降级。A4精细中心线性读出和A5局部Taylor控制未提供正向依据，不能换名为中心修正模块。A9也不支持把通用排序或置信度校准作为当前主要增量。

四、定位方向的最近工作边界
UGS：Uncertainty-Aware Gradient Stabilization for Small Object Detection，ICCV2025。已读作者arXiv v2的完整3.1-3.2.3及部分实验设置/开销分析；涉及非均匀坐标分布、不确定性最小化和梯度扰动。通用“不确定性回归”不构成新的切口。论文数值、推导及资源开销为作者报告，尚未复现。
https://arxiv.org/html/2303.01803v2
https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Uncertainty-Aware_Gradient_Stabilization_for_Small_Object_Detection_ICCV_2025_paper.html

TPRD：Enhanced Knowledge Distillation for Detection Transformers via Teacher Prediction Refinement，arXiv2609.19964v1，2026预印本。已读完整方法III-A至III-D及部分设置/消融；其先前层正负预测修正与质量选择已覆盖“取更好中间层作为教师”的一般思路。未取得并执行其训练源码。
https://arxiv.org/html/2609.19964v1

Shadow-DETR：Alleviating matching conflicts through shadow queries，Neural Networks197，108524，2026。DOI10.1016/j.neunet.2025.108524。当前阅读范围是出版商摘要、引言和方法片段，未获得完整方法；相似负样本分类冲突、额外定位监督已有直接近邻。
https://www.sciencedirect.com/science/article/pii/S0893608025014054

NAN-DETR、CCDN-DETR和RHINO分别提供噪声/多锚、受约束对比去噪和自适应去噪近邻。矩阵逐篇注明阅读范围，不将检索片段描述为全文阅读。

五、下一项源码候选，不是新主线
已核查Dome负去噪生成、正样本索引和实际VFL路径。该生成函数不检查与其他GT的碰撞，负去噪项在分类目标中为零。这只提示可审计“密集目标下跨实例去噪歧义”，尚未量化其发生频率，更没有证明损害训练。Shadow/CCDN/RHINO等的完整近邻排重尚未完成，不能将简单重采样、忽略或重标注包装成创新。
若继续此候选，先固定生成尺度、随机抽样预算与碰撞判据，进行无模型的标签/几何审计；达到事先定义的非偶发现象标准后，再考虑能排除现有方法的最小干预。当前没有批准新模块或训练方案。

六、当前判断
有依据的研究问题方向是微小框几何精度；具体原因和独有算法增量仍未成立。不能宣称已找到论文创新。当前交付是A9容量证据、源码阅读记录和最近工作边界，继续目标保持ACTIVE。
检索为定向非系统调研，12份原始响应已归档。此前11次响应保存了结果但未单独保存全部检索输入，本次另存3条精确标题检索的原样输入与响应；不伪造缺失的原始检索词。详见13记录与14文献矩阵。
'''
report_path.write_text(report, encoding='utf-8')

board = records[names[0]]
board.update({
    'latest_update_utc': NOW,
    'status': 'MAINLINE_NOT_VALIDATED_GEOMETRIC_PRECISION_PROBLEM_FOCUS_ONLY',
    'current_priority_question': 'VisDrone微小框几何精度不足由什么可验证机制导致，编码/解码或训练目标中是否存在未被最近方法覆盖的具体增量？',
    'model_candidates': ['TinyFormer-S-PBM: historical strict-loaded frozen diagnostic comparator', 'Focus-DETR: source-bound target audit only; no full author model executed', 'Dome-DETR-S: strict-loaded author model,15 frozen A8 forwards completed; final paper baseline not selected'],
    'remaining_deliverable': 'Identify a measured geometric-precision mechanism and clear closest prior methods before selecting an algorithm; independent validation and training resource feasibility remain unresolved.',
    'new_empirical_evidence': 'A0-A8:200 full detector forwards;64 diagnostic linear models/96 scalar response heads. A9 adds only cached maximum-matching capacity analysis and90 flow crosschecks; main detector updates0.',
    'latest_report': str(report_path),
    'A9_result': {'directory': str(A9), 'decision': summary['decision'], 'tiny_IoU75': tiny, 'mainline_validated': False},
    'next_decision_protocol': 'No new model forward or training in the completed200-forward phase. Source-only denoising ambiguity is an unquantified candidate requiring nearest-method clearance and a bounded parameter-free protocol.',
    'source_only_candidate_not_mainline': source_audit,
})
board['rejected_generic_novelty_claims'] += [
    'Generic nonuniform coordinate distributions,uncertainty minimization or gradient stabilization without a distinct mechanism beyond UGS.',
    'Generic earlier-layer teacher rectification or selection of a higher-quality intermediate prediction beyond TPRD.',
    'Generic ignore/relabel/reweight of similar negative queries without clearing Shadow-DETR and denoising-specific controls.',
]

raw_files = sorted((ROOT / 'source_audit_v1').glob('A9_*.json'))
log = records[names[1]]
log['supplement_utc'] = NOW
log['A9_evidence_and_search'] = {
    'review_type': 'Targeted non-systematic screening; negative diagnostic and closest-method clearance',
    'archive': [{'file': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in raw_files],
    'search_input_limit': 'Earlier11 stored response strings do not preserve every original query input; do not reconstruct them as original logs. Exact new3-title search input/response is archived separately.',
    'exact_query_log': str((ROOT / 'source_audit_v1' / 'A9_exact_primary_bibliography_check.json').relative_to(ROOT)),
    'new_full_model_forwards': 0,
    'new_fits': 0,
    'latest_report': str(report_path),
}
log['query_groups'].append({
    'name': 'A9 exact primary bibliography verification',
    'queries': ['"Uncertainty-Aware Gradient Stabilization for Small Object Detection"', '"Enhanced Knowledge Distillation" "Teacher Prediction Refinement"', '"Shadow-DETR: Alleviating matching conflicts through shadow queries"'],
    'domains': [['arxiv.org', 'openaccess.thecvf.com'], ['arxiv.org'], ['sciencedirect.com']],
    'result_archive': 'source_audit_v1/A9_exact_primary_bibliography_check.json',
})

matrix = records[names[2]]
additions = [
    {'id': 'UGS', 'title': 'Uncertainty-Aware Gradient Stabilization for Small Object Detection', 'url': 'https://arxiv.org/html/2303.01803v2', 'tier': 'ICCV2025; arXiv v2 2025-07-10', 'read_scope': 'Complete3.1-3.2.3; selected4.1/4.5; CVF bibliographic record. CVF direct access failed; primary arXiv used.', 'author_method': 'Nonuniform coordinate bins,uncertainty minimization and adversarial uncertainty-gradient feature regularization.', 'our_consequence': 'Generic uncertainty-aware localization is prior art; reported training costs are not our feasibility evidence.', 'unresolved': 'No code acquired or executed; equations and supplementary derivations not independently audited.'},
    {'id': 'TPRD', 'title': 'Enhanced Knowledge Distillation for Detection Transformers via Teacher Prediction Refinement', 'url': 'https://arxiv.org/html/2609.19964v1', 'tier': '2026-09-17 preprint', 'read_scope': 'CompleteIII-A toIII-D; selectedIV setup/ablations; official repository README only.', 'author_method': 'Earlier-layer positive/negative teacher prediction rectification and decoupled knowledge distillation.', 'our_consequence': 'A generic best-intermediate-layer teacher is not a new increment.', 'unresolved': 'No author code acquired/executed; external teacher-student scope differs from self-distillation.'},
    {'id': 'Shadow_DETR', 'title': 'Shadow-DETR: Alleviating matching conflicts through shadow queries', 'url': 'https://www.sciencedirect.com/science/article/pii/S0893608025014054', 'doi': '10.1016/j.neunet.2025.108524', 'tier': 'Neural Networks197,108524,May2026', 'read_scope': 'Publisher abstract,introduction and section snippets only; not complete method.', 'author_method': 'Ignore classification on shadow queries and provide balanced additional localization supervision,with feature-aware initialization.', 'our_consequence': 'Similar-negative conflict and extra regression supervision have a close existing method.', 'unresolved': 'Full assignment rules and source comparison not read; no execution.'},
    {'id': 'NAN_DETR', 'title': 'NAN-DETR: noising multi-anchor makes DETR better for object detection', 'url': 'https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1484088/full', 'doi': '10.3389/fnbot.2024.1484088', 'tier': 'Frontiers in Neurorobotics18,1484088,2024', 'read_scope': 'Selected primary3.1,3.4 formulas and experimental setup; not full paper.', 'author_method': 'Multiple independent decoders,multi-anchor queries and centralization noise.', 'our_consequence': 'Generic noise/anchor-conflict reduction requires explicit distinction.', 'unresolved': 'No source acquired/executed; cross-instance negative-DN collision handling not established.'},
    {'id': 'CCDN_DETR', 'title': 'CCDN-DETR: A Detection Transformer Based on Constrained Contrast Denoising for Multi-Class Synthetic Aperture Radar Object Detection', 'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC10974197/', 'tier': '2024 original research; exact publisher DOI pending verification', 'read_scope': 'Primary indexed3.1/3.2,selected3.4.1/Figure4,3.4.2 and experiment/conclusion excerpts; direct fulltext access limited.', 'author_method': 'Constrained positive/negative denoising regions with cross-scale encoding and query initialization.', 'our_consequence': 'A constrained-noise proposal needs exact constraint/source comparison.', 'unresolved': 'Full method/source and DOI not independently obtained; guessed publisher locator excluded.'},
    {'id': 'RHINO', 'title': 'Hausdorff Distance Matching with Adaptive Query Denoising for Rotated Detection Transformer', 'url': 'https://arxiv.org/abs/2305.07598', 'tier': '2023 arXiv record; publication venue not verified in this update', 'read_scope': 'Primary abstract only.', 'author_method': 'Adaptive denoising addresses harmful static noisy queries with matching-based filtering.', 'our_consequence': 'Filtering harmful denoising queries is not unclaimed territory.', 'unresolved': 'Full method,source and exact distinction not read; no execution.'},
]
ids = {r['id'] for r in matrix['rows']}
assert not any(r['id'] in ids for r in additions)
matrix['rows'].extend(additions)
matrix.update({'updated_utc': NOW, 'latest_report': str(report_path), 'latest_empirical_boundary': 'A9:3 exploratory train cases,69 tiny GT; geometry-only IoU75 capacity14 versus native12. Does not identify encoder/decoder cause or demonstrate novelty.'})

status.setdefault('authoritative_phase_history', []).append(copy.deepcopy(status['latest_authoritative_phase']))
status['latest_authoritative_phase'] = {
    'utc': NOW, 'classification': 'progress', 'goal_status': 'active', 'complete_mainline': False,
    'full_model_forwards': 200, 'global_cap': 200, 'bounded_phase_complete': True,
    'main_model_training_updates': 0, 'CPU_diagnostic_linear_models': 64, 'scalar_response_heads': 96,
    'A9_new_full_forwards': 0, 'A9_new_fits': 0, 'A9_decision': summary['decision'],
    'A9_primary_fraction': summary['primary_fraction'], 'independent_graph_counts': 90,
    'latest_report': str(report_path),
    'next': 'Clear concrete localization/denoising mechanism against closest methods; no extra forwards in the completed phase and no generic localization module training.',
    'remaining_uncertainty': ['Causal geometric-precision mechanism', 'Novel operator beyond closest work', 'Independent validation', 'Training time/memory feasibility'],
}
status.update({'recorded_utc': NOW, 'previous_goal_turn': status['current_goal_turn'], 'current_goal_turn': 'progress_A9_cached_geometry_capacity_and_primary_closest_method_clearance', 'current_priority_question': board['current_priority_question'], 'latest_report': str(report_path)})
audit = records[names[4]]
audit.update({'updated_utc': NOW, 'A9_cached_output_analysis': {'summary': str(A9 / 'summary.json'), 'new_model_forwards': 0, 'new_fits': 0, 'decision': summary['decision']}, 'Dome_denoising_static_source': source_audit, 'current_novelty_boundary': 'No approved algorithm. Generic uncertainty localization,earlier-layer teacher rectification and similar-negative conflict handling have close prior work.'})
for name, value in records.items():
    write(ROOT / name, value)

integrity_files = [A9 / 'protocol.json', A9 / 'summary.json', A9 / 'started.json', A9 / 'completed.json', A9 / 'denoising_source_scope.json', ROOT / 'a9_cached_error_capacity.py', Path(__file__), report_path] + [ROOT / n for n in names]
receipt = {
    'utc': NOW, 'status': 'PASS_COMPLETED_CACHED_ANALYSIS_RECONCILED_NO_MODEL_CALL',
    'goal_status': 'ACTIVE', 'mainline_validated': False, 'full_forwards': 200, 'new_forwards': 0,
    'new_fits': 0, 'main_model_updates': 0, 'diagnostic_fitted_models': 64, 'scalar_response_heads': 96,
    'A9_independent_flow_count_crosschecks': 90, 'matrix_rows': len(matrix['rows']),
    'raw_response_archives': len(raw_files), 'snapshot': str(snapshot),
    'files': [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in integrity_files],
    'not_verified': ['Novelty', 'Causal mechanism', 'Independent generalization', 'Training feasibility', 'Denoising collision frequency'],
}
write(ROOT / '31_A9完成核验与当前方向回执.json', receipt)
print(json.dumps({k: receipt[k] for k in ('status', 'goal_status', 'mainline_validated', 'full_forwards', 'new_forwards', 'new_fits', 'matrix_rows', 'raw_response_archives')}, ensure_ascii=False))
