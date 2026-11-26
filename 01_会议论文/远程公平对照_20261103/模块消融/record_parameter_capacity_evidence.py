"""Attach the static capacity decomposition to the current revision evidence."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent
digest = lambda p: sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
report_path = HERE / 'msca_parameter_decomposition_20261108.json'
report = read(report_path)
master = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
index = PAPER / '修订记录_20261103/主要结论证据索引/主要结论证据索引_第三十四轮.json'
expected = '2653ab4164cc454b3fad4d21c4c2e02e80febe00e80e4fc3e13759a7173114ac'
assert digest(master) == read(index)['manuscript']['sha256'] == expected
assert report['status'] == 'FROZEN_MSCA_GATE_PARAMETER_DECOMPOSITION_VERIFIED'
assert report['removed_gate_trainable_parameters'] == 3841280
assert report['model_constructed'] is False and report['model_forwards'] == 0
state_path = HERE / 'current_execution_state.json'
previous_state_sha = digest(state_path)
state = read(state_path)
assert state['paper_revision'] == 34 and state['paper_sha256'] == expected
now = datetime.now(timezone.utc).isoformat()
evidence = {
    'status': 'STATIC_MSCA_CAPACITY_DECOMPOSITION_BOUND_TO_REV34',
    'observed_local_utc': now, 'manuscript_content_revision': 34,
    'manuscript_sha256': expected,
    'evidence_index_sha256': digest(index),
    'parameter_report': str(report_path), 'parameter_report_sha256': digest(report_path),
    'source_auditor_sha256': digest(HERE / 'audit_msca_parameter_decomposition.py'),
    'capacity_summary': {k: report[k] for k in ['full_trainable_parameters', 'NO_MSCA_trainable_parameters',
                                             'removed_gate_trainable_parameters', 'trainable_parameters_by_component',
                                             'context_projection_fraction_of_removed_parameters',
                                             'lite_gate_trainable_parameters']},
    'pending_next_manuscript_revision': {
        'method_sentence': 'At stage 4, the implemented Dilated gate has 3,841,280 trainable parameters, including 3,146,752 in its 3072-to-1024 context projection; the two Lite gates add none.',
        'analysis_scope': 'Relate the actual NO_MSCA outcome to this capacity difference, with no claim that the comparison isolates the gating mechanism from capacity.',
        'integration_timing': 'With the verified final NO_MSCA result and whole-paper revision; do not add unfinished performance values.'},
    'limits': 'Static parameter accounting is not a measurement of FLOPs, latency, memory, performance or a capacity-matched causal effect.',
    'new_model_execution': False, 'manuscript_changed': False, 'submission_ready': False}
output = HERE / 'parameter_capacity_evidence_supplement_20261108.json'
with output.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
state.update(msca_parameter_decomposition_evidence=str(report_path),
             msca_parameter_decomposition_evidence_sha256=digest(report_path),
             parameter_capacity_evidence_supplement=str(output),
             parameter_capacity_evidence_supplement_sha256=digest(output),
             observed_local_utc=now)
temporary = state_path.with_suffix('.parameter_audit.tmp')
assert digest(state_path) == previous_state_sha
with temporary.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
os.replace(temporary, state_path)
with (HERE / '方案与执行记录.md').open('a', encoding='utf-8') as stream:
    stream.write('\n\n## 2026-11-08 MSCA 容量来源静态核查\n\n'
                 '读取已留存初始化张量并核对冻结源码；不构建或执行模型。NO_MSCA 移除的 3,841,280 个可训练参数全部来自第四阶段 Dilated 门：'
                 '三条卷积分支 431,232、上下文投影 3,146,752、通道门 263,296，投影占约 81.92%。两处 Lite 门均无可训练参数。'
                 'BN 运行缓冲区已排除，解析计数与逐张量计数、实际预检参数差一致。'
                 '这只能解释容量来源，不能替代速度、显存、精度或容量匹配对照。证据为 msca_parameter_decomposition_20261108.json 和 parameter_capacity_evidence_supplement_20261108.json。'
                 '当前稿件仍为第三十四轮；待 NO_MSCA 真实完整终点核验后，将容量说明与结果一起并入下一次全文修订。\n')
assert digest(master) == expected
print(json.dumps({'status': evidence['status'], 'manuscript_changed': False,
                  'parameter_report_sha256': digest(report_path), 'submission_ready': False}))
