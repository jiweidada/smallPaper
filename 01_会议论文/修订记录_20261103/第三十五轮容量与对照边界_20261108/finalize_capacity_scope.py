"""Promote the source-checked and visually reviewed capacity clarification."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from zipfile import ZipFile
from lxml import etree

sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
HERE = Path(__file__).resolve().parent
RECORDS = HERE.parent
PAPER = RECORDS.parent
BUNDLE = PAPER / '远程公平对照_20261103/模块消融'
MASTER = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_rev34_backup.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_rev35.docx'
OLD_SHA = '2653ab4164cc454b3fad4d21c4c2e02e80febe00e80e4fc3e13759a7173114ac'
NEW_SHA = 'd3b1d442bad72584082f299e35d7f8fccfa5029de4571af959b065aaae20ff7b'
ORIGINAL_SHA = 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'
NOW = datetime.now(timezone.utc).isoformat()
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W, 'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
digest = lambda p: sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
value = lambda p: ''.join(p.xpath('.//w:t/text()', namespaces=NS))
relative = lambda p: p.relative_to(PAPER).as_posix()

def write_json(path, data):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

def atomic_json(path, data):
    temporary = path.with_suffix('.rev35.tmp')
    write_json(temporary, data)
    os.replace(temporary, path)

assert digest(MASTER) == digest(BACKUP) == OLD_SHA and digest(CANDIDATE) == NEW_SHA
for path in [PAPER / '稿件/MPSMA-DEIM_conference_EN.docx', PAPER / '稿件/备份_20261103/MPSMA-DEIM_conference_EN_original.docx']:
    assert digest(path) == ORIGINAL_SHA
assert read(HERE / 'capacity_scope_candidate.json')['candidate_sha256'] == NEW_SHA
assert read(HERE / 'render_metadata.json')['page_count'] == 8
pages = sorted((HERE / 'rendered_word').glob('page-*.png'))
assert len(pages) == 8 and all(p.stat().st_size > 100000 for p in pages)
assert 'soffice.exe was not found' in (HERE / 'packaged_render_attempt.log').read_text(encoding='utf-8-sig')
gate_path = BUNDLE / 'msca_parameter_decomposition_20261108.json'
capacity_path = BUNDLE / 'base_full_capacity_changes_20261108.json'
gate, capacity = read(gate_path), read(capacity_path)
assert gate['removed_gate_trainable_parameters'] == 3841280
assert gate['trainable_parameters_by_component']['context_conv'] == 3146752
assert capacity['FULL_minus_BASE_total'] == 3803331
assert sum(capacity['FULL_minus_BASE_parameters_by_category'].values()) == 3803331
assert not gate['model_constructed'] and not capacity['model_constructed']

with ZipFile(BACKUP) as before, ZipFile(CANDIDATE) as after:
    assert after.testzip() is None and set(before.namelist()) == set(after.namelist())
    changed = [p for p in before.namelist() if before.read(p) != after.read(p)]
    assert set(changed) == {'word/document.xml', 'word/comments.xml'}
    old = etree.fromstring(before.read('word/document.xml'))
    new = etree.fromstring(after.read('word/document.xml'))
    old_comments = etree.fromstring(before.read('word/comments.xml'))
    comments = etree.fromstring(after.read('word/comments.xml'))
    assert all(b'\xef\xbf\xbd' not in after.read(p) for p in changed)
for path in ['.//m:oMath', './/wp:inline', './/w:tbl']:
    assert [etree.tostring(n) for n in old.findall(path, NS)] == [etree.tostring(n) for n in new.findall(path, NS)]
assert len(new.findall('.//m:oMath', NS)) == len(new.findall('.//wp:inline', NS)) == 6
assert len(new.findall('.//w:tbl', NS)) == 2
for tag in ['w:commentRangeStart', 'w:commentRangeEnd', 'w:commentReference']:
    assert sorted(n.get('{' + W + '}id') for n in new.findall('.//' + tag, NS)) == ['0', '1', '2', '3', '4', '5']
assert [etree.tostring(n) for n in old_comments] == [etree.tostring(n) for n in list(comments)[:5]]
assert [n.get('{' + W + '}id') for n in comments] == ['0', '1', '2', '3', '4', '5']
paragraphs = [value(p) for p in new.findall('w:body/w:p', NS)]
old_paragraphs = [value(p) for p in old.findall('w:body/w:p', NS)]
references = [p for p in paragraphs if re.match(r'^\[\d+\]', p)]
assert references == [p for p in old_paragraphs if re.match(r'^\[\d+\]', p)]
assert len(references) == 13
abstract = next(p for p in paragraphs if p.startswith('Small objects in unmanned aerial vehicle'))
assert abstract == next(p for p in old_paragraphs if p.startswith('Small objects in unmanned aerial vehicle'))
body = '\n'.join(paragraphs[:paragraphs.index('References')])
cited = set()
for match in re.findall(r'(?<![=\w])\[([\d, ]+)\]', body):
    cited.update(int(n) for n in match.split(',') if n.strip())
assert cited == set(range(1, 14))
assert all(s in body for s in ['3,841,280', '3,146,752', 'BASE lacks the ESE/SimAM gates',
                             'The gain cannot be assigned to MPS or MSCA alone.',
                             'NO_MSCA remains in progress', '3.53', '3.23', '1.57', '1.77', '-1.69'])
previous_acronyms = RECORDS / '第三十四轮MPS移除终点_20261108'
for scope, current_name, previous_name in [
    ('body', 'acronym_body_report.json', 'acronym_body_v4_report.json'),
    ('standalone abstract', 'acronym_abstract_report.json', 'acronym_abstract_v4_report.json')]:
    current_report, previous_report = read(HERE / current_name), read(previous_acronyms / previous_name)
    assert {f['acronym'] for f in current_report['findings']} == {f['acronym'] for f in previous_report['findings']}, scope
    assert {f['acronym'] for f in current_report['coverage_limits']} == {f['acronym'] for f in previous_report['coverage_limits']}, scope

manual_path = HERE / 'acronym_manual_adjudication.json'
qa_path = HERE / 'capacity_scope_visual_qa.json'
citation_path = HERE / 'citation_carryforward_rev35.json'
note_path = HERE / '容量说明与对照边界修订记录.txt'
audit_path = HERE / 'current_goal_acceptance_audit_rev35.json'
index_path = RECORDS / '主要结论证据索引/主要结论证据索引_第三十五轮.json'
for path in [manual_path, qa_path, citation_path, note_path, audit_path, index_path, index_path.with_suffix('.txt')]:
    assert not path.exists(), path
write_json(manual_path, {'status': 'NO_NEW_FIRST_USE_FLAGS_MANUALLY_REVIEWED_WITH_INHERITED_LIMITS',
                         'candidate_sha256': NEW_SHA,
                         'body_report_sha256': digest(HERE / 'acronym_body_report.json'),
                         'abstract_report_sha256': digest(HERE / 'acronym_abstract_report.json'),
                         'inherited_adjudication': relative(previous_acronyms / 'acronym_manual_adjudication.json'),
                         'review': 'Abstract unchanged. ESE and SimAM are already introduced before the new scope statements. The sets of advisory findings and coverage limits are unchanged; occurrence counts differ. All are reviewed under the existing named-method, metric, run-label, mathematical and hardware-name adjudication. Raw checker reports remain partial, not zero findings.'})
write_json(qa_path, {'status': 'PASSED', 'revision': 35, 'observed_local_utc': NOW,
                    'source_sha256': OLD_SHA, 'candidate_sha256': NEW_SHA,
                    'renderer': 'Hidden macro-disabled read-only Word export and bundled Poppler',
                    'canonical_render_log': 'packaged_render_attempt.log',
                    'canonical_render_limitation': 'Missing soffice.exe; failure diagnosed from preserved log.',
                    'pages_visually_inspected_individually': list(range(1, 9)),
                    'visual_result': 'All eight final pages reviewed. Added capacity text, scope statements and modified-backbone caption fit cleanly. All equations, six figures, two tables and thirteen references remain readable; no clipping, overlaps or orphan reference page.',
                    'page_png_sha256': {p.name: digest(p) for p in pages},
                    'changed_docx_parts': changed, 'tables_preserved_exactly': 2,
                    'equations_preserved_exactly': 6, 'figures_preserved_exactly': 6,
                    'references_preserved_and_still_cited': 13,
                    'original_comments_and_anchors_preserved': 5, 'new_valid_comment_id': 5,
                    'abstract_preserved_exactly': True, 'performance_values_changed': False,
                    'new_model_execution': False, 'submission_ready': False})
old_citation_path = RECORDS / '引用绑定补核_20261108/citation_binding_current_rev34.json'
write_json(citation_path, {'status': 'UNCHANGED_BIBLIOGRAPHY_CARRIED_FORWARD_WITH_CURRENT_CLAIM_SCOPE_REVIEW',
                          'manuscript_sha256': NEW_SHA, 'content_revision': 35,
                          'source_citation_binding': relative(old_citation_path),
                          'source_citation_binding_sha256': digest(old_citation_path),
                          'reference_entries_identical': 13, 'all_still_cited': True,
                          'previous_local_source_hash_checks': read(old_citation_path)['verified_source_file_count'],
                          'full_text_unavailable_references': [1, 3, 9, 10],
                          'new_claim_sources': [relative(gate_path), relative(capacity_path)],
                          'scope': 'No new external citation or literature performance claim. New architectural and parameter claims are bound to the frozen implementation audits. Prior limited reading scopes remain applicable; this does not claim new full-text reading or a repeated source-download audit.'})
note = (
    '第三十五轮：容量来源与对照边界\n\n'
    '现有配置、源码与留存初始化张量已核对：BASE 默认 agg=se 在该实现中是两次 ConvBNAct 聚合投影，不含独立 ESE/SimAM 门；'
    'FULL 在阶段1/2分别使用 ESE/SimAM，另改投影路径与激活。正文3.3和4.3/4.4节现明确这些差异，图3标题限定为改进骨干，避免把整组结果归为 MPS/MSCA 的独立或可加贡献。\n\n'
    '静态参数差完全闭合：stem +5,325，聚合投影合计 −47,434，阶段1 ESE +4,160，阶段4 Dilated +3,841,280，合计 +3,803,331，'
    '等于 7,540,942−3,737,611。其中一个 HG 块的投影增参，不声称逐块都减参。源码中 se 的名称不被当作另一个独立 ESE 模块。\n\n'
    '3.3节补入 Dilated 门的 3,841,280 个参数及其中上下文投影的 3,146,752 个参数，新增就地批注5指向源码与张量核查记录。'
    '参数核查没有构建或运行模型，不是精度、FLOPs、时延或显存测试。\n\n'
    '两张性能表、摘要、六条公式、六幅图、十三条文献及原五组批注保持，八页逐页检查通过。原稿与原稿备份未改；第三十四轮精确备份保留。'
    '当前十三条引用条目逐字一致，既有20份来源文件哈希核对和四条无全文阅读范围继续适用。\n\n'
    'NO_MSCA 前20轮状态和轨迹已核验，32,340条学习率误差为0；这不是终点结论。尚待完整320轮及权重—预测—评价—指标、轨迹、真实退出与备份核验，'
    '随后按真实结果修订条件分析、摘要和结论并复核全文。选会模板按用户要求后续处理。目前尚不能投稿。\n')
with note_path.open('x', encoding='utf-8') as stream:
    stream.write(note)
health_process = subprocess.run(['D:/Anaconda/python.exe', '-X', 'utf8', '-B', '-u', str(BUNDLE / 'check_backup_health.py')],
                                capture_output=True, encoding='utf-8', errors='backslashreplace', check=True, timeout=25)
health = json.loads(health_process.stdout)
assert health['NO_MSCA']['monitor_alive']
old_audit_path = RECORDS / '引用绑定补核_20261108/current_goal_acceptance_audit_rev34.json'
audit = deepcopy(read(old_audit_path))
audit.update(observed_local_utc=NOW, manuscript_sha256=NEW_SHA, content_revision=35,
             current_backup_health=health, content_stage_complete=False, submission_ready=False,
             previous_audit=relative(old_audit_path), previous_audit_sha256=digest(old_audit_path),
             changed_scope_review='New capacity counts and exact baseline gate/projection differences checked against frozen source and saved tensors; all eight final pages reviewed. Unchanged numerical and citation evidence carried forward through exact table, equation, figure, abstract and bibliography comparisons.',
             this_goal_turn_classification='PROGRESS: parameter-change ledger reconciled, manuscript scope clarified, source binding and full visual QA completed.')
audit['evidence_sha256'].update({relative(p): digest(p) for p in [gate_path, capacity_path, qa_path, citation_path]})
for criterion in audit['criteria']:
    criterion['evidence'].append(relative(qa_path)) if 'evidence' in criterion else None
    if criterion['requirement'].startswith('Citations'):
        criterion['evidence'].append(relative(citation_path))
    if criterion['requirement'].startswith('Original,'):
        criterion['status'] = 'VERIFIED_CURRENT_REV35'
        criterion['proved'] = 'Original and its exact backup unchanged; revision-34 exact backup retained; continuous master is the reviewed revision-35 candidate after atomic promotion.'
write_json(audit_path, audit)
previous_index = RECORDS / '主要结论证据索引/主要结论证据索引_第三十四轮.json'
evidence_paths = [gate_path, capacity_path, HERE / 'capacity_scope_candidate.json', manual_path,
                  qa_path, citation_path, audit_path, note_path,
                  BUNDLE / 'msca_round020_evidence_supplement_20261109.json',
                  BUNDLE / 'NO_MSCA/lineage_through_round020.json']
index = {'revision': 35, 'status': 'CAPACITY_AND_BASELINE_SCOPE_SOURCE_VERIFIED_NO_MSCA_TERMINAL_PENDING',
         'observed_local_utc': NOW, 'path_basis': 'relative to 01_会议论文',
         'manuscript': {'path': relative(MASTER), 'sha256': NEW_SHA, 'content_revision': 35},
         'previous_index': relative(previous_index), 'previous_index_sha256': digest(previous_index),
         'previous_manuscript_backup': {'path': relative(BACKUP), 'sha256': OLD_SHA},
         'original_manuscript_sha256': ORIGINAL_SHA,
         'evidence_files': {relative(p): digest(p) for p in evidence_paths},
         'performance_results_changed': False,
         'current_comparison_scope': 'Complete configuration comparison plus conditional MPS replacement. BASE also differs in stage-1/2 gates, projections and activations and is not a joint removal. No additive, independent or factorial attribution.',
         'remaining': ['NO_MSCA complete 320-round result and lineage/exit/backup verification',
                       'Final conditional analysis and whole-paper review/render', 'Venue-specific adaptation after user selection'],
         'submission_ready': False}
write_json(index_path, index)
with index_path.with_suffix('.txt').open('x', encoding='utf-8') as stream:
    stream.write('第三十五轮主要结论证据索引\n\n稿件 SHA-256: ' + NEW_SHA + '\n\n' + note +
                 '\n证据路径与 SHA-256:\n' + '\n'.join(p + '  ' + h for p, h in index['evidence_files'].items()) + '\n')
checklist_path = PAPER / '远程公平对照_20261103/结果并入与验收清单.txt'
with (HERE / '结果验收清单_更新前.txt').open('xb') as stream:
    stream.write(checklist_path.read_bytes())
assert digest(MASTER) == OLD_SHA and digest(CANDIDATE) == NEW_SHA
temporary = MASTER.with_suffix('.rev35.tmp')
with temporary.open('xb') as stream:
    stream.write(CANDIDATE.read_bytes())
os.replace(temporary, MASTER)
assert digest(MASTER) == NEW_SHA
state_path = BUNDLE / 'current_execution_state.json'
state = read(state_path)
state.update(paper_revision=35, paper_sha256=NEW_SHA, observed_local_utc=NOW,
             msca_parameter_method_statement_incorporated_revision=35,
             base_full_gate_scope_clarified_revision=35,
             base_full_capacity_evidence=str(capacity_path), base_full_capacity_evidence_sha256=digest(capacity_path),
             citation_binding_evidence=str(citation_path), citation_binding_evidence_sha256=digest(citation_path),
             current_goal_acceptance_audit=str(audit_path), current_goal_acceptance_audit_sha256=digest(audit_path))
events = [json.loads(line) for line in (Path(state['NO_MSCA']['backup_root']) / 'monitor_events.jsonl').read_text(encoding='utf-8').splitlines()]
accepted = next(e for e in reversed(events) if e['status'] == 'EPOCH_BACKUP_VERIFIED')
state['NO_MSCA'].update(last_accepted_round=accepted['epoch'] + 1, last_acceptance_utc=accepted['observed_utc'])
header = (
    '【最新稿件与证据入口：2026-11-08，第三十五轮容量与对照边界】\n'
    '持续稿第三十五轮，SHA256 ' + NEW_SHA + '，8页逐页核验通过。方法补入 Dilated 门 3,841,280 参数，其中上下文投影 3,146,752。'
    'BASE 不含 FULL 的 ESE/SimAM 门，投影和激活也不同；正文及图3标题已更明确地限定配置与归因范围。\n'
    '静态参数变化合计 +3,803,331，与完整模型参数差完全一致。两张性能表、六条公式、六幅图、摘要及十三条引用均保持；新增就地批注5。'
    '这不是新训练或精度、速度、显存测试。\n'
    'BASE/FULL 和 NO_MPS 终点已核验；NO_MSCA 前20轮状态及轨迹核验通过，当前已备份 ' + str(state['NO_MSCA']['last_accepted_round']) + ' 轮，仍待完整320轮。'
    '最新运行状态见模块消融/current_execution_state.json；下一步按真实 MSCA 结果完善分析与全文验收。\n'
    '详见第三十五轮证据索引及第三十五轮容量与对照边界_20261108/容量说明与对照边界修订记录.txt。尚不能投稿；以下为历史记录。\n\n')
for path in [PAPER / '当前修订入口.txt', RECORDS / '修订说明与未解决问题.txt', RECORDS / '目标验收与阻塞记录.txt']:
    old_text = path.read_text(encoding='utf-8-sig')
    old_text = old_text.replace('【最新稿件与证据入口：2026-11-08，第三十四轮 MPS 终点并入】',
                                '【历史稿件与证据入口：2026-11-08，第三十四轮 MPS 终点并入】', 1)
    path.write_text(header + old_text, encoding='utf-8')
checklist_path.write_text(header + (HERE / '结果验收清单_更新前.txt').read_text(encoding='utf-8-sig'), encoding='utf-8')
state['acceptance_checklist_sha256'] = digest(checklist_path)
atomic_json(state_path, state)
with (BUNDLE / '方案与执行记录.md').open('a', encoding='utf-8') as stream:
    stream.write('\n\n## 2026-11-08 第三十五轮容量与对照边界修订\n\n' + note + '\n')
print(json.dumps({'status': index['status'], 'revision': 35, 'pages': 8, 'manuscript_sha256': digest(MASTER),
                  'NO_MSCA_backed_up_rounds': state['NO_MSCA']['last_accepted_round'],
                  'NO_MSCA_independently_audited_rounds': state['NO_MSCA']['independent_lineage_audit_round'],
                  'submission_ready': False}))
