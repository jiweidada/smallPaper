"""Record the current acceptance gaps and attach the citation supplement."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
HERE = Path(__file__).resolve().parent
RECORDS = HERE.parent
PAPER = RECORDS.parent
BUNDLE = PAPER / '远程公平对照_20261103/模块消融'
MASTER = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
EXPECTED = '2653ab4164cc454b3fad4d21c4c2e02e80febe00e80e4fc3e13759a7173114ac'
digest = lambda p: sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
relative = lambda p: p.relative_to(PAPER).as_posix()
NOW = datetime.now(timezone.utc).isoformat()

def write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

assert digest(MASTER) == EXPECTED
citation_path = HERE / 'citation_binding_current_rev34.json'
citation = read(citation_path)
assert citation['manuscript_sha256'] == EXPECTED and citation['verified_source_file_count'] == 20
index_path = RECORDS / '主要结论证据索引/主要结论证据索引_第三十四轮.json'
assert read(index_path)['manuscript']['sha256'] == EXPECTED
qa_path = RECORDS / '第三十四轮MPS移除终点_20261108/mps_endpoint_final_visual_qa.json'
assert read(qa_path)['status'] == 'PASSED' and read(qa_path)['candidate_sha256'] == EXPECTED
original_paths = [PAPER / '稿件/MPSMA-DEIM_conference_EN.docx',
                  PAPER / '稿件/备份_20261103/MPSMA-DEIM_conference_EN_original.docx']
assert all(digest(p) == 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323' for p in original_paths)
method_path = BUNDLE / 'method_source_binding_20261108.json'
method = read(method_path)
method_files = []
for file in method['files']:
    current = PAPER / file['method_evidence_path']
    frozen = BUNDLE / 'NO_MSCA' / file['frozen_training_path']
    assert digest(current) == digest(frozen) == file['sha256']
    method_files.append({'method_evidence': relative(current), 'frozen_source': relative(frozen), 'sha256': file['sha256']})
health_run = subprocess.run(['D:/Anaconda/python.exe', '-X', 'utf8', '-B', '-u', str(BUNDLE / 'check_backup_health.py')],
                            capture_output=True, encoding='utf-8', errors='backslashreplace', check=True, timeout=25)
health = json.loads(health_run.stdout)
assert health['NO_MSCA']['monitor_alive']
assert health['NO_MSCA']['accepted_round'] < 320
pair_path = PAPER / '远程公平对照_20261103/断点续训_20261105/BASE_FULL_pair_verified_20261108.json'
mps_path = BUNDLE / 'NO_MPS/lineage_through_round320.json'
assert read(mps_path)['completed_rounds'] == 320
criteria = [
    {'requirement': 'Main and module results trace to actual weights, predictions, evaluator arrays, metrics and trajectory',
     'status': 'PARTIAL_PENDING_NO_MSCA',
     'evidence': [relative(pair_path), relative(mps_path), relative(index_path)],
     'proved': 'BASE/FULL and NO_MPS endpoints and lineages verified; current Tables 1 and 2 recomputed from those results.',
     'remaining': 'NO_MSCA complete endpoint, full trajectory and genuine completion/backup.'},
    {'requirement': 'Experimental comparisons use explicit, reasonable conditions and avoid unsupported attribution',
     'status': 'VERIFIED_CURRENT_REPORTED_COMPARISONS_WITH_LIMITS',
     'evidence': ['远程公平对照_20261103/模块消融/module_shared_conditions_verified_20261108.json', relative(index_path)],
     'limits': 'Single seed; realized continuation/RNG histories and capacity differ. Current paper discloses these differences. Conditional removals do not identify factorial interaction or mechanism independent of capacity.'},
    {'requirement': 'Method text, equations, diagrams and parameters match the implementation used for reported results',
     'status': 'VERIFIED_CURRENT_CONFIGURATION_WITH_DECLARED_SCOPE',
     'evidence': [relative(method_path), relative(qa_path)],
     'proved': 'Four core method files rehashed against actual frozen source. The final eight pages were reviewed individually; six formulas and all six figures preserved. n/s input differences and the exact conditional adapters are explicit.',
     'limits': 'Does not certify unresolved historical training source identity; historical tables remain archived outside the controlled comparisons.'},
    {'requirement': 'Contribution and English expression are precise and supported',
     'status': 'CURRENT_TEXT_REVISED_FINAL_MODULE_ANALYSIS_PENDING',
     'evidence': [relative(qa_path), relative(index_path)],
     'proved': 'Abstract, introduction, related work, methods, analysis, limitations and conclusion revised. Capacity cost and negative large-object recall result retained. No significance, universal-improvement, stability or interaction claim.',
     'remaining': 'Revise conditional analysis and final abstract/conclusion from actual NO_MSCA result, then read and render the whole paper again.'},
    {'requirement': 'Citations are complete, correctly mapped and used within actual source-reading scope',
     'status': 'VERIFIED_FOR_CURRENT_CLAIMS_WITH_READING_LIMITS',
     'evidence': [relative(citation_path)],
     'proved': 'All 13 entries match the audited archive and remain cited in order; 20 local source files rehashed.',
     'limits': 'No full PDF for references 1, 3, 9 and 10. Their uses remain general background or method overviews; reference 13 is static code inspection only.'},
    {'requirement': 'Original, continuous revised manuscript and material change/open-issue records are retained',
     'status': 'VERIFIED_CURRENT_REV34',
     'evidence': [relative(p) for p in original_paths] + [relative(MASTER), relative(index_path)],
     'proved': 'Original and its exact backup unchanged; revision-33 backup and immutable revision-34 candidate retained; continuous master equals reviewed candidate.'},
    {'requirement': 'Venue-specific template, page limit and submission rules',
     'status': 'DEFERRED_BY_USER_UNTIL_VENUE_SELECTION',
     'basis': 'The user said no venue selected and requested content revision first. This is not a reason to stop current content/evidence work or choose a venue without them.'}
]
evidence_paths = [citation_path, HERE / 'publisher_index_refresh_20261108.json', index_path,
                  qa_path, method_path, pair_path, mps_path,
                  BUNDLE / 'module_shared_conditions_verified_20261108.json']
audit_path = HERE / 'current_goal_acceptance_audit_rev34.json'
audit = {'status': 'CONTENT_AND_EVIDENCE_PARTIAL_NO_MSCA_TERMINAL_PENDING',
         'observed_local_utc': NOW, 'manuscript_sha256': EXPECTED, 'content_revision': 34,
         'criteria': criteria, 'verified_method_source_files': method_files,
         'evidence_sha256': {relative(p): digest(p) for p in evidence_paths},
         'current_backup_health': health,
         'previous_goal_turn_classification': 'PROGRESS: revision 34 promoted after numerical, structural and visual QA.',
         'this_goal_turn_classification': 'PROGRESS: current citation/source binding and requirement-level acceptance audit completed; NO_MSCA backup client confirmed live.',
         'content_stage_complete': False, 'submission_ready': False,
         'new_training_groups': 0, 'new_model_forwards': 0, 'new_evaluations': 0}
write(audit_path, audit)
header = (
    '【当前目标验收补核：2026-11-08，稿件仍为第三十四轮】\n'
    '当前稿 SHA-256 ' + EXPECTED + '。BASE/FULL 与 NO_MPS 已完成 320 轮并核验；NO_MSCA 备份客户端本次实际存活，最新已接受 ' + str(health['NO_MSCA']['accepted_round']) + ' 轮，完整终点仍待完成。目标保持 active。\n'
    '当前十三条引用已逐条绑定，八篇 PDF 与文本、两份官方源码和两份正确元数据网页共 20 份来源哈希匹配；文献 1、3、9、10 仍无全文 PDF，引用用途维持在实际阅读范围。'
    '四份方法证据源码与冻结训练源码重新核对一致。稿件字节和实验方案均未改变。\n'
    '按完整目标逐项验收的报告为引用绑定补核_20261108/current_goal_acceptance_audit_rev34.json。剩余实证工作是 NO_MSCA 完整端点与轨迹核验，以及相应结果分析、全文复核和渲染。选会适配按用户要求暂缓。尚不能投稿。以下为历史记录。\n\n')
for name in ['引用核查记录.txt', '目标验收与阻塞记录.txt']:
    path = RECORDS / name
    with (HERE / (path.stem + '_补核前.txt')).open('xb') as stream:
        stream.write(path.read_bytes())
    old = path.read_text(encoding='utf-8-sig')
    path.write_text(header + old, encoding='utf-8')
state_path = BUNDLE / 'current_execution_state.json'
state = read(state_path)
state.update(citation_binding_evidence=str(citation_path), citation_binding_evidence_sha256=digest(citation_path),
             current_goal_acceptance_audit=str(audit_path), current_goal_acceptance_audit_sha256=digest(audit_path),
             observed_local_utc=NOW)
events = [json.loads(line) for line in (Path(state['NO_MSCA']['backup_root']) / 'monitor_events.jsonl').read_text(encoding='utf-8').splitlines()]
accepted = next(e for e in reversed(events) if e['status'] == 'EPOCH_BACKUP_VERIFIED')
state['NO_MSCA']['last_accepted_round'] = accepted['epoch'] + 1
state['NO_MSCA']['last_acceptance_utc'] = accepted['observed_utc']
temporary = state_path.with_suffix('.citation_audit.tmp')
write(temporary, state)
os.replace(temporary, state_path)
with (BUNDLE / '方案与执行记录.md').open('a', encoding='utf-8') as stream:
    stream.write('\n\n## 2026-11-08 当前引用与目标验收补核\n\n' + header + '\n')
assert digest(MASTER) == EXPECTED
print(json.dumps({'status': audit['status'], 'reference_count': 13, 'verified_source_files': 20,
                  'verified_method_files': 4, 'NO_MSCA_accepted_round': state['NO_MSCA']['last_accepted_round'],
                  'backup_client_alive': health['NO_MSCA']['monitor_alive'], 'manuscript_changed': False,
                  'submission_ready': False}))
