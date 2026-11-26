"""Promote the locally verified disclosure and preserve the remaining acceptance gates."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from zipfile import ZipFile
from lxml import etree

sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
HERE = Path(__file__).resolve().parent
RECORDS, PAPER = HERE.parent, HERE.parents[1]
BUNDLE = PAPER / '远程公平对照_20261103/模块消融'
RECOVERY = BUNDLE / 'NO_MSCA/recovery30715_20261109'
MASTER = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_rev36_backup.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_rev37.docx'
OLD = '57fb02f46033266c75f959259028275d8fbd64b139450d0941dab49063a5a343'
NEW = '53060de71535e78591fd5856ff537cf65c12a9463bd532e7525de490eb3c2910'
ORIGINAL = 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
digest = lambda p: sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
relative = lambda p: p.relative_to(PAPER).as_posix()
text = lambda p: ''.join(p.xpath('.//w:t/text()', namespaces=NS))

def write(p, value):
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')

assert digest(MASTER) == digest(BACKUP) == OLD and digest(CANDIDATE) == NEW
for p in [PAPER / '稿件/MPSMA-DEIM_conference_EN.docx', PAPER / '稿件/备份_20261103/MPSMA-DEIM_conference_EN_original.docx']:
    assert digest(p) == ORIGINAL
report = read(HERE / 'candidate_structural_and_evidence_check.json')
assert report['candidate_sha256'] == NEW and report['all_other_document_content_exactly_preserved']
assert read(HERE / 'render_metadata.json')['page_count'] == 9
assert 'soffice.exe was not found' in (HERE / 'packaged_render_attempt.log').read_text('utf-8-sig')
assert read(HERE / 'sandbox_word_render_denied.json')['document_was_opened'] is False
pages = [HERE / 'rendered_word_authorized' / f'page-{n}.png' for n in range(1, 10)]
assert all(p.is_file() and p.stat().st_size > 80000 for p in pages)
with ZipFile(BACKUP) as source, ZipFile(CANDIDATE) as edited:
    assert edited.testzip() is None and source.namelist() == edited.namelist()
    changed = [n for n in source.namelist() if source.read(n) != edited.read(n)]
    assert set(changed) == {'word/document.xml', 'word/comments.xml'}
    old, new = [etree.fromstring(z.read('word/document.xml')) for z in [source, edited]]
    for xpath, count in [('.//m:oMath', 6), ('.//wp:inline', 6), ('.//w:tbl', 2)]:
        a, b = [x.findall(xpath, NS) for x in [old, new]]
        assert len(a) == len(b) == count and [etree.tostring(x) for x in a] == [etree.tostring(x) for x in b]
    a, b = [[text(p) for p in x.findall('w:body/w:p', NS)] for x in [old, new]]
    assert [report['old_paragraph'] if s == report['new_paragraph'] else s for s in b] == a
    references = [s for s in b if re.match(r'^\[\d+\]', s)]
    assert len(references) == 13 and references == [s for s in a if re.match(r'^\[\d+\]', s)]
    assert len(etree.fromstring(edited.read('word/comments.xml'))) == 7

snapshot_path = PAPER / report['evidence'][-1]['file']
snapshot = read(snapshot_path)
assert snapshot['NO_MSCA_independently_verified_rounds'] == 165 and snapshot['per_epoch_csv_rows'] == 1125
for item in snapshot['files']:
    p = (snapshot_path.parent / item['file']).resolve(strict=True)
    assert p.is_relative_to(snapshot_path.parent.resolve())
    assert digest(p) == item['sha256'] and p.stat().st_size == item['bytes']
assert len(snapshot['files']) == 21
previous_index = RECORDS / '主要结论证据索引/主要结论证据索引_第三十六轮.json'
previous_supplement = RECORDS / '主要结论证据索引/主要结论证据索引_第三十六轮_数据快照路径补充_20261109.json'
previous_citation = RECORDS / '第三十六轮断点恢复披露_20261109/citation_carryforward_rev36.json'
for p in [previous_index, previous_supplement, previous_citation]: assert p.is_file()
now = datetime.now(timezone.utc).isoformat()
qa_path = HERE / 'visual_qa_rev37.json'
citation_path = HERE / 'citation_carryforward_rev37.json'
index_path = RECORDS / '主要结论证据索引/主要结论证据索引_第三十七轮.json'
acceptance_path = HERE / 'current_goal_acceptance_audit_rev37.json'
for p in [qa_path, citation_path, index_path, acceptance_path, HERE / 'promotion_verified.json']:
    assert not p.exists()
write(qa_path, {'status': 'PASSED', 'revision': 37, 'candidate_sha256': NEW, 'observed_utc': now,
    'renderer': 'Hidden macro-disabled read-only Word PDF export in authorized channel and bundled Poppler',
    'standard_renderer_limitation': 'LibreOffice soffice.exe missing; preserved failed log',
    'sandbox_word_limitation': 'COM E_ACCESSDENIED; no document opened before authorized read-only retry',
    'pages_visually_inspected_individually': list(range(1, 10)),
    'visual_result': 'All9 pages inspected: text and six formulas readable, diagrams and two tables complete; no clipping or overlap. Updated disclosure is on page8; all13 references remain on page9.',
    'font_render_warnings': 'Poppler emitted Symbol/ArialUnicode font warnings. Visual inspection found the mathematical symbols and text present.',
    'page_png_sha256': {p.name: digest(p) for p in pages},
    'protected_content_preserved': {'native_equations': 6, 'figures': 6, 'result_tables': 2,
                                   'reference_entries': 13, 'comments': 7, 'abstract_and_numerical_endpoints': True},
    'submission_ready': False})
write(citation_path, {'status': 'IDENTICAL_REFERENCES_AND_CLAIM_SCOPE_CARRIED_FORWARD',
    'manuscript_sha256': NEW, 'reference_count': 13,
    'previous_citation_audit': relative(previous_citation), 'previous_citation_audit_sha256': digest(previous_citation),
    'new_external_sources_or_claims': False, 'full_text_unavailable_references': [1, 3, 9, 10],
    'reading_scope_limitations_carried_forward': True,
    'new_sources': report['evidence'], 'new_sources_scope': 'Own archived experiment provenance; no new model evaluation'})
write(index_path, {'revision': 37, 'manuscript': relative(MASTER), 'manuscript_sha256': NEW,
    'previous_evidence_index': relative(previous_index), 'previous_index_sha256': digest(previous_index),
    'previous_index_snapshot_supplement': relative(previous_supplement),
    'previous_index_snapshot_supplement_sha256': digest(previous_supplement),
    'previous_claims_evidence_and_reading_limits_inherited': True,
    'changed_paragraph': report['new_paragraph'], 'updated_comment_id': 6,
    'new_evidence': report['evidence'], 'data_evidence_uses_immutable_snapshot': True,
    'NO_MSCA_independent_evidence_cutoff': 165, 'model_forwards': 0, 'evaluator_runs': 0,
    'NO_MSCA_complete_endpoint_available': False, 'submission_ready': False})
write(acceptance_path, {'revision': 37, 'manuscript_sha256': NEW, 'NO_MSCA_independently_verified_round': 165,
    'data_rows': 1125, 'existing_completed_results_figures_and_formulas_unchanged': True,
    'both_NO_MSCA_recoveries_disclosed': True, 'discarded_partial165_and_unknown_physical_compute_disclosed': True,
    'visual_qa': relative(qa_path), 'visual_qa_sha256': digest(qa_path),
    'new_conclusions_from_incomplete_NO_MSCA': False,
    'remaining': ['NO_MSCA320 retained native backups, genuine successful exit and final capture',
                  'Independent full320 trajectory and actual terminal/secondary endpoint verification',
                  'MSCA conditional result analysis and abstract/conclusion update',
                  'Complete manuscript evidence and visual acceptance review'],
    'venue_selection_and_template_adaptation_deferred_by_user': True, 'submission_ready': False})

temporary = MASTER.with_name('MPSMA-DEIM_conference_EN_revised.rev37.pending.docx')
assert temporary.resolve().parent == MASTER.resolve().parent and not temporary.exists()
with temporary.open('xb') as f: f.write(CANDIDATE.read_bytes())
assert digest(temporary) == NEW
os.replace(temporary, MASTER)
assert digest(MASTER) == NEW and digest(BACKUP) == OLD
state_path = BUNDLE / 'current_execution_state.json'
state = read(state_path)
state.update(paper_revision=37, paper_sha256=NEW, current_goal_acceptance_audit=str(acceptance_path),
             current_goal_acceptance_audit_sha256=digest(acceptance_path),
             citation_binding_evidence=str(citation_path), citation_binding_evidence_sha256=digest(citation_path),
             current_evidence_index=str(index_path), current_evidence_index_sha256=digest(index_path),
             goal_status='active', goal_objective='resume')
state['goal_tool_status_note'] = 'Explicit get_goal receipt confirms active after the user activated resume; original full monitoring and paper task remains incomplete.'
pending = state_path.with_name(state_path.name + '.rev37_partial')
pending.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'); pending.replace(state_path)
note = ('第三十七轮：补充NO_MSCA第二次恢复披露\n\n'
        '4.5已同时说明第29与164轮完整断点恢复，及第二次中断后丢弃的第165轮部分工作。'
        '保留轮数不证明物理训练计算量相等或与无中断训练位级等同；未知部分继续明确披露。'
        '就地批注6更新至已独立核验165轮的封存证据，数据证据固定引用165轮不可变快照。\n'
        '原稿和上一版已保留；仅变更一个正文段落和对应批注。两张结果表、六幅图、六条公式、'
        '摘要、结论及十三条参考文献保持。标准渲染因缺少LibreOffice失败；Word只读导出在普通沙箱中被COM权限限制，'
        '授权本机通道下成功导出，9页逐页检查通过。\n'
        '应用目标现已真实active，显示目标文本为resume；依照用户原意继续原监测、核验、整理和论文修订范围。'
        'NO_MSCA320轮端点及真实退出尚未完成，不宣称论文可投稿。\n')
(HERE / '二次恢复披露修订说明.txt').write_text(note, encoding='utf-8')
heading = ('【第三十七轮稿件与证据修订：2026-11-10】\n'
           '持续稿已补入NO_MSCA两次恢复及第165轮部分工作丢弃的披露；9页逐页检查通过。'
           '当前独立证据与数据快照到165轮、1125行，完整320轮端点仍待完成，尚不能投稿。'
           '应用目标已真实active，继续原监测及稿件任务。详见第三十七轮证据索引。以下为历史记录。\n\n')
for p in [PAPER / '当前修订入口.txt', PAPER / '目标自主执行入口_20261109.txt',
          RECORDS / '目标验收与阻塞记录.txt', PAPER / '远程公平对照_20261103/结果并入与验收清单.txt']:
    p.write_text(heading + p.read_text(encoding='utf-8-sig'), encoding='utf-8')
with (BUNDLE / '方案与执行记录.md').open('a', encoding='utf-8') as f:
    f.write('\n\n## 2026-11-10 第三十七轮二次恢复披露\n\n' + note)
write(HERE / 'promotion_verified.json', {'status': 'PROMOTED_REV37_WITH_VERIFIED_BACKUP_AND_ALL9_PAGE_VISUAL_QA',
    'observed_utc': now, 'current_manuscript_sha256': digest(MASTER), 'backup_rev36_sha256': digest(BACKUP),
    'evidence_index_sha256': digest(index_path), 'immutable_snapshot_manifest_sha256': digest(snapshot_path),
    'immutable_snapshot_files_verified': 21, 'goal_last_verified_active': True,
    'paper_complete_or_submission_ready': False, 'training_actions': 0, 'evaluator_runs': 0})
print(json.dumps({'status': 'PROMOTED_REV37', 'current_manuscript_sha256': digest(MASTER),
                  'pages': 9, 'independent_NO_MSCA_round': 165, 'data_rows': 1125, 'submission_ready': False}))
