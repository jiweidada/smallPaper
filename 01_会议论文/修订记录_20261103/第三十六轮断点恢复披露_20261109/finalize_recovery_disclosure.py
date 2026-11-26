"""Promote the verified recovery disclosure, preserving all previous scientific content."""
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
here = Path(__file__).resolve().parent
records, paper = here.parent, here.parents[1]
bundle = paper / '远程公平对照_20261103/模块消融'
recovery = bundle / 'NO_MSCA/recovery30219_r29_20261108'
master = paper / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
backup = here / 'MPSMA-DEIM_conference_EN_revised_rev35_backup.docx'
candidate = here / 'MPSMA-DEIM_conference_EN_revised_rev36_v2.docx'
old_sha = 'd3b1d442bad72584082f299e35d7f8fccfa5029de4571af959b065aaae20ff7b'
new_sha = '57fb02f46033266c75f959259028275d8fbd64b139450d0941dab49063a5a343'
original_sha = 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
ns = {'w': W, 'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
q = lambda name: '{' + W + '}' + name
sha = lambda path: sha256(path.read_bytes()).hexdigest()
read = lambda path: json.loads(path.read_text(encoding='utf-8-sig'))
text = lambda p: ''.join(p.xpath('.//w:t/text()', namespaces=ns))
relative = lambda path: path.relative_to(paper).as_posix()
def write(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')

assert sha(master) == sha(backup) == old_sha and sha(candidate) == new_sha
for path in [paper / '稿件/MPSMA-DEIM_conference_EN.docx', paper / '稿件/备份_20261103/MPSMA-DEIM_conference_EN_original.docx']:
    assert sha(path) == original_sha
assert read(here / 'render_metadata_v2.json')['page_count'] == 9
pages = sorted((here / 'rendered_word_v2').glob('page-*.png'))
assert len(pages) == 9 and all(path.stat().st_size > 80000 for path in pages)
assert 'soffice.exe was not found' in (here / 'packaged_render_attempt_v2.log').read_text(encoding='utf-8-sig')
with ZipFile(backup) as before, ZipFile(candidate) as after:
    assert after.testzip() is None and set(before.namelist()) == set(after.namelist())
    changed = [name for name in before.namelist() if before.read(name) != after.read(name)]
    assert set(changed) == {'word/document.xml', 'word/comments.xml'}
    old, new = [etree.fromstring(z.read('word/document.xml')) for z in [before, after]]
    old_comments, comments = [etree.fromstring(z.read('word/comments.xml')) for z in [before, after]]
    for xpath, count in [('.//m:oMath', 6), ('.//wp:inline', 6), ('.//w:tbl', 2)]:
        a, b = old.findall(xpath, ns), new.findall(xpath, ns)
        assert len(a) == len(b) == count and [etree.tostring(x) for x in a] == [etree.tostring(x) for x in b]
    assert len(old_comments) == 6 and len(comments) == 7
    assert [etree.tostring(x) for x in old_comments] == [etree.tostring(x) for x in list(comments)[:6]]
    relationships = etree.fromstring(after.read('word/_rels/document.xml.rels'))
    assert len([r for r in relationships if r.get('Type', '').endswith('/comments') and r.get('Target') == 'comments.xml']) == 1
    assert b'/word/comments.xml' in after.read('[Content_Types].xml')
for tag in ['commentRangeStart', 'commentRangeEnd', 'commentReference']:
    assert sorted(n.get(q('id')) for n in new.findall('.//w:' + tag, ns)) == [str(n) for n in range(7)]
old_paragraphs = [text(p) for p in old.findall('w:body/w:p', ns)]
paragraphs = [text(p) for p in new.findall('w:body/w:p', ns)]
addition = read(here / 'recovery_disclosure_candidate.json')['new_disclosure']
assert [p for p in paragraphs if p != addition] == old_paragraphs
assert len(paragraphs) == len(old_paragraphs) + 1
references = [p for p in paragraphs if re.match(r'^\[\d+\]', p)]
assert len(references) == 13 and references == [p for p in old_paragraphs if re.match(r'^\[\d+\]', p)]
abstract = next(p for p in paragraphs if p.startswith('Small objects in unmanned aerial vehicle'))
assert abstract == next(p for p in old_paragraphs if p.startswith('Small objects in unmanned aerial vehicle'))
body = '\n'.join(paragraphs[:paragraphs.index('References')])
cited = set()
for match in re.findall(r'(?<![=\w])\[([\d, ]+)\]', body):
    cited.update(int(n) for n in match.split(',') if n.strip())
assert cited == set(range(1, 14))
assert 'NO_MSCA remains in progress.' in body and '\ufffd' not in body
reference_heading = next(p for p in new.findall('w:body/w:p', ns) if text(p) == 'References')
assert reference_heading.find('w:pPr/w:pageBreakBefore', ns) is not None
audit_path = recovery / 'lineage_recovered_through_round041.json'
audit = read(audit_path)
assert audit['completed_rounds'] == 41 and audit['learning_rates']['recorded_attempts'] == 66297
assert audit['learning_rates']['maximum_absolute_difference'] == audit['maximum_latest_metric_error'] == 0
assert audit['recovery']['new_completed_rounds'] == 12
assert audit['recovery']['recovery_addendum_sha256'] == sha(recovery / 'recovery_addendum.json')
data_root = paper / '实验数据汇总_20261108'
data_manifest_path = data_root / '交付清单.json'
data_manifest = read(data_manifest_path)
assert data_manifest['status'] == 'COMPLETE' and data_manifest['NO_MSCA_accepted_rounds'] == 41
for item in data_manifest['files']:
    path = (data_root / item['file']).resolve(strict=True)
    assert path.is_relative_to(data_root) and sha(path) == item['sha256'], item['file']
assert read(data_root / '第41轮更新核查.json')['per_epoch_csv_rows'] == 1001
now = datetime.now(timezone.utc).isoformat()
qa = {'status': 'PASSED', 'revision': 36, 'observed_utc': now,
    'source_sha256': old_sha, 'candidate_sha256': new_sha,
    'rendered_pages_visually_inspected_individually': list(range(1, 10)),
    'renderer': 'Hidden macro-disabled read-only Word export and bundled Poppler',
    'standard_renderer_limitation': 'Missing soffice.exe, verified from preserved v2 log',
    'initial_layout_rejected': 'Two references alone on page9',
    'final_layout': 'Full reference list starts on page9; no venue page limit selected',
    'visual_result': 'All9 pages reviewed: readable text, diagrams, six native equations, two tables and complete references; no clipping or overlap.',
    'page_png_sha256': {path.name: sha(path) for path in pages},
    'unchanged_exactly': {'performance_tables': 2, 'native_equations': 6, 'figures': 6,
                         'reference_entries': 13, 'old_comments': 6, 'abstract': True},
    'new_comment_id': 6, 'submission_ready': False}
write(here / 'recovery_disclosure_visual_qa.json', qa)
old_citation = records / '第三十五轮容量与对照边界_20261108/citation_carryforward_rev35.json'
write(here / 'citation_carryforward_rev36.json', {'status': 'IDENTICAL_REFERENCES_AND_CLAIM_SCOPE_CARRIED_FORWARD',
    'manuscript_sha256': new_sha, 'reference_count': 13, 'all_cited': True,
    'previous_citation_audit': relative(old_citation), 'previous_citation_audit_sha256': sha(old_citation),
    'new_external_claims': False, 'full_text_unavailable_references': [1, 3, 9, 10],
    'reading_scope_limitations_carried_forward': True, 'new_sources': [relative(audit_path)]})
previous_index = records / '主要结论证据索引/主要结论证据索引_第三十五轮.json'
index = {'revision': 36, 'manuscript': relative(master), 'manuscript_sha256': new_sha,
    'previous_evidence_index': relative(previous_index), 'previous_index_sha256': sha(previous_index),
    'previous_claims_evidence_and_reading_limits_inherited': True,
    'new_change': addition, 'new_evidence': [{'file': relative(path), 'sha256': sha(path)}
        for path in [recovery / 'recovery_addendum.json', audit_path, data_manifest_path]],
    'new_source_scope': 'Own saved experiment artifacts and local source; no new forward or evaluator in this revision',
    'NO_MSCA_complete_endpoint_available': False, 'submission_ready': False}
index_path = records / '主要结论证据索引/主要结论证据索引_第三十六轮.json'
write(index_path, index)
with index_path.with_suffix('.txt').open('x', encoding='utf-8') as handle:
    handle.write('第三十六轮证据索引\n\n' + json.dumps(index, ensure_ascii=False, indent=2) + '\n')
write(here / 'current_goal_acceptance_audit_rev36.json', {'revision': 36, 'manuscript_sha256': new_sha,
    'existing_completed_results_unchanged': True, 'NO_MSCA_independently_verified_round': 41,
    'NO_MSCA_target_round': 320, 'data_rows': 1001, 'recovery_disclosure_added': True,
    'remaining': ['NO_MSCA320 complete lineage/endpoints and genuine exit',
                  'MSCA conditional results analysis and abstract/conclusion update',
                  'final complete manuscript evidence and visual review'],
    'venue_selection_and_template_adaptation_deferred_by_user': True, 'submission_ready': False})
note = ('第三十六轮：NO_MSCA断点恢复披露\n\n'
    '在4.5局限性新增一段，说明NO_MSCA从第29轮完整断点换到同型号RTX3090恢复，保留模型/EMA、优化器、预热和全部保存随机状态。'
    '未保留的物理计算量未知，不据保留轮数主张物理计算相等。新增就地批注6绑定恢复附录和41轮独立核验。\n'
    '两张性能表、六条公式、六幅图、摘要、十三条文献和原六组批注保持。初版只将最后两条引用挤到第9页；'
    '现把完整参考文献另起第9页，未缩小字号，9页逐页检查通过。尚未选会，不把8页当作硬约束。\n'
    '数据汇总已更新至NO_MSCA独立核验41轮/66297更新，共1001条逐轮记录。该前缀不是320轮终点，现有最终结论不变。'
    '尚待完整消融端点、全轨迹和真实退出/备份核验后，再更新结果分析、摘要与结论。\n'
    '当前目标工具仍显示历史blocked，创建新目标被“已有未完成目标”拒绝；用户已要求继续自主执行。已核对官方文档，'
    '恢复控制由用户在客户端/goal resume处理，不虚报工具状态已active。实验与当前修订工作实际继续。\n')
(here / '断点恢复披露修订说明.txt').write_text(note, encoding='utf-8')
# Replace only the maintained manuscript; preserve the immutable original and exact rev35 backup.
assert master.resolve().parent == (paper / '稿件').resolve(strict=True)
temporary = master.with_name('MPSMA-DEIM_conference_EN_revised.rev36.pending.docx')
assert temporary.resolve().parent == master.resolve().parent and not temporary.exists()
with temporary.open('xb') as handle: handle.write(candidate.read_bytes())
assert sha(temporary) == new_sha
os.replace(temporary, master)
assert sha(master) == new_sha and sha(backup) == old_sha
state_path = bundle / 'current_execution_state.json'
state = read(state_path)
state['paper_revision'] = 36
state['NO_MSCA']['data_snapshot_independently_verified_round'] = 41
state['NO_MSCA']['data_handoff'] = {'path': str(data_manifest_path), 'sha256': sha(data_manifest_path),
                                 'data_rows': 1001, 'NO_MSCA_verified_round': 41}
state['goal_user_instruction'] = '2026-11-09: 设定目标 自己执行; continue existing objective autonomously within authorization'
state['goal_tool_status_note'] = 'Existing unfinished goal remains blocked; create_goal rejected duplicate. User cannot find UI control; official /goal resume instruction supplied. Actual authorized experiment and manuscript work continues.'
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
heading = f'【第三十六轮稿件与证据修订：2026-11-09】\n持续稿SHA256 {new_sha}，恢复披露和就地批注已加入，9页逐页核验通过。NO_MSCA已独立核验41轮，数据汇总1001行；完整320轮仍待完成，尚不能投稿。\n详见第三十六轮证据索引及第三十六轮断点恢复披露_20261109/断点恢复披露修订说明.txt。以下为历史记录。\n\n'
for path in [paper / '当前修订入口.txt', records / '目标验收与阻塞记录.txt', paper / '远程公平对照_20261103/结果并入与验收清单.txt']:
    path.write_text(heading + path.read_text(encoding='utf-8'), encoding='utf-8')
with (bundle / '方案与执行记录.md').open('a', encoding='utf-8') as handle:
    handle.write('\n\n## 2026-11-09 第三十六轮恢复披露与41轮证据\n\n' + note + '\n')
write(here / 'promotion_verified.json', {'status': 'PROMOTED_REV36_WITH_VERIFIED_BACKUP_AND_VISUAL_QA',
    'current_manuscript_sha256': sha(master), 'backup_rev35_sha256': sha(backup),
    'data_manifest_sha256': sha(data_manifest_path), 'goal_tool_claimed_active': False,
    'paper_complete_or_submission_ready': False})
print(json.dumps({'status': 'PROMOTED_REV36', 'sha256': sha(master), 'pages': 9, 'data_rows': 1001,
                  'NO_MSCA_verified_round': 41, 'submission_ready': False}))
