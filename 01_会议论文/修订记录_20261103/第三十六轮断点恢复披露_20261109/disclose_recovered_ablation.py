"""Add the verified NO_MSCA recovery disclosure without altering scientific endpoints."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from zipfile import ZipFile
from lxml import etree

sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
here = Path(__file__).resolve().parent
paper = here.parents[1]
master = paper / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
backup = here / 'MPSMA-DEIM_conference_EN_revised_rev35_backup.docx'
target = here / 'MPSMA-DEIM_conference_EN_revised_rev36.docx'
source_sha = 'd3b1d442bad72584082f299e35d7f8fccfa5029de4571af959b065aaae20ff7b'
recovery = paper / '远程公平对照_20261103/模块消融/NO_MSCA/recovery30219_r29_20261108'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W, 'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
q = lambda value: '{' + W + '}' + value
sha = lambda path: sha256(path.read_bytes()).hexdigest()
text = lambda p: ''.join(p.xpath('.//w:t/text()', namespaces=NS))
read = lambda path: json.loads(path.read_text(encoding='utf-8'))
assert sha(master) == source_sha and not backup.exists() and not target.exists()
addendum_path = recovery / 'recovery_addendum.json'
assert sha(addendum_path) == 'e2db4f390756e33e742684e8f1eff526047403ba2db6bad17d5c9e21132e9d54'
audit_path = recovery / 'lineage_recovered_through_round041.json'
audit = read(audit_path)
assert audit['status'] == 'VERIFIED_PARTIAL_ABLATION_LINEAGE' and audit['completed_rounds'] == 41
assert audit['learning_rates']['recorded_attempts'] == 66297 and audit['learning_rates']['maximum_absolute_difference'] == 0
assert audit['recovery']['new_completed_rounds'] == 12 and audit['recovery']['new_optimizer_update_attempts'] == 19404
assert audit['maximum_latest_metric_error'] == 0 and audit['validation_images'] == 548
assert audit['recovery']['prefix']['all_saved_rng_sections_valid'] is True
assert audit['recovery']['ast_checks']['entire_epoch_loop_identical'] is True
with ZipFile(master) as source:
    document = etree.fromstring(source.read('word/document.xml'))
    comments = etree.fromstring(source.read('word/comments.xml'))
body = document.find('w:body', NS)
paragraphs = body.findall('w:p', NS)
matches = [p for p in paragraphs if text(p).startswith('BASE resumed after round 214, excluding 11 later rows')]
assert len(matches) == 1
anchor = matches[0]
assert 'NO_MPS began at round zero.' in text(anchor)
disclosure = ('NO_MSCA resumed after round 29 on another RTX 3090, restoring model and EMA weights, '
              'optimizer and warmup state, and all saved random states. The amount of unretained work '
              'after this checkpoint on the unavailable instance is unknown; retained round counts '
              'therefore do not establish equal physical training computation.')
assert all(disclosure not in text(p) for p in paragraphs)
paragraph = etree.Element(q('p'))
properties = anchor.find('w:pPr', NS)
if properties is not None: paragraph.append(deepcopy(properties))
start = etree.SubElement(paragraph, q('commentRangeStart'))
start.set(q('id'), '6')
run = etree.SubElement(paragraph, q('r'))
properties = anchor.find('w:r/w:rPr', NS)
if properties is not None: run.append(deepcopy(properties))
etree.SubElement(run, q('t')).text = disclosure
end = etree.SubElement(paragraph, q('commentRangeEnd'))
end.set(q('id'), '6')
reference_run = etree.SubElement(paragraph, q('r'))
rpr = etree.SubElement(reference_run, q('rPr'))
etree.SubElement(rpr, q('rStyle')).set(q('val'), 'CommentReference')
etree.SubElement(reference_run, q('commentReference')).set(q('id'), '6')
body.insert(body.index(anchor) + 1, paragraph)
assert not any(c.get(q('id')) == '6' for c in comments)
comment = etree.SubElement(comments, q('comment'))
comment.set(q('id'), '6')
comment.set(q('author'), 'Codex')
comment.set(q('initials'), 'CDX')
comment.set(q('date'), datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
cp = etree.SubElement(comment, q('p'))
cr = etree.SubElement(cp, q('r'))
etree.SubElement(cr, q('t')).text = (
    'NO_MSCA原实例项目盘不可用，已按明确授权从第29轮完整备份在另一张同型号RTX3090恢复，'
    '模型/EMA/优化器/预热及全部保存随机状态均恢复。原冻结科学代码、配方、数据字节和101项依赖保持，'
    'GPU/boot变更由独立恢复附录绑定。现已独立核验41轮、66297条学习率及548图保存预测/评价数组，误差0。'
    '该前缀不是320轮终点；失联后的未保留计算量未知，不能主张物理计算相等或无中断位级等同。'
    '证据：远程公平对照_20261103/模块消融/NO_MSCA/recovery30219_r29_20261108/'
    'recovery_addendum.json及lineage_recovered_through_round041.json。')
with backup.open('xb') as stream: stream.write(master.read_bytes())
with ZipFile(master) as source, ZipFile(target, 'x') as destination:
    for item in source.infolist():
        if item.filename == 'word/document.xml':
            data = etree.tostring(document, encoding='UTF-8', xml_declaration=True, standalone=True)
        elif item.filename == 'word/comments.xml':
            data = etree.tostring(comments, encoding='UTF-8', xml_declaration=True, standalone=True)
        else: data = source.read(item.filename)
        destination.writestr(item, data)
with ZipFile(backup) as source, ZipFile(target) as edited:
    changed = [name for name in source.namelist() if source.read(name) != edited.read(name)]
    assert set(changed) == {'word/document.xml', 'word/comments.xml'}
    old = etree.fromstring(source.read('word/document.xml'))
    for xpath in ['.//m:oMath', './/wp:inline', './/w:tbl']:
        assert [etree.tostring(n) for n in old.findall(xpath, NS)] == [etree.tostring(n) for n in document.findall(xpath, NS)]
old_paragraphs = [text(p) for p in old.findall('w:body/w:p', NS)]
new_paragraphs = [text(p) for p in body.findall('w:p', NS)]
assert [p for p in new_paragraphs if p != disclosure] == old_paragraphs
for tag in ['commentRangeStart', 'commentRangeEnd', 'commentReference']:
    assert sorted(n.get(q('id')) for n in document.findall('.//w:' + tag, NS)) == [str(i) for i in range(7)]
report = {'status': 'RECOVERY_DISCLOSURE_CANDIDATE_PENDING_VISUAL_QA', 'revision': 36,
    'source_sha256': source_sha, 'candidate_sha256': sha(target), 'new_disclosure': disclosure,
    'changed_docx_parts': changed, 'new_comment_id': 6,
    'evidence': [{'file': str(p.relative_to(paper)), 'sha256': sha(p)} for p in [addendum_path, audit_path]],
    'scientific_results_figures_equations_styles_abstract_references_unchanged': True,
    'original_paragraphs_preserved': len(old_paragraphs), 'local_model_forwards': 0,
    'local_evaluator_runs': 0, 'NO_MSCA_complete_endpoint_available': False, 'submission_ready': False}
(here / 'recovery_disclosure_candidate.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(here / 'candidate_text.txt').write_text('\n\n'.join(new_paragraphs) + '\n', encoding='utf-8')
renderer = (here.parent / '第三十五轮容量与对照边界_20261108/render_candidate.ps1').read_text(encoding='utf-8-sig')
renderer = renderer.replace('_rev35.docx', '_rev36.docx')
(here / 'render_candidate.ps1').write_text(renderer, encoding='utf-8')
print(json.dumps({k: report[k] for k in ['status', 'candidate_sha256', 'new_comment_id', 'submission_ready']}))
