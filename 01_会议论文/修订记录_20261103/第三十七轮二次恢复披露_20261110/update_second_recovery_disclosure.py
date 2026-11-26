"""Update only the interrupted-run disclosure from independently checked artifacts."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile
from lxml import etree

sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
RECORDS = HERE.parent
RECOVERY = PAPER / '远程公平对照_20261103/模块消融/NO_MSCA/recovery30715_20261109'
MASTER = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_rev36_backup.docx'
TARGET = HERE / 'MPSMA-DEIM_conference_EN_revised_rev37.docx'
SOURCE_SHA = '57fb02f46033266c75f959259028275d8fbd64b139450d0941dab49063a5a343'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W, 'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
Q = lambda name: '{' + W + '}' + name
digest = lambda p: sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
text = lambda p: ''.join(p.xpath('.//w:t/text()', namespaces=NS))
relative = lambda p: p.relative_to(PAPER).as_posix()

assert digest(MASTER) == SOURCE_SHA
assert not BACKUP.exists() and not TARGET.exists()
evidence = [
    (RECOVERY / 'recovery30715_addendum.json', '5b6fec3b7f4fe2f088cd0176b085fa6074731082307e6215d2ff46ce918a71c9'),
    (RECOVERY / 'segment30715_audit_binding.json', '6de82b5b4bcbf8bbdbd3a61a2fb7ecbd70905b98da0eceefc55c7b451435a41e'),
    (RECOVERY / 'lineage_second_recovery_through_round165.json', '27ff571947fbf99296d8dab7a722eb5064cc3ca41b65d1c3789e4dfe95a3cb21'),
    (PAPER / '实验数据汇总_20261108/verified_snapshots/NO_MSCA_round165_verified_20261109/snapshot_manifest.json',
     'beb46756c1abd556597ff7ce5829226a8bd837ec4749a108df69f0fe261cfc8b'),
]
for p, expected in evidence:
    assert digest(p) == expected, p
addendum, binding, audit, snapshot = [read(p) for p, _ in evidence]
assert addendum['first_epoch_index'] == 164 and addendum['retained_round_limit'] == 320
assert addendum['observed_discarded_partial_round165_updates'] == 1401
assert audit['status'] == 'VERIFIED_PARTIAL_ABLATION_LINEAGE' and audit['completed_rounds'] == 165
assert audit['learning_rates']['recorded_attempts'] == 266805
assert audit['learning_rates']['maximum_absolute_difference'] == audit['maximum_latest_metric_error'] == 0
assert audit['model_forwards'] == audit['evaluator_runs'] == 0
assert audit['recovery']['status'] == 'EXACT_SECOND_RECOVERY_CONTRACT_AND164_PREFIX_VERIFIED'

with ZipFile(MASTER) as z:
    before = etree.fromstring(z.read('word/document.xml'))
    old_comments = etree.fromstring(z.read('word/comments.xml'))
doc, comments = deepcopy(before), deepcopy(old_comments)
paragraphs = doc.findall('w:body/w:p', NS)
matches = [p for p in paragraphs if text(p).startswith('NO_MSCA resumed after round 29 on another RTX 3090')]
assert len(matches) == 1
paragraph = matches[0]
old_text = text(paragraph)
new_text = (
    'NO_MSCA resumed from complete checkpoints after rounds 29 and 164 on replacement RTX 3090 '
    'instances, restoring model and EMA weights, optimizer and warmup state, and all saved random '
    'states. Partial work in round 165 was discarded after the second interruption. The full amount '
    'of unretained work across the interruptions is unknown; retained round counts therefore do not '
    'establish equal physical training computation or bitwise equivalence to uninterrupted training.'
)
runs = paragraph.findall('.//w:t', NS)
assert len(runs) == 1
runs[0].text = new_text
assert len(comments) == 7
comment_matches = [p for p in comments if p.get(Q('id')) == '6']
assert len(comment_matches) == 1
comment = comment_matches[0]
comment_texts = comment.findall('.//w:t', NS)
assert len(comment_texts) == 1
new_comment = (
    '更新断点恢复披露：NO_MSCA先从第29轮、后因实例到期从第164轮完整备份恢复，'
    '两次均恢复模型、EMA、优化器、预热和全部已保存随机状态。第165轮旧分支至少观察到1401次更新，'
    '没有完整断点，已归档而未混入保留轨迹；未观察到的物理计算量未知。'
    '第二次恢复保留原科学epoch循环和状态转换，数据字节及依赖核对一致；'
    '165轮独立轨迹核验通过，266805条学习率及保存预测/评价数组/12指标误差0。'
    '此证据仅到165轮，不是320轮终点，也不证明物理计算量相等或无中断位级等同。'
    '证据：远程公平对照_20261103/模块消融/NO_MSCA/recovery30715_20261109/'
    'recovery30715_addendum.json、segment30715_audit_binding.json及'
    'lineage_second_recovery_through_round165.json；历史第29轮恢复证据保留。'
)
comment_texts[0].text = new_comment

old_paragraphs = [text(p) for p in before.findall('w:body/w:p', NS)]
new_paragraphs = [text(p) for p in paragraphs]
assert [old_text if s == new_text else s for s in new_paragraphs] == old_paragraphs
for xpath, count in [('.//m:oMath', 6), ('.//wp:inline', 6), ('.//w:tbl', 2)]:
    a, b = before.findall(xpath, NS), doc.findall(xpath, NS)
    assert len(a) == len(b) == count
    assert [etree.tostring(n) for n in a] == [etree.tostring(n) for n in b]
for a, b in zip(old_comments, comments):
    if a.get(Q('id')) != '6': assert etree.tostring(a) == etree.tostring(b)
restored = deepcopy(doc)
next(p for p in restored.findall('w:body/w:p', NS) if text(p) == new_text).find('.//w:t', NS).text = old_text
assert etree.tostring(restored) == etree.tostring(before)
references = [s for s in new_paragraphs if re.match(r'^\[\d+\]', s)]
assert len(references) == 13
assert references == [s for s in old_paragraphs if re.match(r'^\[\d+\]', s)]
assert 'NO_MSCA remains in progress.' in '\n'.join(new_paragraphs)
for tag in ['commentRangeStart', 'commentRangeEnd', 'commentReference']:
    assert sorted(n.get(Q('id')) for n in doc.findall('.//w:' + tag, NS)) == [str(n) for n in range(7)]

with BACKUP.open('xb') as f: f.write(MASTER.read_bytes())
with ZipFile(MASTER) as source, ZipFile(TARGET, 'x') as dest:
    for item in source.infolist():
        if item.filename == 'word/document.xml':
            data = etree.tostring(doc, encoding='UTF-8', xml_declaration=True, standalone=True)
        elif item.filename == 'word/comments.xml':
            data = etree.tostring(comments, encoding='UTF-8', xml_declaration=True, standalone=True)
        else: data = source.read(item.filename)
        dest.writestr(item, data)
with ZipFile(BACKUP) as old, ZipFile(TARGET) as new:
    assert new.testzip() is None and old.namelist() == new.namelist()
    changed = [name for name in old.namelist() if old.read(name) != new.read(name)]
    assert set(changed) == {'word/document.xml', 'word/comments.xml'}

report = {
    'status': 'SECOND_RECOVERY_DISCLOSURE_CANDIDATE_PENDING_VISUAL_QA', 'revision': 37,
    'observed_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': SOURCE_SHA,
    'candidate_sha256': digest(TARGET), 'old_paragraph': old_text, 'new_paragraph': new_text,
    'updated_comment_id': 6, 'changed_parts': changed,
    'all_other_document_content_exactly_preserved': True,
    'figures': 6, 'native_equations': 6, 'result_tables': 2, 'reference_entries': 13, 'comments': 7,
    'evidence': [{'file': relative(p), 'sha256': expected} for p, expected in evidence],
    'new_scientific_endpoints': False, 'new_external_sources_or_claims': False,
    'model_forwards': 0, 'evaluator_runs': 0, 'NO_MSCA_complete_endpoint_available': False,
    'submission_ready': False,
}
with (HERE / 'candidate_structural_and_evidence_check.json').open('x', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2); f.write('\n')
(HERE / 'candidate_text.txt').write_text('\n\n'.join(new_paragraphs) + '\n', encoding='utf-8')
renderer = (RECORDS / '第三十六轮断点恢复披露_20261109/render_candidate_v2.ps1').read_text('utf-8-sig')
renderer = renderer.replace('_rev36_v2.docx', '_rev37.docx').replace('rendered_word_v2', 'rendered_word').replace('render_metadata_v2.json', 'render_metadata.json')
(HERE / 'render_candidate.ps1').write_text(renderer, encoding='utf-8')
print(json.dumps({k: report[k] for k in ['status', 'candidate_sha256', 'submission_ready']}))
