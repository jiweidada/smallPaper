"""Clarify evaluator class-wise limits versus the global postprocessor output count."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
from lxml import etree
import json
import shutil

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第十轮备份.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第十一轮候选.docx'
OLD = '2dea983b748b1002f372d138c591db2845a36d000c890275ea750499ce09b0e0'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

assert sha(CURRENT) == OLD and not BACKUP.exists() and not CANDIDATE.exists()
evidence = HERE / 'evaluation_limit_audit.json'
audit = json.loads(evidence.read_text(encoding='utf-8'))
assert audit['remote_sources_identical_on_both_hosts'] and len(audit['settings']) == 8
for item in audit['archived_sources']:
    assert sha(Path(item['path'])) == item['sha256']
with ZipFile(CURRENT) as z:
    original = z.read('word/document.xml')
root = etree.fromstring(original)
ps = root.findall('.//w:p', NS)
texts = [''.join(p.xpath('.//w:t/text()', namespaces=NS)) for p in ps]
data_index = next(i for i, t in enumerate(texts) if t.startswith('The experiments use the VisDrone2019 detection data'))
metric_index = next(i for i, t in enumerate(texts) if 'AR@100 allows at most 100 detections per image.' in t)
assert (data_index, metric_index) == (50, 55)
before = texts[data_index]
assert 'and resize inputs to 640 × 640.' in before
assert 'ten categories and maximum-detection limits of 1, 10 and 100.' in before
data_after = before.replace('and resize inputs to 640 × 640.',
    'and resize inputs to 640 × 640, with 300 postprocessor predictions per image across categories.')
data_after = data_after.replace('ten categories and maximum-detection limits of 1, 10 and 100.',
    'ten categories, class-aware matching (useCats=1) and maxDets=[1, 10, 100] per image and category.')
metric_after = texts[metric_index].replace('AR@100 allows at most 100 detections per image.',
    'AR@100 denotes recall at maxDets=100 in this class-aware evaluation.')
updated = original
changes = []
for index, after in [(data_index, data_after), (metric_index, metric_after)]:
    before = texts[index]
    assert len(ps[index].findall('.//w:t', NS)) == 1
    assert original.count(before.encode('utf-8')) == 1
    assert not any(c in after for c in '<&')
    updated = updated.replace(before.encode('utf-8'), after.encode('utf-8'))
    changes.append({'paragraph': index, 'before': before, 'after': after})
new_root = etree.fromstring(updated)
np = new_root.findall('.//w:p', NS)
assert len(np) == len(ps)
assert [i for i, (a, b) in enumerate(zip(ps, np)) if etree.tostring(a) != etree.tostring(b)] == [50, 55]
shutil.copy2(CURRENT, BACKUP)
with ZipFile(CURRENT) as source, ZipFile(CANDIDATE, 'x') as output:
    for item in source.infolist():
        output.writestr(item, updated if item.filename == 'word/document.xml' else source.read(item.filename))
with ZipFile(CURRENT) as a, ZipFile(CANDIDATE) as b:
    assert a.namelist() == b.namelist() and b.testzip() is None
    assert [n for n in a.namelist() if a.read(n) != b.read(n)] == ['word/document.xml']
counts = {'tables': len(new_root.findall('.//w:tbl', NS)), 'equations': len(new_root.findall('.//m:oMath', NS)),
          'inline_images': len(new_root.findall('.//wp:inline', NS)),
          **{k: len(new_root.findall('.//w:' + k, NS)) for k in ['commentRangeStart', 'commentRangeEnd', 'commentReference']}}
assert counts == {'tables': 6, 'equations': 6, 'inline_images': 7, 'commentRangeStart': 5, 'commentRangeEnd': 5, 'commentReference': 5}
report = {'status': 'CANDIDATE_PENDING_RENDER_AND_REVIEW', 'revision': 11,
          'source_sha256': OLD, 'candidate_sha256': sha(CANDIDATE),
          'backup_path': str(BACKUP), 'backup_sha256': sha(BACKUP), 'changes': changes,
          'only_document_xml_changed': True, 'all_other_paragraphs_unchanged': True,
          'structural_counts': counts, 'evidence_file': str(evidence), 'evidence_sha256': sha(evidence),
          'scope': 'Clarify class-aware maxDets limits and the distinct top-300 postprocessor setting. No metric changes, model execution, evaluator rerun or training-setting changes.'}
(HERE / 'eleventh_revision_pending_qa.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': report['status'], 'candidate_sha256': report['candidate_sha256'], 'structural_counts': counts}))
