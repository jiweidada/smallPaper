"""Clarify the complete aggregation path in method prose and the Figure 3 caption."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
from lxml import etree
import json
import shutil

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent
SOURCE = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第九轮备份.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第十轮候选.docx'
EXPECTED = '41cbe7bd9c42fda9ab5fac813207209d591250076e15d62f3097647d1716fc2f'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

assert sha(SOURCE) == EXPECTED and not BACKUP.exists() and not CANDIDATE.exists()
evidence = HERE / 'aggregation_path_audit.json'
audit = json.loads(evidence.read_text(encoding='utf-8'))
assert audit['status'] == 'AGGREGATION_PROJECTION_PATHS_STATICALLY_TRACED'
assert len(audit['settings']) == 8
for item in audit['sources'].values():
    assert sha(Path(item['path'])) == item['sha256']
for run in ['BASE', 'FULL']:
    source = audit['sources'][audit['settings'][run]['source_filename']]
    assert source['frozen_manifest_match'] is True

with ZipFile(SOURCE) as z:
    original = z.read('word/document.xml')
root = etree.fromstring(original)
ps = root.findall('.//w:p', NS)
changes = {
    25: ('The four stages use ESE, SimAM, MSCA-Lite and MSCA-Dilated (Fig. 3). '
         'Each HG block concatenates its features, projects them directly to its output width C using one 1 × 1 ConvBNAct, and then applies its gate. '
         'The local baseline uses two 1 × 1 ConvBNAct blocks with intermediate width C/2; the change therefore includes the projection path. '
         'B0 has one block in stages 1, 2 and 4 and two in stage 3, giving two MSCA-Lite instances. '
         'Stage outputs are 64, 256, 512 and 1024 channels at strides 4, 8, 16 and 32. '
         'Stage-2 SimAM uses variance denominator HW − 1 [8]; the two MSCA variants use denominator HW, so we describe them as SimAM-style gates.'),
    27: ('Fig. 3. Stage-specific gates within the B0 backbone. Each HG block aggregates concatenated features '
         'with a full-width 1 × 1 ConvBNAct followed by its gate.')
}
assert ''.join(ps[25].xpath('.//w:t/text()', namespaces=NS)).startswith('The four stages use ESE, SimAM, MSCA-Lite and MSCA-Dilated, respectively')
assert ''.join(ps[27].xpath('.//w:t/text()', namespaces=NS)).startswith('Fig. 3. Stage-specific gates within the B0 backbone.')
updated = original
log = []
for index, after in changes.items():
    p = ps[index]
    before = ''.join(p.xpath('.//w:t/text()', namespaces=NS))
    assert len(p.findall('.//w:t', NS)) == 1
    assert original.count(before.encode('utf-8')) == 1
    assert not any(c in after for c in '<&')
    updated = updated.replace(before.encode('utf-8'), after.encode('utf-8'))
    log.append({'paragraph': index, 'before': before, 'after': after})
new_root = etree.fromstring(updated)
np = new_root.findall('.//w:p', NS)
assert len(ps) == len(np)
assert [i for i, (a, b) in enumerate(zip(ps, np)) if etree.tostring(a) != etree.tostring(b)] == [25, 27]
shutil.copy2(SOURCE, BACKUP)
with ZipFile(SOURCE) as src, ZipFile(CANDIDATE, 'x') as dst:
    for item in src.infolist():
        dst.writestr(item, updated if item.filename == 'word/document.xml' else src.read(item.filename))
with ZipFile(SOURCE) as old, ZipFile(CANDIDATE) as new:
    assert old.namelist() == new.namelist() and new.testzip() is None
    assert [n for n in old.namelist() if old.read(n) != new.read(n)] == ['word/document.xml']
counts = {'tables': len(new_root.findall('.//w:tbl', NS)),
          'equations': len(new_root.findall('.//m:oMath', NS)),
          'inline_images': len(new_root.findall('.//wp:inline', NS)),
          **{name: len(new_root.findall('.//w:' + name, NS))
             for name in ['commentRangeStart', 'commentRangeEnd', 'commentReference']}}
assert counts == {'tables': 6, 'equations': 6, 'inline_images': 7,
                  'commentRangeStart': 5, 'commentRangeEnd': 5, 'commentReference': 5}
report = {'status': 'CANDIDATE_PENDING_RENDER_AND_REVIEW', 'revision': 10,
          'source_sha256': EXPECTED, 'candidate_sha256': sha(CANDIDATE),
          'backup_path': str(BACKUP), 'backup_sha256': sha(BACKUP),
          'changes': log, 'only_document_xml_changed': True,
          'all_other_paragraphs_unchanged': True, 'structural_counts': counts,
          'evidence_file': str(evidence), 'evidence_sha256': sha(evidence),
          'scope': 'Make the full-width aggregation projection explicit, including the baseline bottleneck difference. No changes to methods implementation, table values, equations, images, references or experimental execution.'}
(HERE / 'tenth_revision_pending_qa.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: report[k] for k in ['status', 'candidate_sha256', 'structural_counts']}))
