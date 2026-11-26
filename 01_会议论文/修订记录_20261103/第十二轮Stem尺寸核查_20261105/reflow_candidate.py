"""Shorten the new shape qualification to avoid splitting its neighboring expression."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
from lxml import etree
import json

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE / 'MPSMA-DEIM_conference_EN_revised_第十二轮候选.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第十二轮候选_v2.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第十一轮备份.docx'
REPORT = HERE / 'twelfth_revision_v2_pending_qa.json'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
LONG = ('For the reported 640 × 640 inputs, the pooled detail and strided main paths align; '
        'in the inspected implementation, an odd input height or width gives different path sizes '
        'and prevents concatenation. ')
SHORT = 'In the inspected stem, an odd input height or width prevents concatenation of the detail and main paths. '

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def main():
    prior = json.loads((HERE / 'twelfth_revision_pending_qa.json').read_text(encoding='utf-8'))
    assert digest(PREVIOUS) == prior['candidate_sha256'] and digest(BACKUP) == prior['source_sha256']
    assert not CANDIDATE.exists() and not REPORT.exists()
    with ZipFile(PREVIOUS) as z:
        original = z.read('word/document.xml')
    assert original.count(LONG.encode('utf-8')) == 1
    updated = original.replace(LONG.encode('utf-8'), SHORT.encode('utf-8'))
    root = etree.fromstring(updated)
    with ZipFile(PREVIOUS) as source, ZipFile(CANDIDATE, 'x') as target:
        for item in source.infolist():
            target.writestr(item, updated if item.filename == 'word/document.xml' else source.read(item.filename))
    with ZipFile(BACKUP) as old, ZipFile(CANDIDATE) as new:
        assert old.namelist() == new.namelist() and new.testzip() is None
        assert [n for n in old.namelist() if old.read(n) != new.read(n)] == ['word/document.xml']
        before = etree.fromstring(old.read('word/document.xml')).findall('.//w:p', NS)
        after = root.findall('.//w:p', NS)
        changed = [i for i, (a, b) in enumerate(zip(before, after)) if etree.tostring(a) != etree.tostring(b)]
        assert changed == [x['paragraph_index'] for x in prior['changes']]
    counts = {'tables': len(root.findall('.//w:tbl', NS)), 'equations': len(root.findall('.//m:oMath', NS)),
              'inline_images': len(root.findall('.//wp:inline', NS)),
              **{key: len(root.findall('.//w:' + key, NS)) for key in
                 ['commentRangeStart', 'commentRangeEnd', 'commentReference']}}
    assert counts == prior['structural_counts']
    report = {'status': 'CANDIDATE_PENDING_RENDER_AND_REVIEW', 'revision': 12,
              'source_sha256': prior['source_sha256'], 'backup_sha256': digest(BACKUP),
              'first_candidate_sha256': digest(PREVIOUS), 'candidate_sha256': digest(CANDIDATE),
              'reason': 'Shortened the added sentence because the first rendering split the next expression across pages.',
              'added_method_sentence': SHORT.strip(), 'caption': prior['changes'][1]['after'],
              'changed_paragraph_indices': changed, 'structural_counts': counts,
              'evidence_sha256': prior['evidence_sha256'], 'only_document_xml_changed': True}
    with REPORT.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'status': report['status'], 'candidate_sha256': report['candidate_sha256']}))

if __name__ == '__main__':
    main()
