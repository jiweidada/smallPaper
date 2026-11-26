"""Clarify the documented input shape without changing equations, figures or results."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
from lxml import etree
import json
import shutil

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第十一轮备份.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第十二轮候选.docx'
REPORT = HERE / 'twelfth_revision_pending_qa.json'
OLD_SHA = '06c51c534c640fe35348bd88ed6c9a2a45c540b35521e378f008fe5a93126809'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def main():
    assert digest(CURRENT) == OLD_SHA
    assert not any(p.exists() for p in [BACKUP, CANDIDATE, REPORT])
    evidence_path = HERE / 'stem_shape_static_audit.json'
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    assert evidence['status'] == 'STEM_INPUT_SHAPE_STATICALLY_CHECKED_NO_MODEL_EXECUTION'
    assert evidence['manuscript_sha256'] == OLD_SHA
    with ZipFile(CURRENT) as z:
        original = z.read('word/document.xml')
    root = etree.fromstring(original)
    paragraphs = root.findall('.//w:p', NS)
    texts = [''.join(p.xpath('.//w:t/text()', namespaces=NS)) for p in paragraphs]
    method = next(i for i, text in enumerate(texts) if text.startswith('MPS replaces the standard stem with three paths'))
    caption = next(i for i, text in enumerate(texts) if text.startswith('Fig. 2. MPS for HGNetv2-B0.'))
    old_method, old_caption = texts[method], texts[caption]
    marker = 'Concatenating the three outputs produces a 28-channel feature Z at half the input resolution.'
    addition = ('For the reported 640 × 640 inputs, the pooled detail and strided main paths align; '
                'in the inspected implementation, an odd input height or width gives different path sizes '
                'and prevents concatenation. ')
    assert old_method.count(marker) == 1
    new_method = old_method.replace(marker, addition + marker)
    new_caption = old_caption.replace('Fig. 2. MPS for HGNetv2-B0.',
                                      'Fig. 2. MPS for HGNetv2-B0 with H = W = 640 for the reported inputs.')
    updated = original
    changes = []
    for index, before, after in [(method, old_method, new_method), (caption, old_caption, new_caption)]:
        assert len(paragraphs[index].findall('.//w:t', NS)) == 1
        assert original.count(before.encode('utf-8')) == 1 and not any(c in after for c in '<&')
        updated = updated.replace(before.encode('utf-8'), after.encode('utf-8'))
        changes.append({'paragraph_index': index, 'before': before, 'after': after})
    new_root = etree.fromstring(updated)
    new_paragraphs = new_root.findall('.//w:p', NS)
    assert len(new_paragraphs) == len(paragraphs)
    assert [i for i, (a, b) in enumerate(zip(paragraphs, new_paragraphs))
            if etree.tostring(a) != etree.tostring(b)] == [method, caption]
    shutil.copy2(CURRENT, BACKUP)
    assert digest(BACKUP) == OLD_SHA
    with ZipFile(CURRENT) as source, ZipFile(CANDIDATE, 'x') as output:
        for item in source.infolist():
            output.writestr(item, updated if item.filename == 'word/document.xml' else source.read(item.filename))
    with ZipFile(CURRENT) as a, ZipFile(CANDIDATE) as b:
        assert a.namelist() == b.namelist() and b.testzip() is None
        assert [name for name in a.namelist() if a.read(name) != b.read(name)] == ['word/document.xml']
    counts = {'tables': len(new_root.findall('.//w:tbl', NS)),
              'equations': len(new_root.findall('.//m:oMath', NS)),
              'inline_images': len(new_root.findall('.//wp:inline', NS)),
              **{key: len(new_root.findall('.//w:' + key, NS))
                 for key in ['commentRangeStart', 'commentRangeEnd', 'commentReference']}}
    assert counts == {'tables': 6, 'equations': 6, 'inline_images': 7,
                      'commentRangeStart': 5, 'commentRangeEnd': 5, 'commentReference': 5}
    report = {'status': 'CANDIDATE_PENDING_RENDER_AND_REVIEW', 'revision': 12,
              'source_sha256': OLD_SHA, 'backup_sha256': digest(BACKUP), 'candidate_sha256': digest(CANDIDATE),
              'backup': str(BACKUP), 'candidate': str(CANDIDATE), 'changes': changes,
              'only_document_xml_changed': True, 'all_other_paragraphs_unchanged': True,
              'structural_counts': counts, 'evidence_sha256': digest(evidence_path),
              'scope': 'Clarify the inspected stem input-shape condition and the fixed 640 × 640 figure example; no experiment, metric, formula or figure changed.'}
    with REPORT.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'status': report['status'], 'candidate_sha256': report['candidate_sha256'],
                      'structural_counts': counts}, ensure_ascii=False))

if __name__ == '__main__':
    main()
