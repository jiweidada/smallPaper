"""Disclose verified local duplicate-image annotation differences in one paragraph."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
from lxml import etree
import json
import shutil

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
CURRENT = PAPER/'稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE/'MPSMA-DEIM_conference_EN_revised_第十六轮备份.docx'
CANDIDATE = HERE/'MPSMA-DEIM_conference_EN_revised_第十七轮候选.docx'
RECEIPT = HERE/'seventeenth_revision_pending_qa.json'
OLD_HASH = '617ffd2024d42affb7f6b66f88cdca67204a0de3fef97946ffebf0c90f896984'
OLD = ('The local metrics are kept separate from challenge rankings, and confirming that these retained files '
       'are identical to those used in each historical run remains necessary.')
NEW = ('The retained current data also contain two byte-identical training-image pairs and one byte-identical '
       'validation-image pair with differing box annotations. Both validation entries were included in the CPU '
       'checks of Section 4.5; their impact on AP is unknown. Local metrics are kept separate from challenge '
       'rankings, and historical data identity remains unverified.')
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def main():
    assert digest(CURRENT) == OLD_HASH
    assert not any(path.exists() for path in [BACKUP, CANDIDATE, RECEIPT])
    audit_path = HERE/'train_val_byte_overlap_audit.json'
    labels_path = HERE/'duplicate_annotation_audit.json'
    verify_path = HERE/'duplicate_provenance_verification.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    labels = json.loads(labels_path.read_text(encoding='utf-8'))
    verify = json.loads(verify_path.read_text(encoding='utf-8'))
    assert audit['status'] == 'HASHED_CURRENT_FILES' and not audit['cross_split_identical_jpeg_byte_records']
    assert audit['train']['successfully_hashed'] == 6471 and audit['validation']['successfully_hashed'] == 548
    assert len(audit['train']['within_split_identical_byte_groups']) == 2
    assert len(audit['validation']['within_split_identical_byte_groups']) == 1
    assert len(labels['groups']) == 3 and all(not g['annotations_match_exactly_on_bbox_eval_fields'] for g in labels['groups'])
    assert verify['status'] == 'BYTE_IDENTITY_AND_CPU_INPUT_RECEIPTS_VERIFIED'
    assert verify['source_audit_sha256'] == digest(audit_path) and verify['pair_annotation_audit_sha256'] == digest(labels_path)
    assert len(verify['cpu_receipts']) == 3 and all(digest(Path(r['path'])) == r['sha256'] for r in verify['cpu_receipts'])
    with ZipFile(CURRENT) as source:
        original_xml = source.read('word/document.xml')
    root = etree.fromstring(original_xml)
    paragraphs = root.findall('.//w:p', NS)
    matching = [i for i, p in enumerate(paragraphs) if OLD in ''.join(p.xpath('.//w:t/text()', namespaces=NS))]
    assert len(matching) == 1
    index = matching[0]
    assert original_xml.count(OLD.encode('utf-8')) == 1 and not any(c in NEW for c in '<&')
    revised_xml = original_xml.replace(OLD.encode('utf-8'), NEW.encode('utf-8'))
    revised = etree.fromstring(revised_xml)
    new_paragraphs = revised.findall('.//w:p', NS)
    assert len(paragraphs) == len(new_paragraphs)
    assert [i for i, (a, b) in enumerate(zip(paragraphs, new_paragraphs)) if etree.tostring(a) != etree.tostring(b)] == [index]
    shutil.copy2(CURRENT, BACKUP)
    assert digest(BACKUP) == OLD_HASH
    with ZipFile(CURRENT) as source, ZipFile(CANDIDATE, 'x') as target:
        for entry in source.infolist():
            target.writestr(entry, revised_xml if entry.filename == 'word/document.xml' else source.read(entry.filename))
    with ZipFile(CURRENT) as a, ZipFile(CANDIDATE) as b:
        assert b.testzip() is None and a.namelist() == b.namelist()
        assert [n for n in a.namelist() if a.read(n) != b.read(n)] == ['word/document.xml']
    counts = {'tables': len(revised.findall('.//w:tbl', NS)), 'equations': len(revised.findall('.//m:oMath', NS)),
              'inline_images': len(revised.findall('.//wp:inline', NS)),
              **{name: len(revised.findall('.//w:' + name, NS)) for name in ['commentRangeStart', 'commentRangeEnd', 'commentReference']}}
    assert counts == {'tables': 6, 'equations': 6, 'inline_images': 7,
                      'commentRangeStart': 5, 'commentRangeEnd': 5, 'commentReference': 5}
    receipt = {'status': 'CANDIDATE_PENDING_RENDER_AND_REVIEW', 'revision': 17,
               'source_sha256': OLD_HASH, 'backup_sha256': digest(BACKUP),
               'candidate_sha256': digest(CANDIDATE), 'changed_paragraph_index': index,
               'old_text': OLD, 'new_text': NEW, 'only_document_xml_changed': True,
               'structural_counts': counts, 'data_audit_sha256': digest(audit_path),
               'duplicate_provenance_sha256': digest(verify_path),
               'scope': 'Current data caveat only; no AP, tables, figures, formulae or reference changes.'}
    with RECEIPT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'candidate_sha256': receipt['candidate_sha256'], 'changed_paragraph_index': index, 'structural_counts': counts}, ensure_ascii=False))

if __name__ == '__main__':
    main()
