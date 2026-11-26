"""Clarify the inspected solver's validation-dependent restoration, without claiming historical identity."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
from lxml import etree
import json
import shutil

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第十二轮备份.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第十三轮候选.docx'
REPORT = HERE / 'thirteenth_revision_pending_qa.json'
OLD = 'c49695664206778e6d0537b050194ddf1ab9a651021e8f64e8d4dacab7e22da9'
OLD_SENTENCE = ('The available solver can reset last_epoch when restoring a stage-one checkpoint, '
                'so this field alone cannot identify the evaluation that produced a historical value.')
NEW_SENTENCE = ('The inspected solver selects a stage-one checkpoint by validation AP and may restore it '
                'during later training, which can reset last_epoch; whether this solver was used historically '
                'is unverified. This field alone cannot identify the evaluation that produced a historical value.')
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

def main():
    assert sha(CURRENT) == OLD and not any(p.exists() for p in [BACKUP, CANDIDATE, REPORT])
    evidence_path = HERE / 'validation_restore_static_audit.json'
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    assert evidence['status'] == 'RESTORE_SEMANTICS_AND_COLLATE_PATH_STATICALLY_CHECKED'
    assert evidence['historical_solver_identity_certified'] is False
    for source in evidence['sources']:
        assert sha(Path(source['path'])) == source['sha256']
    with ZipFile(CURRENT) as z:
        original = z.read('word/document.xml')
    root = etree.fromstring(original)
    ps = root.findall('.//w:p', NS)
    texts = [''.join(p.xpath('.//w:t/text()', namespaces=NS)) for p in ps]
    index = next(i for i, t in enumerate(texts) if OLD_SENTENCE in t)
    before = texts[index]
    assert len(ps[index].findall('.//w:t', NS)) == 1
    assert before.count(OLD_SENTENCE) == 1 and original.count(before.encode('utf-8')) == 1
    after = before.replace(OLD_SENTENCE, NEW_SENTENCE)
    assert not any(c in after for c in '<&')
    updated = original.replace(before.encode('utf-8'), after.encode('utf-8'))
    new_root = etree.fromstring(updated)
    np = new_root.findall('.//w:p', NS)
    assert len(np) == len(ps)
    assert [i for i, (a, b) in enumerate(zip(ps, np)) if etree.tostring(a) != etree.tostring(b)] == [index]
    shutil.copy2(CURRENT, BACKUP)
    assert sha(BACKUP) == OLD
    with ZipFile(CURRENT) as source, ZipFile(CANDIDATE, 'x') as target:
        for item in source.infolist():
            target.writestr(item, updated if item.filename == 'word/document.xml' else source.read(item.filename))
    with ZipFile(CURRENT) as a, ZipFile(CANDIDATE) as b:
        assert a.namelist() == b.namelist() and b.testzip() is None
        assert [n for n in a.namelist() if a.read(n) != b.read(n)] == ['word/document.xml']
    counts = {'tables': len(new_root.findall('.//w:tbl', NS)), 'equations': len(new_root.findall('.//m:oMath', NS)),
              'inline_images': len(new_root.findall('.//wp:inline', NS)),
              **{key: len(new_root.findall('.//w:' + key, NS)) for key in
                 ['commentRangeStart', 'commentRangeEnd', 'commentReference']}}
    assert counts == {'tables': 6, 'equations': 6, 'inline_images': 7,
                      'commentRangeStart': 5, 'commentRangeEnd': 5, 'commentReference': 5}
    report = {'status': 'CANDIDATE_PENDING_RENDER_AND_REVIEW', 'revision': 13,
              'source_sha256': OLD, 'backup_sha256': sha(BACKUP), 'candidate_sha256': sha(CANDIDATE),
              'changed_paragraph_index': index, 'before': before, 'after': after,
              'only_document_xml_changed': True, 'structural_counts': counts,
              'static_evidence_sha256': sha(evidence_path),
              'scope': 'Describe validation-dependent restoration only for the inspected solver; historical solver identity remains unverified. No model, evaluation, training or numeric result changed.'}
    with REPORT.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'status': report['status'], 'candidate_sha256': report['candidate_sha256'],
                      'changed_paragraph_index': index}, ensure_ascii=False))

if __name__ == '__main__':
    main()
