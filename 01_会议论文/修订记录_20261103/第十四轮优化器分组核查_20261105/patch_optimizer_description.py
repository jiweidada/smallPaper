"""Replace the overbroad optimizer description with the inspected name-based rules."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
from lxml import etree
import json
import shutil

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第十三轮备份.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第十四轮候选.docx'
REPORT = HERE / 'fourteenth_revision_pending_qa.json'
OLD_HASH = '3d6a72252567faebd7dd5bdb7d0f665d723f468edbe7e17f70caf08a76f9c1f9'
OLD = ('AdamW uses weight decay 10⁻⁴ and betas (0.9, 0.999). For the n runs, '
       'the configured backbone learning rate is 4 × 10⁻⁴ and the remaining rate is 8 × 10⁻⁴. '
       'The s configuration uses 2 × 10⁻⁴ for backbone parameters outside batch normalization '
       'and 4 × 10⁻⁴ otherwise; its parameter-group exclusions also differ.')
NEW = ('AdamW has betas (0.9, 0.999), default weight decay 10⁻⁴ and name-matched zero-decay groups. '
       'The n configurations use 4 × 10⁻⁴ for backbone names and 8 × 10⁻⁴ otherwise; '
       'in the inspected implementation, backbone names containing bn but not norm enter the default-decay group. '
       'The inspected s rules assign 2 × 10⁻⁴ to backbone names without bn and 4 × 10⁻⁴ to the others. '
       'Some BatchNorm parameters with sequential names therefore receive the lower rate and default decay. '
       'The saved flat-cosine settings specify 2,000 warm-up iterations and gamma = 1.0; '
       'the inspected scheduler has no post-warm-up decay.')
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

def main():
    assert sha(CURRENT) == OLD_HASH and not any(p.exists() for p in [BACKUP, CANDIDATE, REPORT])
    audit_path = HERE / 'optimizer_groups_audit.json'
    check_path = HERE / 'optimizer_semantic_verification.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    check = json.loads(check_path.read_text(encoding='utf-8'))
    assert audit['status'] == 'OPTIMIZER_NAME_PATTERNS_AND_CHECKPOINT_GROUPS_INSPECTED'
    assert check['status'] == 'BN_LAYER_INDICES_REGEX_AND_SAVED_GROUP_VALUES_VERIFIED'
    assert sha(audit_path) == check['audit_sha256']
    assert not audit['historical_source_identity_certified']
    assert all(audit[x] == 0 for x in ['model_instantiations', 'model_forwards', 'evaluator_executions', 'optimizer_updates'])
    for source in audit['sources']:
        assert sha(Path(source['path'])) == source['sha256']
    for run in audit['runs']:
        assert sha(Path(run['args']['path'])) == run['args']['sha256']
        assert sha(Path(run['checkpoint_metadata']['path'])) == run['checkpoint_metadata']['sha256']
    with ZipFile(CURRENT) as z:
        original = z.read('word/document.xml')
    root = etree.fromstring(original)
    paragraphs = root.findall('.//w:p', NS)
    texts = [''.join(p.xpath('.//w:t/text()', namespaces=NS)) for p in paragraphs]
    index = next(i for i, text in enumerate(texts) if OLD in text)
    assert index == 54 and len(paragraphs[index].findall('.//w:t', NS)) == 1
    before = texts[index]
    assert before.count(OLD) == 1 and original.count(before.encode('utf-8')) == 1
    after = before.replace(OLD, NEW)
    assert not any(c in after for c in '<&')
    updated = original.replace(before.encode('utf-8'), after.encode('utf-8'))
    new_root = etree.fromstring(updated)
    new_paragraphs = new_root.findall('.//w:p', NS)
    assert len(new_paragraphs) == len(paragraphs)
    assert [i for i, (a, b) in enumerate(zip(paragraphs, new_paragraphs)) if etree.tostring(a) != etree.tostring(b)] == [index]
    shutil.copy2(CURRENT, BACKUP)
    assert sha(BACKUP) == OLD_HASH
    with ZipFile(CURRENT) as source, ZipFile(CANDIDATE, 'x') as target:
        for item in source.infolist():
            target.writestr(item, updated if item.filename == 'word/document.xml' else source.read(item.filename))
    with ZipFile(CURRENT) as before_zip, ZipFile(CANDIDATE) as after_zip:
        assert before_zip.namelist() == after_zip.namelist() and after_zip.testzip() is None
        assert [n for n in before_zip.namelist() if before_zip.read(n) != after_zip.read(n)] == ['word/document.xml']
    counts = {'tables': len(new_root.findall('.//w:tbl', NS)), 'equations': len(new_root.findall('.//m:oMath', NS)),
              'inline_images': len(new_root.findall('.//wp:inline', NS)),
              **{key: len(new_root.findall('.//w:' + key, NS)) for key in
                 ['commentRangeStart', 'commentRangeEnd', 'commentReference']}}
    assert counts == {'tables': 6, 'equations': 6, 'inline_images': 7,
                      'commentRangeStart': 5, 'commentRangeEnd': 5, 'commentReference': 5}
    report = {'status': 'CANDIDATE_PENDING_RENDER_AND_REVIEW', 'revision': 14,
              'source_sha256': OLD_HASH, 'backup_sha256': sha(BACKUP), 'candidate_sha256': sha(CANDIDATE),
              'changed_paragraph_index': index, 'before': before, 'after': after,
              'only_document_xml_changed': True, 'structural_counts': counts,
              'optimizer_audit_sha256': sha(audit_path), 'semantic_check_sha256': sha(check_path),
              'scope': 'Corrects the n/s name-based optimizer groups and describes saved schedule settings conditionally on the inspected source. No model, evaluation, training or result changed.'}
    with REPORT.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps({'status': report['status'], 'candidate_sha256': report['candidate_sha256'],
                      'changed_paragraph_index': index}, ensure_ascii=False))

if __name__ == '__main__':
    main()
