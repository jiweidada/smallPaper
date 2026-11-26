"""Make a limited, evidence-bound edit to retained training settings."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
from lxml import etree
import json
import shutil

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第二十轮备份.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第二十一轮候选.docx'
PENDING = HERE / 'twentyfirst_revision_pending_qa.json'
OLD_SHA = 'eae4bbc357b9355003ed6809fbba3077bcd5148c764b1238ffca13f5a21f6409'
OLD1 = 'Most importantly, the historical n baseline uses'
NEW1 = ('The saved configurations disable AMP and set gradient-norm clipping to 0.1. '
        'They select flat-cosine scheduling with 2,000 warm-up iterations and lr_gamma=1.0; '
        'in the inspected implementation it has no post-warm-up decay, and a separate 500-step '
        'LinearWarmup is initialized. Historical source identity remains unverified. '
        'Most importantly, the historical n baseline uses')
OLD2 = 'Batch multi-scale resizing is disabled and exponential moving average (EMA) is enabled.'
NEW2 = ('Batch multi-scale resizing is disabled. The saved exponential moving average (EMA) '
        'configuration specifies decay=0.9999 and warmups=1,000; in the inspected code the latter '
        'is an exponential-ramp time constant.')
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    assert digest(CURRENT) == OLD_SHA and not any(p.exists() for p in [BACKUP, CANDIDATE, PENDING])
    audit_path = HERE / 'training_parameters_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    assert audit['status'] == 'SIX_CONFIGS_AND_INSPECTED_PARAMETER_RESOLUTION_CHECKED'
    assert len(audit['runs']) == 6 and all(audit['static_checks'].values())
    assert not audit['historical_source_identity_certified']
    assert all(audit[k] == 0 for k in ['model_instantiations', 'model_forwards', 'evaluator_executions', 'optimizer_updates'])
    for source in audit['sources'].values():
        assert digest(Path(source['path'])) == digest(Path(source['cpu_snapshot'])) == digest(Path(source['round_snapshot'])) == source['sha256']
    for run in audit['runs']:
        assert digest(Path(run['args_path'])) == run['args_sha256']
    with ZipFile(CURRENT) as archive:
        old_xml = archive.read('word/document.xml')
    root = etree.fromstring(old_xml)
    paragraphs = root.findall('.//w:p', NS)
    texts = [''.join(p.xpath('.//w:t/text()', namespaces=NS)) for p in paragraphs]
    positions = []
    new_xml = old_xml
    for old, new in [(OLD1, NEW1), (OLD2, NEW2)]:
        matches = [i for i, value in enumerate(texts) if old in value]
        assert len(matches) == 1 and new_xml.count(old.encode('utf-8')) == 1
        assert not any(character in new for character in '<&')
        positions.append(matches[0])
        new_xml = new_xml.replace(old.encode('utf-8'), new.encode('utf-8'))
    revised = etree.fromstring(new_xml)
    revised_paragraphs = revised.findall('.//w:p', NS)
    assert len(revised_paragraphs) == len(paragraphs)
    assert [i for i, (a, b) in enumerate(zip(paragraphs, revised_paragraphs)) if etree.tostring(a) != etree.tostring(b)] == positions
    shutil.copy2(CURRENT, BACKUP)
    assert digest(BACKUP) == OLD_SHA
    with ZipFile(CURRENT) as source, ZipFile(CANDIDATE, 'x') as target:
        for item in source.infolist():
            target.writestr(item, new_xml if item.filename == 'word/document.xml' else source.read(item.filename))
    with ZipFile(CURRENT) as before, ZipFile(CANDIDATE) as after:
        assert after.testzip() is None and before.namelist() == after.namelist()
        assert [name for name in before.namelist() if before.read(name) != after.read(name)] == ['word/document.xml']
    counts = {'tables': len(revised.findall('.//w:tbl', NS)), 'equations': len(revised.findall('.//m:oMath', NS)),
              'inline_images': len(revised.findall('.//wp:inline', NS)),
              **{key: len(revised.findall('.//w:' + key, NS)) for key in ['commentRangeStart', 'commentRangeEnd', 'commentReference']}}
    assert counts == {'tables': 6, 'equations': 6, 'inline_images': 7,
                      'commentRangeStart': 5, 'commentRangeEnd': 5, 'commentReference': 5}
    report = {'status': 'CANDIDATE_PENDING_RENDER_AND_REVIEW', 'revision': 21,
              'source_sha256': OLD_SHA, 'backup_sha256': digest(BACKUP), 'candidate_sha256': digest(CANDIDATE),
              'changed_paragraph_indices': positions, 'replacements': [{'before': OLD1, 'after': NEW1}, {'before': OLD2, 'after': NEW2}],
              'only_document_xml_changed': True, 'structural_counts': counts, 'training_audit_sha256': digest(audit_path),
              'scope': 'Reports saved AMP, clipping, EMA and scheduler settings and distinguishes inspected-code behavior from historical execution. No results, tables, figures or formulas changed.'}
    with PENDING.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'candidate_sha256': report['candidate_sha256'], 'changed_paragraph_indices': positions}))


if __name__ == '__main__':
    main()
