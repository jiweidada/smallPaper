"""Describe the frozen comparison protocol in the working manuscript."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

from lxml import etree


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第二十七轮备份.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第三十轮候选.docx'
PREVIOUS_SHA = 'ee5aec68fc4a30cc2e814ab2bb9a08626196151e327570371bcd4deb92961588'
PROTOCOL_SHA = '8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'
PART = 'word/document.xml'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W, 'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}

PROTOCOL_TEXT = (
    'The ongoing n-scale comparison uses seed 0, FP32 and batch size 4 for 320 epochs, with 1,617 optimizer '
    'update attempts per epoch. Both arms use the same frozen training and validation sets and identical initialization for '
    'a predefined set of 554 shared tensors. The primary endpoint uses the EMA evaluated at actual epoch index '
    '319 before any late rollback, with AP and APs as the primary metrics. The highest-AP row is secondary, with '
    'ties resolved at the earliest epoch. BASE (the original n baseline) and FULL (the complete n model) resumed '
    'after 214 and 1 completed epochs, respectively. Each continuation reinitialized the main-process random states with seed 0 because '
    'the old checkpoints lacked those states. The FULL per-batch learning-rate log begins at round 2. The '
    'paired results are pending.'
)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def text(paragraph):
    return ''.join(node.text or '' for node in paragraph.findall('.//w:t', NS))


def edit(paragraphs, prefix, old, new):
    matches = [p for p in paragraphs if text(p).startswith(prefix)]
    assert len(matches) == 1
    paragraph = matches[0]
    matches = [node for node in paragraph.findall('.//w:t', NS) if old in (node.text or '')]
    assert len(matches) == 1 and text(paragraph).count(old) == 1
    node = matches[0]
    node.text = node.text.replace(old, new)
    return paragraph


def main():
    assert digest(CURRENT) == PREVIOUS_SHA
    remote = PAPER / '远程公平对照_20261103'
    protocol_path = remote / 'execution_protocol.json'
    assert digest(protocol_path) == PROTOCOL_SHA
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    budget = protocol['budget']
    assert (budget['seeds'], budget['epochs_per_group'], budget['batch'], budget['batches_per_epoch']) == ([0], 320, 4, 1617)
    assert budget['precision'] == 'FP32, AMP disabled' and budget['validation_images_per_epoch'] == 548
    assert protocol['evaluation']['primary_metrics'] == ['AP', 'APs']
    assert protocol['randomness']['common_initialization']['shared_tensors'] == 554
    resumed = remote / '断点续训_20261105'
    shared = resumed / 'resumed_pair_shared_conditions_20261106.json'
    base = resumed / 'backup_epoch0239/lineage.json'
    full = resumed / 'FULL_30848_20261106/full_lineage_through_round006.json'
    shared_record = json.loads(shared.read_text(encoding='utf-8'))
    base_record = json.loads(base.read_text(encoding='utf-8'))
    full_record = json.loads(full.read_text(encoding='utf-8'))
    assert shared_record['status'] == 'RESUMED_PAIR_SHARED_CONDITIONS_VERIFIED'
    assert shared_record['protocol_sha256'] == PROTOCOL_SHA
    assert base_record['original_prefix_epochs'] == 214 and full_record['original_prefix_epochs'] == 1
    assert full_record['historical_rng_state_available'] is False
    assert full_record['learning_rates']['historical_per_batch_log_available'] is False
    with ZipFile(CURRENT) as source:
        xml = source.read(PART)
    document = etree.fromstring(xml)
    paragraphs = document.findall('w:body/w:p', NS)
    original_paragraphs = [etree.tostring(p) for p in paragraphs]
    abstract = edit(
        paragraphs, 'Small objects in unmanned aerial vehicle',
        'These results describe individual configurations rather than isolated component effects: the historical '
        'baseline uses a different batch size, and some variants also change the aggregation structure. A matched '
        'baseline comparison is needed to determine whether this combination improves detection under the same '
        'training and evaluation conditions.',
        'The historical n baseline used a different batch size, and several variants also changed aggregation. '
        'A separate controlled comparison is in progress for the complete n design.',
    )
    limits = edit(
        paragraphs, 'The 48 retained periodic or latest evaluation arrays',
        'A matched baseline/full-model comparison is needed to assess the complete design; separate controls '
        'are needed to attribute effects to individual components.',
        'Attribution to individual components requires separate controls.',
    )
    conclusion = edit(
        paragraphs, 'MPSMA-DEIM combines a three-path shallow stem',
        'These observations are promising enough to motivate controlled evaluation, but the historical baseline '
        'differs in batch size and aggregation structure, preventing isolated component attribution. The archived '
        'timing results describe GPU network-plus-postprocessor execution and do not establish an onboard '
        'deployment guarantee. The next empirical requirement is a matched baseline/full-model comparison with '
        'fixed data, initialization policy, training budget and evaluation, followed by targeted component '
        'controls if individual contribution claims are retained.',
        'The historical differences in training and aggregation limit these results to descriptions of the '
        'evaluated configurations. The controlled comparison in Section 4.6 focuses on the complete n design '
        'under a common recipe. Separate experiments are needed to attribute effects to individual components.',
    )
    inserted = etree.Element(f'{{{W}}}p')
    properties = limits.find('w:pPr', NS)
    if properties is not None:
        inserted.append(deepcopy(properties))
    run = etree.SubElement(inserted, f'{{{W}}}r')
    run_properties = limits.find('w:r/w:rPr', NS)
    if run_properties is not None:
        run.append(deepcopy(run_properties))
    etree.SubElement(run, f'{{{W}}}t').text = PROTOCOL_TEXT
    limits.addnext(inserted)
    assert [p for i, p in enumerate(paragraphs) if etree.tostring(p) != original_paragraphs[i]] == [abstract, limits, conclusion]
    counts = {tag: len(document.findall('.//' + tag, NS)) for tag in
              ('w:tbl', 'm:oMath', 'wp:inline', 'w:commentRangeStart', 'w:commentRangeEnd', 'w:commentReference')}
    assert counts == {'w:tbl': 6, 'm:oMath': 6, 'wp:inline': 7,
                      'w:commentRangeStart': 5, 'w:commentRangeEnd': 5, 'w:commentReference': 5}
    revised = etree.tostring(document, encoding='UTF-8', xml_declaration=True, standalone=True)
    assert not BACKUP.exists() and not CANDIDATE.exists()
    shutil.copy2(CURRENT, BACKUP)
    assert digest(BACKUP) == PREVIOUS_SHA
    with ZipFile(CURRENT) as source, ZipFile(CANDIDATE, 'x') as target:
        for part in source.infolist():
            target.writestr(part, revised if part.filename == PART else source.read(part.filename))
    with ZipFile(BACKUP) as before, ZipFile(CANDIDATE) as after:
        assert after.testzip() is None and before.namelist() == after.namelist()
        assert [name for name in before.namelist() if before.read(name) != after.read(name)] == [PART]
    report = {'status': 'CANDIDATE_PENDING_VISUAL_QA', 'revision': 30, 'source_sha256': PREVIOUS_SHA,
              'candidate_sha256': digest(CANDIDATE), 'protocol_sha256': PROTOCOL_SHA, 'structure': counts,
              'evidence_sha256': {str(path.relative_to(PAPER)): digest(path) for path in (shared, base, full)},
              'submission_ready': False}
    with (HERE / 'controlled_comparison_candidate.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'sha256': report['candidate_sha256']}))


if __name__ == '__main__':
    main()
