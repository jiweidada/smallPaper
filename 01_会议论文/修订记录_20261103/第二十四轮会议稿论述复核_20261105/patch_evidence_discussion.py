"""Condense the conference-paper evidence discussion without changing results."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
from lxml import etree
import json
import re
import shutil

HERE = Path(__file__).resolve().parent
REC = HERE.parent
PAPER = REC.parent
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第二十三轮备份.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第二十四轮候选.docx'
PENDING = HERE / 'twentyfourth_revision_pending_qa.json'
OLD_SHA = '9060aee70bae6804fbf61a4adbc76ce34a2d4be7e44d17dfa2500877300adbab'
HEAD_OLD = '4.6 Late-epoch Variation and Evidence Limits'
HEAD_NEW = '4.6 Evidence Limits'
PARA1_START = 'For the s run, the last 20 logged validation AP values'
PARA1_NEW = ('The reported best-AP values are selected from single runs on the validation set; '
             'they do not provide an independent test estimate or measure reproducibility across seeds. '
             'Variation across epochs within one run is not a substitute for repeated training runs.')
PARA2_START = 'The available evidence supports a documented architectural design and single-run configuration results.'
PARA2_NEW = ('The 48 retained periodic or latest evaluation arrays match their corresponding 12-metric '
             'log rows, but only N2’s selected best row has such an array, and none contains an explicit '
             'checkpoint hash. Some checkpoint epoch fields differ from the selected log indices. '
             'The inspected solver can restore earlier training state, but its historical version is '
             'unverified; epoch metadata alone cannot bind a checkpoint to a historical metric. '
             'The CPU checks in Section 4.5 bind current evaluations to N0, N4 and S1 weights; '
             'they neither establish the exact historical training state nor resolve the differing training '
             'conditions. A matched baseline/full-model comparison is needed to assess the complete design; '
             'separate controls are needed to attribute effects to individual components.')
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    assert digest(CURRENT) == OLD_SHA and not any(p.exists() for p in [BACKUP, CANDIDATE, PENDING])
    arrays_path = REC / '第十九轮历史评价追溯_20261105/independent_saved_array_verification.json'
    fields_path = REC / '第十九轮历史评价追溯_20261105/saved_evaluation_binding_fields.json'
    restore_path = REC / '第二十轮检查点时序追溯_20261105/n2_source_semantics_verification.json'
    cpu_path = PAPER / '本地CPU核验_20261103/comparison_report.json'
    arrays = json.loads(arrays_path.read_text(encoding='utf-8-sig'))
    fields = json.loads(fields_path.read_text(encoding='utf-8-sig'))
    restore = json.loads(restore_path.read_text(encoding='utf-8-sig'))
    assert arrays['status'] == 'ALL_48_CORRESPONDING_ROWS_VERIFIED' and arrays['arrays_checked'] == 48
    assert arrays['maximum_abs_difference'] < 1e-12
    assert [(item['run'], item['epoch']) for item in arrays['selected_best_log_arrays']] == [('N2', 300)]
    assert fields['status'] == 'SAVED_EVALUATION_EXPLICIT_BINDING_FIELDS_INSPECTED'
    assert not fields['files_with_checkpoint_or_hash_named_fields'] and fields['files'] == 49
    assert restore['status'] == 'INSPECTED_CODE_AND_N2_COUNTER_ARITHMETIC_CONSISTENT'
    assert all(restore['static_source_checks'].values()) and cpu_path.is_file()
    with ZipFile(CURRENT) as archive:
        old_xml = archive.read('word/document.xml')
    root = etree.fromstring(old_xml)
    paragraphs = root.findall('.//w:p', NS)
    replacements = []
    for prefix, replacement in [(HEAD_OLD, HEAD_NEW), (PARA1_START, PARA1_NEW), (PARA2_START, PARA2_NEW)]:
        matches = []
        for i, paragraph in enumerate(paragraphs):
            text_nodes = paragraph.findall('.//w:t', NS)
            content = ''.join(node.text or '' for node in text_nodes)
            if content.startswith(prefix):
                matches.append((i, text_nodes, content))
        assert len(matches) == 1
        index, nodes, before = matches[0]
        assert len(nodes) == 1 and old_xml.count(before.encode('utf-8')) == 1
        assert not any(c in replacement for c in '<&')
        replacements.append({'paragraph_index': index, 'before': before, 'after': replacement})
    new_xml = old_xml
    for item in replacements:
        new_xml = new_xml.replace(item['before'].encode('utf-8'), item['after'].encode('utf-8'))
    revised = etree.fromstring(new_xml)
    revised_paragraphs = revised.findall('.//w:p', NS)
    assert len(paragraphs) == len(revised_paragraphs)
    assert [i for i, (a, b) in enumerate(zip(paragraphs, revised_paragraphs)) if etree.tostring(a) != etree.tostring(b)] == [item['paragraph_index'] for item in replacements]
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
    before_words = len(re.findall(r'\b\w+(?:[’\-]\w+)*\b', ' '.join(item['before'] for item in replacements)))
    after_words = len(re.findall(r'\b\w+(?:[’\-]\w+)*\b', ' '.join(item['after'] for item in replacements)))
    report = {'status': 'CANDIDATE_PENDING_RENDER_AND_REVIEW', 'revision': 24,
              'source_sha256': OLD_SHA, 'backup_sha256': digest(BACKUP), 'candidate_sha256': digest(CANDIDATE),
              'replacements': replacements, 'only_document_xml_changed': True, 'structural_counts': counts,
              'approximate_words_before': before_words, 'approximate_words_after': after_words,
              'source_evidence': {str(path): digest(path) for path in [arrays_path, fields_path, restore_path, cpu_path]},
              'scope': 'Condenses evidence limitations; retains the N2-only best-array qualification, no-checkpoint-hash limit, CPU/historical distinction and need for matched controls. All prior forensic evidence stays in the revision records; no numerical results, tables, figures, equations or references changed.'}
    with PENDING.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'candidate_sha256': report['candidate_sha256'],
                      'paragraphs': [item['paragraph_index'] for item in replacements],
                      'approximate_words_before': before_words, 'approximate_words_after': after_words}))


if __name__ == '__main__':
    main()
