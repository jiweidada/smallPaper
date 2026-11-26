"""Add two source-checked related works and renumber later citations locally."""
from pathlib import Path
from zipfile import ZipFile
from hashlib import sha256
from copy import deepcopy
from lxml import etree
import json
import re
import shutil

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent
SOURCE = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第十四轮备份.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第十五轮候选.docx'
REPORT = HERE / 'fifteenth_revision_pending_qa.json'
EXPECTED = '15c62d7f363775645d73e29710366b32309f2b500ec6a08bd524be826b9eb3f8'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
ADDED = (' Recent UAV detectors explore related designs. DMF-Net combines multi-branch detail extraction '
         'with dual-attention fusion [9], while TF-DEIM-DFINE uses a gated Mamba backbone, '
         'half-channel attention and a tiered feature pyramid [10].')
NEW_REFS = [
    '[9] X. Liu, J. Liu, J. Tao, H. Hu, P. Li, and L. Chen, “A UAV small object detection method based on multi-branch detail enhancement and cross-scale attention fusion,” Symmetry, vol. 18, no. 5, Art. 763, 2026. doi: 10.3390/sym18050763.',
    '[10] H. Guo and S. Liu, “Enhancing UAV small target detection: A balanced accuracy-efficiency algorithm with tiered feature focus,” Image and Vision Computing, vol. 167, Art. 105897, 2026. doi: 10.1016/j.imavis.2026.105897.'
]

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def text(paragraph):
    return ''.join(paragraph.xpath('.//w:t/text()', namespaces=NS))

def counts(root):
    return {'tables': len(root.findall('.//w:tbl', NS)), 'equations': len(root.findall('.//m:oMath', NS)),
            'inline_images': len(root.findall('.//wp:inline', NS)),
            **{key: len(root.findall('.//w:' + key, NS)) for key in
               ['commentRangeStart', 'commentRangeEnd', 'commentReference']}}

def main():
    assert digest(SOURCE) == EXPECTED and not any(x.exists() for x in [BACKUP, CANDIDATE, REPORT])
    evidence_path = HERE / 'literature_evidence.json'
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    assert evidence['status'] == 'TWO_RELEVANT_PRIMARY_PUBLISHER_RECORDS_CHECKED'
    assert [x['new_reference'] for x in evidence['sources']] == [9, 10]
    assert not evidence['cross_paper_metric_comparisons'] and not evidence['new_experiments']
    with ZipFile(SOURCE) as z:
        raw = z.read('word/document.xml')
    root = etree.fromstring(raw)
    assert etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True) == raw
    original = deepcopy(root)
    paragraphs = root.findall('.//w:p', NS)
    assert 'Feature recalibration and multi-scale aggregation' in text(paragraphs[13])
    assert text(paragraphs[13]).endswith('their individual effects require controlled comparisons.')
    nodes = paragraphs[13].findall('.//w:t', NS)
    assert len(nodes) == 1
    nodes[0].text += ADDED
    for index, old, new in [(47, '[9]', '[11]'), (50, '[10]', '[12]'), (51, '[11]', '[13]')]:
        nodes = paragraphs[index].findall('.//w:t', NS)
        assert len(nodes) == 1 and (nodes[0].text or '').count(old) == 1
        nodes[0].text = nodes[0].text.replace(old, new)
    body = root.find('w:body', NS)
    refs = []
    for paragraph in body.findall('w:p', NS):
        match = re.match(r'^\[(\d+)\] ', text(paragraph))
        if match:
            refs.append((int(match.group(1)), paragraph))
    assert [i for i, _ in refs] == list(range(1, 12))
    old_refs = {i: text(p) for i, p in refs}
    for old in [9, 10, 11]:
        paragraph = refs[old - 1][1]
        nodes = paragraph.findall('.//w:t', NS)
        assert len(nodes) == 1 and nodes[0].text.startswith('[' + str(old) + '] ')
        nodes[0].text = nodes[0].text.replace('[' + str(old) + ']', '[' + str(old + 2) + ']', 1)
    ref_template = refs[8][1]
    assert not ref_template.findall('.//w:commentRangeStart', NS) and not ref_template.findall('.//w:bookmarkStart', NS)
    insertion = body.index(ref_template)
    for offset, reference in enumerate(NEW_REFS):
        paragraph = deepcopy(ref_template)
        nodes = paragraph.findall('.//w:t', NS)
        assert len(nodes) == 1
        nodes[0].text = reference
        body.insert(insertion + offset, paragraph)
    actual_refs = [text(p) for p in body.findall('w:p', NS) if re.match(r'^\[\d+\] ', text(p))]
    assert [int(re.match(r'^\[(\d+)\]', item).group(1)) for item in actual_refs] == list(range(1, 14))
    assert actual_refs[8:10] == NEW_REFS
    for old in range(1, 12):
        new = old if old < 9 else old + 2
        assert actual_refs[new - 1] == old_refs[old].replace('[' + str(old) + ']', '[' + str(new) + ']', 1)
    before_paras = original.findall('.//w:p', NS)
    after_paras = root.findall('.//w:p', NS)
    assert len(after_paras) == len(before_paras) + 2
    modified = [13, 47, 50, 51]
    for i in range(240):
        if i not in modified:
            assert etree.tostring(before_paras[i]) == etree.tostring(after_paras[i]), i
    assert counts(root) == counts(original) == {'tables': 6, 'equations': 6, 'inline_images': 7,
                                               'commentRangeStart': 5, 'commentRangeEnd': 5, 'commentReference': 5}
    assert [etree.tostring(x) for x in original.findall('.//m:oMath', NS)] == [etree.tostring(x) for x in root.findall('.//m:oMath', NS)]
    order = []
    for i in [8, 9, 12, 13, 16, 25, 47, 50, 51, 53]:
        for group in re.findall(r'\[(\d+(?:,\s*\d+)*)\]', text(after_paras[i])):
            numbers = [int(x) for x in re.findall(r'\d+', group)]
            if any(x > 13 for x in numbers):
                continue
            for number in numbers:
                if number not in order:
                    order.append(number)
    assert order == list(range(1, 14)), order
    shutil.copy2(SOURCE, BACKUP)
    assert digest(BACKUP) == EXPECTED
    updated = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
    with ZipFile(SOURCE) as source, ZipFile(CANDIDATE, 'x') as target:
        for item in source.infolist():
            target.writestr(item, updated if item.filename == 'word/document.xml' else source.read(item.filename))
    with ZipFile(SOURCE) as a, ZipFile(CANDIDATE) as b:
        assert a.namelist() == b.namelist() and b.testzip() is None
        assert [name for name in a.namelist() if a.read(name) != b.read(name)] == ['word/document.xml']
    report = {'status': 'CANDIDATE_PENDING_RENDER_AND_REVIEW', 'revision': 15,
              'source_sha256': EXPECTED, 'backup_sha256': digest(BACKUP), 'candidate_sha256': digest(CANDIDATE),
              'literature_evidence_sha256': digest(evidence_path), 'changed_original_paragraph_indices': modified,
              'citation_number_mapping_old_to_new': {str(i): i if i < 9 else i + 2 for i in range(1, 12)},
              'new_reference_numbers': [9, 10], 'first_appearance_order': order, 'references': 13,
              'only_document_xml_changed': True, 'structural_counts': counts(root),
              'noncitation_paragraphs_numerical_vectors_and_math_unchanged': True,
              'scope': 'Adds two related architectures using only publisher abstract/section excerpts; no cross-paper accuracy comparison, model, evaluation or training.'}
    with REPORT.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps({'status': report['status'], 'candidate_sha256': report['candidate_sha256'],
                      'first_appearance_order': order, 'references': report['references']}))

if __name__ == '__main__':
    main()
