"""Keep the input restriction while restoring the original paragraph's compact flow."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
from lxml import etree
import json

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE / 'MPSMA-DEIM_conference_EN_revised_第十二轮候选_v2.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第十二轮候选_v3.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第十一轮备份.docx'
REPORT = HERE / 'twelfth_revision_v3_pending_qa.json'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
NEW_METHOD = ('MPS replaces the standard stem with three paths (Fig. 2). In B0, the detail path uses two '
              'stride-one 3 × 3 convolution blocks with 8 channels, then 2 × 2 average pooling. '
              'The main path uses two 16-channel 3 × 3 blocks with strides two and one. '
              'The high-frequency path forms X − AvgPool3×3(X), using stride one and padding one, '
              'then applies a 1 × 1 projection and a depthwise 3 × 3 convolution with 4 channels. '
              'Bilinear interpolation with align_corners=False aligns this path to the main path. '
              'Odd input height or width prevents the branch outputs from being concatenated. '
              'Concatenating the three outputs produces a 28-channel feature Z at half the input resolution.')

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def main():
    prior = json.loads((HERE / 'twelfth_revision_v2_pending_qa.json').read_text(encoding='utf-8'))
    assert digest(PREVIOUS) == prior['candidate_sha256'] and digest(BACKUP) == prior['source_sha256']
    assert not CANDIDATE.exists() and not REPORT.exists()
    with ZipFile(PREVIOUS) as z:
        original = z.read('word/document.xml')
    root = etree.fromstring(original)
    paragraphs = root.findall('.//w:p', NS)
    index = prior['changed_paragraph_indices'][0]
    nodes = paragraphs[index].findall('.//w:t', NS)
    assert len(nodes) == 1 and nodes[0].text.startswith('MPS replaces the standard stem')
    before = nodes[0].text
    assert original.count(before.encode('utf-8')) == 1 and not any(c in NEW_METHOD for c in '<&')
    updated = original.replace(before.encode('utf-8'), NEW_METHOD.encode('utf-8'))
    with ZipFile(PREVIOUS) as source, ZipFile(CANDIDATE, 'x') as target:
        for item in source.infolist():
            target.writestr(item, updated if item.filename == 'word/document.xml' else source.read(item.filename))
    new_root = etree.fromstring(updated)
    with ZipFile(BACKUP) as old, ZipFile(CANDIDATE) as new:
        assert old.namelist() == new.namelist() and new.testzip() is None
        assert [name for name in old.namelist() if old.read(name) != new.read(name)] == ['word/document.xml']
        before_paras = etree.fromstring(old.read('word/document.xml')).findall('.//w:p', NS)
        after_paras = new_root.findall('.//w:p', NS)
        changed = [i for i, (a, b) in enumerate(zip(before_paras, after_paras))
                   if etree.tostring(a) != etree.tostring(b)]
        assert changed == prior['changed_paragraph_indices']
    counts = {'tables': len(new_root.findall('.//w:tbl', NS)), 'equations': len(new_root.findall('.//m:oMath', NS)),
              'inline_images': len(new_root.findall('.//wp:inline', NS)),
              **{key: len(new_root.findall('.//w:' + key, NS)) for key in
                 ['commentRangeStart', 'commentRangeEnd', 'commentReference']}}
    assert counts == prior['structural_counts']
    report = {'status': 'CANDIDATE_PENDING_RENDER_AND_REVIEW', 'revision': 12,
              'source_sha256': prior['source_sha256'], 'backup_sha256': digest(BACKUP),
              'second_candidate_sha256': digest(PREVIOUS), 'candidate_sha256': digest(CANDIDATE),
              'reason': 'Condensed existing stem prose while retaining the new odd-dimension restriction and figure scope.',
              'new_method': NEW_METHOD, 'caption': prior['caption'], 'changed_paragraph_indices': changed,
              'structural_counts': counts, 'evidence_sha256': prior['evidence_sha256'],
              'only_document_xml_changed': True}
    with REPORT.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'status': report['status'], 'candidate_sha256': report['candidate_sha256']}))

if __name__ == '__main__':
    main()
