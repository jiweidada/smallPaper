"""Record visual, citation and structural checks for the related-work revision."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
from zipfile import ZipFile
from lxml import etree
from pypdf import PdfReader
from PIL import Image
import json
import re

HERE = Path(__file__).resolve().parent
REC, PAPER = HERE.parent, HERE.parent.parent
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第十四轮备份.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第十五轮候选_v2.docx'
QA = HERE / 'rendered_word_v2'
OLD_QA = REC / '第十四轮优化器分组核查_20261105/rendered_word_v2'
OUTPUT = HERE / 'fifteenth_revision_visual_qa.json'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def text(node):
    return ''.join(node.xpath('.//w:t/text()', namespaces=NS))

def main():
    assert not OUTPUT.exists()
    patch = json.loads((HERE / 'fifteenth_revision_v2_pending_qa.json').read_text(encoding='utf-8'))
    assert digest(CURRENT) == digest(BACKUP) == patch['source_sha256']
    assert digest(CANDIDATE) == patch['candidate_sha256']
    assert digest(HERE / 'literature_evidence.json') == patch['literature_evidence_sha256']
    original = PAPER / '稿件/MPSMA-DEIM_conference_EN.docx'
    original_backup = PAPER / '稿件/备份_20261103/MPSMA-DEIM_conference_EN_original.docx'
    assert digest(original) == digest(original_backup) == 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'
    with ZipFile(BACKUP) as before, ZipFile(CANDIDATE) as after:
        assert before.namelist() == after.namelist() and after.testzip() is None
        assert [name for name in before.namelist() if before.read(name) != after.read(name)] == ['word/document.xml']
        r0 = etree.fromstring(before.read('word/document.xml'))
        r1 = etree.fromstring(after.read('word/document.xml'))
        p0, p1 = r0.findall('.//w:p', NS), r1.findall('.//w:p', NS)
        assert len(p1) == len(p0) + 2
        assert [i for i in range(240) if etree.tostring(p0[i]) != etree.tostring(p1[i])] == [13, 47, 50, 51]
        counts = {'tables': len(r1.findall('.//w:tbl', NS)), 'equations': len(r1.findall('.//m:oMath', NS)),
                  'inline_images': len(r1.findall('.//wp:inline', NS)),
                  **{key: len(r1.findall('.//w:' + key, NS)) for key in
                     ['commentRangeStart', 'commentRangeEnd', 'commentReference']}}
        assert counts == patch['structural_counts']
        assert [etree.tostring(x) for x in r0.findall('.//m:oMath', NS)] == [etree.tostring(x) for x in r1.findall('.//m:oMath', NS)]
        refs = [text(p) for p in r1.find('w:body', NS).findall('w:p', NS) if re.match(r'^\[\d+\] ', text(p))]
        assert [int(re.match(r'^\[(\d+)\]', t).group(1)) for t in refs] == list(range(1, 14))
        assert '10.3390/sym18050763' in refs[8] and '10.1016/j.imavis.2026.105897' in refs[9]
        assert 'Elfwing' in refs[10] and 'Microsoft COCO' in refs[11] and 'toolkit' in refs[12]
        order = []
        for i in [8, 9, 12, 13, 16, 25, 47, 50, 51, 53]:
            for group in re.findall(r'\[(\d+(?:,\s*\d+)*)\]', text(p1[i])):
                nums = [int(x) for x in re.findall(r'\d+', group)]
                if any(x > 13 for x in nums):
                    continue
                for n in nums:
                    if n not in order:
                        order.append(n)
        assert order == list(range(1, 14))
    pdf = QA / 'manuscript.pdf'
    reader = PdfReader(pdf)
    assert len(reader.pages) == 10
    p2, p10 = (' '.join(reader.pages[i].extract_text().split()) for i in [1, 9])
    assert 'weighted feature is Z · [1 + A(Z)]' in p2
    assert '10.3390/sym18050763' in p10 and '10.1016/j.imavis.2026.105897' in p10
    pngs, dimensions, unchanged = {}, {}, []
    for page in range(1, 11):
        name = 'page-%02d.png' % page
        path = QA / name
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            dimensions[name] = list(image.size)
            assert image.width > 1000 and image.height == 1800
        pngs[name] = digest(path)
        if digest(OLD_QA / name) == pngs[name]:
            unchanged.append(page)
    assert unchanged == [1, 4]
    qa = {'status': 'VISUAL_CITATION_AND_STRUCTURAL_QA_PASSED', 'revision': 15,
          'checked_utc': datetime.now(timezone.utc).isoformat(),
          'candidate_sha256': digest(CANDIDATE), 'previous_docx_sha256': digest(BACKUP),
          'literature_evidence_sha256': patch['literature_evidence_sha256'],
          'pages': 10, 'inspected_pages': list(range(1, 11)), 'reference_count': len(refs),
          'citation_first_appearance_order': order,
          'review_method': 'Opened all ten full-page PNGs. Checked the related-work additions, the lead-in and weighted feature expression on page 2, the reflowed method figures, tables, and all reference entries on pages 9–10.',
          'page_png_sha256': pngs, 'page_png_dimensions': dimensions,
          'pages_pixel_identical_to_prior_revision': unchanged, 'pdf_sha256': digest(pdf),
          'renderer': 'Hidden read-only Microsoft Word PDF export, then bundled Poppler pdftoppm.',
          'packaged_renderer_failure': 'LibreOffice soffice.exe unavailable on PATH; logged before the Word fallback.',
          'poppler_warnings': 'Symbol and ArialUnicode display-font warnings; the unchanged equation page had an independent PDFium check in revision 12.',
          'changed_original_body_paragraph_indices': [13, 47, 50, 51], 'added_reference_paragraphs': 2,
          'structural_counts': counts, 'only_document_xml_changed': True,
          'scope': 'Adds two publisher-sourced architectural comparisons and renumbers three later citations; no figures, equations, table values, cross-paper metric comparisons, model run, evaluation or training changed.'}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(qa, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps({'status': qa['status'], 'pages': qa['pages'], 'references': len(refs), 'candidate_sha256': qa['candidate_sha256']}))

if __name__ == '__main__':
    main()
