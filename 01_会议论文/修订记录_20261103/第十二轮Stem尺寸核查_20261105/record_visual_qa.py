"""Record completed visual and structural review of the twelfth candidate."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
from zipfile import ZipFile
from lxml import etree
from pypdf import PdfReader
from PIL import Image
import json

HERE = Path(__file__).resolve().parent
REC = HERE.parent
PAPER = REC.parent
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第十一轮备份.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第十二轮候选_v3.docx'
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
OUTPUT = HERE / 'twelfth_revision_visual_qa.json'
QA = HERE / 'rendered_word_v3'
OLD_QA = REC / '第十一轮评价上限核查/rendered_word'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def main():
    assert not OUTPUT.exists()
    patch = json.loads((HERE / 'twelfth_revision_v3_pending_qa.json').read_text(encoding='utf-8'))
    assert digest(CURRENT) == digest(BACKUP) == patch['source_sha256']
    assert digest(CANDIDATE) == patch['candidate_sha256']
    assert digest(HERE / 'stem_shape_static_audit.json') == patch['evidence_sha256']
    assert digest(PAPER / '稿件/MPSMA-DEIM_conference_EN.docx') == 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'
    with ZipFile(BACKUP) as before, ZipFile(CANDIDATE) as after:
        assert before.namelist() == after.namelist() and after.testzip() is None
        assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ['word/document.xml']
        p0 = etree.fromstring(before.read('word/document.xml')).findall('.//w:p', NS)
        p1 = etree.fromstring(after.read('word/document.xml')).findall('.//w:p', NS)
        assert len(p0) == len(p1)
        changed = [i for i, (a, b) in enumerate(zip(p0, p1)) if etree.tostring(a) != etree.tostring(b)]
        assert changed == patch['changed_paragraph_indices']
    pdf = QA / 'manuscript.pdf'
    reader = PdfReader(pdf)
    assert len(reader.pages) == 10
    page2, page3 = reader.pages[1].extract_text(), reader.pages[2].extract_text()
    assert 'Odd input height or width' in page2
    assert 'weighted feature is' in page2 and '[1 + A(Z)]' in page2
    assert 'H = W = 640' in page3
    pngs = {}
    dimensions = {}
    unchanged = []
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
    assert unchanged == [1, 4, 5, 6, 7, 8, 9, 10]
    assert (QA / 'page-04-pdfium.png').is_file()
    qa = {'status': 'VISUAL_AND_STRUCTURAL_QA_PASSED', 'revision': 12,
          'checked_utc': datetime.now(timezone.utc).isoformat(),
          'candidate_sha256': digest(CANDIDATE), 'previous_docx_sha256': digest(BACKUP),
          'source_and_configuration_audit_sha256': patch['evidence_sha256'],
          'pages': 10, 'inspected_pages': list(range(1, 11)),
          'review_method': 'Opened all ten full-page PNGs. Checked the revised paragraph, complete expression and Fig. 2 caption on pages 2–3, plus figures, equations, tables and references on the remaining pages. The source restrictions were checked without model execution.',
          'page_png_sha256': pngs, 'page_png_dimensions': dimensions,
          'pages_pixel_identical_to_prior_revision': unchanged,
          'pdf_sha256': digest(pdf),
          'renderer': 'Hidden read-only Microsoft Word PDF export, then bundled Poppler pdftoppm; page 4 was also independently rasterized with PDFium.',
          'packaged_renderer_failure': 'LibreOffice soffice.exe is not available on PATH; the failure was logged before the successful Word fallback.',
          'poppler_warnings': 'Symbol and ArialUnicode display-font warnings; visual checks and an independent PDFium render of page 4 showed the inspected equation glyphs.',
          'changed_paragraph_indices': changed, 'structural_counts': patch['structural_counts'],
          'only_document_xml_changed': True,
          'scope': 'Clarifies the 640×640 figure and odd-dimension restriction. No metric, image, equation, table, citation, model run or evaluation changed.'}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(qa, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'status': qa['status'], 'pages': qa['pages'], 'candidate_sha256': qa['candidate_sha256']}))

if __name__ == '__main__':
    main()
