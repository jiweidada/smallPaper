"""Record completed visual and structural review of the optimizer clarification."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
from zipfile import ZipFile
from lxml import etree
from pypdf import PdfReader
from PIL import Image
import json

HERE = Path(__file__).resolve().parent
REC, PAPER = HERE.parent, HERE.parent.parent
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第十三轮备份.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第十四轮候选_v2.docx'
QA = HERE / 'rendered_word_v2'
OLD_QA = REC / '第十三轮恢复语义核查_20261105/rendered_word'
OUTPUT = HERE / 'fourteenth_revision_visual_qa.json'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def main():
    assert not OUTPUT.exists()
    patch = json.loads((HERE / 'fourteenth_revision_v2_pending_qa.json').read_text(encoding='utf-8'))
    assert digest(CURRENT) == digest(BACKUP) == patch['source_sha256']
    assert digest(CANDIDATE) == patch['candidate_sha256']
    assert digest(HERE / 'optimizer_groups_audit.json') == patch['optimizer_audit_sha256']
    assert digest(HERE / 'optimizer_semantic_verification.json') == patch['semantic_check_sha256']
    original = PAPER / '稿件/MPSMA-DEIM_conference_EN.docx'
    original_backup = PAPER / '稿件/备份_20261103/MPSMA-DEIM_conference_EN_original.docx'
    assert digest(original) == digest(original_backup) == 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'
    with ZipFile(BACKUP) as before, ZipFile(CANDIDATE) as after:
        assert before.namelist() == after.namelist() and after.testzip() is None
        assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ['word/document.xml']
        old_root = etree.fromstring(before.read('word/document.xml'))
        root = etree.fromstring(after.read('word/document.xml'))
        p0, p1 = old_root.findall('.//w:p', NS), root.findall('.//w:p', NS)
        assert len(p0) == len(p1)
        changed = [i for i, (a, b) in enumerate(zip(p0, p1)) if etree.tostring(a) != etree.tostring(b)]
        assert changed == [patch['changed_paragraph_index']] == [54]
        counts = {'tables': len(root.findall('.//w:tbl', NS)), 'equations': len(root.findall('.//m:oMath', NS)),
                  'inline_images': len(root.findall('.//wp:inline', NS)),
                  **{key: len(root.findall('.//w:' + key, NS)) for key in
                     ['commentRangeStart', 'commentRangeEnd', 'commentReference']}}
        assert counts == patch['structural_counts']
    pdf = QA / 'manuscript.pdf'
    reader = PdfReader(pdf)
    assert len(reader.pages) == 10
    text = ' '.join(reader.pages[5].extract_text().split())
    assert 'some BatchNorm parameters consequently receive the lower rate and default decay' in text
    assert 'Table 1.' in text and '*SiLU applies to configurable backbone blocks' in text
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
    assert unchanged == [1, 2, 3, 4, 5, 10]
    qa = {'status': 'VISUAL_AND_STRUCTURAL_QA_PASSED', 'revision': 14,
          'checked_utc': datetime.now(timezone.utc).isoformat(),
          'candidate_sha256': digest(CANDIDATE), 'previous_docx_sha256': digest(BACKUP),
          'optimizer_audit_sha256': patch['optimizer_audit_sha256'], 'semantic_check_sha256': patch['semantic_check_sha256'],
          'pages': 10, 'inspected_pages': list(range(1, 11)),
          'review_method': 'Opened all ten full-page PNGs. Checked the revised optimizer paragraph and Table 1 note on page 6, and inspected the reflowed tables and references on pages 7–10. The first five pages and page 10 are pixel-identical to the previous reviewed revision.',
          'page_png_sha256': pngs, 'page_png_dimensions': dimensions,
          'pages_pixel_identical_to_prior_revision': unchanged, 'pdf_sha256': digest(pdf),
          'renderer': 'Hidden read-only Microsoft Word PDF export, then bundled Poppler pdftoppm.',
          'packaged_renderer_failure': 'LibreOffice soffice.exe unavailable on PATH; logged before the Word fallback.',
          'poppler_warnings': 'Symbol and ArialUnicode display-font warnings; the unchanged equation page had an independent PDFium check in revision 12.',
          'changed_paragraph_indices': changed, 'structural_counts': counts,
          'only_document_xml_changed': True,
          'scope': 'Corrects name-based optimizer grouping in one paragraph. No result, figure, equation, table, citation, model run, evaluation or training changed.'}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(qa, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps({'status': qa['status'], 'pages': qa['pages'], 'candidate_sha256': qa['candidate_sha256']}))

if __name__ == '__main__':
    main()
