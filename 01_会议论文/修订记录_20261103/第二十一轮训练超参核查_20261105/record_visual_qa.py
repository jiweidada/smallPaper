"""Record page-by-page inspection and structural checks for revision 21."""
from hashlib import sha256
from pathlib import Path
from pypdf import PdfReader
import json

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第二十一轮候选.docx'
PDF = HERE / 'rendered_word/manuscript.pdf'
OUTPUT = HERE / 'twentyfirst_revision_visual_qa.json'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUTPUT.exists()
    pending = json.loads((HERE / 'twentyfirst_revision_pending_qa.json').read_text(encoding='utf-8'))
    assert digest(CANDIDATE) == pending['candidate_sha256']
    pages = PdfReader(str(PDF)).pages
    assert len(pages) == 10
    extracted = [page.extract_text() or '' for page in pages]
    assert 'norm clipping to 0.1' in extracted[5] and 'LinearWarmup' in extracted[5]
    assert 'warmups=1,000' in extracted[5] and 'Historical source identity' in extracted[5]
    assert 'Table 1.' in extracted[6] and 'Table 6.' in extracted[8] and '5. Conclusion' in extracted[8]
    assert 'References' in extracted[8] and 'VisDrone object detection evaluation toolkit' in extracted[9]
    images = [HERE / 'rendered_word' / f'page-{i:02}.png' for i in range(1, 11)]
    assert all(path.is_file() for path in images)
    old_render = HERE.parent / '第二十轮检查点时序追溯_20261105/rendered_word'
    unchanged = [i for i, path in enumerate(images, start=1) if digest(path) == digest(old_render / path.name)]
    assert unchanged == list(range(1, 6))
    result = {'status': 'VISUAL_AND_STRUCTURAL_QA_PASSED', 'candidate_sha256': digest(CANDIDATE),
              'pdf_sha256': digest(PDF), 'page_count': 10, 'inspected_pages': list(range(1, 11)),
              'pages_pixel_identical_to_previous_render': unchanged,
              'visual_observations': 'All ten rasterized pages were viewed. The new settings are legible on page 6. The first table moves intact to page 7; later tables and references flow correctly. Figures and equations remain legible.',
              'renderer': 'Hidden Microsoft Word read-only PDF export followed by bundled Poppler pdftoppm',
              'fallback': 'Packaged render_docx.py exited because soffice.exe is not installed; output retained.',
              'poppler_warnings': ['No display font for Symbol', 'No display font for ArialUnicode'],
              'warning_review': 'Formula pages and all other pages were visually inspected after rasterization.',
              'page_png_sha256': {path.name: digest(path) for path in images}}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'candidate_sha256': result['candidate_sha256'], 'pages': 10}))


if __name__ == '__main__':
    main()
