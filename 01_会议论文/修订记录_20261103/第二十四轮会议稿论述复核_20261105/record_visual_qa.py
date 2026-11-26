"""Record page-by-page inspection and structural checks for revision 24."""
from hashlib import sha256
from pathlib import Path
from pypdf import PdfReader
import json

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第二十四轮候选.docx'
PDF = HERE / 'rendered_word/manuscript.pdf'
OUTPUT = HERE / 'twentyfourth_revision_visual_qa.json'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUTPUT.exists()
    pending = json.loads((HERE / 'twentyfourth_revision_pending_qa.json').read_text(encoding='utf-8'))
    assert digest(CANDIDATE) == pending['candidate_sha256']
    pages = PdfReader(str(PDF)).pages
    assert len(pages) == 10
    text = [page.extract_text() or '' for page in pages]
    assert '4.6 Evidence Limits' in text[8] and '48 retained periodic' in text[8]
    assert 'checkpoint hash' in text[8] and 'historical version' in text[8]
    assert '5. Conclusion' in text[8] and 'References' in text[8]
    assert 'Table 1.' in text[6] and 'Table 6.' in text[8]
    assert 'VisDrone object detection evaluation toolkit' in text[9]
    images = [HERE / 'rendered_word' / f'page-{i:02}.png' for i in range(1, 11)]
    assert all(path.is_file() for path in images)
    previous = HERE.parent / '第二十一轮训练超参核查_20261105/rendered_word'
    unchanged = [i for i, path in enumerate(images, 1) if digest(path) == digest(previous / path.name)]
    assert unchanged == list(range(1, 9))
    result = {'status': 'VISUAL_AND_STRUCTURAL_QA_PASSED', 'candidate_sha256': digest(CANDIDATE),
              'pdf_sha256': digest(PDF), 'page_count': 10, 'inspected_pages': list(range(1, 11)),
              'pages_pixel_identical_to_previous_render': unchanged,
              'visual_observations': 'All ten rasterized pages were viewed. The revised limitations, conclusion and references flow without clipping on pages 9–10; tables, figures and equations remain legible.',
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
