"""Record the completed page-by-page inspection of the revision-20 PDF."""
from hashlib import sha256
from pathlib import Path
from pypdf import PdfReader
import json

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第二十轮候选.docx'
PDF = HERE / 'rendered_word/manuscript.pdf'
OUTPUT = HERE / 'twentieth_revision_visual_qa.json'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUTPUT.exists()
    pending = json.loads((HERE / 'twentieth_revision_pending_qa.json').read_text(encoding='utf-8'))
    assert digest(CANDIDATE) == pending['candidate_sha256']
    pages = PdfReader(str(PDF)).pages
    assert len(pages) == 10
    extracted = [page.extract_text() or '' for page in pages]
    assert 'stage boundary' in extracted[8] and 'EMA updates' in extracted[8]
    assert '5. Conclusion' in extracted[8] and 'References' in extracted[8]
    assert 'Table 1.' in extracted[5] and 'Table 6.' in extracted[7]
    images = [HERE / 'rendered_word' / f'page-{i:02}.png' for i in range(1, 11)]
    assert all(path.is_file() for path in images)
    result = {
        'status': 'VISUAL_AND_STRUCTURAL_QA_PASSED',
        'candidate_sha256': digest(CANDIDATE),
        'pdf_sha256': digest(PDF),
        'page_count': 10,
        'inspected_pages': list(range(1, 11)),
        'visual_observations': 'All ten rasterized pages were viewed. The changed conditional provenance paragraph is readable on page 9; conclusion and references flow correctly, and figures, tables and equations remain legible.',
        'renderer': 'Hidden Microsoft Word read-only PDF export followed by bundled Poppler pdftoppm',
        'fallback': 'Packaged render_docx.py exited because soffice.exe is not installed.',
        'poppler_warnings': ['No display font for Symbol', 'No display font for ArialUnicode'],
        'warning_review': 'Formula pages and all other pages were visually inspected after rasterization.',
        'page_png_sha256': {path.name: digest(path) for path in images},
    }
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'candidate_sha256': result['candidate_sha256'], 'pages': 10}))


if __name__ == '__main__':
    main()
