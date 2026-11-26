"""Record visual inspection of changed pages and pixel comparison to the prior rendering."""
from hashlib import sha256
from pathlib import Path
from pypdf import PdfReader
import json

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第二十七轮候选.docx'
PDF = HERE / 'rendered_word_approved/manuscript.pdf'
OUTPUT = HERE / 'complexity_disclosure_visual_qa.json'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUTPUT.exists()
    pending = json.loads((HERE / 'complexity_disclosure_pending_qa.json').read_text(encoding='utf-8'))
    assert digest(CANDIDATE) == pending['candidate_sha256']
    pages = PdfReader(str(PDF)).pages
    assert len(pages) == 10
    text = [page.extract_text() or '' for page in pages]
    assert 'profiler version' in text[7] and 'operator coverage' in text[7]
    assert 'Table 4.' in text[7] and 'Table 5.' in text[7]
    assert 'Table 6.' in text[8] and '4.6 Evidence Limits' in text[8] and '5. Conclusion' in text[8]
    assert 'VisDrone object detection evaluation toolkit' in text[9]
    images = [HERE / 'rendered_word_approved' / f'page-{i:02}.png' for i in range(1, 11)]
    assert all(path.is_file() for path in images)
    previous = HERE.parent / '第二十四轮会议稿论述复核_20261105/rendered_word'
    unchanged = [i for i, path in enumerate(images, 1) if digest(path) == digest(previous / path.name)]
    assert unchanged == list(range(1, 8))
    previous_qa = json.loads((previous.parent / 'twentyfourth_revision_visual_qa.json').read_text(encoding='utf-8'))
    assert previous_qa['status'] == 'VISUAL_AND_STRUCTURAL_QA_PASSED'
    assert previous_qa['inspected_pages'] == list(range(1, 11))
    for i in unchanged:
        assert digest(images[i-1]) == previous_qa['page_png_sha256'][images[i-1].name]
    result = {'status': 'VISUAL_AND_STRUCTURAL_QA_PASSED', 'candidate_sha256': digest(CANDIDATE),
              'pdf_sha256': digest(PDF), 'page_count': 10, 'inspected_new_pages': [8, 9, 10],
              'pages_pixel_identical_to_previously_inspected_render': unchanged,
              'visual_observations': 'Pages 8–10 were viewed at full resolution. The disclosure, tables, limitations, conclusion and references flow without clipping. Pages 1–7 are byte-identical to the previously inspected rendering.',
              'renderer': 'Hidden Microsoft Word read-only PDF export followed by bundled Poppler pdftoppm',
              'fallback': 'Packaged render_docx.py could not locate soffice.exe. The restricted Word COM attempt was denied; the approved read-only Word export succeeded.',
              'poppler_warnings': ['No display font for Symbol', 'No display font for ArialUnicode'],
              'warning_review': 'The seven unchanged pages including formulas match the previously visually inspected rendering.',
              'page_png_sha256': {path.name: digest(path) for path in images}}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'candidate_sha256': result['candidate_sha256'],
                      'pages': 10, 'newly_viewed_pages': [8, 9, 10]}))


if __name__ == '__main__':
    main()
