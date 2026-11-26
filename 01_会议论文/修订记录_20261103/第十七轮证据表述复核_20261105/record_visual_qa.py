"""Record the already-performed visual inspection and basic PDF checks."""
from hashlib import sha256
from pathlib import Path
from pypdf import PdfReader
import json

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE/'MPSMA-DEIM_conference_EN_revised_第十七轮候选.docx'
PDF = HERE/'rendered_word_elevated/manuscript.pdf'
REPORT = HERE/'seventeenth_revision_visual_qa.json'

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def main():
    assert not REPORT.exists()
    pending = json.loads((HERE/'seventeenth_revision_pending_qa.json').read_text(encoding='utf-8'))
    assert digest(CANDIDATE) == pending['candidate_sha256']
    reader = PdfReader(str(PDF))
    assert len(reader.pages) == 10
    pages = [page.extract_text() or '' for page in reader.pages]
    assert 'byte-identical' in pages[4] and 'impact on AP is unknown' in pages[4]
    assert 'Fig. 6.' in pages[5] and 'Table 1.' in pages[5]
    assert 'Table 6.' in pages[7] and 'References' in pages[8]
    images = [HERE/'rendered_word_elevated'/f'page-{i:02}.png' for i in range(1, 11)]
    assert all(x.is_file() for x in images)
    report = {'status': 'VISUAL_AND_STRUCTURAL_QA_PASSED', 'candidate_sha256': digest(CANDIDATE),
              'pdf_sha256': digest(PDF), 'page_count': 10,
              'inspected_pages': list(range(1, 11)),
              'visual_observations': 'All ten rasterized pages were individually viewed. The revised data disclosure fits on page 5; figure 6, Table 1 and its note stay on page 6; equations and references are readable.',
              'renderer': 'Hidden Microsoft Word read-only PDF export followed by bundled Poppler pdftoppm',
              'renderer_fallback': 'Packaged render_docx.py could not locate soffice.exe; the first sandboxed Word COM attempt was access-denied, and the read-only fallback was run with approved escalation.',
              'poppler_warnings': ['No display font for Symbol', 'No display font for ArialUnicode'],
              'warning_review': 'Formula pages and all other pages were visually inspected after rasterization.',
              'page_png_sha256': {p.name: digest(p) for p in images},
              'model_executions': 0, 'evaluation_executions': 0, 'optimizer_updates': 0}
    with REPORT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'candidate_sha256': report['candidate_sha256'], 'page_count': report['page_count']}))

if __name__ == '__main__':
    main()
