"""Record the completed page-by-page visual inspection and PDF checks."""
from hashlib import sha256
from pathlib import Path
from pypdf import PdfReader
import json

HERE=Path(__file__).resolve().parent
CANDIDATE=HERE/'MPSMA-DEIM_conference_EN_revised_第十九轮候选.docx'
PDF=HERE/'rendered_word/manuscript.pdf'
OUTPUT=HERE/'nineteenth_revision_visual_qa.json'

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

def main():
    assert not OUTPUT.exists()
    pending=json.loads((HERE/'nineteenth_revision_pending_qa.json').read_text(encoding='utf-8'))
    assert sha(CANDIDATE)==pending['candidate_sha256']
    reader=PdfReader(str(PDF))
    assert len(reader.pages)==10
    pages=[p.extract_text() or '' for p in reader.pages]
    assert '48 retained periodic' in pages[8] and 'checkpoint hash' in pages[8]
    assert '5. Conclusion' in pages[8] and 'References' in pages[8]
    assert 'Table 1.' in pages[5] and 'Table 6.' in pages[7]
    images=[HERE/'rendered_word'/f'page-{i:02}.png' for i in range(1,11)]
    assert all(p.is_file() for p in images)
    result={'status':'VISUAL_AND_STRUCTURAL_QA_PASSED','candidate_sha256':sha(CANDIDATE),'pdf_sha256':sha(PDF),
            'page_count':10,'inspected_pages':list(range(1,11)),
            'visual_observations':'All ten rasterized pages were individually viewed. The new provenance text is readable on page 9; conclusion and references flow correctly and figures, tables and equations remain legible.',
            'renderer':'Hidden Microsoft Word read-only PDF export followed by bundled Poppler pdftoppm',
            'fallback':'Packaged render_docx.py exited because soffice.exe is not installed.',
            'poppler_warnings':['No display font for Symbol','No display font for ArialUnicode'],
            'warning_review':'Formula pages and all other pages were visually inspected after rasterization.',
            'page_png_sha256':{p.name:sha(p) for p in images}}
    with OUTPUT.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps({'status':result['status'],'candidate_sha256':result['candidate_sha256'],'pages':10}))

if __name__=='__main__':
    main()
