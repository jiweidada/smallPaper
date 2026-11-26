"""Local PDF preflight and extraction; records source bytes, not a full-read claim."""
import hashlib
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, 'C:/Users/jiwei/.codex/skills/academic-research-suite/ars/scripts')
from pdf_read_preflight import run_preflight
from pypdf import PdfReader
ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / '文献_相关原始论文_20261004'
OUT = ROOT / 'data/00_实验计划与证据索引/分类监督责任解耦_20261006_v1'
receipts = []
for name, url in [
    ('分类责任复核_20261006_DCR_arXiv1803.06799.pdf', 'https://arxiv.org/pdf/1803.06799'),
    ('188_DetRefiner_CVPRF2026.pdf', 'https://openaccess.thecvf.com/content/CVPR2026F/papers/Okazaki_DetRefiner_Model-Agnostic_Detection_Refinement_with_Feature_Fusion_Transformer_CVPRF_2026_paper.pdf'),
]:
    p = LIB / name
    preflight = run_preflight(str(p))
    reader = PdfReader(p)
    pages = [page.extract_text() or '' for page in reader.pages]
    dest = LIB / '_text' / (p.stem + '_责任复核逐页.txt')
    with dest.open('x', encoding='utf-8') as f:
        f.write('\n\n'.join(f'===== PDF PAGE {i+1} =====\n{t}' for i,t in enumerate(pages)))
    receipts.append(dict(file=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                         extraction=str(dest), page_count=len(pages), preflight=preflight,
                         source_url=url, read_scope='Extraction only; actual read pages recorded separately.'))
with (OUT / '04_近邻PDF提取与预检.json').open('x', encoding='utf-8') as f:
    json.dump(receipts, f, ensure_ascii=False, indent=2, default=str)
print(json.dumps([{'file':x['file'], 'pages':x['page_count'], 'sha256':x['sha256'], 'preflight':x['preflight']} for x in receipts], ensure_ascii=False))
