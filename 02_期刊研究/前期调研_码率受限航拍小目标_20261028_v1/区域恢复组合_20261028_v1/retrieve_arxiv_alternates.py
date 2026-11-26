"""Retrieve verified arXiv alternates and Ultra's original methods; documents only."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import sys,json
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'深入核查_20261028_v1'))
from retrieve_static import retrieve
archive=root.parent.parent/'文献_相关原始论文_20261004'
jobs=[
 ('AdaptiveISP_arxiv','https://arxiv.org/pdf/2410.22939','20261028_区域组合_AdaptiveISP_arxiv_latest.pdf'),
 ('PixelRL_arxiv','https://arxiv.org/pdf/1811.04323','20261028_区域组合_PixelRL_arxiv_latest.pdf'),
 ('FBCNN_arxiv','https://arxiv.org/pdf/2109.14573','20261028_区域组合_FBCNN_arxiv_latest.pdf'),
 ('Ultra_v1','https://arxiv.org/pdf/2608.16589v1','20261028_区域组合_Ultra_arxiv_v1.pdf')]
rows=[]
with ThreadPoolExecutor(max_workers=3) as pool:
    work=[pool.submit(retrieve,k,u,archive/n) for k,u,n in jobs]
    for f in as_completed(work):
        row=f.result();rows.append(row)
        (root/'alternate_retrieval_receipts.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:row.get(k) for k in ['key','status','bytes','is_pdf','error']},ensure_ascii=True),flush=True)
