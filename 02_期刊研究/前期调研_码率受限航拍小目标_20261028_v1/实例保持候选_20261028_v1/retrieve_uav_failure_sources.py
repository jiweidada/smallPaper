"""Retrieve two UAV failure-source papers; do not fetch model weights or datasets."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, sys
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'深入核查_20261028_v1'))
from retrieve_static import retrieve
archive=root.parent.parent/'文献_相关原始论文_20261004'
jobs=[
 ('R42_DAME_v1','https://arxiv.org/pdf/2604.09313v1','20261028_第二批失败_DAME_arxiv_v1.pdf'),
 ('R42_DAME_html','https://arxiv.org/html/2604.09313v1','20261028_第二批失败_DAME_arxiv_v1.html'),
 ('R43_LAF_v1','https://arxiv.org/pdf/2609.14560v1','20261028_第二批失败_LAF_arxiv_v1.pdf'),
 ('R43_LAF_html','https://arxiv.org/html/2609.14560v1','20261028_第二批失败_LAF_arxiv_v1.html')]
records=[]
with ThreadPoolExecutor(max_workers=3) as pool:
    pending=[pool.submit(retrieve,k,u,archive/name) for k,u,name in jobs]
    for future in as_completed(pending):
        row=future.result();records.append(row)
        (root/'additional_retrieval_receipts.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:row.get(k) for k in ['key','status','bytes','is_pdf','error']},ensure_ascii=True),flush=True)
