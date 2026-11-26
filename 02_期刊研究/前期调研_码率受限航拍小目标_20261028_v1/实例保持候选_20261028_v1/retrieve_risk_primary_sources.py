"""Retrieve public primary papers and HTML for reading; no model/data experiment."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, sys
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'深入核查_20261028_v1'))
from retrieve_static import retrieve
archive=root.parent.parent/'文献_相关原始论文_20261004'
jobs=[
 ('R35_TaskGuard_v1','https://arxiv.org/pdf/2609.08011v1','20261028_实例保持_TaskGuard_arxiv_v1.pdf'),
 ('R35_TaskGuard_html','https://arxiv.org/html/2609.08011v1','20261028_实例保持_TaskGuard_arxiv_v1.html'),
 ('R36_PCT','https://openaccess.thecvf.com/content/CVPR2021/papers/Yan_Positive-Congruent_Training_Towards_Regression-Free_Model_Updates_CVPR_2021_paper.pdf','20261028_实例保持_PCT_CVPR2021.pdf'),
 ('R37_LTT_v5','https://arxiv.org/pdf/2110.01052v5','20261028_实例保持_LTT_arxiv_v5.pdf'),
 ('R38_DetectionCRC','https://proceedings.mlr.press/v204/andeol23a/andeol23a.pdf','20261028_实例保持_DetectionCRC_PMLR2023.pdf'),
 ('R39_CFM_v1','https://arxiv.org/pdf/2604.18001v1','20261028_实例保持_CFM_arxiv_v1.pdf')]
records=[]
with ThreadPoolExecutor(max_workers=3) as pool:
    pending=[pool.submit(retrieve,k,u,archive/name) for k,u,name in jobs]
    for future in as_completed(pending):
        row=future.result();records.append(row)
        (root/'retrieval_receipts.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:row.get(k) for k in ['key','status','bytes','is_pdf','error']},ensure_ascii=True),flush=True)
