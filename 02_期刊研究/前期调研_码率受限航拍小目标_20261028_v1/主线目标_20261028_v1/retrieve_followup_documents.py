"""Retrieve primary papers for reading only; never imports model code."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent / '深入核查_20261028_v1'))
from retrieve_static import retrieve

archive = root.parent.parent / '文献_相关原始论文_20261004'
jobs = [
    ('R21_TTAC_CVF', 'https://openaccess.thecvf.com/content/ICCV2021W/MELEX/papers/Ehrlich_Analyzing_and_Mitigating_JPEG_Compression_Defects_in_Deep_Learning_ICCVW_2021_paper.pdf', archive / '20261028_主线补检_TTAC_CVF.pdf'),
    ('R21_TTAC_arxiv_v2', 'https://arxiv.org/pdf/2011.08932v2', archive / '20261028_主线补检_TTAC_arxiv_v2.pdf'),
    ('R22_QGAC_ECCV', 'https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530290.pdf', archive / '20261028_主线补检_QGAC_ECCV2020.pdf'),
    ('R25_DocQT_v2', 'https://arxiv.org/pdf/2605.19688v2', archive / '20261028_主线补检_DocQT_arxiv_v2.pdf'),
    ('R26_OADG_AAAI', 'https://ojs.aaai.org/index.php/AAAI/article/download/28076/28158', archive / '20261028_主线补检_OADG_AAAI2024.pdf'),
    ('R21_TTAC_HTML', 'https://arxiv.org/html/2011.08932v2', archive / '20261028_主线补检_TTAC_arxiv_v2.html'),
]
records = []
with ThreadPoolExecutor(max_workers=3) as pool:
    pending = [pool.submit(retrieve, key, url, dest) for key, url, dest in jobs]
    for future in as_completed(pending):
        row = future.result()
        records.append(row)
        (root / 'retrieval_receipts.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k:row.get(k) for k in ['key','status','bytes','is_pdf','error']},ensure_ascii=True),flush=True)
