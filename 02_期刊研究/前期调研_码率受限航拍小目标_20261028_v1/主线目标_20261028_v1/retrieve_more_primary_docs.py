"""Bounded public-paper retrieval for subsequent reading only."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys, json
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'深入核查_20261028_v1'))
from retrieve_static import retrieve
archive=root.parent.parent/'文献_相关原始论文_20261004'
jobs=[
 ('R24_JPEG_posterior','https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Man_High-Perceptual_Quality_JPEG_Decoding_via_Posterior_Sampling_CVPRW_2023_paper.pdf','20261028_主线补检_JPEG_posterior_CVPRW2023.pdf'),
 ('R26_OADG_arxiv','https://arxiv.org/pdf/2312.12133','20261028_主线补检_OADG_arxiv.pdf'),
 ('R27_Crop_SAR','https://mdpi-res.com/d_attachment/remotesensing/remotesensing-13-00034/article_deploy/remotesensing-13-00034-v2.pdf?version=1609117717','20261028_主线补检_Crop_SAR_RS2021.pdf'),
 ('R28_DNTOD_arxiv','https://arxiv.org/pdf/2401.08056','20261028_主线补检_DNTOD_arxiv.pdf'),
 ('R29_SoftSampling_v2','https://arxiv.org/pdf/1806.06986v2','20261028_主线补检_SoftSampling_arxiv_v2.pdf')]
records=[]
with ThreadPoolExecutor(max_workers=3) as pool:
    pending=[pool.submit(retrieve,k,u,archive/name) for k,u,name in jobs]
    for future in as_completed(pending):
        row=future.result();records.append(row)
        (root/'additional_retrieval_receipts.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:row.get(k) for k in ['key','status','bytes','is_pdf','error']},ensure_ascii=True),flush=True)
