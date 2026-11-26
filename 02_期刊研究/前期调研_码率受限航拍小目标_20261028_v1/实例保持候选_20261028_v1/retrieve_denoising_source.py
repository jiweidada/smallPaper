"""Retrieve one public original paper; no model or dataset assets."""
from pathlib import Path
import sys,json
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'深入核查_20261028_v1'))
from retrieve_static import retrieve
archive=root.parent.parent/'文献_相关原始论文_20261004'
jobs=[('R44_denoising_pdf','https://mdpi-res.com/d_attachment/computation/computation-13-00234/article_deploy/computation-13-00234.pdf',archive/'20261028_第二批失败_UAV_Denoising_Computation2025.pdf')]
rows=[retrieve(*j) for j in jobs]
(root/'denoising_retrieval_receipt.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
for r in rows:print(json.dumps({k:r.get(k) for k in ['key','status','bytes','is_pdf','error']},ensure_ascii=True))
