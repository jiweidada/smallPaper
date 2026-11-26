"""Download selected original-method papers for reading; no code/weights execution."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import sys,json
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'深入核查_20261028_v1'))
from retrieve_static import retrieve
archive=root.parent.parent/'文献_相关原始论文_20261004'
jobs=[
 ('AdaptiveISP','https://papers.neurips.cc/paper_files/paper/2024/file/cc596a803bedc7a03a87e98c77a22efe-Paper-Conference.pdf','20261028_区域组合_AdaptiveISP_NeurIPS2024.pdf'),
 ('PixelRL','https://ojs.aaai.org/index.php/AAAI/article/download/4240/4118','20261028_区域组合_PixelRL_AAAI2019.pdf'),
 ('FBCNN','https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_Towards_Flexible_Blind_JPEG_Artifacts_Removal_ICCV_2021_paper.pdf','20261028_区域组合_FBCNN_ICCV2021.pdf')]
rows=[]
with ThreadPoolExecutor(max_workers=3) as pool:
    work=[pool.submit(retrieve,k,u,archive/n) for k,u,n in jobs]
    for f in as_completed(work):
        row=f.result();rows.append(row)
        (root/'retrieval_receipts.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:row.get(k) for k in ['key','status','bytes','is_pdf','error']},ensure_ascii=True),flush=True)
