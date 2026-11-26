"""Public paper downloads for static method reading only."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys, json
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'深入核查_20261028_v1'))
from retrieve_static import retrieve
archive=root.parent.parent/'文献_相关原始论文_20261004'
jobs=[
 ('R31_CrossKD','https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.pdf','20261028_主线补检_CrossKD_CVPR2024.pdf'),
 ('R32_ConsistentTeacher','https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Consistent-Teacher_Towards_Reducing_Inconsistent_Pseudo-Targets_in_Semi-Supervised_Object_Detection_CVPR_2023_paper.pdf','20261028_主线补检_ConsistentTeacher_CVPR2023.pdf'),
 ('R33_SSRTDETR','https://mdpi-res.com/d_attachment/sensors/sensors-26-01539/article_deploy/sensors-26-01539.pdf','20261028_主线补检_SSRTDETR_Sensors2026.pdf')]
records=[]
with ThreadPoolExecutor(max_workers=3) as pool:
    pending=[pool.submit(retrieve,k,u,archive/name) for k,u,name in jobs]
    for future in as_completed(pending):
        row=future.result();records.append(row)
        (root/'assignment_retrieval_receipts.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:row.get(k) for k in ['key','status','bytes','is_pdf','error']},ensure_ascii=True),flush=True)
