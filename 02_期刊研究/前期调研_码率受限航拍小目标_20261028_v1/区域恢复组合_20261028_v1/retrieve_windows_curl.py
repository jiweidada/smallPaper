"""Public document retrieval via system TLS implementation; never executes remote code."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import subprocess,hashlib,json
root=Path(__file__).resolve().parent
archive=root.parent.parent/'文献_相关原始论文_20261004'
jobs=[
 ('AdaptiveISP','https://papers.neurips.cc/paper_files/paper/2024/file/cc596a803bedc7a03a87e98c77a22efe-Paper-Conference.pdf','20261028_区域组合_AdaptiveISP_NeurIPS2024.pdf'),
 ('PixelRL','https://ojs.aaai.org/index.php/AAAI/article/download/4240/4118','20261028_区域组合_PixelRL_AAAI2019.pdf'),
 ('FBCNN','https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_Towards_Flexible_Blind_JPEG_Artifacts_Removal_ICCV_2021_paper.pdf','20261028_区域组合_FBCNN_ICCV2021.pdf'),
 ('Ultra_v1','https://arxiv.org/pdf/2608.16589v1','20261028_区域组合_Ultra_arxiv_v1.pdf')]
def fetch(job):
    key,url,name=job
    row={'key':key,'url':url,'research_date':'2026-10-28','retriever':'Windows system curl; public document; no auth','path':str(archive/name)}
    try:
        r=subprocess.run([r'C:\Windows\System32\curl.exe','--location','--fail','--silent','--show-error','--http1.1','--max-time','40',url],capture_output=True,timeout=45)
        if r.returncode:raise RuntimeError(r.stderr.decode('utf-8',errors='replace')[-450:])
        data=r.stdout
        if not data.startswith(b'%PDF') or len(data)>40_000_000:raise ValueError('Not a PDF or too large')
        p=archive/name
        if p.exists():assert p.read_bytes()==data,'Different existing version preserved'
        else:p.write_bytes(data)
        row.update(status='downloaded',is_pdf=True,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    except Exception as exc:row.update(status='unavailable',error=str(exc))
    return row
rows=[]
with ThreadPoolExecutor(max_workers=3) as pool:
    for f in as_completed([pool.submit(fetch,j) for j in jobs]):
        row=f.result();rows.append(row)
        (root/'curl_retrieval_receipts.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:row.get(k) for k in ['key','status','bytes','error']},ensure_ascii=True),flush=True)
