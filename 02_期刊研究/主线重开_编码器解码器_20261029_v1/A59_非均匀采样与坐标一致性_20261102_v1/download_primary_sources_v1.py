"""Archive selected primary PDFs without overwriting originals."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
ARCHIVE=ROOT/'文献_相关原始论文_20261004'
P=Path(__file__).resolve().parent
ITEMS=[
 ('FOVEA_ICCV2021', ['https://arxiv.org/pdf/2108.12102','https://publications.ri.cmu.edu/storage/publications/2022/05/FOVEA.pdf']),
 ('SALISA_ECCV2022', ['https://arxiv.org/pdf/2204.02397']),
 ('TwoPlane_CVPR2023', ['https://openaccess.thecvf.com/content/CVPR2023/papers/Ghosh_Learned_Two-Plane_Perspective_Prior_Based_Image_Resampling_for_Efficient_Object_CVPR_2023_paper.pdf']),
]

def get(item):
 name,urls=item
 out=ARCHIVE/('A59_'+name+'.pdf')
 meta=ARCHIVE/('A59_'+name+'.source.json')
 assert not out.exists() and not meta.exists()
 attempts=[];success=None
 for url in urls:
  try:
   req=urllib.request.Request(url,headers={'User-Agent':'Academic-literature-source-audit/1.0'})
   with urllib.request.urlopen(req,timeout=25) as r:
    raw=r.read(40*1024*1024)
    actual=r.url
   assert raw.startswith(b'%PDF-') and len(raw)>10000
   with out.open('xb') as f:f.write(raw)
   success=dict(path=str(out),source_url=url,final_url=actual,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
   attempts.append(dict(url=url,status='DOWNLOADED'))
   break
  except Exception as e:
   attempts.append(dict(url=url,status='FAILED',error=type(e).__name__+': '+str(e)))
 result=dict(name=name,recorded_utc=datetime.now(timezone.utc).isoformat(),attempts=attempts,
             downloaded=success is not None,pdf=success,read_scope='Downloaded only; not yet read')
 with meta.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
 return result

if __name__=='__main__':
 ARCHIVE.mkdir(exist_ok=True)
 with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(get,ITEMS))
 with (P/'primary_download_receipt_v1.json').open('x',encoding='utf-8') as f:json.dump(results,f,ensure_ascii=False,indent=2)
 print(json.dumps(results,ensure_ascii=False))
