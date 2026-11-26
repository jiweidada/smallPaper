from retrieve_static import retrieve, ROOT, ARCHIVE
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import urllib.request, http.cookiejar, hashlib, json

sha='2004cba021c68a46429be460e26954ab171b2862'
files=['ultralytics/utils/ops.py','ultralytics/engine/trainer.py','ultralytics/engine/predictor.py','ultralytics/engine/validator.py','ultralytics/nn/autobackend.py','ultralytics/data/augment.py','ultralytics/models/yolo/detect/val.py']
rows=[]
with ThreadPoolExecutor(max_workers=4) as pool:
 jobs=[pool.submit(retrieve,'extra_'+p.replace('/','_'),'https://raw.githubusercontent.com/ultralytics/ultralytics/'+sha+'/'+p,ROOT/'source_code'/('v8.4.191_'+sha[:12])/p) for p in files]
 for job in as_completed(jobs):
  r=job.result();rows.append(r)
  (ROOT/'extra_source_receipts.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
  print(json.dumps({'key':r['key'],'status':r['status'],'bytes':r.get('bytes'),'error':r.get('error')},ensure_ascii=True),flush=True)

# Ordinary anonymous publisher session only: no account, credentials or security override.
opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
headers={'User-Agent':'Mozilla/5.0','Accept':'text/html,application/pdf,*/*'}
for key,url in [('PDAD_anonymous_article','https://www.nature.com/articles/s41598-026-63984-0'),('PDAD_accepted_reference_retry','https://www.nature.com/articles/s41598-026-63984-0_reference.pdf')]:
 r={'key':key,'url':url,'date':'2026-10-28','route':'ordinary anonymous publisher session, no credentials'}
 try:
  with opener.open(urllib.request.Request(url,headers=headers),timeout=28) as response:
   data=response.read(40_000_001);r.update(final_url=response.url,content_type=response.headers.get('Content-Type',''))
  if len(data)>40_000_000: raise ValueError('Document exceeds 40 MB')
  pdf=data.startswith(b'%PDF')
  path=ARCHIVE/('20261028_深入核查_'+key+'.pdf') if pdf else ROOT/'sources'/(key+'.html')
  digest=hashlib.sha256(data).hexdigest()
  if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()!=digest: raise ValueError('Existing different file preserved')
  if not path.exists():path.write_bytes(data)
  r.update(status='downloaded',is_pdf=pdf,path=str(path),bytes=len(data),sha256=digest)
 except Exception as exc:r.update(status='unavailable',error=str(exc))
 rows.append(r)
 (ROOT/'extra_source_receipts.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(r,ensure_ascii=True),flush=True)
