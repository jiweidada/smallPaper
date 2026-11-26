"""Fetch public papers/source text only. Never import or run downloaded code."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request, json, hashlib, re

ROOT=Path(__file__).resolve().parent
ARCHIVE=ROOT.parent.parent/'文献_相关原始论文_20261004'
HEADERS={'User-Agent':'Mozilla/5.0 (academic source reading)','Accept':'application/json,text/html,application/pdf,*/*'}

def retrieve(key,url,destination=None):
 r={'key':key,'url':url,'date':'2026-10-28','purpose':'static source/document reading; no experiments'}
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers=HEADERS),timeout=28) as response:
   data=response.read(40_000_001)
   if len(data)>40_000_000: raise ValueError('Source exceeds 40 MB reading limit')
   r.update(final_url=response.url,content_type=response.headers.get('Content-Type',''))
  ispdf=data.startswith(b'%PDF')
  if destination is None:
   if ispdf: path=ARCHIVE/('20261028_深入核查_'+key+'.pdf')
   else: path=ROOT/'sources'/(key+('.json' if 'json' in r['content_type'] else '.html'))
  else: path=Path(destination)
  path.parent.mkdir(parents=True,exist_ok=True)
  digest=hashlib.sha256(data).hexdigest()
  if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
   raise ValueError('Different existing file preserved; a new version name is required')
  if not path.exists(): path.write_bytes(data)
  r.update(status='downloaded',path=str(path),bytes=len(data),sha256=digest,is_pdf=ispdf)
 except Exception as exc: r.update(status='unavailable',error=str(exc))
 return r

def save_receipts(rows):
 (ROOT/'retrieval_receipts.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':
 rows=[]
 initial={
  'github_latest_release':'https://api.github.com/repos/ultralytics/ultralytics/releases/latest',
  'github_main_commit':'https://api.github.com/repos/ultralytics/ultralytics/commits/main',
  'PDAD_nature_page':'https://www.nature.com/articles/s41598-026-63984-0',
  'PDAD_springer_pdf':'https://link.springer.com/content/pdf/10.1038/s41598-026-63984-0.pdf',
  'FSIC_publisher_page':'https://www.sciencedirect.com/science/article/pii/S1051200424004470',
  'FSIC_publisher_pdf_route':'https://www.sciencedirect.com/science/article/pii/S1051200424004470/pdfft',
 }
 with ThreadPoolExecutor(max_workers=4) as pool:
  tasks={pool.submit(retrieve,k,u):k for k,u in initial.items()}
  for f in as_completed(tasks):
   r=f.result();rows.append(r);save_receipts(rows)
   print(json.dumps({k:r.get(k) for k in ['key','status','path','bytes','is_pdf','error']},ensure_ascii=True),flush=True)
 release=next((r for r in rows if r['key']=='github_latest_release' and r['status']=='downloaded'),None)
 release_json=json.loads(Path(release['path']).read_text(encoding='utf-8')) if release else {}
 # Fixed by the signed GitHub release page already read with the web tool.
 fixed={'v8.4.191':'2004cba021c68a46429be460e26954ab171b2862','v8.4.24':None}
 for version,expected in fixed.items():
  r=retrieve('github_commit_'+version,'https://api.github.com/repos/ultralytics/ultralytics/commits/'+version)
  rows.append(r);save_receipts(rows)
  if r['status']=='downloaded': commit=json.loads(Path(r['path']).read_text(encoding='utf-8'))['sha']
  elif expected: commit=expected
  else: print(json.dumps(r,ensure_ascii=True),flush=True);continue
  if not re.fullmatch('[0-9a-f]{40}',commit): raise ValueError('Invalid commit identifier')
  if expected and commit!=expected: raise ValueError('Tag differs from the already observed signed release commit')
  code_root=ROOT/'source_code'/(version+'_'+commit[:12])
  metadata={'tag':version,'commit':commit,'release_published_at':release_json.get('published_at') if release_json.get('tag_name')==version else None,'source':'https://github.com/ultralytics/ultralytics','execution':'source text only','pin_basis':'GitHub API and/or signed release page'}
  code_root.mkdir(parents=True,exist_ok=True)
  (code_root/'pin.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
  files=['ultralytics/utils/tal.py','ultralytics/utils/loss.py','ultralytics/nn/modules/head.py','ultralytics/cfg/models/26/yolo26.yaml','ultralytics/cfg/default.yaml','ultralytics/models/yolo/detect/predict.py','ultralytics/nn/tasks.py','ultralytics/__init__.py']
  if version=='v8.4.24': files=['ultralytics/utils/tal.py','ultralytics/utils/loss.py','ultralytics/nn/modules/head.py','ultralytics/cfg/default.yaml']
  with ThreadPoolExecutor(max_workers=4) as pool:
   tasks={pool.submit(retrieve,version+'_'+p.replace('/','_'),'https://raw.githubusercontent.com/ultralytics/ultralytics/'+commit+'/'+p,code_root/p):p for p in files}
   for f in as_completed(tasks):
    out=f.result();rows.append(out);save_receipts(rows)
    print(json.dumps({'key':out['key'],'status':out['status'],'bytes':out.get('bytes'),'error':out.get('error')},ensure_ascii=True),flush=True)
  print(json.dumps({'pin':metadata,'directory':str(code_root)},ensure_ascii=True),flush=True)
