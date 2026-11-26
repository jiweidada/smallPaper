from pathlib import Path
import datetime,hashlib,json,subprocess,sys,urllib.request
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent;L=D.parent.parent/'文献_相关原始论文_20261004'
rows=[]
for kind,url in [('metadata','https://api.crossref.org/works/10.3390/electronics15071364'),('paper','https://mdpi-res.com/d_attachment/electronics/electronics-15-01364/article_deploy/electronics-15-01364.pdf')]:
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'UAV-primary-method-review'}),timeout=30) as f:data=f.read()
  if kind=='metadata':
   target=D/'Aero_Crossref_primary_metadata.json';target.write_bytes(data)
   m=json.loads(data)['message'];row=dict(kind=kind,url=url,file=str(target),title=m.get('title'),authors=m.get('author'),DOI=m.get('DOI'))
  else:
   assert data.startswith(b'%PDF-'),'Public endpoint did not returnPDF'
   stem='A33_AeroPinWorld_Electronics2026';target=L/(stem+'.pdf')
   with target.open('xb') as f:f.write(data)
   txt=L/'_text'/(stem+'.txt');subprocess.run(['pdftotext','-layout',str(target),str(txt)],check=True)
   row=dict(kind=kind,url=url,file=str(target),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),text_file=str(txt),text_sha256=hashlib.sha256(txt.read_bytes()).hexdigest(),version='PublisherpublicCDN;exactPDFhashbound',read_scope='Pending fullmethodinspection;no full-readclaim.',retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
   (L/(stem+'.source.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
 except Exception as e:row=dict(kind=kind,url=url,error=repr(e))
 rows.append(row);print(json.dumps(row,ensure_ascii=False)[:1500],flush=True)
(D/'Aero_access_receipt.json').write_text(json.dumps(dict(rows=rows),ensure_ascii=False,indent=2),encoding='utf-8')
