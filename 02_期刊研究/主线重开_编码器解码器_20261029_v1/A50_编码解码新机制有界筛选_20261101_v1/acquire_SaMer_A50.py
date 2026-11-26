from pathlib import Path
import hashlib,json,subprocess,requests
ROOT=Path(__file__).resolve().parent
LIB=ROOT.parent.parent/'文献_相关原始论文_20261004'
url='https://arxiv.org/pdf/2607.04605v3'
p=LIB/'A50_SaMer_arXiv2607.04605v3.pdf'
assert not p.exists()
r=requests.get(url,timeout=(15,55))
r.raise_for_status()
assert r.content.startswith(b'%PDF')
p.write_bytes(r.content)
t=ROOT/'selected_text'/(p.stem+'.txt')
subprocess.run([r'C:\texlive\2025\bin\windows\pdftotext.exe','-layout',str(p),str(t)],check=True)
record={'title':'Do All Visual Tokens Matter Equally? Object-Evidence Preserving Token Merging for Vision-Language Retrieval',
 'source_url':url,'version':'arXiv2607.04605v3','http_status':r.status_code,
 'content_type':r.headers.get('Content-Type'),'pdf':str(p),'bytes':len(r.content),
 'sha256':hashlib.sha256(r.content).hexdigest(),'text':str(t),
 'reading_scope':'Downloaded and extracted only. Selected method reading recorded separately.',
 'purpose':'closest mechanism to generic train-time object-instance-preserving token merge prior; retrieval setting, not UAV detector evidence'}
p.with_suffix('.pdf.source.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'SaMer_acquisition_v1.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'downloaded':p.name,'bytes':len(r.content),'sha256':record['sha256']},ensure_ascii=False))
