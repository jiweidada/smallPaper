"""Local archival and document extraction only; no external upload/model execution."""
from pathlib import Path
import hashlib, json, os, subprocess, sys
from pypdf import PdfReader
from retrieve_static import ROOT, ARCHIVE
source=Path(r'D:\微信聊天记录\xwechat_files\wxid_mgvv5yspt9mf22_ab41\msg\file\2026-10\Progressive_Degradation_Aware_Distillat_2d11d9.pdf')
data=source.read_bytes();digest=hashlib.sha256(data).hexdigest();assert data.startswith(b'%PDF')
dest=ARCHIVE/'20261028_深入核查_PDAD_用户提供.pdf'
if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()!=digest: dest=ARCHIVE/('20261028_深入核查_PDAD_用户提供_'+digest[:12]+'.pdf')
if not dest.exists():dest.write_bytes(data)
assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest
sidecar=ROOT/'pdf_preflight'/(dest.stem+'.json');sidecar.parent.mkdir(exist_ok=True)
if not sidecar.exists():
 p=subprocess.run([sys.executable,r'C:\Users\jiwei\.codex\skills\academic-research-suite\ars\scripts\pdf_read_preflight.py',str(dest)],capture_output=True,env={**os.environ,'PYTHONIOENCODING':'utf-8'})
 result=json.loads(p.stdout.decode('utf-8'));sidecar.write_bytes(p.stdout)
else:result=json.loads(sidecar.read_text(encoding='utf-8'))
reader=PdfReader(dest);pages=[p.extract_text() or '' for p in reader.pages]
text='\n\n'.join(f'=== PDF page {i+1} ===\n{p}' for i,p in enumerate(pages))
target=ARCHIVE/'_text'/(dest.stem+'_full.txt')
if not target.exists():target.write_text(text,encoding='utf-8')
receipt={'source':'user supplied local PDF','source_path':str(source),'archive_path':str(dest),'bytes':len(data),'sha256':digest,'pdf_pages':len(pages),'text_path':str(target),'preflight_path':str(sidecar),'structural_verdict':result.get('verdict'),'reading_scope':'extracted; actual reading to be updated after reading','external_upload':False,'model_experiments':0}
(ROOT/'PDAD_user_source_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=True));print(pages[0][:2700].encode('ascii','backslashreplace').decode('ascii'))
for i,p in enumerate(pages):print(json.dumps({'page':i+1,'chars':len(p),'start':p[:100]},ensure_ascii=True))
