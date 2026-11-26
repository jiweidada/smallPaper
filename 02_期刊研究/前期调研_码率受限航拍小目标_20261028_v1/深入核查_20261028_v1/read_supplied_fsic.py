"""Archive and extract the user-supplied FSIC paper locally; no external upload."""
from pathlib import Path
import hashlib, json, os, subprocess, sys
from pypdf import PdfReader
from retrieve_static import ROOT, ARCHIVE

source=Path(r'D:\微信聊天记录\xwechat_files\wxid_mgvv5yspt9mf22_ab41\msg\file\2026-10\FSIC__Frequency_separated_image_compres_b4b413.pdf')
data=source.read_bytes(); digest=hashlib.sha256(data).hexdigest()
assert data.startswith(b'%PDF')
dest=ARCHIVE/'20261028_深入核查_FSIC_用户提供.pdf'
if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()!=digest:
 dest=ARCHIVE/('20261028_深入核查_FSIC_用户提供_'+digest[:12]+'.pdf')
if not dest.exists(): dest.write_bytes(data)
assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest
preflight=ROOT/'pdf_preflight'/(dest.stem+'.json');preflight.parent.mkdir(exist_ok=True)
if not preflight.exists():
 run=subprocess.run([sys.executable,r'C:\Users\jiwei\.codex\skills\academic-research-suite\ars\scripts\pdf_read_preflight.py',str(dest)],capture_output=True,env={**os.environ,'PYTHONIOENCODING':'utf-8'})
 result=json.loads(run.stdout.decode('utf-8'));preflight.write_bytes(run.stdout)
else: result=json.loads(preflight.read_text(encoding='utf-8'))
reader=PdfReader(dest)
pages=[p.extract_text() or '' for p in reader.pages]
text='\n\n'.join(f'=== PDF page {i+1} ===\n{t}' for i,t in enumerate(pages))
textpath=ARCHIVE/'_text'/(dest.stem+'_full.txt')
if not textpath.exists(): textpath.write_text(text,encoding='utf-8')
receipt={'source':'user supplied local PDF','source_path':str(source),'archive_path':str(dest),'bytes':len(data),'sha256':digest,'pdf_pages':len(pages),'text_path':str(textpath),'preflight_path':str(preflight),'structural_verdict':result.get('verdict'),'reading_scope':'extracted, actual reading not yet attested','external_upload':False,'model_experiments':0}
(ROOT/'FSIC_user_source_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=True))
print(pages[0][:3200].encode('ascii','backslashreplace').decode('ascii'))
for i,p in enumerate(pages): print(json.dumps({'pdf_page':i+1,'start':p[:135]},ensure_ascii=True))
