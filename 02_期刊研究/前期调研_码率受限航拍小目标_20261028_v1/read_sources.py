from pathlib import Path
import sys, subprocess, json, hashlib, os
from pypdf import PdfReader
from source_fetch import ROOT, ARCHIVE

preflight=Path(r'C:\Users\jiwei\.codex\skills\academic-research-suite\ars\scripts\pdf_read_preflight.py')
files=list(ARCHIVE.glob('20261028_码率调研_*.pdf'))
files+=list(ARCHIVE.glob('20261027_条件新入口_Lost_in_Compression_CVPRW2022_Publisher.pdf'))
records=[]
for path in files:
 slug=path.stem
 sidecar=ROOT/'pdf_preflight'/(slug+'.json'); sidecar.parent.mkdir(exist_ok=True)
 if not sidecar.exists():
  p=subprocess.run([sys.executable,str(preflight),str(path)],capture_output=True,text=True,encoding='utf-8',errors='replace',env={**os.environ,'PYTHONIOENCODING':'utf-8'})
  try:
   json.loads(p.stdout)
   sidecar.write_text(p.stdout,encoding='utf-8')
  except (ValueError, OSError):
   sidecar.write_text(json.dumps({'verdict':'UNAVAILABLE','reason':p.stderr,'adapter_note':'CLI stdout mode; no successful structural result was obtained.'},ensure_ascii=False,indent=2),encoding='utf-8')
  print(json.dumps({'preflight':path.name,'exit':p.returncode,'stdout':p.stdout[-400:],'stderr':p.stderr[-400:]},ensure_ascii=True),flush=True)
 reader=PdfReader(path)
 textpath=ARCHIVE/'_text'/(slug+'_full.txt'); textpath.parent.mkdir(exist_ok=True)
 pages=[p.extract_text() or '' for p in reader.pages]
 text='\n\n'.join(f'=== PDF page {i+1} ===\n{t}' for i,t in enumerate(pages))
 if not textpath.exists(): textpath.write_text(text,encoding='utf-8')
 records.append({'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'pdf_pages':len(pages),'text_file':str(textpath),'preflight_sidecar':str(sidecar),'text_extraction_only':True,'read_scope':'To be updated after actual reading'})
 print(json.dumps({'file':path.name,'pdf_pages':len(pages),'first_page_start':pages[0][:900]},ensure_ascii=True),flush=True)
(ROOT/'local_pdf_inventory.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
