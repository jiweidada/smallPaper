"""Preflight downloaded originals and extract paper text; no model/data experiments."""
from pathlib import Path
import os,sys,json,subprocess
from pypdf import PdfReader
root=Path(__file__).resolve().parent
archive=root.parent.parent/'文献_相关原始论文_20261004'
(root/'pdf_preflight').mkdir(exist_ok=True)
rows=[]
for name in ['retrieval_receipts.json','alternate_retrieval_receipts.json','curl_retrieval_receipts.json']:
    p=root/name
    if p.exists():rows+=json.loads(p.read_text(encoding='utf-8'))
out=[];seen=set()
for r in rows:
    if r.get('status')!='downloaded' or r['path'] in seen:continue
    seen.add(r['path']);path=Path(r['path'])
    side=root/'pdf_preflight'/(path.stem+'.json')
    if not side.exists():
        p=subprocess.run([sys.executable,r'C:\Users\jiwei\.codex\skills\academic-research-suite\ars\scripts\pdf_read_preflight.py',str(path)],capture_output=True,env={**os.environ,'PYTHONIOENCODING':'utf-8'})
        json.loads(p.stdout.decode('utf-8'));side.write_bytes(p.stdout)
    pre=json.loads(side.read_text(encoding='utf-8'))
    pdf=PdfReader(path)
    texts=[p.extract_text() or '' for p in pdf.pages]
    value='\n\n'.join(f'=== PDF page {i+1} ===\n{t}' for i,t in enumerate(texts))
    value=value.encode('utf-8',errors='backslashreplace').decode('utf-8')
    target=archive/'_text'/(path.stem+'_full.txt')
    if target.exists():assert target.read_text(encoding='utf-8')==value
    else:target.write_text(value,encoding='utf-8')
    out.append({**r,'pages':len(texts),'structural_verdict':pre.get('verdict'),'text_path':str(target),'preflight_path':str(side),'reading_scope':'extracted only; actual selected reading recorded separately'})
    print(json.dumps({'key':r['key'],'pages':len(texts),'structure':pre.get('verdict'),'starts':[t[:65] for t in texts]},ensure_ascii=True))
(root/'document_receipts.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
