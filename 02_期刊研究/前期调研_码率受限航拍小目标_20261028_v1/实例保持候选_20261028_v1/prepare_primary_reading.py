"""Structural PDF preflight and text extraction; no task model is imported."""
from pathlib import Path
import json, os, subprocess, sys
from pypdf import PdfReader
root=Path(__file__).resolve().parent
archive=root.parent.parent/'文献_相关原始论文_20261004'
(root/'pdf_preflight').mkdir(exist_ok=True)
receipts=[]
rows=json.loads((root/'retrieval_receipts.json').read_text(encoding='utf-8'))
if (root/'additional_retrieval_receipts.json').exists():
    rows+=json.loads((root/'additional_retrieval_receipts.json').read_text(encoding='utf-8'))
if (root/'denoising_retrieval_receipt.json').exists():
    rows+=json.loads((root/'denoising_retrieval_receipt.json').read_text(encoding='utf-8'))
for row in rows:
    if row['status']!='downloaded' or not row.get('is_pdf'):continue
    path=Path(row['path']);side=root/'pdf_preflight'/(path.stem+'.json')
    if side.exists():preflight=json.loads(side.read_text(encoding='utf-8'))
    else:
        process=subprocess.run([sys.executable,r'C:\Users\jiwei\.codex\skills\academic-research-suite\ars\scripts\pdf_read_preflight.py',str(path)],capture_output=True,env={**os.environ,'PYTHONIOENCODING':'utf-8'})
        preflight=json.loads(process.stdout.decode('utf-8'));side.write_bytes(process.stdout)
    texts=[page.extract_text() or '' for page in PdfReader(path).pages]
    text_path=archive/'_text'/(path.stem+'_full.txt')
    value='\n\n'.join(f'=== PDF page {i+1} ===\n{text}' for i,text in enumerate(texts))
    surrogates=sum(0xD800<=ord(c)<=0xDFFF for c in value)
    # Preserve invalid extracted Unicode as visible literal escapes; never change the PDF.
    value=value.encode('utf-8',errors='backslashreplace').decode('utf-8')
    if text_path.exists() and text_path.stat().st_size==0:
        assert path.name=='20261028_第二批失败_DAME_arxiv_v1.pdf'
        text_path.write_text(value,encoding='utf-8')
    elif text_path.exists():assert text_path.read_text(encoding='utf-8')==value
    else:text_path.write_text(value,encoding='utf-8')
    receipts.append({**row,'pdf_pages':len(texts),'preflight_path':str(side),'structural_verdict':preflight.get('verdict'),'text_path':str(text_path),'extraction_surrogate_escapes':surrogates,'reading_scope':'text extracted only; actual reading is recorded separately; check formulas visually in original PDF'})
    print(json.dumps({'key':row['key'],'pages':len(texts),'structure':preflight.get('verdict'),'page_starts':[x[:65] for x in texts]},ensure_ascii=True),flush=True)
(root/'document_reading_receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf-8')
