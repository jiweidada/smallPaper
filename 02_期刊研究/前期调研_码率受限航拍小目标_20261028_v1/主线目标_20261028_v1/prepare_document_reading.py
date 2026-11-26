"""Structural PDF preflight and text extraction only, never model execution."""
from pathlib import Path
import hashlib, json, os, subprocess, sys
from pypdf import PdfReader

root = Path(__file__).resolve().parent
archive = root.parent.parent / '文献_相关原始论文_20261004'
records = json.loads((root/'retrieval_receipts.json').read_text(encoding='utf-8'))
if (root/'additional_retrieval_receipts.json').exists():
    records += json.loads((root/'additional_retrieval_receipts.json').read_text(encoding='utf-8'))
if (root/'assignment_retrieval_receipts.json').exists():
    records += json.loads((root/'assignment_retrieval_receipts.json').read_text(encoding='utf-8'))
lad=archive/'20261028_码率调研_LAD_WACV2022.pdf'
if lad.exists():
    records.append({'key':'R15_LAD_existing','status':'downloaded','is_pdf':True,
                    'path':str(lad),'bytes':lad.stat().st_size,
                    'sha256':hashlib.sha256(lad.read_bytes()).hexdigest(),
                    'url':'https://openaccess.thecvf.com/content/WACV2022/papers/Nguyen_Improving_Object_Detection_by_Label_Assignment_Distillation_WACV_2022_paper.pdf',
                    'retrieval_note':'existing archived PDF, not downloaded again'})
preflight_folder = root / 'pdf_preflight'
preflight_folder.mkdir(exist_ok=True)
receipts = []
for row in records:
    if row['status'] != 'downloaded' or not row.get('is_pdf'):
        continue
    path = Path(row['path'])
    sidecar = preflight_folder / (path.stem + '.json')
    if sidecar.exists():
        preflight=json.loads(sidecar.read_text(encoding='utf-8'))
    else:
        process = subprocess.run([sys.executable, r'C:\Users\jiwei\.codex\skills\academic-research-suite\ars\scripts\pdf_read_preflight.py', str(path)],
                                 capture_output=True, env={**os.environ,'PYTHONIOENCODING':'utf-8'})
        preflight = json.loads(process.stdout.decode('utf-8'))
        sidecar.write_bytes(process.stdout)
    document = PdfReader(path)
    texts = [page.extract_text() or '' for page in document.pages]
    text_path = archive / '_text' / (path.stem + '_full.txt')
    value = '\n\n'.join(f'=== PDF page {i+1} ===\n{text}' for i,text in enumerate(texts))
    if text_path.exists() and text_path.read_text(encoding='utf-8') != value:
        raise ValueError('Different existing extraction preserved')
    if not text_path.exists():
        text_path.write_text(value, encoding='utf-8')
    receipts.append({**row,'pdf_pages':len(texts),'text_path':str(text_path),
                     'preflight_path':str(sidecar),'structural_verdict':preflight.get('verdict'),
                     'reading_scope':'extracted only; actual reading recorded separately','model_execution':False})
    print(json.dumps({'key':row['key'],'pages':len(texts),'structure':preflight.get('verdict'),
                      'page_starts':[text[:65] for text in texts]},ensure_ascii=True),flush=True)
(root/'document_reading_receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf-8')
