from pathlib import Path
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parent
source=ROOT/'source_audit_v1'
pdf=source/'EASE-DETR_CVPR2024_official.pdf'
python=r'C:\Users\jiwei\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
preflight=r'C:\Users\jiwei\.codex\skills\academic-research-suite\ars\scripts\pdf_read_preflight.py'
result=subprocess.run([python,'-X','utf8','-s',preflight,str(pdf)],capture_output=True,encoding='utf-8',check=True)
sidecar=json.loads(result.stdout)
assert sidecar['verdict']=='PASS' and sidecar['sha256']==hashlib.sha256(pdf.read_bytes()).hexdigest()
assert sidecar['declared_page_count']==sidecar['enumerated_page_count']==sidecar['reader_page_count']==10
(source/'EASE-DETR_CVPR2024_pdf_preflight.json').write_text(json.dumps(sidecar,ensure_ascii=False,indent=2),encoding='utf-8')
pages=(source/'EASE-DETR_CVPR2024_official.txt').read_text(encoding='utf-8').split('\f')
for i,text in enumerate(pages,1):
    if not text.strip():continue
    if any(x in text for x in ['3.2. Preparation','3.3. Encourage','3.4. Encourage']):
        (source/f'EASE_method_page_{i}.txt').write_text(text,encoding='utf-8')
        print('METHOD_PAGE',i,len(text))
print(json.dumps({'pdf_sha256':sidecar['sha256'],'structural_preflight':'PASS','pages':10}))
