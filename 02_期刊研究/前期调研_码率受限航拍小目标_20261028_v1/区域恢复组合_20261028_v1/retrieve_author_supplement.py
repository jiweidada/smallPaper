"""Retrieve original paper supplement; inspect archive names only, never load weights or execute code."""
from pathlib import Path,PurePosixPath
import subprocess,json,hashlib,zipfile
root=Path(__file__).resolve().parent
archive=root.parent.parent/'文献_相关原始论文_20261004'
url='https://article.accscience.com/journal/DP/3/3/dp026260029/article_dp026260029_supplement_d9d138.zip'
dest=archive/'20261028_区域组合_JCRR_original_supplement.zip'
receipt={'url':url,'research_date':'2026-10-28','purpose':'Original supplementary material for static reading; no code or checkpoint loading.'}
try:
    if dest.exists():data=dest.read_bytes()
    else:
        r=subprocess.run([r'C:\Windows\System32\curl.exe','--location','--fail','--silent','--show-error','--max-time','45',url],capture_output=True,timeout=50)
        if r.returncode:raise RuntimeError(r.stderr.decode('utf-8',errors='replace')[-500:])
        data=r.stdout
    if not data.startswith(b'PK') or len(data)>40_000_000:raise ValueError('Not a ZIP or larger than the reading cap')
    if not dest.exists():dest.write_bytes(data)
    with zipfile.ZipFile(dest) as z:
        entries=[{'name':i.filename,'bytes':i.file_size,'compressed_bytes':i.compress_size} for i in z.infolist()]
    receipt.update(status='downloaded',path=str(dest),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),entries=entries,weight_files_loaded=0,source_files_executed=0)
except Exception as exc:receipt.update(status='unavailable',error=str(exc))
(root/'author_supplement_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:receipt.get(k) for k in ['status','bytes','error','entries']},ensure_ascii=True))
