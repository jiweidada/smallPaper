"""Read-only HTTPS retrieval of public primary sources; no model execution."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.request import Request, urlopen
import hashlib
import json
import ssl

root = Path(__file__).resolve().parent / 'source_audit_v1'
root.mkdir(exist_ok=True)
jobs = [
    ('edge_commit.json', 'https://api.github.com/repos/Intellindust-AI-Lab/EdgeCrafter/commits/main'),
    ('edge_tree.json', 'https://api.github.com/repos/Intellindust-AI-Lab/EdgeCrafter/git/trees/main?recursive=1'),
    ('deimv2_commit.json', 'https://api.github.com/repos/Intellindust-AI-Lab/DEIMv2/commits/main'),
    ('tinyformer_commit.json', 'https://api.github.com/repos/mmpmmpmmpjosh/TinyFormer/commits/main'),
    ('tinyformer_tree.json', 'https://api.github.com/repos/mmpmmpmmpjosh/TinyFormer/git/trees/main?recursive=1'),
]
ctx = ssl.create_default_context()

def fetch(job):
    name, url = job
    receipt = {'name':name, 'url':url, 'fetched_utc':datetime.now(timezone.utc).isoformat()}
    try:
        req = Request(url, headers={'User-Agent':'AcademicArchitectureAudit/1.0', 'Accept':'application/vnd.github+json'})
        with urlopen(req, context=ctx, timeout=20) as response:
            blob = response.read(5_000_001)
            if len(blob)>5_000_000:
                raise ValueError('response exceeds 5 MB bound')
            data = json.loads(blob)
            (root / name).write_bytes(blob)
            receipt.update(status=response.status, final_url=response.url, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
            if 'sha' in data:
                receipt['git_sha'] = data['sha']
            if 'tree' in data:
                receipt['selected_paths']=[x['path'] for x in data['tree'] if x['type']=='blob' and any(k in x['path'].lower() for k in ('decoder','encoder','backbone','dinov3','vitt','ecvit','hybrid','tinyformer'))][:100]
    except Exception as exc:
        receipt.update(status='FAILED', error=type(exc).__name__+': '+str(exc))
    return receipt

with ThreadPoolExecutor(max_workers=3) as pool:
    receipts = list(pool.map(fetch, jobs))
(root / 'retrieval_receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
for rec in receipts:
    print(json.dumps(rec,ensure_ascii=False))
