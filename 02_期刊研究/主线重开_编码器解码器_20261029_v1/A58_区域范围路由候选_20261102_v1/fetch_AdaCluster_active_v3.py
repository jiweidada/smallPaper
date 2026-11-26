"""Last selected source within the four-file overlap-check bound."""
from pathlib import Path
import hashlib
import json
import urllib.request

P = Path(__file__).resolve().parent
index = json.loads((P / 'sources/AdaCluster_index_v1/receipt.json').read_text('utf-8'))
tree = json.loads((P / 'sources/AdaCluster_index_v1/tree.json').read_text('utf-8'))
lookup = {v['path']: v for v in tree['tree']}
name = 'runcog/spas_sage_attn/autotune.py'
revision = index['commit_sha']
url = f'https://raw.githubusercontent.com/USTC-MLSys/Adacluster/{revision}/{name}'
D = P / 'sources/AdaCluster_active_v3'
D.mkdir(exist_ok=False)
req = urllib.request.Request(url, headers={'User-Agent': 'bounded-literature-source-check'})
with urllib.request.urlopen(req, timeout=20) as response:
    assert response.status == 200
    raw = response.read(500001)
assert len(raw) <= 500000
gitsha = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
assert gitsha == lookup[name]['sha']
with (D / 'autotune.py').open('xb') as f:
    f.write(raw)
receipt = dict(commit_sha=revision, source=name, bytes=len(raw), git_blob_sha=gitsha,
               sha256=hashlib.sha256(raw).hexdigest(), url=url,
               downloaded_code_executed=False, new_model_forwards=0,
               total_selected_source_files_fetched=4, selected_source_limit=4)
with (D / 'receipt.json').open('x', encoding='utf-8') as f:
    json.dump(receipt, f, indent=2)
print(json.dumps(receipt))
