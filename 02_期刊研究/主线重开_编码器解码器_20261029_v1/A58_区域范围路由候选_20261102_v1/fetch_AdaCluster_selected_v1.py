"""Fetch two selected files at a verified public commit; never import them."""
from pathlib import Path
import json
import hashlib
import urllib.request

P = Path(__file__).resolve().parent
index = json.loads((P / 'sources/AdaCluster_index_v1/receipt.json').read_text('utf-8'))
tree = json.loads((P / 'sources/AdaCluster_index_v1/tree.json').read_text('utf-8'))
revision = index['commit_sha']
lookup = {v['path']: v for v in tree['tree']}
names = ['runcog/modify_cogvideox/cog_video_attn_processor_kvclus_new.py',
         'triton_kernel/flashinfer_cluster_sparse_attn_v3.py']
D = P / 'sources/AdaCluster_selected_v1'
D.mkdir(parents=True, exist_ok=False)
rows = []
for name in names:
    url = 'https://raw.githubusercontent.com/USTC-MLSys/Adacluster/' + revision + '/' + name
    req = urllib.request.Request(url, headers={'User-Agent': 'bounded-literature-source-check'})
    with urllib.request.urlopen(req, timeout=20) as response:
        assert response.status == 200
        raw = response.read(500001)
    assert len(raw) <= 500000
    gitsha = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    assert gitsha == lookup[name]['sha']
    target = (D / name).resolve()
    assert target.is_relative_to(D.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as f:
        f.write(raw)
    rows.append({'source': name, 'bytes': len(raw), 'git_blob_sha': gitsha,
                 'sha256': hashlib.sha256(raw).hexdigest(),
                 'url': 'https://github.com/USTC-MLSys/Adacluster/blob/' + revision + '/' + name})
receipt = {'status': 'TWO_PINNED_SOURCE_FILES_FETCHED_GIT_BLOBS_VERIFIED', 'commit_sha': revision,
           'files': rows, 'downloaded_code_executed': False, 'new_model_forwards': 0}
with (D / 'receipt.json').open('x', encoding='utf-8') as f:
    json.dump(receipt, f, indent=2)
print(json.dumps(receipt))
