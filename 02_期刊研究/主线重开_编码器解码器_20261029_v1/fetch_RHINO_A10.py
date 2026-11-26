"""Read pinned public author files; never imports the downloaded implementation."""
from pathlib import Path
import hashlib
import json
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent / 'comparison_sources_v1' / 'RHINO_A10_v1'
head = json.loads((ROOT / 'HEAD.json').read_text(encoding='utf-8'))
tree = json.loads((ROOT / 'tree.json').read_text(encoding='utf-8'))
assert head['sha'] == 'd9baf3bc1149ebef2ef03c89b26b93106b82075a'
assert tree['sha'] == head['commit']['tree']['sha'] and not tree['truncated']
blobs = {r['path']: r for r in tree['tree'] if r['type'] == 'blob'}
selected = [
    'README.md',
    'configs/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_dota.py',
    'configs/rhino/rhino-4scale_r50_2xb2-12e_dota.py',
    'mmrotate/models/dense_heads/rhino_phc_head.py',
    'mmrotate/models/dense_heads/rhino_ph_head.py',
    'mmrotate/models/dense_heads/rhino_head.py',
    'mmrotate/models/dense_heads/rhino_align_head.py',
    'mmrotate/models/detectors/rhino.py',
    'mmrotate/models/layers/transformer/rhino_layers.py',
    'mmrotate/models/layers/transformer/rhino_layers_v2.py',
]
records = []
for rel in selected:
    path = ROOT / 'source' / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    url = 'https://raw.githubusercontent.com/SIAnalytics/RHINO/' + head['sha'] + '/' + rel
    if path.exists():
        data = path.read_bytes()
    else:
        req = urllib.request.Request(url, headers={'User-Agent': 'Research-source-audit'})
        with urllib.request.urlopen(req, timeout=25) as r:
            data = r.read()
        path.write_bytes(data)
    actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    assert actual == blobs[rel]['sha'], rel
    records.append({'path': rel, 'url': url, 'git_blob': actual, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
receipt = {'utc': datetime.now(timezone.utc).isoformat(), 'commit': head['sha'], 'status': 'PASS_ALL_SELECTED_GIT_BLOBS_VERIFIED', 'files': records, 'scope': 'Selected source transfer only; imports0,model execution0.'}
(ROOT / 'download_verification.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': receipt['status'], 'files': len(records), 'bytes': sum(r['bytes'] for r in records)}))
