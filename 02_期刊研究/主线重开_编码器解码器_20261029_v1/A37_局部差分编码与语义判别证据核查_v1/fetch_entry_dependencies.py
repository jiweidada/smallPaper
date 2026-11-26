from pathlib import Path
import hashlib, json, sys, urllib.request
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding='utf-8')
D = Path(__file__).resolve().parent
pin = json.loads((D / 'repo_metadata/PiDiViT/receipt.json').read_text(encoding='utf-8'))
tree = json.loads((D / 'repo_metadata/PiDiViT/tree.json').read_text(encoding='utf-8'))
blobs = {x['path']: x for x in tree['tree'] if x['type'] == 'blob'}
paths = ['scripts/train.sh', 'train.py', 'Tools.md', 'configs/Base-RCNN-C4.yaml',
         'configs/Base-RCNN-FPN.yaml', 'configs/RPN/mask_rcnn_R_50_C4_1x_fewshot_14.yaml']
rows = []
for path in paths:
    assert path in blobs, path
    url = 'https://raw.githubusercontent.com/Seaz9/PiDiViT/' + pin['commit'] + '/' + path
    req = urllib.request.Request(url, headers={'User-Agent': 'UAV-architecture-primary-source-review'})
    with urllib.request.urlopen(req, timeout=30) as response:
        raw = response.read()
    git_sha = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    assert git_sha == blobs[path]['sha'], path
    target = D / 'source/PiDiViT' / path
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as output:
        output.write(raw)
    rows.append(dict(path=path, url=url, git_blob_sha=git_sha,
                     sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
with (D / 'entry_dependency_verification.json').open('x', encoding='utf-8') as output:
    json.dump(dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
                   status='SELECTED_SIX_ENTRY_DEPENDENCIES_VERIFIED_STATIC_ONLY',
                   repo='Seaz9/PiDiViT', commit=pin['commit'], files=rows),
              output, ensure_ascii=False, indent=2)
print(json.dumps(dict(verified_extra_blobs=len(rows), commit=pin['commit'])))
