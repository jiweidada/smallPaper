"""Public document/static text retrieval and Git blob verification only."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, hashlib, base64
from retrieve_static import ROOT, retrieve

commit = '2004cba021c68a46429be460e26954ab171b2862'
source_root = ROOT / 'source_code' / 'v8.4.191_2004cba021c6'
jobs = {
    'augment_contents': 'https://api.github.com/repos/ultralytics/ultralytics/contents/ultralytics/data/augment.py?ref=' + commit,
    'detect_val_contents': 'https://api.github.com/repos/ultralytics/ultralytics/contents/ultralytics/models/yolo/detect/val.py?ref=' + commit,
    'ultralytics_git_tree': 'https://api.github.com/repos/ultralytics/ultralytics/git/trees/' + commit + '?recursive=1',
    'AERIS_ECCV2022': 'https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690465.pdf',
}
records = []
with ThreadPoolExecutor(max_workers=3) as pool:
    futures = [pool.submit(retrieve, key, url) for key, url in jobs.items()]
    for future in as_completed(futures):
        row = future.result()
        records.append(row)
        print(json.dumps({key: row.get(key) for key in ['key', 'status', 'bytes', 'error']}, ensure_ascii=True), flush=True)

decoded = []
for row in records:
    if row['status'] != 'downloaded' or not row['key'].endswith('_contents'):
        continue
    item = json.loads(Path(row['path']).read_text(encoding='utf-8'))
    data = base64.b64decode(item['content'])
    blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    if blob != item['sha']:
        raise ValueError('Git blob mismatch')
    destination = source_root / item['path']
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != data:
        raise ValueError('Preserved different existing source')
    if not destination.exists():
        destination.write_bytes(data)
    decoded.append({'path': str(destination), 'repository_path': item['path'], 'sha256': hashlib.sha256(data).hexdigest(), 'git_blob_sha1': blob, 'verification': 'matches GitHub contents API', 'commit': commit})

tree_row = next((r for r in records if r['key'] == 'ultralytics_git_tree' and r['status'] == 'downloaded'), None)
verification = []
if tree_row:
    tree = json.loads(Path(tree_row['path']).read_text(encoding='utf-8'))
    if tree.get('truncated'):
        raise ValueError('Incomplete Git tree')
    expected = {item['path']: item['sha'] for item in tree['tree'] if item['type'] == 'blob'}
    for file in sorted(source_root.rglob('*')):
        if not file.is_file() or file.name == 'pin.json':
            continue
        path = file.relative_to(source_root).as_posix()
        data = file.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        verification.append({'path': path, 'sha256': hashlib.sha256(data).hexdigest(), 'git_blob_sha1': blob, 'expected_blob': expected.get(path), 'matches': blob == expected.get(path)})
    if not all(item['matches'] for item in verification):
        raise ValueError('Source does not match frozen repository tree')

(ROOT / 'complete_static_receipts.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
(ROOT / 'static_blob_verification.json').write_text(json.dumps({'commit': commit, 'decoded_api_files': decoded, 'tree_verified_files': verification, 'executed_source': False}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'decoded_files': len(decoded), 'tree_verified_files': len(verification), 'source_execution': False}), flush=True)
