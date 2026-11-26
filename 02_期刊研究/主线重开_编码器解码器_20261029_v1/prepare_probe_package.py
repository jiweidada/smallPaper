"""Package only previously verified official bytes; no model execution."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'official_assets_v1'
receipt = json.loads((ASSETS / 'source_git_receipt.json').read_text(encoding='utf-8'))
assert receipt['status'] == 'OFFICIAL_COMMIT_ALL_BLOBS_VERIFIED'
source = Path(receipt['path'])
rows = []
archive = ASSETS / 'TinyFormer_9075d9f_verified_source.tar.gz'
assert not archive.exists(), 'Do not silently replace a prepared package'
with tarfile.open(archive, 'w:gz') as tf:
    for row in receipt['verified_files']:
        p = source / row['path']
        assert p.is_file() and not p.is_symlink()
        data = p.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        assert blob == row['git_blob_sha'] and len(data) == row['bytes'], row['path']
        rows.append({**row, 'sha256': hashlib.sha256(data).hexdigest()})
        tf.add(p, arcname='source/' + row['path'], recursive=False)
weight = ASSETS / 'TinyFormer-S-pbm-visdrone.pth'
wh = hashlib.sha256(weight.read_bytes()).hexdigest()
assert wh == '0ba0e997e4bdfae7800f6f3e55fa714b0302efd4aa5327663290c9705eb8a199'
out = {'source_commit': receipt['commit'], 'files': rows,
       'source_archive': {'name': archive.name, 'bytes': archive.stat().st_size,
                          'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()},
       'weight': {'name': weight.name, 'bytes': weight.stat().st_size, 'sha256': wh},
       'provenance_scope': 'Official Git blobs and locally computed weight SHA; no independently published author weight checksum',
       'model_forwards': 0, 'training_updates': 0}
(ASSETS / 'probe_package_manifest.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'verified_source_files': len(rows), 'source_archive': out['source_archive'], 'weight': out['weight']}))
