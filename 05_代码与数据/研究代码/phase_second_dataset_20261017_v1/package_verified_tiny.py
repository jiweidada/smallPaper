"""Portable, append-only data package for the user's second account."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path('/24085403014/datasets_verified_20261017/TinyPerson_v1')
ready = json.loads((root / 'data_ready.json').read_text())
assert ready['status'] == 'data_view_ready_no_training_started'
target = root.parent / 'TinyPerson_v1_verified_portable.tar'
if target.exists(): raise FileExistsError(str(target))
files = []
for path in sorted((root / 'annotations').iterdir()): files.append((path, 'annotations/' + path.name))
for name in ['split_manifest.json', 'data_ready.json']: files.append((root / name, name))
for name, image_folder in [('train.json', 'train_erased'), ('val.json', 'train_original'), ('test.json', 'test_original')]:
    annotation = json.loads((root / 'annotations' / name).read_text())
    for im in annotation['images']:
        files.append((root / image_folder / im['file_name'], image_folder + '/' + im['file_name']))
assert len({name for _, name in files}) == len(files)
image_manifest = []
with tarfile.open(target, 'x') as archive:
    for path, name in files:
        raw_sha = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(2 * 1024 * 1024), b''): raw_sha.update(block)
        image_manifest.append({'relative': name, 'sha256': raw_sha.hexdigest(), 'bytes': path.stat().st_size})
        info = archive.gettarinfo(str(path.resolve()), arcname=name)
        if not info.isfile(): raise RuntimeError('Portable package accepts regular files only')
        with path.open('rb') as handle: archive.addfile(info, handle)
manifest_path = root / 'portable_file_manifest.json'
with manifest_path.open('x') as handle: json.dump(image_manifest, handle, indent=2)
# Keep the package immutable; the manifest is relayed separately and checked before extraction.
digest = hashlib.sha256()
with target.open('rb') as handle:
    for block in iter(lambda: handle.read(4 * 1024 * 1024), b''): digest.update(block)
receipt = {'path': str(target), 'sha256': digest.hexdigest(), 'bytes': target.stat().st_size,
           'regular_files': len(files), 'file_manifest': str(manifest_path),
           'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
with target.with_suffix('.receipt.json').open('x') as handle: json.dump(receipt, handle, indent=2)
print(json.dumps(receipt))
