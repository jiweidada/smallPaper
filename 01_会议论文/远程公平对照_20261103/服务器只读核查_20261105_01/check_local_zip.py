"""Read and check the saved local ZIP and its manifest; write only a small audit report."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import zipfile

out_dir = Path(__file__).resolve().parent
root = out_dir.parent
archive = root / 'verified_dataset.zip'
manifest_path = root / 'dataset_bundle_manifest.json'
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)
digest = hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
entries = manifest['files']
manifest_names = [entry['path'] for entry in entries]
manifest_sizes = {entry['path']: entry['bytes'] for entry in entries}
with zipfile.ZipFile(archive) as zf:
    infos = [info for info in zf.infolist() if not info.is_dir()]
    zip_names = [info.filename for info in infos]
    size_mismatches = [info.filename for info in infos if manifest_sizes.get(info.filename) != info.file_size]
    counts = Counter()
    for name in zip_names:
        if name.startswith('train/images/'):
            counts['train_images'] += 1
        elif name.startswith('val/images/'):
            counts['val_images'] += 1
        elif name.startswith('train/annotations/'):
            counts['train_annotations'] += 1
        elif name.startswith('val/annotations/'):
            counts['val_annotations'] += 1
        else:
            counts['other'] += 1
result = {
    'archive_bytes': archive.stat().st_size,
    'archive_sha256': digest.hexdigest(),
    'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
    'archive_hash_matches_manifest': digest.hexdigest() == manifest['zip_sha256'],
    'manifest_entries': len(entries),
    'zip_entries': len(infos),
    'zip_names_unique': len(zip_names) == len(set(zip_names)),
    'manifest_names_unique': len(manifest_names) == len(set(manifest_names)),
    'missing_in_zip_count': len(set(manifest_names) - set(zip_names)),
    'extra_in_zip_count': len(set(zip_names) - set(manifest_names)),
    'size_mismatch_count': len(size_mismatches),
    'counts': dict(counts),
    'scope': 'ZIP archive hash and directory/manifest names and uncompressed sizes; no decoding or training.'
}
(out_dir / 'local_zip_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False))
