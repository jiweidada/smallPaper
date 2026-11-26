"""Hash current local training images without decoding or model execution."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

OUT = Path(__file__).resolve().parent
protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
annotation = Path(protocol['data']['train']['annotation_path'])
annotation_bytes = annotation.read_bytes()
assert hashlib.sha256(annotation_bytes).hexdigest() == protocol['data']['train']['annotation_sha256']
data = json.loads(annotation_bytes)
image_root = (annotation.parent.parent / 'images').resolve()
target = OUT / 'training_image_manifest.json'
if target.exists():
    raise SystemExit('Training manifest already exists; no automatic replacement or repeated hashing.')
images = data['images']
assert len(images) == 6471 and len({x['id'] for x in images}) == 6471
records = []
for index, item in enumerate(sorted(images, key=lambda x: x['id']), 1):
    path = (image_root / item['file_name']).resolve()
    assert path.is_relative_to(image_root)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    records.append({'image_id': item['id'], 'file_name': item['file_name'], 'path': str(path),
                    'bytes': before.st_size, 'sha256': digest.hexdigest(),
                    'annotation_width': item['width'], 'annotation_height': item['height']})
    if index % 1000 == 0:
        print(f'Current training-image hashes: {index}/6471', flush=True)
manifest = {'status': 'CURRENT_LOCAL_IMAGES_HASHED_NO_EXPERIMENT', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'annotation_sha256': hashlib.sha256(annotation_bytes).hexdigest(),
            'count': len(records), 'bytes': sum(r['bytes'] for r in records), 'image_root': str(image_root),
            'scope': 'Current file bytes only; does not establish historical training-data identity. Dimensions are annotation fields, not a decoded-image integrity check.',
            'images': records}
target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'manifest': str(target), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                  'images': len(records), 'bytes': manifest['bytes'], 'model_execution': False}, ensure_ascii=False))
