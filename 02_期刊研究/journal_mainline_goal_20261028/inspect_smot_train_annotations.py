"""Source-GT inventory only; never load predictions or run matching/evaluation."""
import hashlib
import json
import statistics
import zipfile
from collections import Counter
from pathlib import Path

root = Path(__file__).parent / 'smot_source_metadata_20261029'
archive = root / 'annotations.zip'
receipt = {'scope': 'TRAIN annotation inventory only',
           'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
           'model_run': False, 'prediction_matching': False, 'test_payloads_read': False}
with zipfile.ZipFile(archive) as z:
    catalog = [{'name': i.filename, 'size': i.file_size, 'compressed_size': i.compress_size} for i in z.infolist()]
    receipt['archive_catalog'] = catalog
    train_entries = [i for i in z.infolist() if i.filename.replace('\\', '/').split('/')[-1] == 'train.json']
    if len(train_entries) != 1:
        raise RuntimeError('Expected one explicit train.json; inspect catalog without reading other payloads')
    entry = train_entries[0]
    if entry.file_size > 160_000_000:
        raise RuntimeError('TRAIN annotation source exceeds read cap')
    raw = z.read(entry)
    data = json.loads(raw)
    (root / 'author_train_annotations.json').write_bytes(raw)
receipt['train_entry'] = entry.filename
receipt['top_level_keys'] = list(data)
for key in ('videos', 'images', 'annotations', 'categories'):
    rows = data.get(key, [])
    receipt[key + '_count'] = len(rows)
    receipt[key + '_example'] = rows[:1]
receipt['train_json_sha256'] = hashlib.sha256(raw).hexdigest()
images = data.get('images', [])
anns = data.get('annotations', [])
receipt['image_dimensions'] = {str(k): v for k, v in Counter((i.get('width'), i.get('height')) for i in images).items()}
receipt['annotation_keys'] = list(anns[0]) if anns else []
valid = [a['bbox'] for a in anns if isinstance(a.get('bbox'), list) and len(a['bbox']) == 4 and a['bbox'][2] > 0 and a['bbox'][3] > 0]
area = [b[2]*b[3] for b in valid]
receipt['valid_positive_bboxes'] = len(valid)
receipt['native_bbox_area_lt_1024_count'] = sum(a < 1024 for a in area)
receipt['native_bbox_area_lt_1024_fraction'] = sum(a < 1024 for a in area)/len(area) if area else None
receipt['native_bbox_width_median'] = statistics.median(b[2] for b in valid) if valid else None
receipt['native_bbox_height_median'] = statistics.median(b[3] for b in valid) if valid else None
receipt['interpretation'] = 'Annotation appearances, not independent animals. Bbox area inventory is not COCO AP and is not instance-mask area. No inference about reliable background pairing from labels alone.'
path = root / 'train_annotation_inventory.json'
path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(receipt, ensure_ascii=False, indent=2))
