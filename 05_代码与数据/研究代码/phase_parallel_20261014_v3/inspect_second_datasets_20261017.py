"""Read bounded dataset inventory and annotation summaries; never change source data."""
from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path
import zipfile

storage = Path('/24085403014')
roots = []
top = []
for parent in [storage, storage / 'DEIM']:
    if not parent.is_dir():
        continue
    for path in sorted(parent.iterdir()):
        if any(token in path.name.lower() for token in ['uavdt', 'tinyperson', 'tiny_person']):
            top.append({'path': str(path), 'is_dir': path.is_dir(), 'bytes': path.stat().st_size})
            if path.is_dir():
                roots.append(path)
result = {'remote_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'candidate_roots': top, 'directories': [], 'annotations': [], 'archives': [], 'errors': []}
for root in roots:
    for current, directories, files in os.walk(root):
        path = Path(current)
        depth = len(path.relative_to(root).parts)
        directories[:] = sorted(d for d in directories if d not in ['.git', '__pycache__'])
        counts = Counter(Path(name).suffix.lower() for name in files)
        result['directories'].append({'path': str(path), 'depth': depth,
                                      'subdirectories': directories[:50], 'file_count': len(files),
                                      'suffix_counts': dict(counts), 'sample_files': sorted(files)[:8]})
        for name in sorted(files):
            file = path / name
            if file.suffix.lower() == '.json' and file.stat().st_size <= 160 * 1024 * 1024:
                try:
                    raw = file.read_bytes()
                    value = json.loads(raw)
                    row = {'path': str(file), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
                    if isinstance(value, dict):
                        row['keys'] = list(value)
                        if isinstance(value.get('images'), list):
                            images = value['images']; annotations = value.get('annotations', [])
                            row.update(images=len(images), annotations=len(annotations), categories=value.get('categories'),
                                       sample_images=images[:4], sample_annotations=annotations[:5],
                                       annotation_key_counts=dict(Counter(k for ann in annotations for k in ann)),
                                       category_counts=dict(Counter(str(ann.get('category_id')) for ann in annotations)),
                                       flag_counts={k: dict(Counter(str(ann.get(k, '<missing>')) for ann in annotations))
                                                    for k in ['ignore', 'iscrowd', 'uncertain', 'dense', 'is_dense']})
                    result['annotations'].append(row)
                except Exception as exc:
                    result['errors'].append({'path': str(file), 'error': str(exc)})
            elif file.suffix.lower() in ['.yaml', '.yml'] or (file.suffix.lower() == '.txt' and any(x in name.lower() for x in ['readme', 'split', 'train', 'val', 'test', 'gt_ignore', 'gt_whole'])):
                if len([x for x in result['annotations'] if x.get('format') == 'text']) < 35:
                    raw = file.read_bytes()
                    result['annotations'].append({'path': str(file), 'bytes': len(raw),
                                                  'sha256': hashlib.sha256(raw).hexdigest(), 'format': 'text',
                                                  'sample': raw[:2500].decode(errors='replace')})
            elif file.suffix.lower() == '.zip':
                try:
                    with zipfile.ZipFile(file) as archive:
                        names = archive.namelist()
                        result['archives'].append({'path': str(file), 'bytes': file.stat().st_size,
                                                   'members': len(names), 'samples': names[:15],
                                                   'annotation_members': [n for n in names if n.endswith(('.json', '.yml', '.yaml'))][:100]})
                except Exception as exc:
                    result['errors'].append({'path': str(file), 'error': str(exc)})
        if depth >= 4:
            directories[:] = []
print(json.dumps(result, ensure_ascii=False))
