"""Read-only byte-level split-integrity audit; no models or image decoding."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
DATA = Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone')
TRAIN_MANIFEST = PAPER / '公平对照方案_20261103/training_image_manifest.json'
VAL_MANIFEST = PAPER / '本地CPU核验_20261103/validation_image_manifest.json'
OUTPUT = HERE / 'train_val_byte_overlap_audit.json'

def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read_manifest(path, train):
    record = json.loads(path.read_text(encoding='utf-8-sig'))
    return record['images'] if train else record

def inspect(split, manifest_path, image_root, annotation, train):
    listed = read_manifest(manifest_path, train)
    assert len(listed) == (6471 if train else 548)
    ann = json.loads(annotation.read_text(encoding='utf-8-sig'))
    ann_images = ann['images']
    root = image_root.resolve(strict=True)
    actual = []
    failures = []
    for item in listed:
        name = item['file_name']
        target = (root / name).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            failures.append({'file_name': name, 'issue': 'unsafe_or_missing'})
            continue
        size = target.stat().st_size
        byte_hash = digest(target)
        if item['sha256'] != byte_hash or item['bytes'] != size:
            failures.append({'file_name': name, 'issue': 'manifest_mismatch', 'saved_sha256': item['sha256'], 'current_sha256': byte_hash, 'saved_bytes': item['bytes'], 'current_bytes': size})
        actual.append({'file_name': name, 'sha256': byte_hash, 'bytes': size})
    manifest_names = Counter(x['file_name'] for x in listed)
    annotation_names = Counter(x['file_name'] for x in ann_images)
    groups = defaultdict(list)
    for item in actual:
        groups[item['sha256']].append(item['file_name'])
    return {'split': split, 'image_root': str(root), 'manifest': str(manifest_path),
            'manifest_sha256': digest(manifest_path), 'annotation': str(annotation),
            'annotation_sha256': digest(annotation), 'listed_images': len(listed),
            'annotation_images': len(ann_images), 'successfully_hashed': len(actual),
            'manifest_mismatches_or_missing': failures,
            'manifest_name_duplicates': {n: c for n, c in manifest_names.items() if c > 1},
            'annotation_name_duplicates': {n: c for n, c in annotation_names.items() if c > 1},
            'names_missing_from_manifest': list((annotation_names - manifest_names).elements()),
            'names_extra_in_manifest': list((manifest_names - annotation_names).elements()),
            'within_split_identical_byte_groups': {h: v for h, v in groups.items() if len(v) > 1},
            'files': actual}

def main():
    assert not OUTPUT.exists()
    train = inspect('train', TRAIN_MANIFEST, DATA/'VisDrone2019-DET-train/images',
                    DATA/'VisDrone2019-DET-train/annotations/train.json', True)
    val = inspect('val', VAL_MANIFEST, DATA/'VisDrone2019-DET-val/images',
                  DATA/'VisDrone2019-DET-val/annotations/val.json', False)
    train_hash = defaultdict(list)
    for item in train['files']:
        train_hash[item['sha256']].append(item['file_name'])
    byte_overlap = [{'sha256': item['sha256'], 'validation_file': item['file_name'],
                     'train_files': train_hash[item['sha256']]}
                    for item in val['files'] if item['sha256'] in train_hash]
    train_names = {item['file_name'] for item in train['files']}
    name_overlap = sorted({item['file_name'] for item in val['files']} & train_names)
    complete = all(not x['manifest_mismatches_or_missing'] and not x['names_missing_from_manifest']
                   and not x['names_extra_in_manifest'] and not x['manifest_name_duplicates']
                   and not x['annotation_name_duplicates'] for x in (train, val))
    audit = {'checked_utc': datetime.now(timezone.utc).isoformat(),
             'status': 'HASHED_CURRENT_FILES' if complete else 'INCOMPLETE_OR_MISMATCH',
             'train': train, 'validation': val,
             'cross_split_identical_jpeg_byte_records': byte_overlap,
             'cross_split_shared_filenames': name_overlap,
             'limits': ['SHA256 compares exact current JPEG bytes, not decoded pixels, near-duplicate frames or semantic overlap.',
                        'The present data and manifests do not authenticate the images used in historical training.',
                        'No model, prediction, evaluator or optimizer was executed.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': audit['status'], 'train_count': len(train['files']), 'validation_count': len(val['files']),
                      'failures': len(train['manifest_mismatches_or_missing']) + len(val['manifest_mismatches_or_missing']),
                      'cross_split_byte_records': len(byte_overlap), 'cross_split_shared_filenames': len(name_overlap),
                      'train_within_split_groups': len(train['within_split_identical_byte_groups']),
                      'val_within_split_groups': len(val['within_split_identical_byte_groups'])}, ensure_ascii=False))

if __name__ == '__main__':
    main()
