"""Build a new TinyPerson data view from preserved original archives and labels."""
from collections import Counter
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import tarfile

source = Path('/24085403014/DEIM/tiny_set')
dest = Path('/24085403014/datasets_verified_20261017/TinyPerson_v1')
dest.parent.mkdir(parents=True, exist_ok=True)
dest.mkdir(exist_ok=False)


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(2 * 1024 * 1024), b''): value.update(block)
    return value.hexdigest()


expected = {
    'annotations/tiny_set_train.json': '4c2683bf695d6be08667c52e262ec4f1790ebeeba04cb5200424a05464a5eeea',
    'annotations/tiny_set_test.json': '0cfcbbcc7433ad5bdc6b856df1d109188cda48035506bda1f16b7e2261c63d57',
    'annotations/task/tiny_set_test_all.json': '8c7703d7058a75de37c45d4b0023769ad08ee14a6767ad4939c53e4763d1f8b1',
    'erase_with_uncertain_dataset/annotations/task/tiny_set_train_all.json': 'c4f5bef58be3bd7b6b622a4a2ac7030255d2321ef5e59dcc2d153b999e0de407',
}
annotations = {}; receipt = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'source': str(source), 'dest': str(dest), 'status': 'preparing', 'sources': [],
    'source_release': 'TinyPerson v1 original-format copy attributed to UCAS-VG by the existing OpenDataLab metadata; no author-published byte-level checksum available.'}
for relative, expected_sha in expected.items():
    path = source / relative
    assert sha(path) == expected_sha, relative
    annotations[relative] = json.loads(path.read_text())
    receipt['sources'].append({'path': str(path), 'sha256': expected_sha, 'bytes': path.stat().st_size})
raw_train = annotations['annotations/tiny_set_train.json']
raw_test = annotations['annotations/tiny_set_test.json']
clean_train = annotations['erase_with_uncertain_dataset/annotations/task/tiny_set_train_all.json']
assert len(raw_train['images']) == len(clean_train['images']) == 746
assert len(raw_test['images']) == 786
assert len(clean_train['annotations']) == 18433
assert all(not ann.get('ignore') and not ann.get('uncertain') for ann in clean_train['annotations'])

erase_archive = source / 'erase_with_uncertain_dataset/train.tar.gz'
receipt['sources'].append({'path': str(erase_archive), 'sha256': sha(erase_archive), 'bytes': erase_archive.stat().st_size})
erase = dest / 'erased_original'; erase.mkdir()
extracted = 0
with tarfile.open(erase_archive, 'r:gz') as archive:
    for member in archive:
        relative = Path(member.name)
        if relative.is_absolute() or '..' in relative.parts or member.issym() or member.islnk():
            raise RuntimeError('Unsafe archive member; refusing extraction')
        target = erase / relative
        assert target.resolve().is_relative_to(erase.resolve())
        if member.isdir(): target.mkdir(parents=True, exist_ok=True)
        elif member.isfile():
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as incoming, target.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing, 2 * 1024 * 1024)
            extracted += 1
        else: raise RuntimeError('Unsupported special archive member')
candidates = [erase, erase / 'train', erase / 'erase_with_uncertain_dataset/train']
image_root = next((path for path in candidates if all((path / im['file_name']).is_file() for im in clean_train['images'])), None)
assert image_root is not None, 'Erased images do not match official clean train annotations'
(dest / 'train_erased').symlink_to(image_root, target_is_directory=True)
(dest / 'train_original').symlink_to(source / 'train', target_is_directory=True)
(dest / 'test_original').symlink_to(source / 'test', target_is_directory=True)
for split, data in [('train', raw_train), ('test', raw_test)]:
    assert all((source / split / im['file_name']).is_file() for im in data['images'])
    assert len({im['file_name'] for im in data['images']}) == len(data['images'])

def group(im):
    stem = Path(im['file_name']).stem
    return stem.rsplit('_I', 1)[0] if '_I' in stem else stem

groups = {}
for im in raw_train['images']: groups.setdefault(group(im), []).append(im['id'])
ordered = sorted(groups, key=lambda name: hashlib.sha256(('tinyperson-validation-v1-20261017|' + name).encode()).hexdigest())
val_groups = []; count = 0
for name in ordered:
    if count >= math.ceil(0.2 * len(raw_train['images'])): break
    val_groups.append(name); count += len(groups[name])
val_ids = {image_id for name in val_groups for image_id in groups[name]}
train_ids = {im['id'] for im in raw_train['images']} - val_ids
assert train_ids and val_ids and not train_ids & val_ids
assert not ({group(im) for im in raw_train['images'] if im['id'] in train_ids} & set(val_groups))

out_anns = dest / 'annotations'; out_anns.mkdir()
for relative in expected:
    name = ('clean_' if relative.startswith('erase_') else 'raw_') + Path(relative).name
    shutil.copyfile(source / relative, out_anns / name)


def convert(data, ids=None):
    result = dict(data)
    result['images'] = [dict(im) for im in data['images'] if ids is None or im['id'] in ids]
    image_ids = {im['id'] for im in result['images']}
    result['annotations'] = [dict(ann, category_id=0) for ann in data['annotations'] if ann['image_id'] in image_ids]
    result['categories'] = [{'id': 0, 'name': 'person', 'supercategory': 'person'}]
    return result

derived = {'train.json': convert(clean_train, train_ids), 'val.json': convert(raw_train, val_ids),
           'test.json': convert(annotations['annotations/task/tiny_set_test_all.json'])}
for name, data in derived.items():
    with (out_anns / name).open('x') as handle: json.dump(data, handle)
split = {'seed_tag': 'tinyperson-validation-v1-20261017', 'rule': 'Fixed hash ordering of original training video groups until at least 20% of images; no test use or metric-dependent choice.',
         'train_ids': sorted(train_ids), 'val_ids': sorted(val_ids), 'val_groups': val_groups,
         'train_images': len(train_ids), 'val_images': len(val_ids), 'official_test_images': 786}
with (dest / 'split_manifest.json').open('x') as handle: json.dump(split, handle, indent=2)
receipt['split'] = split
receipt['extracted_erased_files'] = extracted
receipt['normal_definition'] = 'Official evaluator: not ignore and not uncertain. Logo is not independently overridden.'
receipt['counts'] = {}
for name, data in derived.items():
    anns = data['annotations']
    receipt['counts'][name] = {'images': len(data['images']), 'annotations': len(anns),
        'positive': sum(not a.get('ignore', False) and not a.get('uncertain', False) for a in anns),
        'ignore': sum(bool(a.get('ignore', False)) for a in anns),
        'uncertain': sum(bool(a.get('uncertain', False)) for a in anns),
        'ignore_and_uncertain': sum(bool(a.get('ignore', False)) and bool(a.get('uncertain', False)) for a in anns)}
receipt['annotation_files'] = [{'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size} for path in sorted(out_anns.iterdir())]
receipt['status'] = 'data_view_ready_no_training_started'
with (dest / 'data_ready.json').open('x') as handle: json.dump(receipt, handle, indent=2)
print(json.dumps(receipt, ensure_ascii=False))
