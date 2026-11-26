"""Read the original dataset paths and check for FULL under this project's mount."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

MOUNT = Path('/25085400120')
ROOT = MOUNT / 'mpsma_fair_20261103_v1'
ORIGINAL = MOUNT / 'DEIM/work/DEIM_1572862469/dataset_visdrone'
MANIFEST_SHA = 'ba463a04fe688d5c3005cb3cf9ca6e13aa699b0b11016e15e7fa9b76b52e2a56'
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect_split(split, manifest):
    folder = ORIGINAL / f'VisDrone2019-DET-{split}'
    images = folder / 'images'
    annotation = folder / 'annotations' / f'{split}.json'
    result = {'image_directory': str(images), 'annotation': str(annotation),
              'image_directory_present': images.is_dir(), 'annotation_present': annotation.is_file()}
    if result['image_directory_present']:
        actual = {p.name: p.stat().st_size for p in images.iterdir()
                  if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()}
        result['image_count'] = len(actual)
        if manifest is not None:
            prefix = f'{split}/images/'
            expected = {f['path'][len(prefix):]: f['bytes'] for f in manifest if f['path'].startswith(prefix)}
            result.update(missing_image_names=len(expected.keys() - actual.keys()),
                          extra_image_names=len(actual.keys() - expected.keys()),
                          differing_image_sizes=sum(actual[name] != expected[name] for name in actual.keys() & expected.keys()))
    if result['annotation_present']:
        result['annotation_sha256'] = sha256(annotation)
        if manifest is not None:
            expected = next(f['sha256'] for f in manifest if f['path'] == f'{split}/annotations/{split}.json')
            result['annotation_matches_manifest'] = result['annotation_sha256'] == expected
    return result


def inspect_full_locations():
    folders = sorted(p for p in MOUNT.iterdir()
                     if p.name.lower().startswith('mpsma') and not p.is_symlink() and p.is_dir()) if MOUNT.is_dir() else []
    locations = []
    for folder in folders[:32]:
        full = folder / 'runs' / 'FULL'
        log = folder / 'FULL.train.log'
        location = {'project_dir': str(folder), 'full_run_present': full.is_dir(),
                    'secondary_queue_present': (folder / 'secondary_queue.json').is_file()}
        if log.is_file() and not log.is_symlink():
            location['full_log_bytes'] = log.stat().st_size
            if location['full_log_bytes'] <= 32 * 1024:
                location['full_log_sha256'] = sha256(log)
        if full.is_dir() and not full.is_symlink():
            location['full_artifacts'] = [p.name for p in sorted(full.iterdir())[:100]
                                          if p.is_file() and not p.is_symlink()]
        locations.append(location)
    return {'total_project_dirs': len(folders), 'locations': locations}


manifest_path = ROOT / 'dataset_bundle_manifest.json'
manifest = None
manifest_sha = None
if manifest_path.is_file():
    raw = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if manifest_sha == MANIFEST_SHA:
        manifest = json.loads(raw)['files']

print(json.dumps({
    'observed_remote_utc': datetime.now(timezone.utc).isoformat(),
    'manifest_sha256': manifest_sha,
    'manifest_matches_frozen': manifest_sha == MANIFEST_SHA,
    'original_data': {split: inspect_split(split, manifest) for split in ('train', 'val')},
    'full_locations': inspect_full_locations(),
    'scope': 'Known original data paths, annotation hashes and image names/sizes, plus first 32 MPSMA-named directories under the project mount.'
}, ensure_ascii=False))
