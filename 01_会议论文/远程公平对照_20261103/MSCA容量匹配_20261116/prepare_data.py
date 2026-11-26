"""Restore the frozen VisDrone bundle into a dedicated per-instance cache."""

import hashlib
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(volume, preserved):
    manifest = json.loads((preserved / 'dataset_bundle_manifest.json').read_text())
    assert len(manifest['files']) == 7021
    candidates = (
        volume / 'mpsma_fresh_pair_20261112_v1/inputs/verified_dataset.zip',
        volume / 'mpsma_fair_20261103_v1/verified_dataset.zip',
    )
    archive = next((p for p in candidates if p.is_file() and p.stat().st_size == manifest['bytes']), None)
    if archive is None:
        raise FileNotFoundError('Verified dataset archive missing from both preserved locations')
    assert sha(archive) == manifest['zip_sha256'], 'Dataset archive hash differs'
    cache = Path('/tmp/mpsma_msca_pair_20261116_data')
    assert shutil.disk_usage(cache.parent).free > 8 * 1024**3
    cache.mkdir(exist_ok=True)
    root = cache.resolve()
    assert root == cache
    with ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == {r['path'] for r in manifest['files']}
        for item in manifest['files']:
            path = root / item['path']
            assert path.resolve().is_relative_to(root)
            if path.is_file() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256']:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + '.partial')
            with bundle.open(item['path']) as source, temporary.open('wb') as target:
                shutil.copyfileobj(source, target)
            assert temporary.stat().st_size == item['bytes'] and sha(temporary) == item['sha256']
            temporary.replace(path)
    print(json.dumps({'cache': str(cache), 'archive': str(archive), 'archive_sha256': manifest['zip_sha256'],
                      'verified_files': len(manifest['files'])}))


if __name__ == '__main__':
    prepare(Path(sys.argv[1]), Path(sys.argv[2]))
