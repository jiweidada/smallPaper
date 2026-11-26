"""Restore only the frozen data bytes at the original configured cache path."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
import zipfile

BASE = Path('/25085400120/mpsma_fair_20261103_v1')
ARM = BASE / 'module_ablation_20261107_v1/no_msca_20261107_v1'
HERE = ARM / 'recovery30715_preparation'
ARCHIVE = BASE / 'verified_dataset.zip'
CACHE = Path('/tmp/mpsma_module_ablation_20261107_data')
STAGING = Path('/tmp/mpsma_module_ablation_20261107_data_restore30715')
EXPECTED_MANIFEST = 'ba463a04fe688d5c3005cb3cf9ca6e13aa699b0b11016e15e7fa9b76b52e2a56'
EXPECTED_ARCHIVE = 'fdde141711867fa1bc2de3a1e6342cca257eb056e2a1700795eb04068aa90266'


def digest(path):
    value = sha256()
    with path.open('rb') as handle:
        while block := handle.read(4 * 1024 ** 2):
            value.update(block)
    return value.hexdigest()


def status(value):
    value.update(observed_utc=datetime.now(timezone.utc).isoformat(), training_started=False)
    temporary = HERE / 'data_status.json.partial'
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, HERE / 'data_status.json')


assert HERE.is_dir() and not CACHE.exists() and not STAGING.exists()
assert digest(ARM / 'execution_protocol.json') == '3e5e8937e0889aa360ffe3f7acb722ad1ab5f8e6c66cd6b3991a71516212dcfc'
manifest_path = ARM / 'dataset_bundle_manifest.json'
assert digest(manifest_path) == EXPECTED_MANIFEST
manifest = json.loads(manifest_path.read_text())
assert manifest['zip_sha256'] == EXPECTED_ARCHIVE and len(manifest['files']) == 7021
assert shutil.disk_usage('/tmp').free > 10 * 1024 ** 3
status({'status': 'VERIFYING_PRESERVED_ARCHIVE'})
assert ARCHIVE.stat().st_size == manifest['bytes'] and digest(ARCHIVE) == EXPECTED_ARCHIVE
STAGING.mkdir(exist_ok=False)
try:
    with zipfile.ZipFile(ARCHIVE) as bundle:
        entries = bundle.infolist()
        expected = {item['path']: item for item in manifest['files']}
        assert len(entries) == len(expected) and {e.filename for e in entries} == set(expected)
        for index, entry in enumerate(entries):
            relative = PurePosixPath(entry.filename)
            assert not relative.is_absolute() and '..' not in relative.parts
            assert not stat.S_ISLNK(entry.external_attr >> 16)
            target = (STAGING / entry.filename).resolve()
            assert target.is_relative_to(STAGING.resolve()) and not entry.is_dir()
            assert entry.file_size == expected[entry.filename]['bytes']
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(entry) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output, length=4 * 1024 ** 2)
            assert target.stat().st_size == expected[entry.filename]['bytes']
            assert digest(target) == expected[entry.filename]['sha256']
            if index % 500 == 0:
                status({'status': 'RESTORING_AND_VERIFYING_EXACT_BYTES', 'verified_files': index + 1, 'total_files': 7021})
    assert not CACHE.exists()
    os.replace(STAGING, CACHE)
    report = {'status': 'ORIGINAL_CACHE_PATH_7021_FROZEN_FILES_RESTORED_AND_VERIFIED',
        'archive': str(ARCHIVE), 'archive_sha256': EXPECTED_ARCHIVE,
        'manifest_sha256': EXPECTED_MANIFEST, 'cache_root': str(CACHE),
        'files': 7021, 'train_images': 6471, 'validation_images': 548,
        'source_config_changed': False, 'original_cache_verification_record_changed': False,
        'restoration_source_sha256': digest(Path(__file__)), 'training_started': False,
        'completed_utc': datetime.now(timezone.utc).isoformat()}
    with (HERE / 'data_restoration.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    status(report)
    print(json.dumps(report), flush=True)
except Exception as error:
    status({'status': 'RESTORATION_FAILED_STAGING_PRESERVED', 'error_type': type(error).__name__, 'error': str(error)})
    raise
