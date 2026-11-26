"""Prepare verified inputs on the authorized spare host, without training."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zipfile

PARENT = Path('/25085400120/mpsma_fair_20261103_v1')
ROOT = PARENT / 'module_ablation_20261107_v1'
CACHE = Path('/tmp/mpsma_module_ablation_20261107_data')
PARENT_PROTOCOL = '8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'
GPU = 'GPU-683147c9-ac52-967b-67f0-9eb4a813b418'


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    ROOT.mkdir(exist_ok=False)
    state = {'status': 'VERIFYING_FROZEN_INPUTS', 'pid': os.getpid(), 'started_utc': timestamp(),
             'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'optimizer_updates': 0}
    status = ROOT / 'preparation.json'
    write_json(status, state)
    try:
        assert sha(PARENT / 'execution_protocol.json') == PARENT_PROTOCOL
        protocol = json.loads((PARENT / 'execution_protocol.json').read_text())
        for item in protocol['frozen_files']:
            assert sha(PARENT / item['path']) == item['sha256'], item['path']
        shared = ROOT / 'shared'
        shared.mkdir()
        for item in protocol['frozen_files']:
            relative = Path(item['path'])
            if relative.parts[0] == 'source_snapshot':
                target = shared / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(PARENT / relative, target)
                assert sha(target) == item['sha256']
        copies = {
            'source_loader.py': 'source_loader.py',
            'fair_runner.py': 'fair_runner.reference.py',
            'preflight/BASE.initial.pt': 'BASE.initial.pt',
            'preflight/FULL.initial.pt': 'FULL.initial.pt',
            'preflight/BASE.first100.json': 'reference.first100.json',
            'execution_configs/FULL.yml': 'FULL.reference.yml',
            'dataset_bundle_manifest.json': 'dataset_bundle_manifest.json',
        }
        bindings = {}
        for source, name in copies.items():
            shutil.copyfile(PARENT / source, shared / name)
            assert sha(shared / name) == sha(PARENT / source)
            bindings[name] = sha(shared / name)
        manifest = json.loads((shared / 'dataset_bundle_manifest.json').read_text())
        state.update(status='VERIFYING_DATA_ARCHIVE', shared_files=bindings)
        write_json(status, state)
        assert shutil.disk_usage('/tmp').free > manifest['bytes'] * 2 + 5 * 1024**3
        assert sha(PARENT / 'verified_dataset.zip') == manifest['zip_sha256']
        CACHE.mkdir(exist_ok=False)
        with zipfile.ZipFile(PARENT / 'verified_dataset.zip') as archive:
            assert set(archive.namelist()) == {item['path'] for item in manifest['files']}
            for count, item in enumerate(manifest['files'], 1):
                destination = (CACHE / item['path']).resolve()
                assert destination.is_relative_to(CACHE)
                destination.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size = 0
                with archive.open(item['path']) as source, destination.open('xb') as target:
                    for block in iter(lambda: source.read(4 * 1024**2), b''):
                        target.write(block)
                        digest.update(block)
                        size += len(block)
                assert size == item['bytes'] and digest.hexdigest() == item['sha256'], item['path']
                if count % 1000 == 0:
                    state.update(status='EXTRACTING_VERIFIED_DATA', completed_files=count, updated_utc=timestamp())
                    write_json(status, state)
                    print(json.dumps({'completed_files': count}), flush=True)
        expected = protocol['runtime']
        import torch
        for actual, key in ((sys.version, 'python'), (torch.__version__, 'torch'),
                            (torch.version.cuda, 'cuda'), (torch.backends.cudnn.version(), 'cudnn'),
                            (torch.__config__.show(), 'torch_build')):
            assert actual == expected[key], 'Runtime differs at ' + key
        packages = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()
                    if dist.metadata.get('Name')}
        normalized = lambda values: {name.lower().replace('_', '-'): version for name, version in values.items()}
        assert normalized(packages) == normalized(expected['packages']), 'Dependency versions differ'
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.free',
                                       '--format=csv,noheader,nounits'], text=True).strip().split(', ')
        assert gpu[0] == GPU and gpu[1] == 'NVIDIA GeForce RTX 3090' and int(gpu[3]) >= 20000
        runtime = {**expected, 'executable': sys.executable, 'packages': packages,
                   'gpu': gpu, 'boot_id': state['boot_id']}
        write_json(ROOT / 'environment.current.json', runtime)
        write_json(ROOT / 'cache_verification.json', {
            'status': 'ALL_INPUT_BYTES_VERIFIED', 'cache_root': str(CACHE), 'files': len(manifest['files']),
            'train_images': 6471, 'validation_images': 548, 'archive_sha256': manifest['zip_sha256'],
            'manifest_sha256': bindings['dataset_bundle_manifest.json'], 'completed_utc': timestamp()})
        state.update(status='INPUTS_AND_RUNTIME_READY', completed_files=len(manifest['files']),
                     gpu=gpu, completed_utc=timestamp())
        write_json(status, state)
        print(json.dumps(state), flush=True)
    except BaseException:
        state.update(status='PREPARATION_FAILED_NO_TRAINING', traceback=traceback.format_exc(), ended_utc=timestamp())
        write_json(status, state)
        raise


if __name__ == '__main__':
    main()
