"""Verify extracted inputs after matching FULL's frozen AVX2 dispatch."""

import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['ATEN_CPU_CAPABILITY'] = 'avx2'

import prepare_module_inputs_30848 as prep


def main():
    status_path = prep.ROOT / 'preparation.json'
    failed = json.loads(status_path.read_text())
    boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert failed['status'] == 'PREPARATION_FAILED_NO_TRAINING'
    assert 'Runtime differs at torch_build' in failed['traceback']
    assert failed['boot_id'] == boot_id and failed['optimizer_updates'] == 0
    assert not Path('/proc', str(failed['pid'])).exists()
    history = prep.ROOT / 'preparation_attempt1_cpu_dispatch_mismatch.json'
    with history.open('xb') as archive:
        archive.write(status_path.read_bytes())

    protocol_path = prep.PARENT / 'execution_protocol.json'
    assert prep.sha(protocol_path) == prep.PARENT_PROTOCOL
    protocol = json.loads(protocol_path.read_text())
    shared = prep.ROOT / 'shared'
    for item in protocol['frozen_files']:
        assert prep.sha(prep.PARENT / item['path']) == item['sha256'], item['path']
        if item['path'].startswith('source_snapshot/'):
            assert prep.sha(shared / item['path']) == item['sha256'], item['path']
    for name, digest in failed['shared_files'].items():
        assert prep.sha(shared / name) == digest, name

    manifest = json.loads((shared / 'dataset_bundle_manifest.json').read_text())
    assert prep.sha(prep.PARENT / 'verified_dataset.zip') == manifest['zip_sha256']
    for item in manifest['files']:
        path = (prep.CACHE / item['path']).resolve()
        assert path.is_relative_to(prep.CACHE)
        assert path.stat().st_size == item['bytes'], item['path']
        assert prep.sha(path) == item['sha256'], item['path']

    import torch

    expected = protocol['runtime']
    actual = {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'cudnn': torch.backends.cudnn.version(), 'torch_build': torch.__config__.show()}
    assert torch.backends.cpu.get_cpu_capability() == 'AVX2'
    assert all(actual[key] == expected[key] for key in actual)
    packages = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()
                if dist.metadata.get('Name')}
    normalized = lambda values: {name.lower().replace('_', '-'): version
                                 for name, version in values.items()}
    assert normalized(packages) == normalized(expected['packages'])
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    gpu = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.free',
         '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    assert gpu[0] == prep.GPU and gpu[1] == 'NVIDIA GeForce RTX 3090' and int(gpu[3]) >= 20000

    prep.write_json(prep.ROOT / 'environment.current.json', {
        **expected, 'executable': sys.executable, 'packages': packages, 'gpu': gpu,
        'boot_id': boot_id, 'aten_cpu_capability': 'avx2'})
    prep.write_json(prep.ROOT / 'cache_verification.json', {
        'status': 'ALL_INPUT_BYTES_VERIFIED', 'cache_root': str(prep.CACHE),
        'files': len(manifest['files']), 'train_images': 6471, 'validation_images': 548,
        'archive_sha256': manifest['zip_sha256'],
        'manifest_sha256': failed['shared_files']['dataset_bundle_manifest.json'],
        'completed_utc': prep.timestamp()})
    result = {
        'status': 'INPUTS_AND_RUNTIME_READY', 'pid': os.getpid(), 'boot_id': boot_id,
        'completed_files': len(manifest['files']), 'shared_files': failed['shared_files'],
        'gpu': gpu, 'optimizer_updates': 0, 'aten_cpu_capability': 'avx2',
        'prior_preparation': history.name, 'completed_utc': prep.timestamp()}
    prep.write_json(status_path, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
