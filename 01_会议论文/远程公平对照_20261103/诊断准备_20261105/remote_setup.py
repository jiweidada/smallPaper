"""Prepare the authorized diagnostic in an isolated directory; no model execution."""
from pathlib import Path, PurePosixPath
import copy
import importlib.metadata as metadata
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import traceback
import zipfile

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
import diagnostic_contract as c

def safe_relative(value):
    p = PurePosixPath(value)
    c.require(not p.is_absolute() and p.parts and '..' not in p.parts, 'Unsafe relative path')
    return p

def copy_verified(source, dest, expected):
    c.require(source.is_file() and not source.is_symlink() and c.sha(source) == expected,
              'Source hash mismatch: ' + str(source))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as src, dest.open('xb') as out:
        shutil.copyfileobj(src, out, length=4 * 1024 * 1024)
        out.flush()
        os.fsync(out.fileno())
    c.require(c.sha(dest) == expected, 'Copied file hash mismatch: ' + str(dest))

def progress(status, **values):
    c.atomic_json(c.ROOT / 'setup_progress.json', {'status': status, 'updated_utc': c.now(), **values})

def main():
    c.auth_check()
    c.package_check()
    with (c.ROOT / 'setup.one_shot.lock').open('x', encoding='utf-8') as f:
        f.write(str(os.getpid()) + '\n')
        f.flush()
        os.fsync(f.fileno())
    c.fsync_dir(c.ROOT)
    try:
        progress('VERIFYING_INPUTS_NO_MODEL_EXECUTION', model_executions=0)
        c.require(c.sha(c.ORIGINAL / 'execution_protocol.json') == c.PROTOCOL_SHA, 'Original protocol changed')
        protocol = c.read(c.ORIGINAL / 'execution_protocol.json')
        frozen = {item['path']: item['sha256'] for item in protocol['frozen_files']}
        c.require(len(frozen) == len(protocol['frozen_files']) == 247, 'Unexpected frozen source list')
        for relative, expected in frozen.items():
            rel = safe_relative(relative)
            source = c.ORIGINAL.joinpath(*rel.parts)
            c.require(source.is_file() and not source.is_symlink() and c.sha(source) == expected,
                      'Frozen file changed: ' + relative)
        c.require(c.sha(c.ORIGINAL / 'verified_dataset.zip') == c.DATA_SHA, 'Dataset archive changed')
        c.require(c.sha(c.ORIGINAL / 'dataset_bundle_manifest.json') == c.DATA_MANIFEST_SHA, 'Dataset manifest changed')
        c.require(c.sha(c.ORIGINAL / 'runs/BASE/best.pt') == c.CHECKPOINT_SHA, 'Checkpoint changed')
        checkpoint_metrics = c.read(c.ORIGINAL / 'runs/BASE/best.metrics.json')
        c.require(checkpoint_metrics['epoch'] == c.START_EPOCH and
                  checkpoint_metrics['checkpoint_sha256'] == c.CHECKPOINT_SHA and
                  checkpoint_metrics['protocol_sha256'] == c.PROTOCOL_SHA and
                  checkpoint_metrics['config_sha256'] == c.BASE_CONFIG_SHA, 'Checkpoint metrics binding changed')

        for relative, expected in frozen.items():
            if relative.startswith('source_snapshot/') or relative in ['fair_runner.py', 'source_loader.py']:
                rel = safe_relative(relative)
                copy_verified(c.ORIGINAL.joinpath(*rel.parts), c.ROOT.joinpath(*rel.parts), expected)
        copy_verified(c.ORIGINAL / 'runs/BASE/best.pt', c.ROOT / 'input/best.pt', c.CHECKPOINT_SHA)
        metrics_source = c.ORIGINAL / 'runs/BASE/best.metrics.json'
        copy_verified(metrics_source, c.ROOT / 'input/best.metrics.json', c.sha(metrics_source))
        copy_verified(c.ORIGINAL / 'dataset_bundle_manifest.json', c.ROOT / 'dataset_bundle_manifest.json', c.DATA_MANIFEST_SHA)
        copy_verified(c.ORIGINAL / 'execution_protocol.json', c.ROOT / 'original_protocol.json', c.PROTOCOL_SHA)

        manifest = c.read(c.ROOT / 'dataset_bundle_manifest.json')
        c.require(manifest['zip_sha256'] == c.DATA_SHA and len(manifest['files']) == 7021, 'Unexpected dataset manifest')
        total_uncompressed = sum(item['bytes'] for item in manifest['files'])
        c.require(not c.CACHE.exists() and not c.CACHE.is_symlink(), 'Diagnostic cache already exists')
        c.require(shutil.disk_usage('/tmp').free > total_uncompressed + 2 * 1024**3, 'Insufficient temporary space')
        c.CACHE.mkdir(exist_ok=False)
        expected_names = {item['path'] for item in manifest['files']}
        c.require(len(expected_names) == len(manifest['files']), 'Duplicate dataset manifest paths')
        with zipfile.ZipFile(c.ORIGINAL / 'verified_dataset.zip') as archive:
            c.require(set(archive.namelist()) == expected_names and len(archive.namelist()) == len(expected_names),
                      'Dataset archive membership mismatch')
            for i, item in enumerate(manifest['files'], 1):
                rel = safe_relative(item['path'])
                dest = c.CACHE.joinpath(*rel.parts)
                c.require(dest.resolve().is_relative_to(c.CACHE.resolve()), 'Dataset destination escapes cache')
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item['path']) as src, dest.open('xb') as out:
                    shutil.copyfileobj(src, out, length=4 * 1024 * 1024)
                c.require(dest.stat().st_size == item['bytes'] and c.sha(dest) == item['sha256'],
                          'Extracted dataset mismatch: ' + item['path'])
                if i % 1000 == 0:
                    progress('EXTRACTING_VERIFIED_DATA_NO_MODEL_EXECUTION', files_verified=i)

        original_config = c.read(c.ORIGINAL / 'execution_configs/BASE.yml')
        config = copy.deepcopy(original_config)
        for split in ['train', 'val']:
            config[split + '_dataloader']['dataset']['ann_file'] = str(c.CACHE / split / 'annotations' / (split + '.json'))
            config[split + '_dataloader']['dataset']['img_folder'] = str(c.CACHE / split / 'images')
        config['output_dir'] = str(c.ROOT / 'run_output')
        restored = copy.deepcopy(config)
        for split in ['train', 'val']:
            for key in ['ann_file', 'img_folder']:
                restored[split + '_dataloader']['dataset'][key] = original_config[split + '_dataloader']['dataset'][key]
        restored['output_dir'] = original_config['output_dir']
        c.require(restored == original_config, 'Unexpected configuration change')
        (c.ROOT / 'execution_configs').mkdir(exist_ok=False)
        c.atomic_json(c.ROOT / 'execution_configs/BASE.yml', config, new=True)

        # Check the exact installed distribution set before importing torch.
        def normalize(items):
            return {k.lower().replace('_', '-'): v for k, v in items.items()}
        expected_packages = protocol['runtime']['packages']
        installed = {d.metadata['Name']: d.version for d in metadata.distributions(path=[sysconfig.get_path('purelib')])}
        before = {d.metadata['Name']: d.version for d in metadata.distributions()}
        c.require(normalize(installed) == normalize(before) == normalize(expected_packages), 'Installed package mismatch')
        import torch
        import torchvision
        direct = {name: metadata.version(name) for name in expected_packages}
        c.require(normalize(direct) == normalize(installed), 'Direct package resolution mismatch')
        expected = protocol['runtime']
        actual = {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'cudnn': torch.backends.cudnn.version(), 'torch_build': torch.__config__.show()}
        for key in actual:
            c.require(actual[key] == expected[key], 'Runtime mismatch: ' + key)
        c.require(torch.cuda.is_available(), 'CUDA unavailable')
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,memory.used',
                                       '--format=csv,noheader,nounits'], text=True).strip().split(', ')
        c.require(len(gpu) == 5 and gpu[0] == 'NVIDIA GeForce RTX 3090' and gpu[2] == '550.100'
                  and int(gpu[3]) == 24576 and int(gpu[4]) < 256, 'GPU is not the expected free 3090')
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                            text=True).strip()
        c.require(not processes, 'GPU already has a compute process')
        mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(c.ROOT), '-o',
                                                     'TARGET,SOURCE,FSTYPE,OPTIONS'], text=True, timeout=10))
        c.require(len(mount.get('filesystems', [])) == 1, 'Could not identify filesystem mount')
        c.require(shutil.disk_usage(c.ROOT).free > 5 * 1024**3, 'Insufficient filesystem free space')
        probe = c.ROOT / 'isolated_write_probe.bin'
        with probe.open('xb') as f:
            f.write(b'\0' * (1024 * 1024))
            f.flush()
            os.fsync(f.fileno())
        c.fsync_dir(c.ROOT)
        c.require(probe.stat().st_size == 1024 * 1024, 'Write probe incomplete')
        # Keep the tiny probe in the isolated directory as evidence; do not delete it.
        for name in ['phases', 'locks', 'logs']:
            (c.ROOT / name).mkdir(exist_ok=False)
        c.fsync_dir(c.ROOT)
        ready = {'status': 'SETUP_READY_NO_MODEL_EXECUTION', 'completed_utc': c.now(),
                 'run_id': c.RUN_ID, 'authorization_sha256': c.sha(c.ROOT / 'authorization.json'),
                 'code_manifest_sha256': c.sha(c.ROOT / 'code_manifest.json'),
                 'original_protocol_sha256': c.PROTOCOL_SHA, 'original_config_sha256': c.BASE_CONFIG_SHA,
                 'diagnostic_config_sha256': c.sha(c.ROOT / 'execution_configs/BASE.yml'),
                 'checkpoint_sha256': c.CHECKPOINT_SHA, 'dataset_sha256': c.DATA_SHA,
                 'dataset_manifest_sha256': c.DATA_MANIFEST_SHA, 'cache': str(c.CACHE),
                 'dataset_files_verified': len(manifest['files']), 'source_files_copied': sum(k.startswith('source_snapshot/') for k in frozen),
                 'runtime': actual, 'gpu': {'name': gpu[0], 'uuid': gpu[1], 'driver': gpu[2],
                                            'memory_total_mib': int(gpu[3]), 'memory_used_mib': int(gpu[4])},
                 'filesystem_mount': mount['filesystems'][0],
                 'filesystem_free_bytes': shutil.disk_usage(c.ROOT).free,
                 'quota_note': 'Filesystem free space and a 1 MiB write do not establish user quota or future retention.',
                 'model_executions': 0, 'evaluator_executions': 0, 'optimizer_updates': 0}
        c.atomic_json(c.ROOT / 'setup_ready.json', ready, new=True)
        progress('SETUP_READY_NO_MODEL_EXECUTION', files_verified=len(manifest['files']))
        print(json.dumps({'status': ready['status'], 'gpu': ready['gpu'], 'dataset_files_verified': ready['dataset_files_verified']}), flush=True)
    except BaseException:
        c.atomic_json(c.ROOT / 'setup_failure.json', {'status': 'SETUP_FAILED_NO_AUTOMATIC_RETRY',
                                                      'traceback': traceback.format_exc(), 'failed_utc': c.now()}, new=True)
        raise

if __name__ == '__main__':
    main()
