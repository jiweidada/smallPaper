"""Install one isolated arm from verified shared inputs and a closed code bundle."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path('/25085400120/mpsma_fair_20261103_v1/module_ablation_20261107_v1')
FILES = {'fair_helpers.py', 'train_runner.py', 'backup_protocol.py', 'backup_monitor.py',
         'ablation_model.py', 'preflight_arm.py'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['NO_MPS', 'NO_MSCA'], required=True)
    parser.add_argument('--bundle-sha256', required=True)
    args = parser.parse_args()
    preparation = json.loads((ROOT / 'preparation.json').read_text())
    assert preparation['status'] == 'INPUTS_AND_RUNTIME_READY'
    archive = ROOT / 'module_runtime.zip'
    assert sha(archive) == args.bundle_sha256
    arm = args.arm
    target = ROOT / (arm.lower() + '_20261107_v1')
    target.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == FILES
        for name in sorted(FILES):
            with (target / name).open('xb') as output:
                output.write(bundle.read(name))
    shared = ROOT / 'shared'
    shutil.copytree(shared / 'source_snapshot', target / 'source_snapshot')
    reference = target / 'reference'
    reference.mkdir()
    for name in ('FULL.initial.pt', 'BASE.initial.pt', 'reference.first100.json'):
        shutil.copyfile(shared / name, reference / name)
    for name in ('source_loader.py', 'dataset_bundle_manifest.json'):
        shutil.copyfile(shared / name, target / name)
    for name in ('environment.current.json', 'cache_verification.json'):
        shutil.copyfile(ROOT / name, target / name)
    config = json.loads((shared / 'FULL.reference.yml').read_text())
    cache = json.loads((ROOT / 'cache_verification.json').read_text())['cache_root']
    for split in ('train', 'val'):
        dataset = config[f'{split}_dataloader']['dataset']
        dataset['img_folder'] = f'{cache}/{split}/images'
        dataset['ann_file'] = f'{cache}/{split}/annotations/{split}.json'
    config['output_dir'] = str(target / 'run')
    config['ablation'] = {
        'arm': arm,
        'construction': 'Frozen FULL constructor followed by the hash-bound ablation_model.apply adapter',
        'effective_stem': 'hgnetv2.StemBlock with SiLU' if arm == 'NO_MPS' else 'EnhancedStemBlock',
        'msca': 'retained' if arm == 'NO_MPS' else 'three gates replaced with Identity; projections retained',
        'activation': 'silu',
        'reference_config_sha256': preparation['shared_files']['FULL.reference.yml'],
    }
    configs = target / 'execution_configs'
    configs.mkdir()
    (configs / f'{arm}.yml').write_text(json.dumps(config, indent=2) + '\n')
    specification = {'arm': arm, 'gpu_uuid': preparation['gpu'][0], 'gpu_model': preparation['gpu'][1],
                     'reference_initialization': preparation['shared_files'],
                     'effective_architecture': config['ablation'], 'dataset_files': 7021,
                     'optimizer_update_limit': 517440, 'evaluation_limit': 320}
    (target / 'arm_spec.json').write_text(json.dumps(specification, indent=2) + '\n')
    print(json.dumps({'status': 'ARM_INSTALLED_NO_TRAINING', 'arm': arm, 'root': str(target),
                      'bundle_sha256': args.bundle_sha256, 'configuration_sha256': sha(configs / f'{arm}.yml')}))


if __name__ == '__main__':
    main()
