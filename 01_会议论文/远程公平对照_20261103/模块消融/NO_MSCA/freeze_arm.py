"""Freeze one passed ablation preflight before transport proof and training."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT_PROTOCOL = '8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def main():
    destination = HERE / 'execution_protocol.json'
    if destination.exists():
        raise RuntimeError('An execution protocol already exists')
    preflight = json.loads((HERE / 'inputs/preflight.json').read_text())
    spec = json.loads((HERE / 'arm_spec.json').read_text())
    arm = spec['arm']
    assert arm in ('NO_MPS', 'NO_MSCA')
    assert preflight['arm'] == arm and preflight['status'] == 'PREFLIGHT_PASSED_NO_OPTIMIZER_UPDATE'
    assert preflight['optimizer_updates'] == 0 and preflight['finite_loss_and_gradients'] is True
    assert sha(HERE / 'inputs/initial.pt') == preflight['initial_file_sha256']
    assert sha(HERE / 'inputs/first100.json') == preflight['first100_batches_sha256']
    config = HERE / 'execution_configs' / f'{arm}.yml'
    assert sha(config) == preflight['configuration_sha256']
    paths = set(HERE.rglob('*.py'))
    paths.update(path for directory in ('reference', 'inputs', 'execution_configs')
                 for path in (HERE / directory).rglob('*') if path.is_file())
    paths.update(HERE / name for name in ('arm_spec.json', 'environment.current.json',
                                         'cache_verification.json', 'dataset_bundle_manifest.json'))
    assert all(path.is_file() and not path.is_symlink() for path in paths)
    assert (HERE / 'launch_arm.py') in paths
    plan = {
        'status': 'FROZEN_AUTHORIZED_ABLATION_READY_FOR_TRANSPORT_PROOF',
        'authorized': True, 'arm': arm, 'epochs': 320, 'seed': 0, 'batch_size': 4, 'precision': 'FP32',
        'authorization': 'User requested running the previously described module ablations and supplied the spare server SSH entry.',
        'question': 'Conditional contribution of the removed module within the frozen FULL architecture and recipe.',
        'authorized_additional_arms': ['NO_MPS', 'NO_MSCA'],
        'total_additional_training_epoch_limit': 640, 'evaluation_limit_per_arm': 320,
        'optimizer_update_limit_per_arm': 517440, 'hard_timeout_seconds': 604800,
        'parent_protocol_sha256': PARENT_PROTOCOL,
        'configuration_sha256': sha(config), 'gpu_uuid': spec['gpu_uuid'],
        'architecture': spec['effective_architecture'],
        'initialization': preflight['initialization'],
        'probe_files': {'initial.pt': sha(HERE / 'inputs/initial.pt'),
                        'preflight.json': sha(HERE / 'inputs/preflight.json')},
        'primary_endpoint': 'Actual round 320 EMA before validation-dependent rollback',
        'secondary_endpoint': 'Highest AP across all 320 rounds, earliest round on ties',
        'inference': 'Single-seed conditional removals; report neither a factorial interaction nor a significance test.',
        'frozen_files': [{'path': path.relative_to(HERE).as_posix(), 'sha256': sha(path)}
                         for path in sorted(paths)],
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    with destination.open('x') as stream:
        stream.write(json.dumps(plan, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': plan['status'], 'arm': arm, 'protocol_sha256': sha(destination),
                      'frozen_files': len(plan['frozen_files'])}))


if __name__ == '__main__':
    main()
