"""Freeze once after bounded capacity success; launch only the authorized queue."""
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time
from runtime_common import R, digest, save_json


def ticks(pid):
    return int(pathlib.Path('/proc/%s/stat' % pid).read_text().split(') ', 1)[1].split()[19])


def main():
    assert not (R / 'execution_protocol_FROZEN.json').exists()
    assert not (R / 'supervisor_started.json').exists()
    cap = json.loads((R / 'capacity_preflight_result.json').read_text())
    assert cap['status'] == 'PASS' and cap['fixed_microbatch'] == 8
    assert cap['new_student_forwards'] == 8 and cap['new_teacher_forwards'] == 4 and cap['optimizer_updates'] == 5
    assert cap['M_Q_actual_teacher_stream_identical'] and cap['teacher_frozen_unchanged']
    assert cap['auxiliary_gradient_reaches_stem'] and cap['zero_output_influence_bitwise_equal']
    with (R / 'formal_launch_once.lock').open('x') as f:
        f.write(str(os.getpid()))
    spec = json.loads((R / 'pre_capacity_spec.json').read_text())
    files = list(R.glob('*.py'))
    files += [p for p in (R / 'source').rglob('*') if p.is_file() and p.suffix in ['.py', '.yml', '.yaml']]
    files += [R / n for n in ['pre_capacity_spec.json', 'common_init_10class.pth', 'teacher_B_epoch12.pth',
        'train_data_manifest.json', 'val_data_manifest.json', 'cpu_interface_validation.json',
        'capacity_preflight_result.json', 'remote_asset_preparation.json', 'local_source_preparation.json']]
    manifest = {str(p.relative_to(R)): digest(p) for p in sorted(set(files))}
    assert manifest['common_init_10class.pth'] == spec['common_init_sha256']
    assert manifest['teacher_B_epoch12.pth'] == spec['teacher_checkpoint_sha256']
    spec.update(status='FROZEN_BEFORE_FORMAL_TRAINING', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                files_sha256=manifest, capacity_student_forwards=8, capacity_teacher_forwards=4,
                capacity_optimizer_updates=5, formal_training_before_freeze=0,
                all_capacity_weights_discarded=True)
    save_json(R / 'execution_protocol_FROZEN.json', spec)
    with (R / 'supervisor_stdout.log').open('w') as stdout, (R / 'supervisor_stderr.log').open('w') as stderr:
        p = subprocess.Popen([sys.executable, '-u', str(R / 'supervise_three_arms.py')],
                             cwd=str(R), stdout=stdout, stderr=stderr, start_new_session=True)
    result = {'status': 'LAUNCHED', 'pid': p.pid, 'start_ticks': ticks(p.pid),
              'protocol_sha256': digest(R / 'execution_protocol_FROZEN.json'), 'queue': ['B', 'M', 'Q'],
              'max_epochs_each': 12, 'automatic_retry': False, 'other_direction_training_queued': False}
    save_json(R / 'launch_receipt.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
