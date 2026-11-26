"""Read-only observer; no model imports, GPU operations, training or restarts."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time

ROOT = Path('/root/uav_a49_20261101_v1')


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def identity(path):
    expected = read(path)
    if expected is None:
        return None
    proc = Path('/proc', str(expected['pid']))
    result = {'pid': expected['pid'], 'expected_start_ticks': expected['start_ticks'], 'present': proc.exists()}
    if proc.exists():
        try:
            stat = (proc / 'stat').read_text().split(') ', 1)[1].split()
            cmd = (proc / 'cmdline').read_bytes().decode(errors='replace')
            result.update(state=stat[0], same_start_ticks=int(stat[19]) == expected['start_ticks'],
                          owned_command=str(ROOT) in cmd)
        except FileNotFoundError:
            result['present'] = False
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-freeze', action='store_true')
    args = parser.parse_args()
    result = {'observation_only': True, 'observed_server_unix': time.time(),
              'supervisor': read(ROOT / 'supervisor_progress.json'),
              'supervisor_identity': identity(ROOT / 'supervisor_identity.json'),
              'three_arm_result': read(ROOT / 'three_arm_result.json'), 'arms': {}}
    for arm in ['B', 'M', 'Q']:
        folder = ROOT / ('arm_' + arm)
        completed = read(folder / 'completed_epochs.json') or {}
        progress = read(folder / 'progress.json')
        stderr = folder / 'stderr.log'
        result['arms'][arm] = {'identity': identity(folder / 'child_identity.json'),
            'progress': progress, 'result': read(folder / 'result.json'), 'exit': read(folder / 'exit.json'),
            'completed_epochs': completed.get('completed_epochs', 0),
            'evaluations': {str(e['epoch']): e['EMA_validation'] for e in completed.get('events', []) if 'EMA_validation' in e},
            'reported_loss_grad_finite': all(math.isfinite(progress[k]) for k in ['loss', 'grad_norm'] if progress and k in progress),
            'stderr_tail': stderr.read_text()[-2000:] if stderr.is_file() else None}
    keys = ['order_sha256', 'first_eight_actual_input_sha256', 'first_eight_pre_forward_rng_sha256',
            'all_batches_targets_including_ignore_tiles_sha256']
    result['common_epoch_checks'] = {}
    for arm in ['M', 'Q']:
        rows = []
        for epoch in range(1, result['arms'][arm]['completed_epochs'] + 1):
            a = read(ROOT / 'arm_B' / ('epoch%02d_input_flow.json' % epoch))
            b = read(ROOT / ('arm_' + arm) / ('epoch%02d_input_flow.json' % epoch))
            if a and b:
                rows.append({'epoch': epoch, 'all_pass': all(a[k] == b[k] for k in keys)})
        result['common_epoch_checks'][arm + '_vs_B'] = rows
    capacity = read(ROOT / 'capacity_result.json') or {}
    observed = [r['result'] or r['progress'] or {} for r in result['arms'].values()]
    result['observed_counters_lower_bounds'] = {
        'model_forwards': capacity.get('model_forwards', 0) + sum(r.get('full_model_forwards', 0) for r in observed),
        'formal_optimizer_updates': sum(r.get('optimizer_updates', 0) for r in observed),
        'capacity_optimizer_updates': capacity.get('capacity_optimizer_updates', 0),
        'formal_epoch_sum': sum(r['completed_epochs'] for r in result['arms'].values()),
        'extra_GPU_mask_core_reads': capacity.get('extra_mask_core_reads', 0) + sum(r.get('extra_mask_core_reads', 0) for r in observed)}
    if args.verify_freeze:
        protocol = read(ROOT / 'protocol_FROZEN.json')
        checks = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
                  for name, expected in protocol['files_sha256'].items()}
        result['frozen_file_checks'] = {'count': len(checks), 'all_pass': all(checks.values()),
                                        'failed': [k for k, v in checks.items() if not v]}
        result['protocol_sha256'] = hashlib.sha256((ROOT / 'protocol_FROZEN.json').read_bytes()).hexdigest()
    print(json.dumps(result, allow_nan=False), flush=True)
