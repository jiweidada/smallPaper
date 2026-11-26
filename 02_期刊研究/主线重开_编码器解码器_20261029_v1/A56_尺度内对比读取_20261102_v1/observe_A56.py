"""Read only bounded A56 files and exact owned process identities."""
from pathlib import Path
import hashlib
import json

P = Path('/root/uav_a56_20261102_v1')


def read(relative):
    q = P/relative
    return json.loads(q.read_text()) if q.exists() else None


def process(pid, ticks, expected):
    q = Path(f'/proc/{pid}')
    if not q.exists():
        return {'pid': pid, 'same_process_alive': False}
    stat = (q/'stat').read_text().split()
    cmd = (q/'cmdline').read_bytes().replace(b'\0', b' ').decode()
    same = int(stat[21]) == ticks and str(P/expected) in cmd
    return {'pid': pid, 'same_process_alive': same, 'state': stat[2],
            'start_ticks': int(stat[21]), 'expected_script_matched': str(P/expected) in cmd}


def main():
    launch = read('formal_launch_receipt.json')
    processes = read('formal_processes.json')
    result = {'launch': launch, 'processes': processes, 'exit': read('formal_exit.json'),
              'capacity_counts': {'forwards': 9, 'temporary_updates': 6},
              'arms': {}, 'process_identity': []}
    if launch:
        result['process_identity'].append(process(launch['supervisor_PID'],
                                                 launch['supervisor_start_ticks'], 'supervise_A56.py'))
    if processes:
        for r in processes['records']:
            result['process_identity'].append(process(r['child_PID'], r['child_start_ticks'], 'train_A56.py'))
    for arm in ['M', 'Q']:
        progress = read(f'arm_{arm}/progress.json')
        if progress:
            progress = {k: v for k, v in progress.items() if k not in ['contrast_diagnostics', 'contrast_diagnostics_EMA']}
        completed = read(f'arm_{arm}/completed_epochs.json')
        flow = []
        if completed:
            for event in completed['events']:
                flow.append({k: event[k] for k in ['epoch', 'optimizer_updates', 'mean_loss', 'train_seconds', 'EMA_validation'] if k in event})
        result['arms'][arm] = {'progress': progress, 'initialization': read(f'arm_{arm}/initialization.json'),
                               'epochs': flow, 'result': read(f'arm_{arm}/result.json')}
    result['protocol_sha256'] = hashlib.sha256((P/'protocol_FROZEN_v1.json').read_bytes()).hexdigest()
    print(json.dumps(result))


if __name__ == '__main__':
    main()
