"""Read-only status for the one authorized diagnostic."""
from pathlib import Path
import json
import subprocess

ROOT = Path('/25085400120/mpsma_fair_20261103_v1/diagnostics/mpsma_diag_base_apdrop_20261105_v1')

def read_if(path):
    if not path.is_file() or path.is_symlink():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as e:
        return {'read_error': type(e).__name__}

def process(pid, ticks=None):
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        stat = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        actual = stat[19]
        return {'pid': pid, 'start_ticks': actual, 'state': stat[0],
                'identity_matches': ticks is None or str(ticks) == actual}
    except (OSError, IndexError):
        return {'pid': pid, 'present': False, 'identity_matches': False}

def main():
    data = {'root_exists': ROOT.is_dir() and not ROOT.is_symlink()}
    if data['root_exists']:
        for key, name in [('setup_progress', 'setup_progress.json'), ('setup_ready', 'setup_ready.json'),
                          ('setup_failure', 'setup_failure.json'), ('controller_state', 'controller_state.json'),
                          ('worker_live', 'worker_live.json'), ('result', 'diagnostic_result.json'),
                          ('setup_launch', 'setup_launch.json'), ('controller_launch', 'controller_launch.json')]:
            value = read_if(ROOT / name)
            if value is not None:
                if key == 'setup_ready':
                    value = {k: value.get(k) for k in ['status', 'completed_utc', 'dataset_files_verified',
                             'source_files_copied', 'gpu', 'code_manifest_sha256', 'model_executions',
                             'evaluator_executions', 'optimizer_updates']}
                data[key] = value
        for label in ['setup_launch', 'controller_launch']:
            entry = data.get(label) or {}
            data[label + '_process'] = process(entry.get('pid'), entry.get('start_ticks'))
        state = data.get('controller_state') or {}
        data['controller_process'] = process(state.get('controller_pid'), state.get('controller_start_ticks'))
        data['child_process'] = process(state.get('child_pid'), state.get('child_start_ticks'))
        phases = []
        parent = ROOT / 'phases'
        names = ['initial_ema', 'initial_raw'] + ['epoch_%04d' % e for e in range(214, 225)] + ['terminal_raw']
        for name in names:
            stage = parent / name
            if stage.is_dir() and not stage.is_symlink():
                manifest = read_if(stage / 'manifest.json')
                phases.append({'phase': name, 'manifest': manifest,
                               'ack_uploading': (stage / 'ack.uploading').exists(),
                               'ack_ready': (stage / 'ack.ready').exists()})
        data['phases'] = phases
    try:
        data['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.used,utilization.gpu',
                                               '--format=csv,noheader,nounits'], text=True, timeout=8).strip()
    except Exception as e:
        data['gpu_error'] = type(e).__name__
    print(json.dumps(data, ensure_ascii=False))

if __name__ == '__main__':
    main()
