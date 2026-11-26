"""Read the current backup receipts and verify the recorded client processes."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import psutil


HERE = Path(__file__).resolve().parent


def client_alive(state):
    try:
        process = psutil.Process(state['monitor_pid'])
        return abs(process.create_time() - state['monitor_create_time']) < 0.001
    except psutil.Error:
        return False


def observe(name, state, backup_root):
    events = [json.loads(line) for line in
              (backup_root / 'monitor_events.jsonl').read_text(encoding='utf-8').splitlines()]
    accepted = next((event for event in reversed(events)
                     if event['status'] == 'EPOCH_BACKUP_VERIFIED'), None)
    reference = accepted or next((event for event in reversed(events)
                                  if event['status'] == 'PROBE_VERIFIED'), events[0])
    status = json.loads((backup_root / 'monitor_status.json').read_text(encoding='utf-8'))
    now = datetime.now(timezone.utc)
    return {
        'arm': name,
        'monitor_alive': client_alive(state),
        'accepted_round': accepted['epoch'] + 1 if accepted else 0,
        'minutes_since_acceptance':
            (now - datetime.fromisoformat(reference['observed_utc'])).total_seconds() / 60,
        'status': status['status'],
        'free_gib': shutil.disk_usage(backup_root).free / 1024**3,
    }


if __name__ == '__main__':
    state = json.loads((HERE / 'current_execution_state.json').read_text(encoding='utf-8'))
    roots = {
        'FULL': HERE.parent / '断点续训_20261105/FULL_30848_20261106/backups',
        'NO_MPS': Path(state['NO_MPS']['backup_root']),
    }
    if 'monitor_pid' in state['NO_MSCA']:
        roots['NO_MSCA'] = Path(state['NO_MSCA']['backup_root'])
    print(json.dumps({name: observe(name, state[name], root) for name, root in roots.items()}))
