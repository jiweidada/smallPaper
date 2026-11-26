"""Read-only inspection after SSH reconnected to an unexpected runtime."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import subprocess

ROOT = Path('/25085400120/mpsma_fair_20261103_v1')
ENV = Path('/25085400120/envs/deim/bin/python')

def identity(pid, expected):
    p = Path('/proc') / str(pid)
    try:
        stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'state': stat[0], 'start_ticks': stat[19],
                'matches_prior_identity': stat[19] == expected,
                'live': stat[0] not in ['Z', 'X']}
    except (FileNotFoundError, ProcessLookupError):
        return {'pid': pid, 'present': False, 'matches_prior_identity': False, 'live': False}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def inspect_file(p):
    d = {'path': str(p), 'exists': p.exists()}
    if p.is_file():
        d.update(bytes=p.stat().st_size, sha256=sha(p))
        if p.name in ['queue.json', 'secondary_queue.json', 'receipt.json']:
            try:
                obj = json.loads(p.read_text())
                d['selected_fields'] = {k: obj.get(k) for k in ['status', 'arm', 'supervisor_pid',
                                         'supervisor_start_ticks', 'completed_epochs', 'actual_loop_epoch',
                                         'optimizer_step_attempts', 'protocol_sha256'] if k in obj}
            except (ValueError, OSError):
                d['json_readable'] = False
    return d

def run(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return {'exit_code': p.returncode, 'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'error': type(error).__name__, 'message': str(error)}

try:
    boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
except OSError:
    boot_id = None
paths = [Path('/25085400120'), ROOT, ENV, ROOT / 'execution_protocol.json',
         ROOT / 'queue.json', ROOT / 'secondary_queue.json',
         ROOT / 'runs/BASE/receipt.json', ROOT / 'runs/FULL/receipt.json',
         Path('/tmp/mpsma_fair_20261103_data')]
out = {'observed_remote_utc': datetime.now(timezone.utc).isoformat(),
       'hostname': platform.node(), 'boot_id': boot_id, 'uid': os.getuid(),
       'gpu': run(['nvidia-smi', '--query-gpu=name,uuid,memory.total,memory.used', '--format=csv,noheader,nounits']),
       'paths': [inspect_file(p) for p in paths],
       'previous_processes': [identity(pid, ticks) for pid, ticks in
                              [(1193, '39931198'), (1194, '39931668'),
                               (564, '42790973'), (567, '42791005')]],
       'scope': 'Read-only hardware, target paths and original process identities; no model, evaluator, training or data changes.'}
print(json.dumps(out))
