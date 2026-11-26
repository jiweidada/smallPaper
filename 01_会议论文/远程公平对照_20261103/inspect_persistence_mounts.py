"""Read mount metadata for experiment-relevant paths only; do not write data."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import re
import platform

def unescape(s):
    return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m.group(1), 8)), s)

mounts = []
for line in Path('/proc/self/mountinfo').read_text().splitlines():
    before, after = line.split(' - ', 1)
    left, right = before.split(), after.split()
    mounts.append({'mountpoint': unescape(left[4]), 'filesystem': right[0],
                   'source': unescape(right[1]), 'mount_options': left[5]})

def inspect(path):
    p = Path(path)
    current = p
    while not current.exists() and current != current.parent:
        current = current.parent
    resolved = str(current.resolve())
    candidates = [m for m in mounts if m['mountpoint'] == '/' or resolved == m['mountpoint']
                  or resolved.startswith(m['mountpoint'].rstrip('/') + '/')]
    mount = max(candidates, key=lambda m: len(m['mountpoint']))
    return {'path': path, 'exists': p.exists(), 'nearest_existing_path': str(current),
            'resolved_existing_path': resolved, 'device_id': current.stat().st_dev,
            'backing_mount': mount}

print(json.dumps({'observed_remote_utc': datetime.now(timezone.utc).isoformat(), 'hostname': platform.node(),
                  'paths': [inspect(p) for p in ['/25085400120', '/25085400120/mpsma_fair_20261103_v1',
                                                '/tmp', '/workspace']],
                  'scope': 'Current mount metadata only; no write test, backup guarantee, model, evaluation or training.'}))
