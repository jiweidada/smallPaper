"""Read-only inspection of the existing setup process and host identity."""
from pathlib import Path
import json
import os
import time

pid = 255
base = Path('/proc') / str(pid)
result = {'observed_utc': time.time()}
for key, path in [('boot_id', Path('/proc/sys/kernel/random/boot_id')),
                  ('uptime', Path('/proc/uptime')), ('stat', base / 'stat'),
                  ('wchan', base / 'wchan'), ('syscall', base / 'syscall')]:
    try:
        result[key] = path.read_text().strip()
    except OSError as e:
        result[key] = {'error': type(e).__name__, 'errno': e.errno}
print(json.dumps(result))
