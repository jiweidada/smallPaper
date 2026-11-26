"""Read process activity only; do not attach a debugger or alter the worker."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

pid = 1779501
folder = Path('/proc') / str(pid)
out = {'observed_server_utc': datetime.now(timezone.utc).isoformat(), 'pid': pid, 'exists': folder.is_dir()}
if folder.is_dir():
    out['status'] = {k.strip(): v.strip() for line in (folder / 'status').read_text().splitlines()
                     if ':' in line for k, v in [line.split(':', 1)]
                     if k in ['Name', 'State', 'Threads', 'VmRSS', 'VmSize', 'voluntary_ctxt_switches', 'nonvoluntary_ctxt_switches']}
    out['stat'] = (folder / 'stat').read_text()
    out['io'] = (folder / 'io').read_text()
    out['wchan'] = (folder / 'wchan').read_text()
    out['open_files'] = []
    for p in (folder / 'fd').iterdir():
        try:
            value = os.readlink(p)
            if value.startswith('/25085400120/') or value.startswith('/tmp/'):
                out['open_files'].append(value)
        except (FileNotFoundError, PermissionError): pass
    out['uptime'] = Path('/proc/uptime').read_text().split()[0]
print(json.dumps(out))
