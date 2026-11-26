from pathlib import Path
import json
import os
from datetime import datetime, timezone

proc = Path('/proc/413')
value = {'observed_server_utc': datetime.now(timezone.utc).isoformat(), 'pid': 413, 'exists': proc.is_dir()}
if proc.is_dir():
    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    assert int(fields[19]) == 26597622
    value.update(start_ticks=int(fields[19]), state=fields[0], wchan=(proc / 'wchan').read_text(),
        io=(proc / 'io').read_text(), status='\n'.join(v for v in (proc / 'status').read_text().splitlines()
            if v.startswith(('Name:', 'State:', 'VmRSS:', 'Threads:'))))
    for label in ['syscall', 'stack']:
        try:
            value[label] = (proc / label).read_text()
        except PermissionError:
            value[label] = 'Read not permitted'
    value['mapped_environment_libraries'] = sorted({line.split()[-1] for line in (proc / 'maps').read_text().splitlines()
        if '/25085400120/envs/deim/' in line and '.so' in line})
    value['project_open_files'] = []
    for path in (proc / 'fd').iterdir():
        try:
            target = os.readlink(path)
            if target.startswith('/25085400120/'):
                value['project_open_files'].append({'fd': path.name, 'path': target,
                    'position': (proc / 'fdinfo' / path.name).read_text().splitlines()[0]})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
print(json.dumps(value))
