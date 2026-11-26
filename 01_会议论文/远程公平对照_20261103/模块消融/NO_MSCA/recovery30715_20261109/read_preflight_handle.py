import json
from pathlib import Path
from datetime import datetime, timezone

root = Path('/25085400120/mpsma_fair_20261103_v1/module_ablation_20261107_v1/no_msca_20261107_v1/recovery30715_preparation')
record = root / 'runtime_and_state_preflight.json'
value = {'observed_server_utc': datetime.now(timezone.utc).isoformat(),
    'record_exists': record.is_file(), 'processes': []}
for name in ['runtime_cache_warm_launch.json', 'runtime_cache_warm_status.json', 'runtime_cache_warm_completed.json']:
    path = root / name
    if path.is_file():
        value[name] = json.loads(path.read_text())
for proc in Path('/proc').iterdir():
    if proc.name.isdigit():
        try:
            command = (proc / 'cmdline').read_bytes()
            if str(root / 'read_runtime_and_state_preflight.py').encode() in command:
                fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
                value['processes'].append({'pid': int(proc.name), 'start_ticks': int(fields[19]),
                    'state': fields[0], 'wchan': (proc / 'wchan').read_text(),
                    'io': (proc / 'io').read_text(),
                    'status': '\n'.join(x for x in (proc / 'status').read_text().splitlines()
                        if x.startswith(('Name:', 'State:', 'VmRSS:', 'Threads:'))),
                    'command': command.replace(b'\0', b' ').decode(errors='replace')})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
if record.is_file():
    result = json.loads(record.read_text())
    value['summary'] = {key: result[key] for key in ['status', 'rolling_metadata', 'source_restore_required', 'gpu_raw']}
print(json.dumps(value))
