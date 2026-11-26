from pathlib import Path
import json

root = Path('/25085400120/mpsma_fair_20261103_v1/module_ablation_20261107_v1/no_msca_20261107_v1/recovery30715_preparation')
value = {'records': {}, 'processes': [], 'cache_exists': Path('/tmp/mpsma_module_ablation_20261107_data').is_dir(),
    'staging_exists': Path('/tmp/mpsma_module_ablation_20261107_data_restore30715').is_dir()}
for name in ['data_status.json', 'data_restoration.json', 'data_status_attempt1_failed.json']:
    path = root / name
    value['records'][name] = json.loads(path.read_text()) if path.is_file() else None
for path in Path('/proc').iterdir():
    if not path.name.isdigit():
        continue
    try:
        command = (path / 'cmdline').read_bytes()
        if str(root / 'restore_exact_data_cache.py').encode() in command:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            value['processes'].append({'pid': int(path.name), 'state': fields[0], 'start_ticks': int(fields[19])})
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
print(json.dumps(value))
