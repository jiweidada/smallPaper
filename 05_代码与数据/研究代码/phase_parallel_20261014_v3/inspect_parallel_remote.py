import datetime
import json
import pathlib
import subprocess

account = 'A' if pathlib.Path('/25085400120/DEIM').is_dir() else 'B'
phase = pathlib.Path('/25085400120/DEIM/experiments/phase_mechanism_20261012' if account == 'A' else '/24085403014/experiments/phase_mechanism_20261012')
stage = phase/'artifacts/round2_20261014_v1/execution_v3'
result = {'account': account, 'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'files': {}, 'logs': {}}
for name in ['preparation_receipt.json', 'queue_state.json', 'launch_receipt.json', 'adopted_training_exit.json']:
    path = stage/name
    if path.exists():
        data = json.loads(path.read_text())
        data.pop('source_sha256', None)
        result['files'][name] = data
for path in stage.glob('*.log'):
    result['logs'][path.name] = {'bytes': path.stat().st_size, 'last_lines': path.read_text(errors='replace').splitlines()[-6:]}
for command, name in [(['nvidia-smi', '--query-gpu=memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], 'gpu'),
                      (['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], 'compute')]:
    query = subprocess.run(command, capture_output=True, text=True)
    result[name] = query.stdout
print(json.dumps(result, indent=2))
