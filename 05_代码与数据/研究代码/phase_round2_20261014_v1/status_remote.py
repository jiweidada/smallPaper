"""Compact live readback of the bounded round-two pipeline."""
import json
import pathlib
import subprocess
import time

storage = pathlib.Path('/25085400120' if pathlib.Path('/25085400120/DEIM').is_dir() else '/24085403014')
phase = next(p for p in [storage/'DEIM/experiments/phase_mechanism_20261012', storage/'experiments/phase_mechanism_20261012'] if p.is_dir())
stage = phase/'artifacts/round2_20261014_v1'
result = {'remote_time': time.time(), 'phase_root': str(phase), 'queue': None, 'logs': {}, 'files': {}}
for name in ['queue_state.json', 'launch_receipt.json', 'activation_receipt.json',
             'addons/soep_cost_audit.json', 'addons/prelaunch_four_images/receipt.json']:
    p = stage/name
    if p.is_file():
        result['files'][name] = json.loads(p.read_text())
for pattern in ['addons/prelaunch_four_images/*.log', 'queue_console.log', 'analysis_tests.log', 'soep_tests.log', 'soep_cost_audit.log']:
    for p in stage.glob(pattern):
        result['logs'][str(p.relative_to(stage))] = p.read_text(errors='replace')[-4000:]
result['jobs'] = {p.name: json.loads(p.read_text()) for p in (phase/'jobs').glob('*.json')}
for name, job in result['jobs'].items():
    if job.get('state') == 'running':
        log = phase/'logs'/(name[:-5]+'.log')
        if log.exists():
            with log.open('rb') as f:
                f.seek(max(0, log.stat().st_size-12000))
                tail = f.read().decode(errors='replace')
            result['logs']['training_'+name] = '\n'.join(tail.splitlines()[-8:])
result['gpu'] = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True).stdout
print(json.dumps(result, indent=2))
