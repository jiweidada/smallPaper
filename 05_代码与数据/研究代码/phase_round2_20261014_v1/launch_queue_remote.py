"""Launch exactly one frozen six-job controller; never interrupts old training."""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

account = 'A' if pathlib.Path('/25085400120/DEIM').is_dir() else 'B'
storage = pathlib.Path('/25085400120' if account == 'A' else '/24085403014')
phase = next(p for p in [storage/'DEIM/experiments/phase_mechanism_20261012', storage/'experiments/phase_mechanism_20261012'] if p.is_dir())
stage = phase/'artifacts/round2_20261014_v1'
addons = stage/'addons'
plan = json.loads((addons/'queue_plan.json').read_text())
expected = ([mode+':0' for mode in ['c_mean_full96','rank16_full96','rank16_mean_full96','soep_lite_full96']]+['c_full96:2','c_mean_full96:2']
            if account == 'A' else [mode+':1' for mode in ['c_mean_full96','rank16_full96','rank16_mean_full96','soep_lite_full96']]+['rank16_full96:2','rank16_mean_full96:2'])
assert plan['jobs'] == expected and plan['training_epochs'] == 96 and not plan['baseline_seed2']
assert not (stage/'launch_receipt.json').exists() and not (stage/'queue_execution.lock').exists()
manifest = json.loads((addons/'deployment_manifest.json').read_text())
assert all(hashlib.sha256((addons/name).read_bytes()).hexdigest() == digest for name, digest in manifest.items())
if account == 'A':
    assert json.loads((addons/'prelaunch_four_images/receipt.json').read_text())['status'] == 'passed'
for model in plan['existing_models']:
    if model['name'] != 'existing_singlephase' or account == 'A':
        assert pathlib.Path(model['checkpoint']).is_file(), 'Missing reference checkpoint: '+model['checkpoint']
environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
with (stage/'queue_console.log').open('xb') as log:
    process = subprocess.Popen([sys.executable, '-u', str(addons/'run_round2_queue.py')], cwd=phase/'code',
                               env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
time.sleep(2)
assert process.poll() is None, 'Queue failed during startup; inspect its console log'
start_ticks = int(pathlib.Path('/proc', str(process.pid), 'stat').read_text().rsplit(')', 1)[1].split()[19])
receipt = {'account': account, 'pid': process.pid, 'start_ticks': start_ticks,
           'launched_at': time.time(), 'jobs': expected, 'epochs_per_job': 96,
           'controller_alive': True, 'does_not_interrupt_existing_training': True,
           'console': str(stage/'queue_console.log')}
with (stage/'launch_receipt.json').open('x') as output:
    json.dump(receipt, output, indent=2)
print(json.dumps(receipt, indent=2))
