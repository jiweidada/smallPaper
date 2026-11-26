"""Launch one new controller, with no existing trainer or controller present."""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time

account = 'A' if pathlib.Path('/25085400120/DEIM').is_dir() else 'B'
phase = pathlib.Path('/25085400120/DEIM/experiments/phase_mechanism_20261012' if account == 'A'
                     else '/24085403014/experiments/phase_mechanism_20261012')
parent = phase/'artifacts/round2_20261014_v1'
stage = parent/'execution_v4'
assert not (stage/'launch_receipt.json').exists() and not (stage/'controller.lock').exists()
assert json.loads((stage/'resume_plan.json').read_text())['status'] == 'ready_to_resume'
compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
assert not any(line.strip().isdigit() for line in compute.splitlines()), 'GPU already has compute processes'
spec = importlib.util.spec_from_file_location('original_parallel', parent/'execution_v3/run_parallel_queue.py')
helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
assert not helper.main_training_processes(phase)
env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
with (stage/'queue_console.log').open('xb') as output:
    child = subprocess.Popen([sys.executable, '-u', str(stage/'resume_parallel_queue_v4.py')], cwd=parent/'execution_v3/code',
                             env=env, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
deadline = time.time()+30
while not (stage/'queue_state.json').exists():
    assert child.poll() is None, 'Controller exited before state creation'
    assert time.time() < deadline
    time.sleep(.2)
identity = helper.identity(child.pid)
assert identity and child.poll() is None
pointer = {'execution_directory': 'execution_v4', 'queue_state': str(stage/'queue_state.json'),
           'version': 'resume_parallel_v4', 'max_concurrent_trainings': 2, 'updated_at': time.time()}
temporary = parent/'current_execution.v4.tmp'
temporary.write_text(json.dumps(pointer, indent=2))
os.replace(temporary, parent/'current_execution.json')
receipt = {'account': account, 'pid': child.pid, 'identity': identity,
           'queue_state': json.loads((stage/'queue_state.json').read_text()), 'pointer': pointer,
           'authorization': 'User provided new endpoints and explicitly requested continuing experiments.'}
with (stage/'launch_receipt.json').open('x') as handle:
    json.dump(receipt, handle, indent=2)
print(json.dumps(receipt))
