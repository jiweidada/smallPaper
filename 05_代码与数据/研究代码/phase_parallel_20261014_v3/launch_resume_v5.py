"""Launch exactly one repaired controller on an idle allocated GPU."""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time

account = sys.argv[1]
assert account in ['A','B']
phase = pathlib.Path('/25085400120/DEIM/experiments/phase_mechanism_20261012' if account=='A'
                     else '/24085403014/experiments/phase_mechanism_20261012')
parent = phase/'artifacts/round2_20261014_v1'
stage = parent/'execution_v5'
assert json.loads((parent/'current_execution.json').read_text())['execution_directory']=='execution_v4'
assert json.loads((stage/'resume_plan.json').read_text())['status']=='ready_to_resume'
assert not (stage/'controller.lock').exists() and not (stage/'launch_receipt.json').exists()
spec=importlib.util.spec_from_file_location('frozen_parallel_v3',parent/'execution_v3/run_parallel_queue.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
assert not helper.main_training_processes(phase)
for path in pathlib.Path('/proc').iterdir():
    if path.name.isdigit():
        try:
            row=helper.identity(int(path.name))
            if row and row['state']!='Z' and any(str(parent) in a for a in row['argv']):
                assert not any(pathlib.PurePosixPath(a).name in ['run_parallel_queue.py','resume_parallel_queue_v4.py','resume_parallel_queue_v5.py'] for a in row['argv']), 'Live controller exists'
        except (FileNotFoundError,ProcessLookupError):
            pass
import torch
assert torch.cuda.is_available(), 'GPU not allocated; do not run training on CPU'
free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
assert int(free.strip().splitlines()[0])>=10240
compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
assert not any(x.strip().isdigit() for x in compute.splitlines()), 'GPU already has compute processes'
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
with (stage/'queue_console.log').open('xb') as handle:
    child=subprocess.Popen([sys.executable,'-u',str(stage/'resume_parallel_queue_v5.py')],
        cwd=parent/'execution_v3/code',env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
        start_new_session=True)
deadline=time.monotonic()+30
while not (stage/'queue_state.json').exists():
    assert child.poll() is None, 'Controller exited before state creation'
    assert time.monotonic()<deadline
    time.sleep(.2)
identity=helper.identity(child.pid)
assert identity and child.poll() is None
pointer={'execution_directory':'execution_v5','queue_state':str(stage/'queue_state.json'),
         'version':'resume_parallel_v5','max_concurrent_trainings':2,'updated_at':time.time()}
temporary=parent/'current_execution.v5.tmp';temporary.write_text(json.dumps(pointer,indent=2))
os.replace(temporary,parent/'current_execution.json')
receipt={'account':account,'pid':child.pid,'identity':identity,'pointer':pointer,
         'queue_state':json.loads((stage/'queue_state.json').read_text()),
         'authorization':'User explicitly requested reconnecting after maintenance, repairing save paths and resuming remaining experiments.'}
with (stage/'launch_receipt.json').open('x') as handle:json.dump(receipt,handle,indent=2)
print(json.dumps(receipt))
