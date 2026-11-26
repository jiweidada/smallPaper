"""Identity-checked handoff of the controller, preserving live trainers."""
import importlib.util
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

account = 'A' if pathlib.Path('/25085400120/DEIM').is_dir() else 'B'
phase = pathlib.Path('/25085400120/DEIM/experiments/phase_mechanism_20261012' if account == 'A' else '/24085403014/experiments/phase_mechanism_20261012')
acceptance = phase/'artifacts/round2_20261014_v1'
stage = acceptance/'execution_v3'
assert json.loads((stage/'preparation_receipt.json').read_text())['status'] == 'ready_for_noninterrupting_handoff'
assert not (stage/'handoff_committed.json').exists() and not (stage/'launch_receipt.json').exists()
spec = importlib.util.spec_from_file_location('parallel_tools', stage/'run_parallel_queue.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
old_launch = json.loads((acceptance/'execution_v2/launch_receipt.json').read_text())
old = module.identity(old_launch['pid'])
assert old and old['start_ticks'] == old_launch['start_ticks'] and any(x.endswith('/run_round2_queue.py') for x in old['argv'])
old_state = json.loads((acceptance/'execution_v2/queue_state.json').read_text())
adopted = None
if account == 'A':
    assert old_state['state'] == 'training_c_mean_full96_seed0'
    trainer = module.identity(old_state['child_pid'])
    assert trainer and trainer['ppid'] == old['pid'] and trainer['state'] not in ['Z', 'T', 't']
    assert any(x.endswith('/train.py') for x in trainer['argv']) and 'c_mean_full96_seed0' in ' '.join(trainer['argv'])
    adopted = {'job': 'c_mean_full96_seed0', 'identity': trainer}
else:
    assert old_state['state'] == 'waiting_existing_experiments'
    legacy = json.loads((phase/'jobs/singlephase_full96_seed1.json').read_text())
    assert legacy['state'] in ['running', 'complete']
for path in pathlib.Path('/proc').iterdir():
    if not path.name.isdigit():
        continue
    child = module.identity(int(path.name))
    if child and child['ppid'] == old['pid'] and child['state'] != 'Z':
        assert adopted and child['pid'] == adopted['identity']['pid'], 'Unexpected active child of old controller'
environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
with (stage/'queue_console.log').open('xb') as output:
    controller = subprocess.Popen([sys.executable, '-u', str(stage/'run_parallel_queue.py')], cwd=stage/'code', env=environment,
                                  stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
deadline = time.time()+20
while not (stage/'queue_state.json').exists():
    assert controller.poll() is None, 'New controller failed before handoff'
    assert time.time() < deadline, 'New controller did not become ready'
    time.sleep(.2)
assert json.loads((stage/'queue_state.json').read_text())['state'] == 'ready_for_handoff'
with (stage/'previous_v2_state.json').open('x') as handle:
    json.dump(old_state, handle, indent=2)
if adopted:
    module.signal_exact(old, signal.SIGSTOP)
    time.sleep(.2)
    assert module.same_process(old)['state'] in ['T', 't']
    assert module.same_process(trainer)['state'] not in ['T', 't']
else:
    module.signal_exact(old, signal.SIGTERM)
    time.sleep(.2)
handoff = {'account': account, 'committed_at': time.time(), 'old_controller': old, 'adopted_training': adopted,
           'old_controller_action': 'hold_parent_only_until_adopted_child_exit' if adopted else 'terminate_waiting_controller_only',
           'maximum_concurrent_trainings': 2, 'trainer_not_signaled': True}
with (stage/'handoff_committed.json').open('x') as handle:
    json.dump(handoff, handle, indent=2)
pointer = {'execution_directory': 'execution_v3', 'queue_state': str(stage/'queue_state.json'),
           'version': 'parallel_v3', 'max_concurrent_trainings': 2, 'updated_at': time.time()}
with (acceptance/'current_execution.json').open('x') as handle:
    json.dump(pointer, handle, indent=2)
time.sleep(2)
assert controller.poll() is None, 'New controller exited after handoff; inspect its preserved state'
receipt = {'account': account, 'pid': controller.pid, 'identity': module.identity(controller.pid),
           'handoff': handoff, 'queue_state': json.loads((stage/'queue_state.json').read_text())}
with (stage/'launch_receipt.json').open('x') as handle:
    json.dump(receipt, handle, indent=2)
print(json.dumps(receipt, indent=2))
