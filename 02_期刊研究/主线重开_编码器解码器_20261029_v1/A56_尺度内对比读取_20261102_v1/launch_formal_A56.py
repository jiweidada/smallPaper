from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

P = Path(__file__).resolve().parent
assert P == Path('/root/uav_a56_20261102_v1')
assert json.loads((P/'capacity_result.json').read_text())['status'] == 'PASS'
assert json.loads((P/'capacity_exit.json').read_text())['child_exit_code'] == 0
assert all(not (P/f'arm_{arm}').exists() for arm in ['M', 'Q'])
with (P/'formal_launch_once.lock').open('x') as f:
    f.write(str(os.getpid()))
env = dict(os.environ)
env['PYTHONDONTWRITEBYTECODE'] = '1'
env['PYTHONPATH'] = str(P)
env['OMP_NUM_THREADS'] = '4'
log = (P/'formal_supervisor.log').open('xb')
child = subprocess.Popen([sys.executable, '-u', str(P/'supervise_A56.py')],
                         cwd=P, env=env, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)
stat = Path(f'/proc/{child.pid}/stat').read_text().split()
receipt = {'status': 'FORMAL_SUPERVISOR_STARTED', 'supervisor_PID': child.pid,
           'supervisor_start_ticks': int(stat[21]),
           'protocol_sha256': hashlib.sha256((P/'protocol_FROZEN_v1.json').read_bytes()).hexdigest()}
(P/'formal_launch_receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
