"""Launch the explicitly authorized controller exactly once."""
from pathlib import Path
import os
import subprocess
import sys

root = Path('/25085400120/mpsma_fair_20261103_v1/diagnostics/mpsma_diag_base_apdrop_20261105_v1')
sys.path.insert(0, str(root))
import diagnostic_contract as c
c.auth_check()
c.package_check()
assert sys.executable == '/25085400120/envs/deim/bin/python'
ready = c.read(root / 'setup_ready.json')
assert ready['status'] == 'SETUP_READY_NO_MODEL_EXECUTION'
assert ready['code_manifest_sha256'] == c.sha(root / 'code_manifest.json')
assert not (root / 'controller_launch.json').exists()
with (root / 'controller_launcher.one_shot.lock').open('x') as lock:
    lock.write('one launch only\n')
    lock.flush()
    os.fsync(lock.fileno())
with (root / 'controller.log').open('x') as log:
    child = subprocess.Popen([sys.executable, '-B', '-u', str(root / 'diagnostic_controller.py')],
                             cwd=str(root), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
    ticks = (Path('/proc') / str(child.pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    receipt = {'status': 'CONTROLLER_LAUNCHED_ONE_BOUNDED_DIAGNOSTIC', 'pid': child.pid,
               'start_ticks': ticks, 'created_utc': c.now(), 'authorization_sha256': c.AUTH_SHA,
               'automatic_retries': False}
    c.atomic_json(root / 'controller_launch.json', receipt, new=True)
print(__import__('json').dumps(receipt))
