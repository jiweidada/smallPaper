"""One-shot isolated setup launch; no training or inference."""
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
assert not (root / 'setup_launch.json').exists()
with (root / 'setup_launcher.one_shot.lock').open('x') as lock:
    lock.write('one launch only\n')
    lock.flush()
    os.fsync(lock.fileno())
with (root / 'setup.log').open('x') as log:
    child = subprocess.Popen([sys.executable, '-B', '-u', str(root / 'remote_setup.py')],
                             cwd=str(root), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
    ticks = (Path('/proc') / str(child.pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    receipt = {'status': 'SETUP_LAUNCHED_NO_MODEL_EXECUTION', 'pid': child.pid, 'start_ticks': ticks,
               'created_utc': c.now(), 'authorization_sha256': c.AUTH_SHA, 'automatic_retries': False}
    c.atomic_json(root / 'setup_launch.json', receipt, new=True)
print(__import__('json').dumps(receipt))
