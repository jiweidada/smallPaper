"""Detach one CPU-only hash/inventory preflight; no experiment is launched."""
import json
from pathlib import Path
import subprocess

here = Path('/25085400120/mpsma_fair_20261103_v1/module_ablation_20261107_v1/recovery_no_msca_r29_20261108')
assert not (here / 'preflight_launch.json').exists()
with (here / 'preflight.log').open('x') as log:
    process = subprocess.Popen(['/25085400120/envs/deim/bin/python', '-B', str(here / 'read_target_preflight.py')],
        cwd=here, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
value = {'pid': process.pid, 'scope': 'CPU file hashes and installed runtime metadata', 'training_started': False}
(here / 'preflight_launch.json').write_text(json.dumps(value) + '\n')
print(json.dumps(value))
