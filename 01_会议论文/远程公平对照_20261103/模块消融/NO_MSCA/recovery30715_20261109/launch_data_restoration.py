"""Detach the data-only restoration, with an observable exact process identity."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import json
import os
import subprocess
import sys

assert sys.executable == '/25085400120/envs/deim/bin/python'
assert sys.version_info[:2] == (3, 10)
root = Path('/25085400120/mpsma_fair_20261103_v1/module_ablation_20261107_v1/no_msca_20261107_v1/recovery30715_preparation')
source = root / 'restore_exact_data_cache.py'
assert not Path('/tmp/mpsma_module_ablation_20261107_data').exists()
assert not Path('/tmp/mpsma_module_ablation_20261107_data_restore30715').exists()
for proc in Path('/proc').iterdir():
    if proc.name.isdigit():
        try:
            assert str(source).encode() not in (proc / 'cmdline').read_bytes(), 'Existing restoration handle must be inspected'
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
with (root / 'data_restore_python310.log').open('x') as log:
    process = subprocess.Popen([sys.executable, '-B', '-u', str(source)],
        cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), start_new_session=True)
    ticks = int(Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
receipt = {'status': 'DATA_ONLY_RESTORATION_STARTED', 'pid': process.pid, 'start_ticks': ticks,
    'interpreter': sys.executable, 'python': sys.version,
    'source_sha256': sha256(source.read_bytes()).hexdigest(),
    'observed_utc': datetime.now(timezone.utc).isoformat(), 'training_started': False}
with (root / 'data_restore_launch_python310.json').open('x') as handle:
    json.dump(receipt, handle, indent=2)
    handle.write('\n')
print(json.dumps(receipt))
