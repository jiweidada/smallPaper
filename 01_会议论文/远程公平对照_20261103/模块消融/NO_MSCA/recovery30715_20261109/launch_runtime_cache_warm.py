from pathlib import Path
from hashlib import sha256
import json
import subprocess
import sys

root = Path('/25085400120/mpsma_fair_20261103_v1/module_ablation_20261107_v1/no_msca_20261107_v1/recovery30715_preparation')
source = root / 'warm_preserved_runtime_files.py'
if (root / 'runtime_and_state_preflight.json').is_file():
    print(json.dumps({'status': 'PREFLIGHT_ALREADY_COMPLETED_NO_CACHE_WARM_NEEDED', 'training_actions': 0}))
else:
    assert not (root / 'runtime_cache_warm_launch.json').exists()
    with (root / 'runtime_cache_warm.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-B', str(source)], cwd=root,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        ticks = int(Path('/proc/{}/stat'.format(process.pid)).read_text().rsplit(')', 1)[1].split()[19])
    receipt = {'status': 'READ_ONLY_CACHE_WARM_STARTED', 'pid': process.pid, 'start_ticks': ticks,
        'source_sha256': sha256(source.read_bytes()).hexdigest(), 'training_actions': 0, 'files_modified': 0}
    with (root / 'runtime_cache_warm_launch.json').open('x') as handle:
        json.dump(receipt, handle, indent=2)
    print(json.dumps(receipt))
