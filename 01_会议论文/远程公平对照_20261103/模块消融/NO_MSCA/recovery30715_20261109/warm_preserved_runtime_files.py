"""Sequentially read unchanged shared-library files to warm the original runtime cache."""
from pathlib import Path
import json
import os
import time
from datetime import datetime, timezone

ENV = Path('/25085400120/envs/deim')
PREP = Path('/25085400120/mpsma_fair_20261103_v1/module_ablation_20261107_v1/no_msca_20261107_v1/recovery30715_preparation')
paths = set()
maps = Path('/proc/227/maps')
if maps.is_file():
    for line in maps.read_text().splitlines():
        part = line.split()[-1]
        if part.startswith(str(ENV) + '/') and '.so' in part and Path(part).is_file():
            paths.add(Path(part))
paths.update(p for p in (ENV / 'lib/python3.10/site-packages/torch/lib').glob('*.so*') if p.is_file())
paths = sorted(paths, key=lambda p: (p.stat().st_size, str(p)))
started = time.monotonic()
total = 0
for index, path in enumerate(paths):
    with path.open('rb') as handle:
        while block := handle.read(4 * 1024 ** 2):
            total += len(block)
    state = {'status': 'WARMING_UNCHANGED_RUNTIME_FILE_CACHE', 'completed_files': index + 1,
        'total_files': len(paths), 'last_file': str(path), 'bytes_read': total,
        'elapsed_seconds': time.monotonic() - started, 'training_actions': 0,
        'files_modified': 0, 'observed_utc': datetime.now(timezone.utc).isoformat()}
    temporary = PREP / 'runtime_cache_warm_status.json.partial'
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    os.replace(temporary, PREP / 'runtime_cache_warm_status.json')
state['status'] = 'UNCHANGED_RUNTIME_CACHE_READ_COMPLETED'
with (PREP / 'runtime_cache_warm_completed.json').open('x') as handle:
    json.dump(state, handle, indent=2)
print(json.dumps(state))
