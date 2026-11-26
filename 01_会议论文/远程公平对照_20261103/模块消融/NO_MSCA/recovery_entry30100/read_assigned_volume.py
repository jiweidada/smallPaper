"""Inspect the currently assigned user volume and known project/environment locations."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil

volume = Path('/24085403014')
result = {'observed_server_utc': datetime.now(timezone.utc).isoformat(),
          'assigned_volume': str(volume), 'volume_exists': volume.is_dir(), 'paths': {}}
if volume.is_dir():
    result['top_level_entries'] = [{'name': p.name, 'directory': p.is_dir(), 'bytes': p.stat().st_size}
                                    for p in sorted(volume.iterdir())][:100]
    usage = shutil.disk_usage(volume)
    result['free_bytes'], result['total_bytes'] = usage.free, usage.total
paths = [volume / 'mpsma_fair_20261103_v1', volume / 'DEIM', volume / 'envs',
         volume / 'envs/deim/bin/python', Path('/opt/conda/envs'),
         volume / 'dataset_visdrone', volume / 'datasets']
for path in paths:
    info = {'exists': path.exists(), 'directory': path.is_dir()}
    if path.is_dir():
        info['entries'] = [p.name for p in sorted(path.iterdir())][:60]
    result['paths'][str(path)] = info
root = volume / 'mpsma_fair_20261103_v1/module_ablation_20261107_v1/no_msca_20261107_v1'
result['relocated_original_run_exists'] = root.is_dir()
if root.is_dir():
    for name in ['execution_protocol.json', 'supervisor.json', 'run/receipt.json']:
        path = root / name
        if path.is_file():
            data = json.loads(path.read_text())
            result[name] = {'sha256': sha256(path.read_bytes()).hexdigest(),
                             'fields': {k: data[k] for k in ['arm', 'status', 'phase', 'pid', 'completed_epochs', 'next_epoch', 'optimizer_step_attempts'] if k in data}}
print(json.dumps(result))
