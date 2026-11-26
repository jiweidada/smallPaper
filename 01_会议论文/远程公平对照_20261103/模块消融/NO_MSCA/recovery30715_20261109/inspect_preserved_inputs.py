"""Read only the preserved project, data sources, runtime metadata and164 cursor."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil

BASE = Path('/25085400120/mpsma_fair_20261103_v1')
ARM = BASE / 'module_ablation_20261107_v1/no_msca_20261107_v1'
result = {'observed_server_utc': datetime.now(timezone.utc).isoformat(), 'directories': {}, 'records': {}}
for path in [BASE, BASE / 'inputs', BASE / 'module_ablation_20261107_v1',
             BASE / 'module_ablation_20261107_v1/inputs', Path('/25085400120'), Path('/tmp')]:
    if path.is_dir():
        result['directories'][str(path)] = [{'name': p.name, 'directory': p.is_dir(),
            'bytes': p.stat().st_size} for p in sorted(path.iterdir())][:100]
        result.setdefault('free_space', {})[str(path)] = shutil.disk_usage(path)._asdict()
for name in ['cache_verification.json', 'environment.current.json', 'execution_protocol.json',
             'recovery_addendum.json', 'run/receipt.json']:
    path = ARM / name
    if path.is_file():
        value = json.loads(path.read_text())
        result['records'][name] = {'sha256': sha256(path.read_bytes()).hexdigest(), 'value': value}
result['rolling_checkpoint_sha256'] = sha256((ARM / 'run/next_epoch.pt').read_bytes()).hexdigest()
result['retained_epoch163_manifest_sha256'] = sha256((ARM / 'backup_gate/epoch0163/manifest.json').read_bytes()).hexdigest()
print(json.dumps(result))
