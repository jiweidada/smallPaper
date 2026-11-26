"""Check the retained input archive and current device, without preparing or running training."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess

ROOT = Path('/25085400120/mpsma_fair_20261103_v1')
EXPECTED_ARCHIVE = 'fdde141711867fa1bc2de3a1e6342cca257eb056e2a1700795eb04068aa90266'
EXPECTED_PROTOCOL = '8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def inspect(path, hash_bytes=True):
    item = {'path': str(path), 'exists': path.exists()}
    if path.is_file():
        before = path.stat()
        item['bytes'] = before.st_size
        if hash_bytes:
            item['sha256'] = sha(path)
            after = path.stat()
            item['unchanged_during_read'] = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    return item

out = {'observed_remote_utc': datetime.now(timezone.utc).isoformat(),
       'archive': inspect(ROOT / 'verified_dataset.zip'),
       'protocol': inspect(ROOT / 'execution_protocol.json'),
       'dataset_manifest': inspect(ROOT / 'dataset_bundle_manifest.json'),
       'baseline_checkpoint': inspect(ROOT / 'runs/BASE/best.pt'),
       'cache_present': Path('/tmp/mpsma_fair_20261103_data').exists(),
       'scope': 'Read-only hash and resource observation. Does not extract data, change a configuration, execute a model, evaluate, or train.'}
out['archive_matches_original'] = out['archive'].get('sha256') == EXPECTED_ARCHIVE
out['protocol_matches_original'] = out['protocol'].get('sha256') == EXPECTED_PROTOCOL
try:
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.total,memory.used,utilization.gpu',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    out['gpu'] = {'exit_code': gpu.returncode, 'stdout': gpu.stdout.strip(), 'stderr': gpu.stderr.strip()}
except (OSError, subprocess.TimeoutExpired) as error:
    out['gpu'] = {'error': type(error).__name__, 'message': str(error)}
for path, key in [(ROOT, 'experiment_disk_free_bytes'), (Path('/tmp'), 'temporary_disk_free_bytes')]:
    if path.exists():
        out[key] = shutil.disk_usage(path).free
out['completed_remote_utc'] = datetime.now(timezone.utc).isoformat()
print(json.dumps(out))
