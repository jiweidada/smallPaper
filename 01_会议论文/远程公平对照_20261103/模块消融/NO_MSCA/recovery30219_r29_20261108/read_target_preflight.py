"""Read-only recovery target audit; no models, evaluations, or optimizer updates."""
import os
os.environ['ATEN_CPU_CAPABILITY'] = 'avx2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

audit_root = root = Path('/25085400120/mpsma_fair_20261103_v1/module_ablation_20261107_v1/recovery_no_msca_r29_20261108')
def progress(value, name='preflight_progress.json'):
    temporary = audit_root / (name + '.partial')
    temporary.write_text(json.dumps(value) + '\n')
    os.replace(temporary, audit_root / name)

root = Path('/25085400120/mpsma_fair_20261103_v1')
module = root / 'module_ablation_20261107_v1'
existing = module / 'no_mps_20261107_v1'
target = module / 'no_msca_20261107_v1'
result = {'observed_server_utc': datetime.now(timezone.utc).isoformat(),
          'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          'target_exists': target.exists(), 'python': sys.version, 'executable': sys.executable,
          'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu',
                                          '--format=csv,noheader,nounits'], text=True).strip(),
          'gpu_processes': subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                                                    '--format=csv,noheader,nounits'], text=True).strip(),
          'available_storage': {}, 'existing_no_mps': {}}
for location in [root, Path('/tmp')]:
    usage = shutil.disk_usage(location)
    result['available_storage'][str(location)] = {'total': usage.total, 'free': usage.free}
for name in ['execution_protocol.json', 'dataset_bundle_manifest.json', 'cache_verification.json',
             'environment.current.json', 'supervisor.json', 'run/receipt.json']:
    path = existing / name
    result['existing_no_mps'][name] = {'exists': path.is_file()}
    if path.is_file():
        result['existing_no_mps'][name].update(sha256=sha256(path.read_bytes()).hexdigest(),
                                                content=json.loads(path.read_text()))
import torch
result['runtime'] = {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                     'cudnn': torch.backends.cudnn.version(), 'torch_build': torch.__config__.show(),
                     'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}
cache = json.loads((existing / 'cache_verification.json').read_text())
manifest_path = existing / 'dataset_bundle_manifest.json'
assert sha256(manifest_path.read_bytes()).hexdigest() == 'ba463a04fe688d5c3005cb3cf9ca6e13aa699b0b11016e15e7fa9b76b52e2a56'
manifest = json.loads(manifest_path.read_text())
data_root = Path(cache['cache_root']).resolve(strict=True)
started = time.monotonic()
bad = []
checked_bytes = 0
for index, item in enumerate(manifest['files']):
    path = (data_root / item['path']).resolve(strict=True)
    assert path.is_relative_to(data_root)
    h = sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024**2):
            h.update(block)
    checked_bytes += path.stat().st_size
    if h.hexdigest() != item['sha256'] or path.stat().st_size != item['bytes']:
        bad.append(item['path'])
    if index % 1000 == 0:
        progress({'status': 'VERIFYING_INPUT_BYTES', 'checked_files': index + 1,
                  'total_files': len(manifest['files']), 'seconds': time.monotonic() - started,
                  'mismatch_count': len(bad)})
result['dataset_current_byte_verification'] = {'root': str(data_root), 'files': len(manifest['files']),
    'bytes': checked_bytes, 'mismatches': bad, 'seconds': time.monotonic() - started}
result['training_started'] = False
progress(result, 'target_preflight_readonly.json')
progress({'status': 'COMPLETED', 'files': len(manifest['files']), 'mismatch_count': len(bad)})
print(json.dumps(result))
