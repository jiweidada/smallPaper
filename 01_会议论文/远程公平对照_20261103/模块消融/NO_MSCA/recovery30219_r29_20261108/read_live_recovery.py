"""Read the exact recovery process, progress receipt, GPU and bounded logs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

here = Path('/25085400120/mpsma_fair_20261103_v1/module_ablation_20261107_v1/no_msca_20261107_v1')
out = {'observed_server_utc': datetime.now(timezone.utc).isoformat(),
       'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'records': {}}
for name in ['supervisor.json', 'run/receipt.json']:
    path = here / name
    out['records'][name] = json.loads(path.read_text()) if path.is_file() else None
for name in ['supervisor.log', 'training.log']:
    path = here / name
    if path.is_file():
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 9000))
            out[name + '_tail'] = stream.read().decode(errors='replace')
state = out['records']['supervisor.json'] or {}
out['processes'] = {}
for key in ['supervisor_pid', 'pid']:
    if key not in state: continue
    folder = Path('/proc') / str(state[key])
    info = {'pid': state[key], 'exists': folder.is_dir()}
    if folder.is_dir():
        info['start_ticks'] = int((folder / 'stat').read_text().rsplit(') ', 1)[1].split()[19])
        info['command'] = (folder / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
    out['processes'][key] = info
out['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.total,utilization.gpu',
                                     '--format=csv,noheader,nounits'], text=True).strip()
out['gpu_compute_processes'] = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                                       '--format=csv,noheader,nounits'], text=True).strip()
print(json.dumps(out))
