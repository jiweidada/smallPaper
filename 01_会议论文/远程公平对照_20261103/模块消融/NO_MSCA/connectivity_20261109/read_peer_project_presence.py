"""Check the known project path on the other already-authorized server."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

root = Path('/25085400120/mpsma_fair_20261103_v1/module_ablation_20261107_v1')
result = {'observed_server_utc': datetime.now(timezone.utc).isoformat(),
          'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          'project_parent_exists': root.is_dir(), 'arm_paths': {},
          'scope': 'Known project paths on the peer server only; absence of a PID in this different namespace would not prove the original trainer stopped.'}
for name in ['no_mps_20261107_v1', 'no_msca_20261107_v1']:
    path = root / name
    item = {'exists': path.is_dir()}
    for label, relative in [('receipt', 'run/receipt.json'), ('supervisor', 'supervisor.json')]:
        target = path / relative
        if target.is_file():
            data = json.loads(target.read_text())
            item[label] = {k: data[k] for k in ['arm', 'status', 'phase', 'completed_epochs', 'next_epoch', 'ended_utc'] if k in data}
    result['arm_paths'][name] = item
result['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader'], text=True).strip()
result['gpu_processes'] = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True).strip()
print(json.dumps(result))
