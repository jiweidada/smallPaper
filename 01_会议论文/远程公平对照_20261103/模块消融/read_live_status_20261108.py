"""Read-only process, receipt and GPU snapshot of the authorized ablations."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

parent = Path('/25085400120/mpsma_fair_20261103_v1/module_ablation_20261107_v1')
result = {'observed_utc': datetime.now(timezone.utc).isoformat(),
          'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          'arms': {}}
for arm in ('no_mps_20261107_v1', 'no_msca_20261107_v1'):
    root = parent / arm
    if not root.is_dir():
        continue
    info = {'root': str(root)}
    receipt_path = root / 'run/receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        info['receipt'] = {key: receipt[key] for key in (
            'arm', 'status', 'phase', 'actual_loop_epoch', 'completed_epochs',
            'next_epoch', 'batch', 'last_loss', 'optimizer_step_attempts',
            'updated_utc', 'ended_utc') if key in receipt}
    for path in sorted(root.glob('*.json')):
        if any(word in path.name for word in ('supervisor', 'receipt')):
            data = json.loads(path.read_text())
            info[path.name] = data
            for key in ('pid', 'supervisor_pid'):
                pid = data.get(key)
                if isinstance(pid, int):
                    stat = Path('/proc') / str(pid) / 'stat'
                    info[str(pid)] = {'exists': stat.exists()}
                    if stat.exists():
                        info[str(pid)]['start_ticks'] = int(stat.read_text().rsplit(')', 1)[1].split()[19])
    result['arms'][arm] = info
result['gpu'] = subprocess.check_output([
    'nvidia-smi', '--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu',
    '--format=csv,noheader'], text=True).strip()
result['gpu_processes'] = subprocess.check_output([
    'nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'],
    text=True).strip()
print(json.dumps(result))
