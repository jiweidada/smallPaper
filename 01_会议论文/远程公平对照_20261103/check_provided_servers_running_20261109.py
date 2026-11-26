"""Read-only live GPU/process/mount snapshot for user-provided SSH endpoints."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=15)
    return {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr[-700:]}

out = {'observed_server_utc': datetime.now(timezone.utc).isoformat(),
       'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
       'gpu': command(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu',
                       '--format=csv,noheader,nounits']),
       'gpu_compute_processes': command(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                                         '--format=csv,noheader,nounits']),
       'training_or_project_processes': [], 'known_project_roots': {}}
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit() or int(entry.name) == os.getpid():
        continue
    try:
        argv = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
        if not argv or not any(token in argv.lower() for token in
                               ['train.py', 'torchrun', 'train_fair', 'no_msca', 'no_mps',
                                'mpsma_fair_20261103', 'native_supervisor']):
            continue
        out['training_or_project_processes'].append({'pid': int(entry.name), 'command': argv[:1600]})
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
for volume_name in ['/25085400120', '/24085403014']:
    volume = Path(volume_name)
    root = volume / 'mpsma_fair_20261103_v1'
    info = {'volume_exists': volume.is_dir(), 'project_root_exists': root.is_dir(),
            'frozen_python_exists': (volume / 'envs/deim/bin/python').is_file()}
    if root.is_dir():
        ablation = root / 'module_ablation_20261107_v1'
        info['project_entries'] = [p.name for p in sorted(root.iterdir())][:45]
        info['module_ablation_entries'] = [p.name for p in sorted(ablation.iterdir())] if ablation.is_dir() else []
        for name in ['no_msca_20261107_v1', 'no_mps_20261107_v1']:
            run = ablation / name
            info[name] = {'exists': run.is_dir()}
            if run.is_dir():
                info[name]['entries'] = [p.name for p in sorted(run.iterdir())][:45]
    lease = volume / '.deim_gpu0_owner_lease_v2.json'
    if lease.is_file():
        value = json.loads(lease.read_text())
        keys = ['owner', 'owner_id', 'task', 'task_id', 'run_id', 'pid', 'trainer_pid',
                'supervisor_pid', 'boot_id', 'gpu_uuid', 'phase', 'status', 'updated_at',
                'heartbeat_utc', 'expires_at', 'expires_utc', 'project_root']
        info['existing_lease_selected_fields'] = {k: value[k] for k in keys if k in value}
        info['existing_lease_keys'] = sorted(value)
    out['known_project_roots'][str(root)] = info
out['mounts_for_known_volumes'] = [line for line in Path('/proc/mounts').read_text().splitlines()
                                  if any((' ' + x + ' ') in line for x in ['/25085400120', '/24085403014'])]
print(json.dumps(out))
