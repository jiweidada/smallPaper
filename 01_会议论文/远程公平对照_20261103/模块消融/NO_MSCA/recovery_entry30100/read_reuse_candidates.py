"""Read current assigned volume candidates only; do not import models or run CUDA."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

volume = Path('/24085403014')
out = {'observed_server_utc': datetime.now(timezone.utc).isoformat(), 'gpu_work_started': False}
lease = volume / '.deim_gpu0_owner_lease_v2.json'
if lease.is_file():
    value = json.loads(lease.read_text())
    allowed = {'owner', 'owner_id', 'task', 'task_id', 'run_id', 'project_root', 'pid',
               'trainer_pid', 'supervisor_pid', 'boot_id', 'gpu_uuid', 'phase', 'status',
               'created_at', 'updated_at', 'heartbeat_utc', 'expires_at', 'expires_utc'}
    out['existing_gpu_lease'] = {'keys': sorted(value),
                                  'selected_fields': {k: v for k, v in value.items() if k in allowed}}
out['candidate_directories'] = {}
for relative in ['DEIM/dataset_visdrone', 'DEIM/dataset', 'DEIM/datasets',
                 'datasets_verified_20261017', 'deim_migration_runtime',
                 'DEIM_migration_20260929', 'experiments']:
    path = volume / relative
    info = {'exists': path.exists(), 'directory': path.is_dir()}
    if path.is_dir():
        info['entries'] = [{'name': p.name, 'directory': p.is_dir(), 'bytes': p.stat().st_size}
                           for p in sorted(path.iterdir())][:40]
    out['candidate_directories'][str(path)] = info

program = """import sys,json,importlib.metadata as m
names=['torch','torchvision','faster-coco-eval','numpy','pillow','opencv-python']
v={}
for n in names:
 try: v[n]=m.version(n)
 except m.PackageNotFoundError: v[n]=None
print(json.dumps({'python':sys.version,'executable':sys.executable,'packages':v}))
"""
out['python_metadata'] = {}
for name in ['/opt/conda/envs/esm/bin/python', '/opt/conda/envs/mamba/bin/python',
             '/opt/conda/bin/python', '/24085403014/deim_migration_runtime/bin/python']:
    if not Path(name).is_file():
        continue
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    completed = subprocess.run([name, '-B', '-c', program], capture_output=True, text=True,
                               timeout=12, env=env)
    out['python_metadata'][name] = {'exit_code': completed.returncode,
                                    'stdout': completed.stdout, 'stderr': completed.stderr[-800:]}
print(json.dumps(out))
