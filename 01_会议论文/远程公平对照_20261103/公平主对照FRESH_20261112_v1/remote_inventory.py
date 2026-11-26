"""Read-only server/GPU/project inventory. Does not import any model or evaluator."""
from pathlib import Path
import datetime
import json
import os
import platform
import shutil
import subprocess

def command(args):
    try:
        p=subprocess.run(args,capture_output=True,text=True,timeout=15)
        return {"exit_code":p.returncode,"stdout":p.stdout[-20000:].strip(),"stderr":p.stderr[-1000:].strip()}
    except Exception as error:return {"error":type(error).__name__,"message":str(error)}

roots=[Path('/25085400120'),Path('/24085403014')]
projects=[]
for root in roots:
    if not root.is_dir():continue
    disk=shutil.disk_usage(str(root))
    children=[]
    for p in sorted(root.iterdir()):
        if p.name.startswith('.'):continue
        children.append({"name":p.name,"directory":p.is_dir()})
    paths=[root/'envs/deim/bin/python',root/'mpsma_fair_20261103_v1',root/'mpsma_fair_20261103_v1/execution_protocol.json',
           root/'mpsma_fair_20261103_v1/verified_dataset.zip',root/'mpsma_fair_20261103_v1/dataset_bundle_manifest.json']
    projects.append({"root":str(root),"total_bytes":disk.total,"free_bytes":disk.free,"is_mount":root.is_mount(),"children":children[:35],
                     "known_paths":[{"path":str(p),"exists":p.exists(),"bytes":p.stat().st_size if p.is_file() else None} for p in paths]})
processes=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        raw=(p/'cmdline').read_bytes()
        if not any(x in raw.lower() for x in [b'train',b'launch_arm',b'supervisor',b'backup_monitor',b'mpsma']):continue
        args=raw.replace(b'\0',b' ').decode('utf-8',errors='replace').strip()
        # Do not echo unrelated process environments or arbitrary secret-bearing arguments.
        if any(x in args.lower() for x in ['password','passwd','token','secret']):args='[credential-related command omitted]'
        processes.append({"pid":int(p.name),"command":args[:1400]})
    except (OSError,ValueError):continue
out={"observed_remote_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"hostname":platform.node(),
     "boot_id":Path('/proc/sys/kernel/random/boot_id').read_text().strip(),"uid":os.getuid(),
     "gpu":command(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits']),
     "compute_processes":command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits']),
     "mounts":command(['findmnt','-rn','-o','TARGET,SOURCE,FSTYPE,OPTIONS']),"projects":projects,"processes":processes,
     "scope":"Read-only. No training, evaluation, model forward or project modification."}
print(json.dumps(out,ensure_ascii=False))
