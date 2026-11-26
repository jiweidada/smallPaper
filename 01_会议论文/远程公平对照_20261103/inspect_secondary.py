"""Read-only inventory of the user's second experiment server."""
import datetime, hashlib, json, os, pathlib, platform, shutil, subprocess, sys

def command(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}
    except Exception as e:
        return {'error': type(e).__name__, 'message': str(e)}

paths = ['/25085400120', '/25085400120/envs/deim/bin/python',
         '/25085400120/mpsma_fair_20261103_v1', '/tmp/mpsma_fair_20261103_data',
         '/opt/conda/bin/python', '/root/autodl-tmp', '/workspace', '/data']
out = {'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'hostname': platform.node(), 'platform': platform.platform(),
       'python': sys.version, 'executable': sys.executable,
       'cpu_count': os.cpu_count(), 'paths': {},
       'gpu': command(['nvidia-smi', '--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,driver_version,utilization.gpu', '--format=csv,noheader,nounits']),
       'gpu_processes': command(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory,process_name', '--format=csv,noheader,nounits'])}
for name in paths:
    p = pathlib.Path(name)
    item = {'exists': p.exists(), 'resolved': str(p.resolve())}
    if p.exists():
        item.update(device=p.stat().st_dev, inode=p.stat().st_ino)
        if p.is_dir():
            item.update(disk=dict(zip(['total','used','free'], shutil.disk_usage(p))))
    out['paths'][name] = item
root = pathlib.Path('/25085400120/mpsma_fair_20261103_v1')
for name in ['execution_protocol.json','queue.json','environment.current.json','cache_verification.json']:
    p = root/name
    if p.is_file():
        out[name] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'json': json.loads(p.read_text()) if name != 'execution_protocol.json' else {'bytes':p.stat().st_size}}
print(json.dumps(out, indent=2))
