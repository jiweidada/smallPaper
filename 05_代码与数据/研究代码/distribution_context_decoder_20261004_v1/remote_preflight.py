import hashlib
import json
import pathlib
import shutil
import subprocess
import time

base = pathlib.Path('/25085400120')
old = base/'DEIM/experiments/dual_view_main_seed1_batch8_pair96_20261004_v1'
data = base/'DEIM/work/DEIM_1572862469/dataset_visdrone'
init = base/'DEIM/experiments/visdrone_light_20260928_090716/artifacts/common_init_seed0.pth'
python_candidates = [base/'deim_migration_runtime/envs/deim/bin/python', base/'envs/deim/bin/python']
d = {'time': time.time(), 'GPU': subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total,memory.used', '--format=csv,noheader'], text=True),
     'compute_pids': subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True),
     'free_disk_GiB': shutil.disk_usage(base).free/2**30,
     'python': [str(p) for p in python_candidates if p.exists()],
     'init': {'path': str(init), 'exists': init.exists(), 'sha256': hashlib.sha256(init.read_bytes()).hexdigest() if init.exists() else None}}
d['data'] = {}
for s in ['train', 'val']:
    folder = data/('VisDrone2019-DET-'+s)
    ann = folder/'annotations'/(s+'.json')
    payload = json.loads(ann.read_text())
    missing = sum(not (folder/'images'/im['file_name']).is_file() for im in payload['images'])
    d['data'][s] = {'images': str(folder/'images'), 'ann': str(ann), 'count': len(payload['images']), 'missing': missing, 'sha256': hashlib.sha256(ann.read_bytes()).hexdigest()}
d['old_jobs_files'] = [p.name for p in (old/'jobs').glob('*.json')]
for name in ['launch_receipt.json', 'state.json', 'owner.json']:
    p = old/'jobs'/name
    if p.exists():
        row = json.loads(p.read_text())
        d['old_'+name] = {k:row[k] for k in ['status', 'pid', 'start_ticks', 'phase', 'time'] if k in row}
d['old_checkpoints'] = [str(p.relative_to(old)) for p in (old/'outputs').glob('*/checkpoints/epoch_*.pth')]
d['processes'] = subprocess.check_output(['ps', '-eo', 'pid,ppid,args'], text=True).splitlines()
d['processes'] = [x for x in d['processes'] if any(s in x for s in ['supervisor.py', 'run_candidate.py', 'train.py']) and 'ps -eo' not in x]
print(json.dumps(d))
