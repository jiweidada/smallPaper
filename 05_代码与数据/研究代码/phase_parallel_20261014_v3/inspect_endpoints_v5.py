"""Read-only CPU/GPU inventory for the approved output-directory repair."""
import datetime
import json
import pathlib
import shutil
import subprocess

roots = [pathlib.Path('/25085400120/DEIM/experiments/phase_mechanism_20261012'),
         pathlib.Path('/24085403014/experiments/phase_mechanism_20261012')]
result = {'remote_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'hostname': pathlib.Path('/etc/hostname').read_text().strip(), 'roots': [], 'processes': []}
for root in roots:
    if not root.is_dir():
        continue
    account = 'A' if str(root).startswith('/250') else 'B'
    python = '/25085400120/envs/deim/bin/python' if account == 'A' else '/24085403014/deim_migration_runtime/envs/deim/bin/python'
    row = {'account': account, 'root': str(root), 'python': python, 'python_exists': pathlib.Path(python).is_file(),
           'disk_free_bytes': shutil.disk_usage(root).free, 'jobs': {}}
    parent = root/'artifacts/round2_20261014_v1'
    pointer = parent/'current_execution.json'
    row['pointer'] = json.loads(pointer.read_text()) if pointer.is_file() else None
    if row['pointer']:
        stage = parent/row['pointer']['execution_directory']
        for name in ['queue_state.json', 'queue_plan.json', 'resume_plan.json']:
            path = stage/name
            row[name] = json.loads(path.read_text()) if path.is_file() else None
    for path in sorted((root/'jobs').glob('*.json')):
        job = json.loads(path.read_text())
        output = root/'outputs'/path.stem
        job['checkpoints'] = [{'name': p.name, 'bytes': p.stat().st_size, 'mtime': p.stat().st_mtime}
                              for p in output.glob('*.pth')]
        log = output/'log.txt'
        if log.is_file():
            rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
            job['logged_epochs'] = [r['epoch'] for r in rows if 'test_coco_eval_bbox' in r]
        row['jobs'][path.stem] = job
    result['roots'].append(row)
for path in pathlib.Path('/proc').iterdir():
    if not path.name.isdigit():
        continue
    try:
        args = [x.decode(errors='replace') for x in (path/'cmdline').read_bytes().split(b'\0') if x]
        if not any('phase_mechanism_20261012' in x for x in args):
            continue
        stat = (path/'stat').read_text().rsplit(')', 1)[1].split()
        result['processes'].append({'pid': int(path.name), 'ppid': int(stat[1]), 'state': stat[0],
                                     'start_ticks': int(stat[19]), 'argv': args})
    except (OSError, ValueError, IndexError):
        continue
for name, args in [('gpu', ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader']),
                   ('gpu_compute', ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'])]:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        result[name] = {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        result[name] = {'returncode': 127, 'stdout': '', 'stderr': type(exc).__name__+': '+str(exc)}
print(json.dumps(result, ensure_ascii=False))
