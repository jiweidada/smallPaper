"""Remote read-only inventory before any resume or deployment."""
import datetime
import json
import pathlib
import subprocess

roots = [pathlib.Path('/25085400120/DEIM/experiments/phase_mechanism_20261012'),
         pathlib.Path('/24085403014/experiments/phase_mechanism_20261012')]
result = {'remote_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'hostname': pathlib.Path('/etc/hostname').read_text().strip(), 'roots': [], 'processes': []}
for root in roots:
    if not root.is_dir():
        continue
    account = 'A' if str(root).startswith('/250') else 'B'
    env = '/25085400120/envs/deim/bin/python' if account == 'A' else '/24085403014/deim_migration_runtime/envs/deim/bin/python'
    row = {'account': account, 'root': str(root), 'python': env, 'python_exists': pathlib.Path(env).exists(), 'jobs': {}}
    stage = root/'artifacts/round2_20261014_v1'
    pointer = stage/'current_execution.json'
    row['execution_pointer'] = json.loads(pointer.read_text()) if pointer.is_file() else None
    if row['execution_pointer']:
        directory = stage/row['execution_pointer']['execution_directory']
        for filename in ['queue_state.json', 'queue_plan.json']:
            path = directory/filename
            row[filename] = json.loads(path.read_text()) if path.is_file() else None
    for path in sorted((root/'jobs').glob('*.json')):
        value = json.loads(path.read_text())
        output = root/'outputs'/path.stem
        value['checkpoint_inventory'] = [{'name': p.name, 'bytes': p.stat().st_size, 'mtime': p.stat().st_mtime}
                                         for p in output.glob('*.pth')]
        log = output/'log.txt'
        if log.is_file():
            values = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
            values = [v for v in values if 'test_coco_eval_bbox' in v]
            value['logged_epochs'] = [v['epoch'] for v in values]
        row['jobs'][path.stem] = value
    result['roots'].append(row)
for path in pathlib.Path('/proc').iterdir():
    if not path.name.isdigit():
        continue
    try:
        args = [a.decode(errors='replace') for a in (path/'cmdline').read_bytes().split(b'\0') if a]
        if not any('phase_mechanism_20261012' in a for a in args):
            continue
        stat = (path/'stat').read_text().rsplit(')', 1)[1].split()
        result['processes'].append({'pid': int(path.name), 'ppid': int(stat[1]),
                                    'state': stat[0], 'start_ticks': int(stat[19]), 'argv': args})
    except (OSError, ValueError):
        pass
for name, command in [('gpu', ['nvidia-smi', '--query-gpu=name,uuid,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader']),
                      ('gpu_compute', ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'])]:
    process = subprocess.run(command, capture_output=True, text=True, timeout=20)
    result[name] = {'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr}
print(json.dumps(result, ensure_ascii=False))
