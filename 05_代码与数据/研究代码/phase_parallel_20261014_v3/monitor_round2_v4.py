"""Read-only new-endpoint monitor; authentication arrives on stdin only."""
import concurrent.futures
import datetime
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
from zoneinfo import ZoneInfo

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parents[2]/'英文小论文/data/06_第二轮_20261014'
DEST = DATA/'每半小时跟进_v4'
DEST.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location('readonly_monitor_v3', DATA/'每半小时跟进/monitor_round2.py')
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
source = old.REMOTE.replace('["execution_v2", "execution_v3"]', '["execution_v2", "execution_v3", "execution_v4", "execution_v5"]')
source = source.replace('"run_parallel_queue.py"}', '"run_parallel_queue.py", "resume_parallel_queue_v4.py", "resume_parallel_queue_v5.py"}')
assert '"execution_v5"' in source and '"resume_parallel_queue_v5.py"' in source
old_run = '''def run(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=20)
    return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}'''
cpu_safe_run = '''def run(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": 127 if isinstance(exc, FileNotFoundError) else 1,
                "stdout": "", "stderr": type(exc).__name__ + ": " + str(exc),
                "gpu_query_unavailable": args[0] == "nvidia-smi"}'''
assert source.count(old_run) == 1
source = source.replace(old_run, cpu_safe_run)
old_tail = '        text = tail(log)\n        if text is not None:'
new_tail = '''        text = tail(log)
        if text is not None and 'RESUME_V4 ' in text:
            text = text.rsplit('RESUME_V4 ', 1)[1]
            record['console_segment'] = 'after_latest_resume_marker'
        if text is not None:'''
assert source.count(old_tail)==1
source=source.replace(old_tail,new_tail)
save_path_check = r'''
# BaseSolver._setup uses dirname(resume) during training, overriding --output-dir.
# Expose the actual save directory; never attribute a shared log to one arm.
save_directories = {}
for process in result['main_trainers']:
    args = process['argv']
    if '--output-dir' not in args:
        continue
    expected = pathlib.Path(args[args.index('--output-dir')+1])
    actual = pathlib.Path(args[args.index('--resume')+1]).parent if '--resume' in args else expected
    job = expected.name
    record = result['jobs'].get(job)
    if record is None:
        continue
    record['expected_output_directory'] = str(expected)
    record['runtime_output_directory'] = str(actual)
    record['runtime_save_path_verified'] = actual == expected
    save_directories.setdefault(str(actual), []).append(job)
    if actual != expected:
        record['output_path_error'] = 'Resume forces its checkpoint parent as output directory; expected job directory differs'
        record['latest_evaluation_is_pre_resume'] = True
        result['errors'].append({'job': job, 'kind': 'resume_output_path_mismatch',
                                  'expected': str(expected), 'actual': str(actual)})
for directory, jobs in save_directories.items():
    if len(jobs) > 1:
        result['errors'].append({'kind': 'shared_training_output_directory', 'path': directory, 'jobs': jobs})
        for job in jobs:
            result['jobs'][job]['output_directory_collision'] = jobs
print(json.dumps(result, ensure_ascii=False))
'''
source = source.replace('print(json.dumps(result, ensure_ascii=False))', save_path_check)
script = DEST/'remote_snapshot_v5_2.py'
if script.exists():
    assert script.read_text(encoding='utf-8') == source
else:
    script.write_text(source, encoding='utf-8')
credentials = json.load(sys.stdin)
now = datetime.datetime.now(datetime.timezone.utc)
stamp = now.strftime('%Y%m%dT%H%M%S_%fZ')
PORTS = {'A': 30512, 'B': 30059}


def inspect(account):
    _, root, python, jobs = old.SERVERS[account]
    raw = DEST/(stamp+'_'+account+'.stdout.json')
    command = [sys.executable, '-W', 'ignore', '-X', 'utf8', '-B', str(HERE/'remote_session_v4.py'),
               '--port', str(PORTS[account]), '--python', python, '--script', str(script),
               '--script-arg', root, '--output', str(raw), '--timeout', '90']
    for job in jobs:
        command.extend(['--script-arg', job])
    try:
        process = subprocess.run(command, input=json.dumps(credentials).encode(), capture_output=True, timeout=100)
        if process.returncode:
            return account, {'port': PORTS[account], 'inspection_error': process.stderr.decode(errors='replace')[-1200:]}
        contents = raw.read_bytes()
        return account, {'port': PORTS[account], 'ssh_returncode': 0, 'raw_file': raw.name,
                         'raw_sha256': hashlib.sha256(contents).hexdigest(),
                         'snapshot': json.loads(contents)}
    except Exception as exc:
        return account, {'port': PORTS[account], 'inspection_error': type(exc).__name__ + ': ' + str(exc)}


try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        servers = dict(pool.map(inspect, ['A', 'B']))
finally:
    credentials.clear()
result = {'checked_utc': now.isoformat(), 'checked_los_angeles': now.astimezone(ZoneInfo('America/Los_Angeles')).isoformat(),
          'servers': servers}
latest = DEST/'latest_snapshot.json'
if latest.exists():
    (DEST/'previous_snapshot.json').write_bytes(latest.read_bytes())
content = json.dumps(result, ensure_ascii=False, indent=2)
(DEST/(stamp+'_snapshot.json')).write_text(content, encoding='utf-8')
temporary = latest.with_suffix('.tmp')
temporary.write_text(content, encoding='utf-8')
temporary.replace(latest)
print('CHECKED_LA', result['checked_los_angeles'])
for account, row in servers.items():
    if 'snapshot' not in row:
        print(account, row)
        continue
    s = row['snapshot']; q = s['round2']['current_queue']
    print(account, 'PORT', row['port'], 'QUEUE', q.get('state'), 'TRAINERS', len(s['main_trainers']),
          'GPU', s['gpu']['stdout'].strip(), 'ERRORS', q.get('errors'), q.get('error'))
    print('CONTROLLER', [(p['pid'], p['process_state']) for p in s['processes'] if p['kind'] == 'queue'])
    if s.get('errors'):
        print('VERIFIED_ANOMALIES', json.dumps(s['errors'], ensure_ascii=False))
    for job, j in s['jobs'].items():
        if j['state'] in ['running', 'failed']:
            print(job, 'EVALUATED', j.get('evaluated_epochs'), 'STEP', j.get('latest_step'), 'TAIL', str(j.get('tail_lines', []))[-1200:])
