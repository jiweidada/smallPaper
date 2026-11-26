"""Launch a single frozen arm and retain its genuine exit status."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent
TIMEOUT_SECONDS = 604800


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def ticks(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()[19])


def save(path, value, exclusive=False):
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('x') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def ready(expected):
    if sha(HERE / 'execution_protocol.json') != expected:
        raise RuntimeError('Execution protocol differs')
    plan = read(HERE / 'execution_protocol.json')
    if plan['authorized'] is not True or plan['epochs'] != 320 or plan['arm'] not in ('BASE_FRESH', 'FULL_FRESH'):
        raise RuntimeError('Unexpected experiment scope')
    for item in plan['frozen_files']:
        path = (HERE / item['path']).resolve(strict=True)
        if not path.is_relative_to(HERE) or sha(path) != item['sha256']:
            raise RuntimeError('Frozen launch input changed')
    import backup_protocol
    backup_protocol.EpochGate(HERE / 'run').require_probe()
    if (HERE / 'run').exists():
        raise RuntimeError('Run output already exists')
    return plan


def supervise(expected):
    plan = ready(expected)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(plan['cuda_visible_device']), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable, '-B', '-u', str(HERE / 'train_runner.py'), '--protocol-sha256', expected]
    path = HERE / 'supervisor.json'
    state = {'status': 'STARTING', 'arm': plan['arm'], 'protocol_sha256': expected,
             'supervisor_pid': os.getpid(), 'supervisor_start_ticks': ticks(os.getpid()),
             'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
             'runner_sha256': sha(HERE / 'train_runner.py'), 'command': command,
             'started_utc': now(), 'hard_timeout_seconds': TIMEOUT_SECONDS,
             'automatic_training_retry': False}
    save(path, state, exclusive=True)
    with (HERE / 'training.log').open('x') as log:
        worker = subprocess.Popen(command, cwd=HERE, env=environment, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(status='RUNNING', pid=worker.pid, start_ticks=ticks(worker.pid))
        save(path, state)
        try:
            code = worker.wait(timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            state.update(status='HARD_TIMEOUT', timeout_utc=now())
            save(path, state)
            os.killpg(worker.pid, signal.SIGTERM)
            try:
                code = worker.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGKILL)
                code = worker.wait()
    final = read(HERE / 'run/receipt.json') if (HERE / 'run/receipt.json').exists() else {}
    if state['status'] != 'HARD_TIMEOUT':
        state['status'] = 'COMPLETED' if code == 0 and final.get('status') == 'COMPLETED' else 'STOPPED'
    state.update(exit_code=code, ended_utc=now(), final_receipt=final)
    save(path, state)


def launch(expected):
    ready(expected)
    save(HERE / 'launch.lock', {'protocol_sha256': expected, 'started_utc': now()}, exclusive=True)
    with (HERE / 'supervisor.log').open('x') as log:
        supervisor = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__)), 'supervise', expected],
                                      cwd=HERE, stdin=subprocess.DEVNULL, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'status': 'SUPERVISOR_SPAWNED', 'protocol_sha256': expected,
                      'supervisor_pid': supervisor.pid, 'start_ticks': ticks(supervisor.pid)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['launch', 'supervise'])
    parser.add_argument('protocol_sha256')
    args = parser.parse_args()
    {'launch': launch, 'supervise': supervise}[args.mode](args.protocol_sha256)
