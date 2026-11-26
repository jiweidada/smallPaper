"""Start one separately logged continuation under the original hard deadline."""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from continuation_guard import read, remaining_seconds, require, sha, verify_continuation

HERE = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def ticks(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()[19])


def save(path, data, exclusive=False):
    from backup_protocol import encode, write_atomic

    if exclusive:
        with path.open('xb') as stream:
            stream.write(encode(data))
            stream.flush()
            os.fsync(stream.fileno())
    else:
        write_atomic(path, encode(data))


def ready(protocol_sha, authorization_sha):
    recovery = verify_continuation(HERE, protocol_sha, authorization_sha)
    source = Path(recovery['original_root'])
    for name, expected in recovery['original_failure_records'].items():
        require(sha(source / name) == expected, 'Original failure record changed: ' + name)
    original = read(source / 'supervisor.json')
    require(all(not Path(f'/proc/{original[key]}').exists() for key in ('supervisor_pid', 'pid')),
            'Original job PIDs are still present')
    plan = read(HERE / 'execution_protocol.json')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free',
                                   '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                       '--format=csv,noheader,nounits'], text=True).strip()
    require(len(gpu) == 2 and gpu[0] == recovery['gpu_uuid'] and int(gpu[1]) >= 20000 and not compute,
            'Expected GPU unavailable or busy')
    return plan, recovery


def launch(protocol_sha, authorization_sha):
    _, recovery = ready(protocol_sha, authorization_sha)
    save(HERE / 'launch.lock', {'protocol_sha256': protocol_sha, 'authorization_sha256': authorization_sha,
                                'started_utc': now(), 'accepted_rounds': recovery['accepted_rounds']}, exclusive=True)
    with (HERE / 'supervisor.log').open('x') as log:
        supervisor = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__)), 'supervise',
                                       protocol_sha, authorization_sha], cwd=HERE, stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'status': 'SUPERVISOR_SPAWNED', 'supervisor_pid': supervisor.pid,
                      'supervisor_start_ticks': ticks(supervisor.pid), 'recovery_root': str(HERE)}), flush=True)


def supervise(protocol_sha, authorization_sha):
    lock = read(HERE / 'launch.lock')
    require(lock['protocol_sha256'] == protocol_sha and lock['authorization_sha256'] == authorization_sha,
            'Missing matching launch reservation')
    plan, recovery = ready(protocol_sha, authorization_sha)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(plan['cuda_visible_device']),
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       PYTHONDONTWRITEBYTECODE='1')
    runner = HERE / 'continuation_train.py'
    command = [sys.executable, '-B', '-u', str(runner), '--protocol-sha256', protocol_sha,
               '--authorization-sha256', authorization_sha]
    path = HERE / 'supervisor.json'
    state = {'status': 'STARTING', 'arm': plan['arm'], 'protocol_sha256': protocol_sha,
             'authorization_sha256': authorization_sha, 'supervisor_pid': os.getpid(),
             'supervisor_start_ticks': ticks(os.getpid()), 'boot_id': recovery['boot_id'],
             'runner_sha256': sha(runner), 'command': command, 'started_utc': now(),
             'deadline_remote_utc': recovery['deadline_remote_utc'], 'automatic_training_retry': False,
             'accepted_rounds_before_start': recovery['accepted_rounds']}
    save(path, state, exclusive=True)
    with (HERE / 'training.log').open('x') as log:
        require(remaining_seconds(recovery) > 0, 'Original hard deadline elapsed before launch')
        worker = subprocess.Popen(command, cwd=HERE, env=environment, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(status='RUNNING', pid=worker.pid, start_ticks=ticks(worker.pid))
        save(path, state)
        try:
            code = worker.wait(timeout=max(0, remaining_seconds(recovery)))
        except subprocess.TimeoutExpired:
            state.update(status='HARD_TIMEOUT', timeout_utc=now())
            save(path, state)
            os.killpg(worker.pid, signal.SIGTERM)
            try:
                code = worker.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGKILL)
                code = worker.wait()
    receipt = HERE / 'run/receipt.json'
    final = read(receipt) if receipt.exists() else {}
    if state['status'] != 'HARD_TIMEOUT':
        state['status'] = 'COMPLETED' if code == 0 and final.get('status') == 'COMPLETED' else 'STOPPED'
    state.update(exit_code=code, ended_utc=now(), final_receipt=final)
    save(path, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('launch', 'supervise'))
    parser.add_argument('protocol_sha256')
    parser.add_argument('authorization_sha256')
    args = parser.parse_args()
    {'launch': launch, 'supervise': supervise}[args.mode](args.protocol_sha256, args.authorization_sha256)
