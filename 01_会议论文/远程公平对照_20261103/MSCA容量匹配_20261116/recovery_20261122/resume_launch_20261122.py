"""Supervise a continuation within the original updates and wall-time limits."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from launch_arm import now, read, save, sha, ticks


def ready(protocol_sha, recovery_sha):
    assert sha(HERE / 'execution_protocol.json') == protocol_sha
    assert sha(HERE / 'recovery_plan_20261122.json') == recovery_sha
    plan = read(HERE / 'recovery_plan_20261122.json')
    original = read(HERE / 'execution_protocol.json')
    assert original['authorized'] and original['epochs'] == 320
    assert plan['arm'] == original['arm'] and plan['protocol_sha256'] == protocol_sha
    assert plan['repeated_updates'] == plan['repeated_evaluations'] == 0
    assert sha(HERE / 'resume_train_20261122.py') == plan['runner_sha256']
    assert sha(HERE / 'resume_launch_20261122.py') == plan['supervisor_sha256']
    for entry in original['frozen_files']:
        target = (HERE / entry['path']).resolve(strict=True)
        assert target.is_relative_to(HERE) and sha(target) == entry['sha256']
    assert sha(HERE / 'run/next_epoch.pt') == plan['checkpoint_sha256']
    assert sha(HERE / 'recovery_20261122_files/original_supervisor.json') == plan['original_supervisor_sha256']
    assert sha(HERE / 'recovery_20261122_files/original_failed_receipt.json') == plan['original_failed_receipt_sha256']
    assert sha(HERE / 'recovery_plan.json') == plan['previous_recovery_plan_sha256']
    assert sha(HERE / 'recovery_20261122_files/environment.json') == plan['environment_snapshot_sha256']
    import json
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader,nounits'], text=True).strip()
    expected = plan['hardware_migration']
    assert gpu == expected['current_gpu_uuid'] + ', ' + expected['gpu_name']
    assert expected['previous_gpu_uuid'] == original['gpu_uuid']
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    assert not active, 'GPU has another compute process'
    old = read(HERE / 'recovery_20261122_files/original_supervisor.json')
    assert old['status'] == 'STOPPED' and old['exit_code'] == 1
    assert read(HERE / 'supervisor.json')['supervisor_pid'] == old['supervisor_pid']
    for pid, expected in [(old['pid'], old['start_ticks']),
                          (old['supervisor_pid'], old['supervisor_start_ticks'])]:
        try:
            current = ticks(pid)
        except FileNotFoundError:
            continue
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        assert boot != old['boot_id'] or current != expected, 'Original process is still live'
    receipt = read(HERE / 'run/receipt.json')
    assert receipt['next_epoch'] == plan['accepted_rounds']
    assert receipt['optimizer_step_attempts'] == plan['accepted_rounds'] * 1617
    assert receipt['phase'] == 'awaiting_local_backup'
    import backup_protocol as backup
    gate = backup.EpochGate(HERE / 'run')
    epoch = plan['accepted_rounds'] - 1
    payload = backup.local_metadata(gate.stage(epoch) / 'manifest.json')
    manifest = backup.parse_manifest(payload, plan['arm'], epoch)
    assert manifest['rolling_sha256'] == plan['checkpoint_sha256']
    gate.verify_ack(payload, manifest, 0)
    remaining = int((datetime.fromisoformat(plan['deadline_utc']) - datetime.now(timezone.utc)).total_seconds())
    assert remaining > 0
    return plan, remaining


def supervise(protocol_sha, recovery_sha):
    plan, remaining = ready(protocol_sha, recovery_sha)
    original = read(HERE / 'execution_protocol.json')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(original['cuda_visible_device']),
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0')
    command = [sys.executable, '-B', '-u', str(HERE / 'resume_train_20261122.py'),
               '--protocol-sha256', protocol_sha, '--recovery-sha256', recovery_sha]
    state = {'status': 'STARTING', 'arm': plan['arm'], 'protocol_sha256': protocol_sha,
             'recovery_plan_sha256': recovery_sha, 'supervisor_pid': os.getpid(),
             'supervisor_start_ticks': ticks(os.getpid()),
             'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
             'runner_sha256': plan['runner_sha256'], 'command': command, 'started_utc': now(),
             'hard_timeout_seconds': remaining, 'original_deadline_utc': plan['deadline_utc'],
             'automatic_training_retry': False, 'hardware_migration': plan['hardware_migration'],
             'previous_recovery_plan_sha256': plan['previous_recovery_plan_sha256']}
    save(HERE / 'supervisor.json', state)
    with (HERE / 'resume_training_20261122.log').open('x') as log:
        worker = subprocess.Popen(command, cwd=HERE, env=environment, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(status='RUNNING', pid=worker.pid, start_ticks=ticks(worker.pid))
        save(HERE / 'supervisor.json', state)
        try:
            code = worker.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            state.update(status='HARD_TIMEOUT', timeout_utc=now())
            save(HERE / 'supervisor.json', state)
            os.killpg(worker.pid, signal.SIGTERM)
            try:
                code = worker.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGKILL)
                code = worker.wait()
    final = read(HERE / 'run/receipt.json')
    if state['status'] != 'HARD_TIMEOUT':
        state['status'] = 'COMPLETED' if code == 0 and final.get('status') == 'COMPLETED' else 'STOPPED'
    state.update(exit_code=code, ended_utc=now(), final_receipt=final)
    save(HERE / 'supervisor.json', state)


def launch(protocol_sha, recovery_sha):
    ready(protocol_sha, recovery_sha)
    save(HERE / 'resume_launch_20261122.lock', {'protocol_sha256': protocol_sha,
                                      'recovery_plan_sha256': recovery_sha, 'started_utc': now()}, exclusive=True)
    with (HERE / 'resume_supervisor_20261122.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__)),
                                    'supervise', protocol_sha, recovery_sha], cwd=HERE,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    import json
    print(json.dumps({'status': 'SUPERVISOR_SPAWNED', 'protocol_sha256': protocol_sha,
                      'recovery_plan_sha256': recovery_sha, 'supervisor_pid': process.pid,
                      'start_ticks': ticks(process.pid)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['launch', 'supervise'])
    parser.add_argument('protocol_sha256')
    parser.add_argument('recovery_sha256')
    args = parser.parse_args()
    {'launch': launch, 'supervise': supervise}[args.mode](args.protocol_sha256, args.recovery_sha256)
