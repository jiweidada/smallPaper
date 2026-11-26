"""Launch only the reviewed arms named in a confirmed continuation receipt."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import psutil

from continuation_guard import read, require, sha

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
from ssh_control import connect, run_script


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def confirmed(authorization):
    scope = read(authorization)
    require(scope['authorized'] is True and isinstance(scope['user_instruction'], str)
            and bool(scope['user_instruction'].strip()), 'Continuation needs a recorded user instruction')
    require(scope['arms'] and set(scope['arms']) <= {'BASE_FRESH', 'FULL_FRESH'}, 'Unexpected authorized arms')
    plans = []
    for arm, limits in scope['arms'].items():
        path = HERE / arm / 'continuation_plan.json'
        plan = read(path)
        require(plan['arm'] == arm and limits['plan_sha256'] == sha(path)
                and limits['replayed_attempts_upper_bound'] == plan['replayed_attempts_upper_bound'],
                'Authorization differs from the staged plan')
        if scope.get('authorization_basis') == 'original_run_remaining_budget':
            original = PROJECT / 'user_authorization_20261112.json'
            authority = read(original)
            require(sha(original) == scope['original_authorization_sha256']
                    and authority['training_authorized'] is True
                    and scope['user_instruction'] in authority['user_instructions']
                    and plan['port'] in authority['ports'], 'Original experiment authorization differs')
            require(plan['lost_attempts_lower_bound'] == plan['replayed_attempts_upper_bound'] == 0
                    and plan['physical_attempts_upper_bound'] == 517440
                    and plan['retained_attempts'] + plan['remaining_attempts'] == 517440
                    and plan['completed_full_validations_before_recovery'] + plan['remaining_full_validations'] == 320,
                    'Recovery extends the original experiment budget')
        require(not (HERE / arm / 'local_backup_launch.json').exists(),
                'A local continuation launch is already recorded')
        plans.append(plan)
    return scope, plans


def start_backup(plan):
    arm = plan['arm']
    folder = HERE / arm
    control = Path(plan['local_backup_root']).parent
    command = [sys.executable, '-B', '-W', 'ignore', '-X', 'utf8', str(HERE / 'continuation_backup.py'), arm]
    stdout_path, stderr_path = (folder / f'backup.{name}.log' for name in ('stdout', 'stderr'))
    with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
        worker = subprocess.Popen(command, cwd=control, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                  creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
    entry = {'arm': arm, 'port': plan['port'], 'pid': worker.pid, 'started_local_utc': now(),
             'created_timestamp': psutil.Process(worker.pid).create_time(), 'command': command,
             'local_control': str(control), 'stderr_path': str(stderr_path), 'stdout_path': str(stdout_path)}
    save(folder / 'local_backup_launch.json', entry)
    return worker, entry


def await_backup(plan, worker, entry):
    status = Path(plan['local_backup_root']) / 'monitor_status.json'
    started = datetime.fromtimestamp(entry['created_timestamp'], timezone.utc)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        require(worker.poll() is None, 'Backup client exited: ' + entry['stderr_path'])
        if status.is_file():
            receipt = read(status)
            if datetime.fromisoformat(receipt['observed_utc']) >= started:
                if receipt['status'] == 'PINNED_SFTP_CONNECTED':
                    require(receipt['next_backup_epoch'] == plan['accepted_rounds'], 'Backup cursor differs')
                    return
                require(receipt['status'] != 'MONITOR_FAILED_NO_FURTHER_ACKS',
                        'Backup client reported a failure')
        time.sleep(2)
    raise TimeoutError('Backup client did not verify the retained prefix and connect')


def remote_launch(plan, limits, scope, authorization_sha):
    arm = plan['arm']
    root = plan['continuation_root']
    folder = HERE / arm
    auth = {'authorized': True, **limits, 'recorded_local_utc': now(),
            'pair_authorization_sha256': authorization_sha, 'user_instruction': scope['user_instruction']}
    if scope.get('authorization_basis') == 'original_run_remaining_budget':
        auth['authorization_basis'] = scope['authorization_basis']
        auth['original_authorization_sha256'] = scope['original_authorization_sha256']
    save(folder / 'continuation_authorization.json', auth)
    auth_path = folder / 'continuation_authorization.json'
    auth_sha = sha(auth_path)
    runtime = plan['runtime']
    command = ['env', 'ATEN_CPU_CAPABILITY=avx2', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONHASHSEED=0',
               'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'OMP_NUM_THREADS=4', 'MKL_NUM_THREADS=4',
               'OPENBLAS_NUM_THREADS=4', 'CUBLAS_WORKSPACE_CONFIG=:4096:8', runtime, '-B',
               root + '/continuation_launch.py', 'launch', plan['original_protocol_sha256'], auth_sha]
    with connect(plan['port']) as client:
        with client.open_sftp() as sftp:
            sftp.get_channel().settimeout(30)
            with sftp.open(root + '/continuation_authorization.json', 'wx') as stream:
                stream.write(auth_path.read_bytes())
        stdin, stdout, stderr = client.exec_command(shlex.join(command), timeout=180)
        stdin.channel.shutdown_write()
        result = {'command': command, 'stdout': stdout.read().decode(errors='replace'),
                  'stderr': stderr.read().decode(errors='replace'),
                  'exit_code': stdout.channel.recv_exit_status(), 'recorded_local_utc': now()}
    save(folder / 'remote_launch.json', result)
    require(result['exit_code'] == 0, result['stderr'])
    return json.loads(result['stdout'])


def observe(plan, launch):
    arm, port, root = (plan[k] for k in ('arm', 'port', 'continuation_root'))
    source = 'observed_run_root = ' + repr(root) + '\n' + (PROJECT / 'observe_remote_run.py').read_text()
    deadline = time.monotonic() + 240
    with connect(port) as client:
        while time.monotonic() < deadline:
            result = run_script(client, source, python=plan['runtime'], timeout=45)
            require(result['exit_code'] == 0, result['stderr'])
            obs = json.loads(result['stdout'])
            sup, receipt = (obs['records'][k] or {} for k in ('supervisor.json', 'run/receipt.json'))
            require(sup.get('status') not in ('HARD_TIMEOUT', 'STOPPED') and receipt.get('status') != 'FAILED',
                    'Continuation worker failed; inspect remote supervisor and training log')
            if sup.get('status') == 'RUNNING' and receipt.get('status') == 'RUNNING':
                require(sup['supervisor_pid'] == launch['supervisor_pid']
                        and sup['supervisor_start_ticks'] == launch['supervisor_start_ticks']
                        and sup['protocol_sha256'] == receipt['protocol_sha256'] == plan['original_protocol_sha256']
                        and sup['runner_sha256'] == receipt['runner_sha256'] == plan['continuation_code']['continuation_train.py']
                        and receipt['first_epoch_in_this_segment'] == plan['accepted_rounds']
                        and receipt['resumed_from_checkpoint_sha256'] == plan['checkpoint_sha256']
                        and obs['boot_id'] == plan['boot_id'] and obs['arm'] == arm
                        and obs['processes'] and all(x.get('live') for x in obs['processes'].values()),
                        'Continuation worker or checkpoint identity differs')
                return {**obs, 'port': port, 'remote_root': root}
            time.sleep(5)
    raise TimeoutError('Unable to confirm a live continuation worker')


def launch_arm(plan, scope, authorization_sha):
    worker, entry = start_backup(plan)
    await_backup(plan, worker, entry)
    launch = remote_launch(plan, scope['arms'][plan['arm']], scope, authorization_sha)
    observed = observe(plan, launch)
    require(worker.poll() is None, 'Backup client exited while starting worker')
    return observed, entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('authorization', type=Path)
    args = parser.parse_args()
    scope, plans = confirmed(args.authorization)
    digest = sha(args.authorization)
    output = HERE / f'continuation_bindings_{digest[:12]}.json'
    require(not output.exists(), 'This authorization already has a launch record')
    observations, clients, errors = [], [], []
    with ThreadPoolExecutor(len(plans)) as pool:
        work = [pool.submit(launch_arm, plan, scope, digest) for plan in plans]
        for plan, task in zip(plans, work):
            try:
                obs, client = task.result()
                observations.append(obs)
                clients.append(client)
            except Exception as error:
                errors.append({'arm': plan['arm'], 'error': f'{type(error).__name__}: {error}'})
    report = {'recorded_local_utc': now(), 'authorization_sha256': digest,
              'remote': observations, 'local_backup_clients': clients, 'issues': errors}
    save(output, report)
    print(json.dumps({'bindings': str(output), 'verified_live_arms': [x['arm'] for x in observations],
                      'issues': errors}, ensure_ascii=True))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
