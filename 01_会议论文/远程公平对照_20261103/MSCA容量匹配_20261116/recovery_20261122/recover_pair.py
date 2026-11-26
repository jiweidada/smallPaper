"""Continue the authorized pair from complete states on renewed RTX 3090s."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import pair

GPU = {30737: 'GPU-4bd279c5-4e39-bd08-618e-20da276d13ed',
       30959: 'GPU-43038ae9-d619-20e2-0216-822fd1c50fed'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_once(path, value):
    if path.exists():
        assert read(path) == value, f'Existing evidence differs: {path}'
    else:
        pair.save(path, value)


def epoch_loop(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_run')
    block = next(n for n in fn.body if isinstance(n, ast.Try))
    loop = next(n for n in block.body if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'epoch')
    return ast.dump(loop, include_attributes=False)


def stage(client, server, evidence):
    directory = server.control / 'recovery_20261122_files'
    assert pair.run(client, pair.cmd(['cat', str(directory / 'preparation.log') + '.exit'])) == '0'
    for name in ('environment.json', 'data_preparation.json'):
        pair.download(client, directory / name, evidence / name)
    pair.download(client, server.control / 'dataset_bundle_manifest.json', evidence / 'dataset_bundle_manifest.json')
    frozen_protocol = read(pair.HERE / f'protocol_{server.arm}.json')
    frozen = {entry['path']: entry['sha256'] for entry in frozen_protocol['frozen_files']}
    assert sha(evidence / 'dataset_bundle_manifest.json') == frozen['dataset_bundle_manifest.json']
    current = read(evidence / 'environment.json')
    original = read(pair.HERE / f'environment_{server.port}.json')
    assert sha(pair.HERE / f'environment_{server.port}.json') == frozen['environment.current.json']
    for key in ('torch', 'torchvision', 'cuda', 'cudnn', 'packages'):
        assert current[key] == original[key], f'Runtime changed: {key}'
    data = read(evidence / 'data_preparation.json')
    assert data['cache'] == pair.CACHE and data['verified_files'] == 7021
    data_manifest = read(evidence / 'dataset_bundle_manifest.json')
    assert data['archive_sha256'] == data_manifest['zip_sha256'] and len(data_manifest['files']) == 7021
    boundary = read(evidence / 'salvaged_boundary.json')
    prior = read(evidence / 'original_supervisor.json')
    previous = pair.HERE / 'recovery_20261118' / server.arm
    previous_launch = read(previous / 'launch.json')
    previous_plan = read(previous / 'recovery_plan.json')
    assert prior['supervisor_pid'] == previous_launch['supervisor_pid']
    assert prior['supervisor_start_ticks'] == previous_launch['start_ticks']
    assert prior['recovery_plan_sha256'] == sha(previous / 'recovery_plan.json')
    assert prior['original_deadline_utc'] == previous_plan['deadline_utc']
    assert datetime.fromisoformat(prior['original_deadline_utc']) > datetime.now(timezone.utc)
    protocol_sha = previous_launch['protocol_sha256']
    observed = pair.run(client, pair.cmd(['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader,nounits']))
    assert observed == GPU[server.port] + ', NVIDIA GeForce RTX 3090'
    driver = pair.run(client, pair.cmd(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits']))
    plan = {
        'arm': server.arm, 'protocol_sha256': protocol_sha,
        'accepted_rounds': boundary['accepted_rounds'], 'checkpoint_sha256': boundary['checkpoint_sha256'],
        'best': boundary['manifest']['best'], 'stage_one': boundary['manifest']['stage_one'],
        'runner_sha256': sha(HERE / 'resume_train_20261122.py'),
        'supervisor_sha256': sha(HERE / 'resume_launch_20261122.py'),
        'original_supervisor_sha256': sha(evidence / 'original_supervisor.json'),
        'original_failed_receipt_sha256': sha(evidence / 'original_failed_receipt.json'),
        'previous_recovery_plan_sha256': sha(previous / 'recovery_plan.json'),
        'environment_snapshot_sha256': sha(evidence / 'environment.json'),
        'data_preparation_sha256': sha(evidence / 'data_preparation.json'),
        'hardware_migration': {'previous_gpu_uuid': server.gpu, 'current_gpu_uuid': GPU[server.port],
                               'gpu_name': 'NVIDIA GeForce RTX 3090',
                               'current_driver_version': driver,
                               'reason': 'User renewed the two GPU instances and requested checkpoint continuation.'},
        'repeated_updates': 0, 'repeated_evaluations': 0,
        'total_optimizer_update_limit': 517440, 'total_evaluation_limit': 320,
        'deadline_utc': prior['original_deadline_utc'],
        'authorization_basis': 'Original authorized pair, resumed goal, renewed server entries and user checkpoint continuation request on 2026-11-22.',
        'training_loop': 'AST-identical to frozen runner. Restore complete post-policy model, optimizer, EMA and all saved RNG fields.',
    }
    assert epoch_loop(HERE / 'resume_train_20261122.py') == epoch_loop(pair.HERE / 'driver/train_runner.py')
    save_once(evidence / 'recovery_plan.json', plan)
    with client.open_sftp() as sftp:
        pair.upload(sftp, server.control / 'recovery_plan_20261122.json', (evidence / 'recovery_plan.json').read_bytes())
        for name in ('resume_train_20261122.py', 'resume_launch_20261122.py'):
            pair.upload(sftp, server.control / name, (HERE / name).read_bytes())
    probe = '''
import json,sys
from pathlib import Path
sys.path.insert(0,ROOT)
import resume_launch_20261122 as supervisor
plan,remaining=supervisor.ready(PROTOCOL,RECOVERY)
print(json.dumps({'status':'RECOVERY_BOUNDARY_READY','next_round':plan['accepted_rounds']+1,
                  'remaining_seconds':remaining,'hardware_migration':plan['hardware_migration']}))
'''
    prelude = f'ROOT={str(server.control)!r}\nPROTOCOL={protocol_sha!r}\nRECOVERY={sha(evidence / "recovery_plan.json")!r}\n'
    result = pair.ssh.run_script(client, prelude + probe, python=str(server.python), timeout=180)
    assert result['exit_code'] == 0, result
    save_once(evidence / 'readiness.json', json.loads(result['stdout']))
    print(server.arm, result['stdout'].strip(), flush=True)


def monitor(server, evidence):
    import psutil
    control = pair.BACKUP / server.arm
    path = control / 'monitor_launch.json'
    old = read(path)
    try:
        process = psutil.Process(old['pid'])
        assert process.cmdline()[1:] != old['command'][1:], 'Backup client is already live'
    except psutil.NoSuchProcess:
        pass
    save_once(evidence / 'original_monitor_launch.json', old)
    command = [sys.executable, '-B', '-W', 'ignore', '-X', 'utf8', str(pair.HERE / 'pair.py'),
               'backup-client', str(server.port)]
    with (evidence / 'backup.stdout.log').open('xb') as out, (evidence / 'backup.stderr.log').open('xb') as err:
        process = subprocess.Popen(command, cwd=control, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                   creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
    metadata = {'pid': process.pid, 'arm': server.arm, 'port': server.port, 'command': command,
                'recovery_plan_sha256': sha(evidence / 'recovery_plan.json')}
    temporary = path.with_suffix('.json.partial')
    pair.save(temporary, metadata)
    os.replace(temporary, path)
    pair.save(evidence / 'monitor_launch.json', metadata)
    print(server.arm, 'BACKUP_CLIENT_SPAWNED', process.pid, flush=True)


def launch(client, server, evidence):
    plan = read(evidence / 'recovery_plan.json')
    result = json.loads(pair.run(client, pair.cmd(pair.ENV + [server.python, '-B',
                         server.control / 'resume_launch_20261122.py', 'launch', plan['protocol_sha256'],
                         sha(evidence / 'recovery_plan.json')]), timeout=180))
    pair.save(evidence / 'launch.json', result)
    print(server.arm, result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['stage', 'monitor', 'launch'])
    parser.add_argument('port', type=int, choices=pair.ssh.PORTS)
    args = parser.parse_args()
    server = next(s for s in pair.SERVERS if s.port == args.port)
    evidence = HERE / server.arm
    if args.phase == 'monitor':
        monitor(server, evidence)
    else:
        with pair.ssh.connect(args.port) as client:
            {'stage': stage, 'launch': launch}[args.phase](client, server, evidence)
