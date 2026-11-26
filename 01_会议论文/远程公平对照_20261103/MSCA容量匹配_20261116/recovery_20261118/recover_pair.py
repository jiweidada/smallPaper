"""Stage and launch continuation from independently accepted complete states."""

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import pair


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage(client, server, evidence):
    boundary = read(evidence / 'salvaged_boundary.json')
    prior = read(evidence / 'original_supervisor.json')
    original_launch = read(pair.HERE / f'launch_{server.arm}.json')
    assert prior['supervisor_pid'] == original_launch['supervisor_pid']
    assert prior['supervisor_start_ticks'] == original_launch['start_ticks']
    assert prior['protocol_sha256'] == original_launch['protocol_sha256']
    plan = {'arm': server.arm, 'protocol_sha256': prior['protocol_sha256'],
            'accepted_rounds': boundary['accepted_rounds'],
            'checkpoint_sha256': boundary['checkpoint_sha256'],
            'best': boundary['manifest']['best'], 'stage_one': boundary['manifest']['stage_one'],
            'runner_sha256': sha(HERE / 'resume_train.py'),
            'supervisor_sha256': sha(HERE / 'resume_launch.py'),
            'original_supervisor_sha256': sha(evidence / 'original_supervisor.json'),
            'original_failed_receipt_sha256': sha(evidence / 'original_failed_receipt.json'),
            'repeated_updates': 0, 'repeated_evaluations': 0,
            'total_optimizer_update_limit': 517440, 'total_evaluation_limit': 320,
            'deadline_utc': (datetime.fromisoformat(prior['started_utc']) + timedelta(days=7)).isoformat(),
            'authorization_basis': 'Continue the two authorized runs under the resumed monitoring goal.',
            'training_loop': 'AST-identical to the frozen runner. Restore saved complete post-policy state and all RNG fields.'}
    plan_path = evidence / 'recovery_plan.json'
    if plan_path.exists():
        assert read(plan_path) == plan
    else:
        pair.save(plan_path, plan)
    with client.open_sftp() as sftp:
        for local_name, remote_name in [('original_supervisor.json', 'recovery_original_supervisor.json'),
                                        ('original_failed_receipt.json', 'recovery_original_failed_receipt.json'),
                                        ('recovery_plan.json', 'recovery_plan.json')]:
            pair.upload(sftp, server.control / remote_name, (evidence / local_name).read_bytes())
        for name in ('resume_train.py', 'resume_launch.py'):
            pair.upload(sftp, server.control / name, (HERE / name).read_bytes())
    script = '''
import ast, sys
from pathlib import Path
here = Path(ROOT)
sys.path.insert(0,str(here))
import resume_launch
plan, remaining = resume_launch.ready(PROTOCOL, RECOVERY)
original=ast.parse((here/'train_runner.py').read_text())
resumed=ast.parse((here/'resume_train.py').read_text())
def epoch_loop(tree):
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_run')
    block=next(n for n in fn.body if isinstance(n,ast.Try))
    return next(n for n in block.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch')
assert ast.dump(epoch_loop(original),include_attributes=False)==ast.dump(epoch_loop(resumed),include_attributes=False)
print({'status':'RECOVERY_BOUNDARY_READY','next_round':plan['accepted_rounds']+1,'remaining_seconds':remaining})
'''
    result = pair.ssh.run_script(client, f'ROOT={str(server.control)!r}\nPROTOCOL={plan["protocol_sha256"]!r}\n'
                                  f'RECOVERY={sha(plan_path)!r}\n' + script,
                                  python=str(server.python), timeout=120)
    assert result['exit_code'] == 0, result
    print(server.arm, result['stdout'].strip(), flush=True)


def monitor(server, evidence):
    control = pair.BACKUP / server.arm
    path = control / 'monitor_launch.json'
    old = read(path)
    import psutil
    try:
        p = psutil.Process(old['pid'])
        assert p.cmdline()[1:] != old['command'][1:], 'Backup client is already running'
    except psutil.NoSuchProcess:
        pass
    history = evidence / 'original_monitor_launch.json'
    if history.exists():
        assert read(history) == old
    else:
        pair.save(history, old)
    command = [sys.executable, '-B', '-W', 'ignore', '-X', 'utf8', str(pair.HERE / 'pair.py'),
               'backup-client', str(server.port)]
    with (evidence / 'backup.stdout.log').open('xb') as out, (evidence / 'backup.stderr.log').open('xb') as err:
        p = subprocess.Popen(command, cwd=control, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                             creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
    metadata = {'pid': p.pid, 'arm': server.arm, 'port': server.port, 'command': command,
                'recovery_plan_sha256': sha(evidence / 'recovery_plan.json')}
    temporary = path.with_suffix('.json.partial')
    pair.save(temporary, metadata)
    os.replace(temporary, path)
    pair.save(evidence / 'monitor_launch.json', metadata)
    print(server.arm, 'BACKUP_CLIENT_SPAWNED', p.pid, flush=True)


def launch(client, server, evidence):
    plan = read(evidence / 'recovery_plan.json')
    metadata = json.loads(pair.run(client, pair.cmd(pair.ENV + [server.python, '-B',
                         server.control / 'resume_launch.py', 'launch', plan['protocol_sha256'],
                         sha(evidence / 'recovery_plan.json')]), timeout=120))
    pair.save(evidence / 'launch.json', metadata)
    print(server.arm, metadata, flush=True)


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
        with pair.ssh.connect(server.port) as client:
            {'stage': stage, 'launch': launch}[args.phase](client, server, evidence)
