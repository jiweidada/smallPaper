"""Run and archive the two authorized 320-round MSCA comparisons."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / '公平主对照FRESH_20261112_v1'
sys.path.insert(0, str(PREVIOUS))
import ssh_control as ssh

ssh.HERE = Path(os.environ['LOCALAPPDATA']) / 'MPSMA_DEIM' / 'capacity_pair_20261116'
ssh.PORTS = (30737, 30959)
BACKUP = Path('D:/MPSMA_DEIM_experiment_backups/MSCA_CAPACITY_20261116')
CACHE = '/tmp/mpsma_msca_pair_20261116_data'
TORCH_SITE = HERE.parents[1] / '本地CPU核验_20261103/.runtime/site-packages'
CPU_PYTHON = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
ENV = ['env', 'ATEN_CPU_CAPABILITY=avx2', 'PYTHONDONTWRITEBYTECODE=1', 'HF_HUB_OFFLINE=1',
       'TRANSFORMERS_OFFLINE=1', 'OMP_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4',
       'CUBLAS_WORKSPACE_CONFIG=:4096:8', 'PYTHONHASHSEED=0']
DRIVER_NAMES = ('stage_arm.py', 'fair_helpers.py', 'train_runner.py', 'launch_arm.py',
                'backup_protocol.py', 'preflight.py')


@dataclass(frozen=True)
class Server:
    port: int
    arm: str
    mount: str
    old_arm: str
    env_path: str
    gpu: str
    peer: int

    @property
    def root(self):
        return PurePosixPath(self.mount) / 'mpsma_msca_capacity_20261116'

    @property
    def control(self):
        return self.root / self.arm

    @property
    def previous(self):
        return PurePosixPath(self.mount) / 'mpsma_fresh_pair_20261112_v1' / self.old_arm

    @property
    def python(self):
        return PurePosixPath(self.mount) / self.env_path


SERVERS = [
    Server(30737, 'FULL_MSCA', '/24085403014', 'FULL_FRESH', 'deim_migration_runtime/envs/deim/bin/python',
           'GPU-43038ae9-d619-20e2-0216-822fd1c50fed', 30959),
    Server(30959, 'NO_MSCA_MATCHED', '/25085400120', 'BASE_FRESH', 'envs/deim/bin/python',
           'GPU-ed6ef6ab-7568-a8e8-a42c-039312c8c0bd', 30737),
]


def cmd(args):
    return shlex.join([str(x) for x in args])


def run(client, command, timeout=35):
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    stdin.channel.shutdown_write()
    out, err = stdout.read().decode(), stderr.read().decode(errors='replace')
    status = stdout.channel.recv_exit_status()
    if status != 0:
        raise RuntimeError(f'Remote command exit {status}: {(err or out)[-4500:]}')
    return out.strip()


def upload(sftp, remote, payload):
    path = str(remote)
    try:
        with sftp.open(path, 'rb') as stream:
            assert stream.read() == payload, path
        return
    except FileNotFoundError:
        pass
    with sftp.open(path, 'wx') as stream:
        stream.write(payload)
    with sftp.open(path, 'rb') as stream:
        assert stream.read() == payload, path


def download(client, remote, dest):
    with client.open_sftp() as sftp, sftp.open(str(remote), 'rb') as stream:
        content = stream.read()
    if dest.exists():
        assert dest.read_bytes() == content, dest
    else:
        with dest.open('xb') as stream:
            stream.write(content)
    return json.loads(content)


def save(dest, value):
    with dest.open('x', encoding='utf-8') as out:
        json.dump(value, out, ensure_ascii=False, indent=2)
        out.write('\n')


def runlog(client, logfile, command):
    remote = str(logfile)
    latch = shlex.quote(remote + '.lock')
    receipt = shlex.quote(remote + '.exit')
    quoted_log = shlex.quote(remote)
    inner = f'{cmd(command)} >{quoted_log} 2>&1; printf "%s\\n" "$?" >{receipt}'
    return run(client, f'mkdir {latch} && {{ nohup sh -c {shlex.quote(inner)} </dev/null >/dev/null 2>&1 & echo $!; }}')


def transfer_scaffold(client, s):
    run(client, cmd(['mkdir','-p',s.root/'driver',s.root/'staging_inputs']))
    with client.open_sftp() as f:
        upload(f, s.root/'driver'/'prepare_data.py', (HERE/'prepare_data.py').read_bytes())
        upload(f, s.root/'driver'/'environment_snapshot.py', (PREVIOUS/'environment_snapshot.py').read_bytes())


def prepare(client, s):
    transfer_scaffold(client, s)
    # Background the expensive imports and bundle checks so transient SSH sessions do not interrupt them.
    shell_script = (
        f'{cmd(ENV + [s.python, "-B", s.root/"driver/environment_snapshot.py"])} >{shlex.quote(str(s.root/"environment.json"))}'
        ' && ' + cmd([s.python, '-B', s.root/'driver/prepare_data.py', s.mount, s.previous])
    )
    return runlog(client, s.root/'inputs.log', ['sh','-c',shell_script])


def collect_inputs(client, s):
    run(client, f'test "$(cat {shlex.quote(str(s.root/"inputs.log.exit"))})" = 0')
    data = json.loads(run(client, cmd(['cat',s.root/'inputs.log'])).splitlines()[-1])
    assert data['cache'] == CACHE and data['verified_files'] == 7021
    env = download(client, s.root/'environment.json', HERE/f'environment_{s.port}.json')
    assert env['cuda_available'] and env['torch'] == '2.5.1+cu124' and env['cuda'] == '12.4'
    save(HERE/f'data_{s.port}.json',data)
    return data


def stage(client, s):
    local_env = HERE/f'environment_{s.port}.json'
    other_env = HERE/f'environment_{s.peer}.json'
    with client.open_sftp() as f:
        for name in DRIVER_NAMES:
            upload(f,s.root/'driver'/name,(HERE/'driver'/name).read_bytes())
        upload(f,s.root/'driver/capacity_match.py',(HERE/'capacity_match.py').read_bytes())
        upload(f,s.root/'staging_inputs/authorization.json',(HERE/'authorization.json').read_bytes())
        for item in (local_env,other_env):
            upload(f,s.root/'staging_inputs'/item.name,item.read_bytes())
    script = [
        s.python,'-B',s.root/'driver/stage_arm.py','prepare','--root',s.previous/'frozen_parent',
        '--data-root', CACHE, '--arm-dir',s.control,'--arm',s.arm,'--gpu-uuid',s.gpu,
        '--authorization-file',s.root/'staging_inputs/authorization.json',
        '--environment-file',s.root/'staging_inputs'/local_env.name,
        '--peer-environment-file',s.root/'staging_inputs'/other_env.name,
    ]
    return runlog(client,s.root/'stage.log',ENV+script)


def preflight(client, s):
    run(client,f'test "$(cat {shlex.quote(str(s.root/"stage.log.exit"))})" = 0')
    return runlog(client,s.root/'preflight.log',ENV+[s.python,'-B',s.control/'preflight.py',
                  '--arm-dir',s.control,'--arm',s.arm])


def freeze(client, s):
    run(client,f'test "$(cat {shlex.quote(str(s.root/"preflight.log.exit"))})" = 0')
    return runlog(client,s.root/'freeze.log',ENV+[s.python,'-B',s.root/'driver/stage_arm.py',
                  'finalize','--arm-dir',s.control])


def archive_binding(client, s):
    run(client,f'test "$(cat {shlex.quote(str(s.root/"freeze.log.exit"))})" = 0')
    plan = download(client,s.control/'execution_protocol.json', HERE/f'protocol_{s.arm}.json')
    pre = download(client,s.control/'inputs/preflight.json',HERE/f'preflight_{s.arm}.json')
    assert pre['status']=='PREFLIGHT_PASSED_NO_OPTIMIZER_UPDATE' and pre['optimizer_updates']==0
    assert plan['arm']==s.arm and plan['gpu_uuid']==s.gpu
    return pre


def publish_probe(client, s):
    return run(client,cmd([s.python,'-B',s.control/'backup_protocol.py','publish-probe',
                            '--output',s.control/'run','--anchor',s.control/'inputs']),timeout=65)


def verified_probe(client, s):
    return run(client,cmd([s.python,'-B',s.control/'backup_protocol.py','verify-probe',
                            '--output',s.control/'run']),timeout=65)


def launch(client, s):
    local = json.loads((BACKUP/s.arm/'backups/monitor_status.json').read_text())
    assert local['status']=='PROBE_VERIFIED', s.arm
    verified_probe(client,s)
    protocol = sha256((HERE/f'protocol_{s.arm}.json').read_bytes()).hexdigest()
    result = json.loads(run(client,cmd(ENV+[s.python,'-B',s.control/'launch_arm.py','launch',protocol]),timeout=120))
    save(HERE/f'launch_{s.arm}.json', result)
    return result


def verify_pair_preflight():
    full, matched = [json.loads((HERE/f'preflight_{s.arm}.json').read_text()) for s in SERVERS]
    for field in ('trainable_parameters', 'first100_batches_sha256', 'optimizer_groups'):
        assert full[field] == matched[field], field
    assert full['trainable_parameters'] == 7_540_942
    full_init, matched_init = (r['initialization'] for r in (full, matched))
    assert full_init['reference_initial_sha256'] == matched_init['reference_initial_sha256']
    assert full_init['all_retained_tensors_verified'] and matched_init['all_retained_tensors_verified']
    assert full_init['arm'] == 'FULL_MSCA' and matched_init['arm'] == 'NO_MSCA_MATCHED'
    assert matched_init['replacement_parameter_count'] == 3_841_280
    assert len(matched['replacement_gradient_check']) == 3
    for server in SERVERS:
        local = json.loads((BACKUP/server.arm/'backups/monitor_status.json').read_text())
        assert local['status'] == 'PROBE_VERIFIED'


def status(client,s):
    script = ['sh','-c',f'cd {shlex.quote(str(s.root))} && for p in inputs stage preflight freeze; do '
              'if test -f "$p.log.exit"; then printf "%s exit " "$p"; cat "$p.log.exit"; '
              'if test "$(cat "$p.log.exit")" != 0; then tail -8 "$p.log"; fi; fi; done; '
              f'if test -f {s.arm}/run/receipt.json; then cat {s.arm}/run/receipt.json; '
              f'elif test -f {s.arm}/supervisor.json; then cat {s.arm}/supervisor.json; fi']
    return run(client,cmd(script))


def recovery_segments(arm):
    return sorted(directory / arm for directory in HERE.glob('recovery_[0-9]*')
                  if (directory / arm / 'launch.json').is_file())


def live_status(client, s):
    remote = """
import json, subprocess
from pathlib import Path
root = Path(RUN_ROOT)
sup = json.loads((root/'supervisor.json').read_text())
receipt_path = root/'run/receipt.json'
receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
def bound(pid, expected):
    if pid is None or expected is None:
        return False
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
    except FileNotFoundError:
        return False
    boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    return boot_id == sup['boot_id'] and fields[0] != 'Z' and int(fields[19]) == expected
gpu = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total,utilization.gpu',
                               '--format=csv,noheader,nounits'], text=True).strip()
print(json.dumps({'arm': sup['arm'], 'protocol_sha256': sup['protocol_sha256'],
                  'recovery_plan_sha256': sup.get('recovery_plan_sha256'),
                  'supervisor_status': sup['status'], 'exit_code': sup.get('exit_code'),
                  'worker_pid': sup.get('pid'), 'worker_live': bound(sup.get('pid'), sup.get('start_ticks')),
                  'supervisor_pid': sup['supervisor_pid'],
                  'supervisor_start_ticks': sup['supervisor_start_ticks'],
                  'supervisor_live': bound(sup['supervisor_pid'], sup['supervisor_start_ticks']),
                  'receipt_pid': receipt.get('pid'), 'receipt_worker_bound': receipt.get('pid') == sup.get('pid'),
                  'worker_status': receipt.get('status') if receipt.get('pid') == sup.get('pid') else 'INITIALIZING',
                  'phase': (receipt.get('phase') or 'model_state_restore') if receipt.get('pid') == sup.get('pid') else 'startup_verification',
                  'completed_rounds': receipt.get('completed_epochs', 0), 'batch': receipt.get('batch'),
                  'optimizer_attempts': receipt.get('optimizer_step_attempts', 0),
                  'details_on_failure': receipt.get('traceback'), 'gpu': gpu}))
"""
    result = ssh.run_script(client, f'RUN_ROOT={str(s.control)!r}\n' + remote, timeout=35)
    if result['exit_code'] != 0:
        raise RuntimeError(result['stderr'])
    evidence = json.loads(result['stdout'])
    assert evidence['arm'] == s.arm
    launch = json.loads((HERE/f'launch_{s.arm}.json').read_text())
    expected_gpu = s.gpu
    for recovery in recovery_segments(s.arm):
        prior = json.loads((recovery/'original_supervisor.json').read_text())
        assert prior['supervisor_pid'] == launch['supervisor_pid']
        assert prior['supervisor_start_ticks'] == launch['start_ticks']
        launch = json.loads((recovery/'launch.json').read_text())
        recovery_plan = json.loads((recovery/'recovery_plan.json').read_text())
        assert launch['recovery_plan_sha256'] == sha256((recovery/'recovery_plan.json').read_bytes()).hexdigest()
        if 'hardware_migration' in recovery_plan:
            assert recovery_plan['hardware_migration']['previous_gpu_uuid'] == expected_gpu
            expected_gpu = recovery_plan['hardware_migration']['current_gpu_uuid']
    if recovery_segments(s.arm):
        assert evidence['recovery_plan_sha256'] == launch['recovery_plan_sha256']
    assert evidence['gpu'].split(',')[0] == expected_gpu
    assert evidence['protocol_sha256'] == launch['protocol_sha256']
    assert evidence['supervisor_pid'] == launch['supervisor_pid']
    assert evidence['supervisor_start_ticks'] == launch['start_ticks']
    archive = BACKUP / s.arm
    evidence['backup'] = json.loads((archive/'backups/monitor_status.json').read_text())
    monitor = json.loads((archive/'monitor_launch.json').read_text())
    import psutil
    try:
        process = psutil.Process(monitor['pid'])
        evidence['backup_client_live'] = process.is_running() and process.cmdline()[1:] == monitor['command'][1:]
    except psutil.NoSuchProcess:
        evidence['backup_client_live'] = False
    return evidence


def local_monitors():
    assert shutil.disk_usage(BACKUP.parent).free >= 232 * 1024**3
    assert CPU_PYTHON.is_file() and TORCH_SITE.is_dir()
    BACKUP.mkdir(exist_ok=True)
    for s in SERVERS:
        control = BACKUP/s.arm
        control.mkdir()
        for name in ('backup_protocol.py','backup_monitor.py','checkpoint_validator.py'):
            shutil.copyfile(HERE/'driver'/name,control/name)
        shutil.copyfile(HERE/f'protocol_{s.arm}.json',control/'execution_protocol.json')
        command = [sys.executable,'-B','-W','ignore','-X','utf8',str(__file__),'backup-client',str(s.port)]
        with (control/'monitor.stdout.log').open('xb') as out,(control/'monitor.stderr.log').open('xb') as err:
            process = subprocess.Popen(command,cwd=control,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                                       creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
        save(control/'monitor_launch.json',{'pid':process.pid,'arm':s.arm,'port':s.port,'command':command})
        print(s.arm, process.pid, flush=True)


def backup_client(port):
    s = next(s for s in SERVERS if s.port==port)
    control = BACKUP/s.arm
    sys.path.insert(0,str(control))
    import backup_monitor
    sys.argv = [str(control/'backup_monitor.py'),'--host',ssh.HOST,'--port',str(port),'--user','root',
                '--known-hosts',str(ssh.HERE/'known_hosts'),'--remote-root',str(s.control),
                '--local-root',str(control),'--checkpoint-python',str(CPU_PYTHON),'--torch-site',str(TORCH_SITE)]
    backup_monitor.main(password_provider=lambda host,p,user: ssh.protect(
        ssh.credential_path(p).read_bytes(),decode=True).decode('utf-8'))


def pair(phase, selected):
    def one(server):
        with ssh.connect(server.port) as client:
            result = phase(client,server)
        print(server.port,server.arm,result,flush=True)
        return result
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(one,s) for s in selected]
        return [f.result() for f in futures]


def main():
    phases = {'prepare':prepare,'collect-inputs':collect_inputs,'stage':stage,'preflight':preflight,
              'freeze':freeze,'bindings':archive_binding,'publish-probe':publish_probe,
              'accept-probe': verified_probe, 'launch':launch,'status':status,'live':live_status}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=[*phases,'monitors','backup-client'])
    parser.add_argument('port',nargs='?',type=int,choices=ssh.PORTS)
    args = parser.parse_args()
    if args.phase=='monitors':
        local_monitors()
    elif args.phase=='backup-client':
        backup_client(args.port)
    else:
        if args.phase == 'launch':
            verify_pair_preflight()
        pair(phases[args.phase], [s for s in SERVERS if args.port is None or args.port==s.port])


if __name__ == '__main__':
    main()
