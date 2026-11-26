"""Restore only frozen data and record the environment on renewed instances."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import pair


def prepare(server):
    evidence = HERE / server.arm
    evidence.mkdir(exist_ok=True)
    directory = server.control / 'recovery_20261122_files'
    with pair.ssh.connect(server.port) as client:
        pair.run(client, pair.cmd(['mkdir', '-p', directory]))
        with client.open_sftp() as sftp:
            pair.upload(sftp, directory / 'prepare_data.py', (pair.HERE / 'prepare_data.py').read_bytes())
            pair.upload(sftp, directory / 'environment_snapshot.py', (pair.PREVIOUS / 'environment_snapshot.py').read_bytes())
        shell = (pair.cmd(pair.ENV + [server.python, '-B', directory / 'environment_snapshot.py'])
                 + ' >' + pair.shlex.quote(str(directory / 'environment.json')) + ' && '
                 + pair.cmd(pair.ENV + [server.python, '-B', directory / 'prepare_data.py', server.mount, server.previous])
                 + ' >' + pair.shlex.quote(str(directory / 'data_preparation.json')))
        pid = int(pair.runlog(client, directory / 'preparation.log', ['sh', '-c', shell]))
        probe = '''
import json
from pathlib import Path
path=Path('/proc')/str(PID)/'stat'
print(json.dumps({'pid':PID,'start_ticks':int(path.read_text().rsplit(') ',1)[1].split()[19]) if path.exists() else None,
                  'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))
'''
        checked = pair.ssh.run_script(client, f'PID={pid}\n' + probe, timeout=30)
        assert checked['exit_code'] == 0, checked
        launch = {**json.loads(checked['stdout']), 'arm': server.arm, 'port': server.port,
                  'remote_directory': str(directory), 'training_started': False}
        pair.save(evidence / 'preparation_launch.json', launch)
        return launch


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(prepare, pair.SERVERS):
            print(json.dumps(result), flush=True)
