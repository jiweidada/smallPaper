from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def identity(pid):
    stat = Path('/proc', str(pid), 'stat').read_text().split(') ', 1)[1].split()
    return {'pid': pid, 'start_ticks': int(stat[19])}


if __name__ == '__main__':
    stage = ROOT / 'capacity_stage'
    stage.mkdir(exist_ok=False)
    save(stage / 'supervisor_identity.json', identity(os.getpid()))
    with (stage / 'stdout.log').open('x') as stdout, (stage / 'stderr.log').open('x') as stderr:
        child = subprocess.Popen([sys.executable, '-u', str(ROOT / 'capacity_A49.py')],
            cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True)
        child_identity = identity(child.pid)
        save(stage / 'child_identity.json', child_identity)
        start = time.monotonic()
        while child.poll() is None:
            save(stage / 'supervisor_progress.json', {'status': 'RUNNING', 'child': child_identity,
                'elapsed_seconds': time.monotonic() - start, 'auto_retry': False})
            if time.monotonic() - start > 1200:
                assert identity(child.pid) == child_identity
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
                break
            time.sleep(2)
        final = {'status': 'COMPLETED' if child.returncode == 0 else 'FAILED_STOPPED_NO_RETRY',
                 'exit_code': child.returncode, 'child': child_identity,
                 'elapsed_seconds': time.monotonic() - start, 'formal_training_queued': False}
        save(stage / 'exit.json', final)
        save(stage / 'supervisor_progress.json', final)
