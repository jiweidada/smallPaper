"""Sequential capacity only. Never starts formal training."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent

def save(name, obj):
    p = ROOT / name
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    os.replace(tmp, p)

with (ROOT / 'capacity_supervisor_once.lock').open('x') as f:
    f.write(str(os.getpid()))
start = time.monotonic()
results = {}
p = None
try:
    plan = json.loads((ROOT / 'capacity_protocol_FROZEN_v1.json').read_text())
    for name, digest in plan['files_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    for arm in ['B', 'M', 'Q']:
        log = (ROOT / ('capacity_' + arm + '_stdout.log')).open('x')
        p = subprocess.Popen([sys.executable, '-u', str(ROOT / 'capacity_A58.py'), '--arm', arm],
                             cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save('capacity_supervisor_status.json', dict(status='CAPACITY_RUNNING', arm=arm,
             child_pid=p.pid, child_start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],
             completed=list(results), training_queue=[], supervisor_pid=os.getpid()))
        try:
            rc = p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait()
            raise RuntimeError(f'{arm} capacity exceeded 600 seconds; no retry')
        log.close()
        assert rc == 0, (arm, rc)
        result = json.loads((ROOT / ('capacity_' + arm) / 'result.json').read_text())
        assert result['status'] == 'PASS'
        assert result['counts']['full_model_forwards_started'] == result['counts']['optimizer_updates'] == 2
        results[arm] = result
    result = dict(status='PASS', arms=results, real_detector_forwards=6, discarded_optimizer_updates=6,
                  elapsed_seconds=time.monotonic()-start, formal_training_started=False,
                  formal_training_queue=[], protocol_sha256=hashlib.sha256((ROOT / 'capacity_protocol_FROZEN_v1.json').read_bytes()).hexdigest())
    save('capacity_result.json', result)
    save('capacity_supervisor_status.json', dict(status='CAPACITY_COMPLETED', results=list(results),
         training_queue=[], child_pid=None, supervisor_pid=os.getpid()))
    save('capacity_supervisor_exit.json', dict(exit=0))
except BaseException:
    if p is not None and p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL); p.wait()
    save('capacity_supervisor_status.json', dict(status='FAILED_STOPPED_NO_RETRY', completed=list(results),
         training_queue=[], supervisor_pid=os.getpid(), traceback=traceback.format_exc()))
    save('capacity_supervisor_exit.json', dict(exit=1))
    raise
