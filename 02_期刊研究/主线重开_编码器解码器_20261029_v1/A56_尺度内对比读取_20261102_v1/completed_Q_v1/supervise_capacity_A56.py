from pathlib import Path
import json
import os
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
with (P / 'capacity_worker.log').open('xb') as log:
    child = subprocess.Popen([sys.executable, '-u', str(P/'capacity_A56.py')],
                             cwd=P, stdout=log, stderr=subprocess.STDOUT)
    stat = Path(f'/proc/{child.pid}/stat').read_text().split()
    (P/'capacity_process.json').write_text(json.dumps(
        {'supervisor_PID': os.getpid(), 'child_PID': child.pid,
         'child_start_ticks': int(stat[21])}, indent=2))
    start = time.monotonic()
    while child.poll() is None:
        if time.monotonic() - start > 3600:
            child.terminate()
            break
        time.sleep(5)
    code = child.wait()
    (P/'capacity_exit.json').write_text(json.dumps(
        {'child_exit_code': code, 'seconds': time.monotonic()-start,
         'no_restart_or_formal_queue': True}, indent=2))
    sys.exit(0 if code == 0 else 1)
