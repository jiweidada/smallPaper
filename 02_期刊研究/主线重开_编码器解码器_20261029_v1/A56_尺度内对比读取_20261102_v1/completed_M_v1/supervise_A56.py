from pathlib import Path
import json
import os
import subprocess
import sys
import time
from runtime_A56 import validate_files

P = Path(__file__).resolve().parent


def write(name, value):
    (P/name).write_text(json.dumps(value, indent=2))


def main():
    validate_files('protocol_FROZEN_v1.json')
    with (P/'formal_supervisor_once.lock').open('x') as f:
        f.write(str(os.getpid()))
    records = []
    for arm in ['M', 'Q']:
        validate_files('protocol_FROZEN_v1.json')
        log = (P/f'train_{arm}.log').open('xb')
        child = subprocess.Popen([sys.executable, '-u', str(P/'train_A56.py'), '--arm', arm],
                                 cwd=P, stdout=log, stderr=subprocess.STDOUT)
        stat = Path(f'/proc/{child.pid}/stat').read_text().split()
        record = {'arm': arm, 'child_PID': child.pid, 'child_start_ticks': int(stat[21]),
                  'supervisor_PID': os.getpid(), 'status': 'RUNNING'}
        records.append(record)
        write('formal_processes.json', {'records': records})
        begin = time.monotonic()
        timed_out = False
        while child.poll() is None:
            if time.monotonic()-begin > 24*3600:
                timed_out = True
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                break
            time.sleep(10)
        code = child.wait()
        log.close()
        record.update({'child_exit_code': code, 'seconds': time.monotonic()-begin,
                       'timed_out': timed_out, 'status': 'EXITED'})
        write('formal_processes.json', {'records': records})
        result_path = P/f'arm_{arm}/result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        if code != 0 or timed_out or result is None or result.get('status') != 'COMPLETED':
            write('formal_exit.json', {'status': 'STOPPED_NO_RETRY', 'records': records,
                                      'unstarted_arms_cancelled': [a for a in ['M', 'Q'] if a not in [r['arm'] for r in records]],
                                      'post_Q_queue': None})
            return 1
        assert result['epochs'] == 12 and result['optimizer_updates'] == 9708
    write('formal_exit.json', {'status': 'COMPLETED_BOTH_FIXED12', 'records': records,
                              'post_Q_queue': None})
    return 0


if __name__ == '__main__':
    sys.exit(main())
