"""No retry; bound one CPU probe and record its live/terminal process state."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(); p.add_argument('--prefix', required=True); a = p.parse_args()
    root = Path(a.prefix)/'DEIM/experiments/scale_protection12_20261010_v1'
    out = root/'diagnostics'; state_path = out/'controller_state.json'
    assert not state_path.exists() and not (out/'data_probe.json').exists()
    protocol = json.loads((root/'protocol.json').read_text())
    timeout = protocol['data_only_timeout_seconds']; assert timeout == 1200
    started = time.monotonic(); state = dict(controller_pid=os.getpid(),
        start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        timeout_seconds=timeout, automatic_retry=False, optimizer_updates=0, model_forwards=0)
    def record(**updates):
        state.update(updates)
        temp = state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2), encoding='utf8'); os.replace(temp, state_path)
    command = [sys.executable, '-u', str(root/'code/probe_data.py'), '--prefix', a.prefix,
               '--output', str(out/'data_probe.json')]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    with (out/'data_probe.log').open('x') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        record(status='RUNNING', child_pid=child.pid, command=command)
        timed_out = False
        while child.poll() is None:
            elapsed = time.monotonic()-started
            record(elapsed_seconds=elapsed)
            if elapsed >= timeout:
                timed_out = True; os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
                break
            time.sleep(5)
        result = json.loads((out/'data_probe.json').read_text()) if (out/'data_probe.json').exists() else None
        success = child.returncode == 0 and result and result['status'] == 'COMPLETE_DATA_ONLY_SCALE_PROTECTION'
        record(status='TIMEOUT' if timed_out else 'COMPLETE' if success else 'FAILED',
               exit_code=child.returncode, elapsed_seconds=time.monotonic()-started,
               complete_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               data_only_physical_gate_passed=result['data_only_physical_gate_passed'] if success else None)
    print(json.dumps(state), flush=True)
    if not success:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
