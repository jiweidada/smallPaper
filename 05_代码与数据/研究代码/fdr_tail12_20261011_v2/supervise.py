"""One-shot process, resource and timeout guard; no baseline execution."""
from pathlib import Path
import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from runtime import HERE, verify_bundle, read, write, sha


def main():
    p = argparse.ArgumentParser(); p.add_argument('--bundle', required=True); a = p.parse_args()
    receipt, plan, paths = verify_bundle(a.bundle)
    protocol = read(HERE/'training_protocol.json'); root = paths['screen_root']
    uuid = subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert uuid == protocol['GPU_uuid'][str(plan['seed'])] and not apps, 'Wrong or busy GPU'
    with (root/'one_shot_launch.lock').open('x') as f:
        f.write(datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(root/'controller_start.json',dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21],
        identity=receipt['identity'], automatic_retry=False, native_baseline_forwards=0, baseline_epochs_added=0))
    try:
        from reuse_baseline import attach_cached_baseline
        attach_cached_baseline(receipt, plan, paths)
        command = [sys.executable,'-B',str(HERE/'experiment.py'),'--bundle',a.bundle]
        started = time.monotonic(); timed_out = False; timeout = protocol['candidate_timeout_seconds']
        with (root/'candidate.log').open('xb') as stream:
            child = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1'))
            write(root/'candidate_start.json',dict(controller_pid=os.getpid(), child_pid=child.pid,
                child_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21], command=command,
                timeout_seconds=timeout, automatic_retry=False))
            while child.poll() is None:
                elapsed = time.monotonic()-started
                (root/'controller_live.json').write_text(json.dumps(dict(phase='candidate', child_pid=child.pid,
                    elapsed_seconds=elapsed, observed_remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))
                if elapsed >= timeout:
                    timed_out = True; os.killpg(child.pid,signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL); child.wait()
                    break
                time.sleep(5)
        write(root/'candidate_completion.json',dict(exit_code=child.returncode, timed_out=timed_out,
            elapsed_seconds=time.monotonic()-started, automatic_retry=False))
        assert child.returncode == 0 and not timed_out, 'Candidate failed; no retry or following phase'
        write(root/'controller_completion.json',dict(status='COMPLETE_FIXED12', automatic_extension=False,
            automatic_retry=False, automatic_control_launch=False))

    except BaseException as e:
        write(root/'controller_failure.json',dict(status='FAILED_NO_RETRY', error=repr(e), traceback=traceback.format_exc(),
                                                 automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
