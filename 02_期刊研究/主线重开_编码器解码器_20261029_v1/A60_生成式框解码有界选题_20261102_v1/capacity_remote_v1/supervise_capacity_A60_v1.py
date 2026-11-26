from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path('/root/uav_a60_20261102_v1')

def save(n,x):
    with (ROOT/n).open('x') as f:json.dump(x,f,indent=2)

def main():
    scope=json.loads((ROOT/'capacity_scope_FROZEN_v1.json').read_text())
    for n,h in scope['source_sha256'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
    assert not (ROOT/'capacity_v1').exists() and not (ROOT/'capacity_supervisor_start_v1.json').exists()
    child=None;start=time.monotonic()
    try:
        with (ROOT/'capacity_stdout_v1.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/'capacity_A60_v1.py')],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().split(') ',1)[1].split()[19]
            save('capacity_supervisor_start_v1.json',dict(supervisor_pid=os.getpid(),child_pid=child.pid,
                child_start_ticks=ticks,seconds_cap=scope['maximum_seconds'],maximum_model_forwards=9,maximum_updates=6))
            try:code=child.wait(timeout=scope['maximum_seconds'])
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise RuntimeError('Fixed capacity time cap exceeded; no retry')
        assert code==0
        r=json.loads((ROOT/'capacity_v1/result.json').read_text())
        assert r['status']=='PASS_REAL_GPU_CAPACITY_AND_SHARED_INPUTS'
        assert r['counts']==dict(started=9,completed=9,updates=6)
        save('capacity_supervisor_exit_v1.json',dict(exit=0,status=r['status'],elapsed_seconds=time.monotonic()-start,
            formal_training_started=False,remaining_queue=[]))
    except BaseException:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        save('capacity_supervisor_exit_v1.json',dict(exit=1,status='CAPACITY_FAILED_NO_RETRY',
            traceback=traceback.format_exc(),formal_training_started=False,remaining_queue=[]))
        raise

if __name__=='__main__':main()
