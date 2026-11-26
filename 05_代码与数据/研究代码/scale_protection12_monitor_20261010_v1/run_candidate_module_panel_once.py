"""One owned candidate-panel process; record real exit, never retry training."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

def write(p,v):
    with Path(p).open('x',encoding='utf8') as f: json.dump(v,f,indent=2)
def proc(pid):
    p=Path(f'/proc/{pid}/stat')
    if not p.exists(): return None
    s=p.read_text();v=s[s.rfind(')')+2:].split()
    return dict(state=v[0],start_ticks=v[19])
def run(a):
    root=Path(a.root).resolve();seed=a.seed
    destination=root/f'module_candidate_panel_seed{seed}_v1'
    launch=root/f'module_candidate_launch_seed{seed}_v1.json'
    completion=root/f'module_candidate_completion_seed{seed}_v1.json'
    log=root/f'module_candidate_panel_seed{seed}_v1.log'
    assert not any(p.exists() for p in (destination,launch,completion,log))
    cmd=[a.python,'-B',str(root/'module_capture_tools_v1/capture_candidate_panel.py'),
        '--bundle',str(root/f'seed{seed}_target_protect/bundle'),
        '--closeout-tools',str(root/'closeout_tools_v1'),
        '--protocol',str(root/'module_capture_tools_v1/panel_protocol.json'),
        '--destination',str(destination)]
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    started=time.monotonic();timedout=False
    with log.open('xb') as stream:
        child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        owned=proc(child.pid);assert owned is not None
        write(launch,dict(pid=child.pid,start_ticks=owned['start_ticks'],controller_pid=os.getpid(),
            controller_start_ticks=proc(os.getpid())['start_ticks'],command=cmd,
            timeout_seconds=1800,automatic_retry=False,baseline_model_forwards=0,epochs_added=0))
        try:
            code=child.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            timedout=True
            actual=proc(child.pid)
            assert actual and actual['start_ticks']==owned['start_ticks'] and os.getpgid(child.pid)==child.pid
            os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                actual=proc(child.pid);assert actual and actual['start_ticks']==owned['start_ticks']
                os.killpg(child.pid,signal.SIGKILL);code=child.wait()
    verification=destination/'verification.json'
    proof=json.loads(verification.read_text()) if verification.exists() else None
    ok=(code==0 and not timedout and proof and proof['status']=='PASS_ALL_CALLED_MODULE_RAW_OUTPUTS_SELECTED_PANEL')
    write(completion,dict(status='CANDIDATE_MODULE_PANEL_COMPLETE' if ok else 'CANDIDATE_MODULE_PANEL_FAILED',
        seed=seed,exit_code=code,timed_out=timedout,elapsed_seconds=time.monotonic()-started,
        completed_remote_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        verification=proof,verification_sha256=hashlib.sha256(verification.read_bytes()).hexdigest() if proof else None,
        original_training_restarted=False,automatic_retry=False,baseline_model_forwards=0,epochs_added=0))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--python',required=True);parser.add_argument('--seed',type=int,choices=(1,2),required=True)
    run(parser.parse_args())
