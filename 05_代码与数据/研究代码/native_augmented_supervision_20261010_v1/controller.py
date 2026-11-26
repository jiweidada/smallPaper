"""One-shot 600-second native data/model observation; never retries."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

def save(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2)
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--seed',type=int,choices=[1,2],required=True);p.add_argument('--expected-code-sha',required=True);a=p.parse_args()
    root=Path(a.root)/'DEIM/experiments/native_augmented_supervision_20261010_v1';code=root/'code/capture_coverage.py'
    with (root/'one_shot.lock').open('x') as f:f.write(str(os.getpid()))
    assert hashlib.sha256(code.read_bytes()).hexdigest()==a.expected_code_sha
    assert hashlib.sha256((code.parent/'training_evidence.py').read_bytes()).hexdigest()=='26013101b0777b709b3b1a0b9bbf6d4022b5698e1120e2d490c4eb5aa1c45258'
    output=root/'capture_v1';assert not output.exists();command=[sys.executable,'-B',str(code),'--root',a.root,'--seed',str(a.seed),'--output',str(output)]
    started=time.monotonic();env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0')
    with (root/'console_v1.txt').open('x') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        save(root/'controller_start.json',dict(status='RUNNING',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split(')')[1].split()[19],child_pid=child.pid,child_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(')')[1].split()[19],command=command,timeout_seconds=600,automatic_retry=False,epochs_added=0,optimizer_steps=0))
        timed_out=False
        try:exit_code=child.wait(timeout=600)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:exit_code=child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);exit_code=child.wait()
    result=dict(status='TIMED_OUT_NO_RETRY' if timed_out else 'COMPLETE' if exit_code==0 else 'FAILED_NO_RETRY',exit_code=exit_code,timed_out=timed_out,elapsed_seconds=time.monotonic()-started,output_completion_exists=(output/'completion.json').is_file(),child_pid=child.pid,automatic_retry=False,epochs_added=0,optimizer_steps=0)
    save(root/'controller_completion.json',result);print(json.dumps(result),flush=True)
    if exit_code!=0:raise SystemExit(1)
if __name__=='__main__':main()
