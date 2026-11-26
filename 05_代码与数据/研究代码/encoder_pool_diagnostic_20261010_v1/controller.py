"""One-shot bounded controller, preserving all failures without retry."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,signal,subprocess,sys,time
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2)
def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();root=Path(a.prefix)/'DEIM/experiments/encoder_pool_diagnostic_20261010_v1';protocol=json.loads((root/'protocol.json').read_text());code=root/'code/capture_pool.py'
    assert hashlib.sha256(code.read_bytes()).hexdigest()==protocol['code_sha256']['capture_pool.py']
    with (root/'one_shot.lock').open('x') as f:f.write(str(os.getpid()))
    command=[sys.executable,'-B',str(code),'--prefix',a.prefix,'--seed',str(a.seed)];started=time.monotonic();env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0');timeout=protocol['budget_per_seed']['hard_timeout_seconds']
    with (root/'console.log').open('x') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        ticks=lambda pid:Path(f'/proc/{pid}/stat').read_text().split(')')[1].split()[19]
        save(root/'controller_start.json',{'pid':os.getpid(),'start_ticks':ticks(os.getpid()),'child_pid':child.pid,'child_start_ticks':ticks(child.pid),'started_remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(),'timeout_seconds':timeout,'command':command,'automatic_retry':False})
        timed_out=False
        try:exit_code=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:exit_code=child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);exit_code=child.wait()
    result={'status':'TIMEOUT_NO_RETRY' if timed_out else 'COMPLETE' if exit_code==0 else 'FAILED_NO_RETRY','exit_code':exit_code,'timed_out':timed_out,'elapsed_seconds':time.monotonic()-started,'child_pid':child.pid,'automatic_retry':False,'epochs_added':0,'optimizer_steps':0};save(root/'controller_completion.json',result);print(json.dumps(result),flush=True)
    if exit_code:raise SystemExit(exit_code)
if __name__=='__main__':main()
