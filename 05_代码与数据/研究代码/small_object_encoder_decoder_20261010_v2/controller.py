"""One-shot bound preflight controller. Does not create training jobs."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,signal,subprocess,sys,time
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2)
def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);a=p.parse_args()
    prefix=Path(a.prefix);root=prefix/'DEIM/experiments/small_object_encoder_decoder_20261010_v2';code=root/'code';protocol=json.loads((code/'preflight_protocol.json').read_text())
    assert all(sha(code/k)==v for k,v in protocol['implementation_sha256'].items())
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert uuid==protocol['GPU_uuid'] and not apps,'Resource identity/busy check failed'
    with (root/'one_shot_preflight.lock').open('x') as f:f.write(datetime.datetime.now(datetime.timezone.utc).isoformat())
    command=[sys.executable,'-B',str(code/'preflight.py'),'--prefix',str(prefix)]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    started=time.monotonic()
    with (root/'preflight.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        save(root/'controller_start.json',{'controller_pid':os.getpid(),'child_pid':child.pid,'child_start_ticks':Path(f'/proc/{child.pid}/stat').read_text().split()[21],'command':command,'started_remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':sha(code/'preflight_protocol.json'),'GPU_uuid':uuid,'timeout_seconds':protocol['timeout_seconds'],'automatic_retry':False})
        timed_out=False
        try:exit_code=child.wait(timeout=protocol['timeout_seconds'])
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:exit_code=child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);exit_code=child.wait()
    save(root/'controller_completion.json',{'status':'COMPLETE' if exit_code==0 else 'FAILED','exit_code':exit_code,'timed_out':timed_out,'elapsed_seconds':time.monotonic()-started,'child_pid':child.pid,'automatic_retry':False,'optimizer_steps':0,'epochs_added':0})
    print(json.dumps({'exit_code':exit_code,'timed_out':timed_out}),flush=True)
if __name__=='__main__':main()
