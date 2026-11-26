"""One local host, one BASE12 evaluation and one fixed12 candidate. No queue."""
from pathlib import Path
import argparse,datetime,json,os,signal,subprocess,sys,time,traceback
from runtime import verify_bundle,write,read,sha
def phase(root,name,command,timeout):
    started=time.monotonic();exit_code=None;timed_out=False
    with (root/(name+'.log')).open('xb') as log:
        child=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'))
        write(root/(name+'_start.json'),{'controller_pid':os.getpid(),'child_pid':child.pid,'child_start_ticks':Path(f'/proc/{child.pid}/stat').read_text().split()[21],'command':command,'timeout_seconds':timeout,'started_remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'automatic_retry':False})
        while exit_code is None:
            try:exit_code=child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                (root/'controller_live.json').write_text(json.dumps({'phase':name,'child_pid':child.pid,'elapsed_seconds':time.monotonic()-started,'observed_remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
                if time.monotonic()-started>=timeout:
                    timed_out=True;os.killpg(child.pid,signal.SIGTERM)
                    try:exit_code=child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);exit_code=child.wait()
    write(root/(name+'_completion.json'),{'exit_code':exit_code,'timed_out':timed_out,'elapsed_seconds':time.monotonic()-started,'automatic_retry':False})
    assert exit_code==0 and not timed_out,name+' failed; no auto retry or next phase'
def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);a=p.parse_args()
    receipt,plan,paths=verify_bundle(a.bundle);root=paths['screen_root'];protocol=read(Path(__file__).with_name('protocol.json'))
    assert plan['seed'] in protocol['execution_allowlist'][plan['variant']],'Configuration not authorized in this deployment'
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert uuid==protocol['GPU_uuid'][str(plan['seed'])] and not apps,'Resource identity/busy check failed'
    with (root/'one_shot_launch.lock').open('x') as f:f.write(datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(root/'controller_start.json',{'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'protocol_sha256':sha(Path(__file__).with_name('protocol.json')),'identity':receipt['identity'],'automatic_retry':False})
    try:
        phase(root,'baseline',[sys.executable,'-B',str(Path(__file__).with_name('baseline_eval.py')),'--bundle',a.bundle],1800)
        phase(root,'candidate',[sys.executable,'-B',str(Path(__file__).with_name('experiment.py')),'--bundle',a.bundle],64800)
        write(root/'controller_completion.json',{'status':'COMPLETE_FIXED12','automatic_control_launch':False,'automatic_extension':False,'automatic_retry':False})
    except BaseException as e:
        write(root/'controller_failure.json',{'status':'FAILED_NO_RETRY','error':repr(e),'traceback':traceback.format_exc(),'automatic_retry':False});raise
if __name__=='__main__':main()
