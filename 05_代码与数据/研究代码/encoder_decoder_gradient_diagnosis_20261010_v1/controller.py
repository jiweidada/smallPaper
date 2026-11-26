"""Single diagnostic child, 600-second timeout, no retry or subsequent queue."""
import argparse,datetime,json,os,pathlib,signal,subprocess,sys,time
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--runtime-root',required=True);a=p.parse_args();r=pathlib.Path(a.root)
    with (r/'one_shot.lock').open('x') as f:f.write(datetime.datetime.now(datetime.timezone.utc).isoformat())
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()=='GPU-d29c55c5-02ab-1e69-c978-0b28f2102003'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    with (r/'probe.log').open('xb') as log:
        started=time.monotonic();c=subprocess.Popen([sys.executable,'-B',str(r/'probe.py'),'--root',str(r),'--runtime-root',a.runtime_root],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'))
        (r/'start.json').write_text(json.dumps({'controller_pid':os.getpid(),'controller_start_ticks':pathlib.Path('/proc/self/stat').read_text().split()[21],'child_pid':c.pid,'child_start_ticks':pathlib.Path(f'/proc/{c.pid}/stat').read_text().split()[21],'timeout_seconds':600,'automatic_retry':False},indent=2))
        timeout=False
        while c.poll() is None:
            try:c.wait(timeout=15)
            except subprocess.TimeoutExpired:
                (r/'live.json').write_text(json.dumps({'elapsed_seconds':time.monotonic()-started,'child_pid':c.pid}))
                if time.monotonic()-started>=600:
                    timeout=True;os.killpg(c.pid,signal.SIGTERM)
                    try:c.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
    (r/'controller_completion.json').write_text(json.dumps({'exit_code':c.returncode,'timed_out':timeout,'elapsed_seconds':time.monotonic()-started,'automatic_retry':False,'optimizer_steps':0},indent=2));assert c.returncode==0 and not timeout
if __name__=='__main__':main()
