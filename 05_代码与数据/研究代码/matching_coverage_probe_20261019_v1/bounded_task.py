"""Single-use finite task wrapper, no automatic retry or subsequent training."""
import argparse,datetime,json,os,pathlib,signal,subprocess,time
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):
    tmp=p.with_suffix('.next');tmp.write_text(json.dumps(v,indent=2));os.replace(tmp,p)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',type=pathlib.Path);a=p.parse_args();cfg=json.loads(a.task.read_text())
    root=pathlib.Path(cfg['root']);out=root/'task_records';out.mkdir(exist_ok=True);name=cfg['name'];lock=out/(name+'.lock')
    with lock.open('x') as f:f.write(now())
    state={'name':name,'started_utc':now(),'command':cfg['command'],'state':'starting','automatic_retry':False,'timeout_seconds':cfg['timeout_seconds']};sp=out/(name+'.json')
    with (out/(name+'.log')).open('xb') as log:
        env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        if cfg.get('cpu_only'):env['CUDA_VISIBLE_DEVICES']=''
        child=subprocess.Popen(cfg['command'],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(state='running',pid=child.pid,start_ticks=pathlib.Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]);write(sp,state);start=time.monotonic()
        while child.poll() is None:
            if time.monotonic()-start>cfg['timeout_seconds']:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                state['hard_timeout']=True;break
            state.update(checked_utc=now(),elapsed_seconds=time.monotonic()-start,console_bytes=(out/(name+'.log')).stat().st_size);write(sp,state);time.sleep(5)
        state.update(state='complete' if child.returncode==0 else 'failed',returncode=child.returncode,finished_utc=now());write(sp,state)
