"""Exactly one SPATIAL24 run per seed after bounded CUDA preflight; no retry."""
import argparse,datetime,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def identity(pid):
    p=Path('/proc')/str(pid);fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'start_ticks':fields[19],'cmd':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()}
def verify_package():
    for rel,h in json.loads((ROOT/'deployment_manifest.json').read_text())['files'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen file changed: '+rel)
def gpu_idle():
    r=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    if r.stdout.strip():raise RuntimeError('GPU busy; existing task untouched: '+r.stdout)
def run_one(name,command,seconds,state):
    verify_package();gpu_idle()
    log=ROOT/f'{name}.console.log'
    with log.open('xb') as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
             env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4'})
        start=time.monotonic();state.update(stage=name,status='RUNNING',child=identity(child.pid),started_utc=now(),timeout_seconds=seconds)
        write(ROOT/'pipeline_state.json',state)
        while child.poll() is None:
            if time.monotonic()-start>seconds:
                state.update(status='HARD_TIMEOUT_OWN_PROCESS_GROUP',utc=now());write(ROOT/'pipeline_state.json',state)
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise RuntimeError('Hard timeout: '+name)
            state.update(utc=now(),elapsed_seconds=time.monotonic()-start,console_bytes=log.stat().st_size)
            write(ROOT/'pipeline_state.json',state);time.sleep(30)
        state.update(exit_code=child.returncode,finished_utc=now())
        with (ROOT/'pipeline_events.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
        if child.returncode:raise RuntimeError(f'{name} failed with exit {child.returncode}; no retry')
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=(1,2),required=True);a=p.parse_args()
    fd=os.open(ROOT/'one_shot.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    state={'seed':a.seed,'supervisor':identity(os.getpid()),'arms':['spatial'],'authorized_epochs_each':24,'retries':0,'automatic_132_promotion':False}
    try:
        from compare_reused import verify_baseline,compare
        verify_baseline(a.seed)
        base=str(ROOT/'bundles'/f'baseline_seed{a.seed}');candidate=str(ROOT/'bundles'/f'spatial_seed{a.seed}')
        run_one('preflight',[sys.executable,'-B',str(ROOT/'code/preflight.py'),'--baseline-bundle',base,'--candidate-bundle',candidate,'--output',str(ROOT/'preflight_result.json')],900,state)
        if json.loads((ROOT/'preflight_result.json').read_text())['status']!='PASS_BOUNDED_CUDA_PREFLIGHT_NOT_AP_EVIDENCE':raise RuntimeError('Preflight failed')
        run_one('spatial',[sys.executable,'-B',str(ROOT/'code/experiment.py'),'train','--bundle',candidate],18*3600,state)
        from experiment import verify_bundle
        receipt,plan,paths=verify_bundle(candidate);out=paths['output']
        rows=[json.loads(l) for l in (out/'log.txt').read_text().splitlines() if l.strip()]
        complete=json.loads((out/'endpoint_epoch24/training_completion.json').read_text())
        if [r['epoch'] for r in rows]!=list(range(24)) or complete['identity']!=receipt['identity']:
            raise RuntimeError('Missing 24-epoch endpoint identity')
        if hashlib.sha256((out/'last.pth').read_bytes()).hexdigest()!=complete['last_checkpoint_sha256']:raise RuntimeError('Last checkpoint changed')
        compare(out,a.seed)
        state.update(status='COMPLETED_SPATIAL24_AND_REUSED_BASELINE_COMPARISON',utc=now());write(ROOT/'pipeline_state.json',state)
    except BaseException:
        state.update(status='FAILED_NO_RETRY',utc=now(),error=traceback.format_exc());write(ROOT/'pipeline_state.json',state);raise
if __name__=='__main__':main()
