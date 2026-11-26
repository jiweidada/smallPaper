"""One-shot two-arm queue, fixed seed, no retries or full132 promotion."""
import argparse, datetime, hashlib, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def identity(pid):
    p=Path('/proc')/str(pid);fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'start_ticks':fields[19],'cmd':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()}
def verify_package():
    manifest=json.loads((ROOT/'deployment_manifest.json').read_text())
    for rel,expected in manifest['files'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=expected:raise RuntimeError('Frozen source changed: '+rel)
def gpu_idle():
    r=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    if r.stdout.strip():raise RuntimeError('GPU has an existing compute process; queue did not interfere: '+r.stdout)
def run_one(name,command,seconds,state):
    verify_package();gpu_idle()
    log=ROOT/f'{name}.console.log'
    with log.open('xb') as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
            start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4'})
        started=time.monotonic();state.update(stage=name,status='RUNNING',child=identity(child.pid),command=command,started_utc=now(),timeout_seconds=seconds)
        write(ROOT/'pipeline_state.json',state)
        while child.poll() is None:
            if time.monotonic()-started>seconds:
                state.update(status='HARD_TIMEOUT_TERMINATING_OWN_PROCESS_GROUP',utc=now());write(ROOT/'pipeline_state.json',state)
                # PID is our unreaped child; its process group was created by this invocation.
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise RuntimeError(name+' exceeded its hard timeout')
            state.update(utc=now(),elapsed_seconds=time.monotonic()-started,console_bytes=log.stat().st_size)
            write(ROOT/'pipeline_state.json',state)
            time.sleep(30)
        state.update(exit_code=child.returncode,finished_utc=now(),elapsed_seconds=time.monotonic()-started)
        with (ROOT/'pipeline_events.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
        if child.returncode:raise RuntimeError(f'{name} failed with exit {child.returncode}; no retry')
        return child.returncode
def validate_finish(arm,seed):
    from experiment import verify_bundle
    receipt,plan,paths=verify_bundle(ROOT/'bundles'/f'{arm}_seed{seed}')
    out=paths['output'];rows=[json.loads(x) for x in (out/'log.txt').read_text().splitlines() if x.strip()]
    if [r['epoch'] for r in rows]!=list(range(24)):raise RuntimeError('Expected exactly epochs 0..23')
    complete=json.loads((out/'endpoint_epoch24/training_completion.json').read_text())
    if complete['identity']!=receipt['identity'] or complete['status']!='NATIVE_FIT_RETURNED_AND_LAST_CHECKPOINT_PRESENT':raise RuntimeError('Missing completion identity')
    if hashlib.sha256((out/'last.pth').read_bytes()).hexdigest()!=complete['last_checkpoint_sha256']:raise RuntimeError('Final checkpoint changed')
    return out
def compare_seed(base,fine,seed):
    from results import load_endpoint,pair
    left,lh,_=load_endpoint(base/'endpoint_epoch24');right,rh,_=load_endpoint(fine/'endpoint_epoch24')
    from evaluator_binding import same_result_evaluator
    same_result_evaluator(left,right)
    if left['protocol']!=right['protocol'] or left['environment']!=right['environment']:raise RuntimeError('Pair evaluation/runtime mismatch')
    inputs=[[json.loads(l) for l in (d/'first100_training_inputs.jsonl').read_text().splitlines()] for d in (base,fine)]
    aligned=len(inputs[0])==100 and inputs[0]==inputs[1]
    result={'seed':seed,'endpoint_epoch_one_based':24,'first100_inputs_and_pre_forward_rng_equal':aligned,
        'differences_fine_minus_baseline':pair(left,right,lh,rh),'status':'OBSERVED_SCREEN_ONLY',
        'claim_boundary':'24-epoch prefix; no mechanism isolation, full132 efficacy, novelty, or significance claim'}
    write(ROOT/f'paired_seed{seed}.json',result)
    if not aligned:raise RuntimeError('First100 input/RNG alignment failed; results retained for review')
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=(1,2),required=True);a=p.parse_args()
    fd=os.open(ROOT/'one_shot.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    state={'seed':a.seed,'supervisor':identity(os.getpid()),'authorized_epochs_each':24,
           'arms':['baseline','fine_residual'],'automatic_132_promotion':False,'retries':0}
    try:
        bundles=[str(ROOT/'bundles'/f'{arm}_seed{a.seed}') for arm in state['arms']]
        run_one('preflight',[sys.executable,'-B',str(ROOT/'code/preflight.py'),'--baseline-bundle',bundles[0],
            '--candidate-bundle',bundles[1],'--output',str(ROOT/'preflight_result.json')],900,state)
        preflight=json.loads((ROOT/'preflight_result.json').read_text())
        if preflight['status']!='PASS_BOUNDED_CUDA_PREFLIGHT_NOT_AP_EVIDENCE':raise RuntimeError('Preflight not passed')
        outputs=[]
        for arm,bundle in zip(state['arms'],bundles):
            run_one(arm,[sys.executable,'-B',str(ROOT/'code/experiment.py'),'train','--bundle',bundle],18*3600,state)
            outputs.append(validate_finish(arm,a.seed))
        compare_seed(*outputs,a.seed)
        state.update(status='COMPLETED_24_EPOCHS_BOTH_ARMS',utc=now());write(ROOT/'pipeline_state.json',state)
    except BaseException:
        state.update(status='FAILED_NO_RETRY',utc=now(),error=traceback.format_exc());write(ROOT/'pipeline_state.json',state)
        raise
if __name__=='__main__':main()
