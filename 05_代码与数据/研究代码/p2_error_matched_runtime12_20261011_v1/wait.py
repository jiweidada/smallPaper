"""Single altered-weight diagnostic after current exact owners and actual GPU idle."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,sys,time,subprocess,traceback,signal
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def live(h):
    pid=h.get('child_pid',h.get('pid'));ticks=h.get('child_start_ticks',h.get('start_ticks'))
    try:
        s=Path('/proc/%s/stat'%pid).read_text().rsplit(')',1)[1].split();return s[19]==str(ticks) and s[0]!='Z'
    except FileNotFoundError:return False
def main(seed):
    spec=read(HERE/'protocol.json');assert {n:sha(HERE/n) for n in spec['remote_sources']}==spec['remote_sources']
    assert seed in (1,2) and not (ROOT/('seed%d_GPU'%seed)).exists()
    queue=ROOT/('queue_seed%d'%seed);queue.mkdir(exist_ok=False)
    start=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],seed=seed,source_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),automatic_retry=False,training_epochs=0,BASE_forwards=0,parent_forwards=0)
    write(queue/'start.json',start);begin=time.monotonic()
    try:
        base=Path('/25085400120' if seed==1 else '/24085403014')/'DEIM/experiments'
        while True:
            owners=list(spec['predecessor_handles'][str(seed)])
            dynamic=[base/'p2_lowrank12_20261011_v3'/('seed%d_p2_lowrank16/finisher_dispatch.json'%seed)]
            if seed==1:dynamic += [base/'user_remaining_formal_closeout12_20261011_v1/backbone_silu_dispatch.json',base/'user_remaining_formal_closeout12_20261011_v1/backbone_attention_dispatch.json',base/'p2_direction_diagnostic12_20261011_v2/GPU_dispatch.json']
            for p in dynamic:
                if p.is_file():owners.append(read(p))
            observed=[dict(h,same_live=live(h)) for h in owners]
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if not any(x['same_live'] for x in observed) and not apps:break
            temp=queue/'live.tmp';temp.write_text(json.dumps(dict(phase='waiting_exact_existing_owners_and_actual_GPU_idle',owners=observed,GPU_apps=apps,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())));temp.replace(queue/'live.json')
            assert time.monotonic()-begin<spec['max_wait_seconds'],'Only this CPU waiter exits; no existing-process signals or restart'
            time.sleep(30)
        cmd=[sys.executable,'-B',str(HERE/'evaluate.py'),'--seed',str(seed)]
        with (queue/'GPU.log').open('xb') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'))
        handle=dict(pid=child.pid,start_ticks=Path('/proc/%s/stat'%child.pid).read_text().rsplit(')',1)[1].split()[19],command=cmd,automatic_retry=False)
        write(queue/'GPU_dispatch.json',handle);launched=time.monotonic();timed_out=False
        while child.poll() is None:
            if time.monotonic()-launched>=spec['private_GPU_timeout_seconds']:
                timed_out=True
                if live(handle):os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if live(handle):os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
                break
            time.sleep(5)
        write(queue/'GPU_completion.json',dict(exit_code=child.returncode,timed_out=timed_out,automatic_retry=False,existing_process_signals=0))
        assert child.returncode==0 and not timed_out,'Preserve all outputs; no retry'
        assert read(ROOT/('seed%d_GPU/completion.json'%seed))['exit_code']==0
        write(queue/'completion.json',dict(status='ONE_ERROR_MATCHED_NATIVE_DIAGNOSTIC_COMPLETED_ONCE',training_epochs=0,BASE_forwards=0,parent_forwards=0,automatic_retry=False,existing_process_signals=0))
    except BaseException:
        write(queue/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False,training_epochs=0,BASE_forwards=0,parent_forwards=0,existing_process_signals=0));raise
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--seed',type=int,choices=(1,2),required=True);main(a.parse_args().seed)
