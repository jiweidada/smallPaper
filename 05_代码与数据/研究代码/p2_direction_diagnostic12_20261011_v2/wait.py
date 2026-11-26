"""Wait for reviewed exact owners; attempt two new controls once without signals."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):
    with Path(p).open('x') as f:json.dump(o,f,indent=2,allow_nan=False)
def live(h):
    pid=h.get('child_pid',h.get('pid'));expected=h.get('child_start_ticks',h.get('start_ticks'))
    try:
        state=Path('/proc/%s/stat'%pid).read_text().rsplit(')',1)[1].split()
        return state[19]==str(expected) and state[0]!='Z'
    except FileNotFoundError:return False
def main():
    p=read(HERE/'protocol.json');assert {n:sha(HERE/n) for n in p['remote_sources']}==p['remote_sources']
    assert not (ROOT/'seed1_GPU').exists()
    start=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],source_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),automatic_retry=False,training_epochs=0,BASE_forwards=0,parent_forwards=0)
    write(ROOT/'queue_start.json',start);begin=time.monotonic()
    try:
        base=Path('/25085400120/DEIM/experiments');low=base/'p2_lowrank12_20261011_v3';remaining=base/'user_remaining_formal_closeout12_20261011_v1'
        while True:
            owners=list(p['predecessor_handles'])
            for f in [low/'seed1_p2_lowrank16/finisher_dispatch.json',remaining/'backbone_silu_dispatch.json',remaining/'backbone_attention_dispatch.json']:
                if f.is_file():owners.append(read(f))
            observed=[dict(h,same_live=live(h)) for h in owners]
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if not any(x['same_live'] for x in observed) and not apps:break
            temp=ROOT/'queue_live.tmp';temp.write_text(json.dumps(dict(phase='waiting_exact_rank16_and_first_SiLU_attention_closeouts_GPU_idle',owners=observed,GPU_apps=apps,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())));temp.replace(ROOT/'queue_live.json')
            assert time.monotonic()-begin<p['max_wait_seconds'],'Timeout stops only this CPU waiter; no signals or restart'
            time.sleep(30)
        cmd=[sys.executable,'-B',str(HERE/'evaluate.py'),'--prefix','/25085400120','--seed','1']
        with (ROOT/'seed1_GPU.log').open('xb') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'))
        write(ROOT/'GPU_dispatch.json',dict(pid=child.pid,start_ticks=Path('/proc/%s/stat'%child.pid).read_text().rsplit(')',1)[1].split()[19],command=cmd,automatic_retry=False,training_epochs=0,BASE_forwards=0,parent_forwards=0))
        rc=child.wait();assert rc==0,'Preserve partial outputs; no retry'
        done=read(ROOT/'seed1_GPU/completion.json');assert done['exit_code']==0
        write(ROOT/'queue_completion.json',dict(status='TWO_UNATTEMPTED_DIRECTION_CONTROLS_COMPLETED_ONCE',completion_sha256=sha(ROOT/'seed1_GPU/completion.json'),process_signals=0,automatic_retry=False,training_epochs=0,BASE_forwards=0,parent_forwards=0))
    except BaseException:
        write(ROOT/'queue_failure.json',dict(status='FAILED_NO_RETRY_NO_PROCESS_SIGNALS',traceback=traceback.format_exc(),process_signals=0,automatic_retry=False,training_epochs=0,BASE_forwards=0,parent_forwards=0));raise
if __name__=='__main__':main()
