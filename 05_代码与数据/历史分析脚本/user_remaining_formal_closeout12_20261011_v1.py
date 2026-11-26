"""Two previously unattempted old closeouts after current exact live owners stop."""
from pathlib import Path
import json,hashlib,datetime,subprocess,sys,os,time,traceback
R=Path('/25085400120/DEIM/experiments');OUT=R/'user_remaining_formal_closeout12_20261011_v1'
USER=R/'user_backbone12_20261011_v1';LOW=R/'p2_lowrank12_20261011_v3'
read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def live(r,k='pid',t='start_ticks'):
    try:
        f=Path('/proc/'+str(r[k])+'/stat').read_text().split()
        return f[21]==str(r[t]) and f[2]!='Z'
    except FileNotFoundError:return False
def validate_unattempted(arm):
    s=USER/('seed1_'+arm);d=read(s/'run/training_completion.json')
    assert d['status']=='NATIVE_FIT_FINISHED_EXACT12' and d['completed_epochs']==12
    assert sha(s/'run/last.pth')==d['last_checkpoint_sha256']
    assert read(s/'controller_completion.json')['status']=='COMPLETE_FIXED12'
    assert not live(read(s/'controller_start.json')) and not live(read(s/'candidate_start.json'),'child_pid','child_start_ticks')
    assert not (s/'post_training').exists() and not (s/'candidate_reload_v1').exists()
    return s
def main():
    write(OUT/'start.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],source_sha256=sha(__file__),training_epochs=0,BASE_forwards=0))
    try:
        expected=read(OUT/'reviewed_implementation.json')
        assert {p.name:sha(p) for p in (USER/'code').iterdir() if p.is_file()}==expected
        for arm in ['backbone_silu','backbone_attention']:validate_unattempted(arm)
        newqueue=read(LOW/'queue_dispatch.json');started=time.monotonic()
        while True:
            screen=LOW/'seed1_p2_lowrank16';active=live(newqueue)
            for name,k,t in [('controller_start.json','pid','start_ticks'),('candidate_start.json','child_pid','child_start_ticks'),('finisher_dispatch.json','pid','start_ticks')]:
                p=screen/name
                if p.exists():active=active or live(read(p),k,t)
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if not active and not apps:break
            p=OUT/'live.tmp';p.write_text(json.dumps(dict(phase='waiting_exact_rank16_owners_and_actual_GPU_idle',queue_same_live=live(newqueue),any_recorded_owner_live=active,GPU_apps=apps,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())));p.replace(OUT/'live.json')
            assert time.monotonic()-started<172800,'Wait expired; do not stop or restart any process';time.sleep(30)
        results=[]
        for arm in ['backbone_silu','backbone_attention']:
            assert {p.name:sha(p) for p in (USER/'code').iterdir() if p.is_file()}==expected
            s=validate_unattempted(arm);c=read(s/'controller_start.json')
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            command=[sys.executable,'-B',str(USER/'code/finish.py'),'--bundle',str(s/'bundle'),'--controller-pid',str(c['pid']),'--controller-start-ticks',str(c['start_ticks'])]
            with (OUT/(arm+'.log')).open('xb') as f:p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'))
            write(OUT/(arm+'_dispatch.json'),dict(pid=p.pid,start_ticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],command=command,training_epochs=0,BASE_forwards=0))
            rc=p.wait();v=s/'candidate_reload_v1/verification.json'
            result=dict(arm=arm,exit_code=rc,verification_status=read(v)['status'] if v.exists() else None,automatic_retry=False,training_epochs=0,BASE_forwards=0)
            write(OUT/(arm+'_completion.json'),result);results.append(result)
        write(OUT/'completion.json',dict(status='TWO_PREVIOUSLY_UNATTEMPTED_OLD_CLOSEOUTS_ATTEMPTED_ONCE',results=results,training_epochs=0,BASE_forwards=0,automatic_retry=False,process_signals=0))
    except BaseException:
        write(OUT/'failure.json',dict(status='NO_RETRY_NO_PROCESS_SIGNALS',traceback=traceback.format_exc(),training_epochs=0,BASE_forwards=0));raise
if __name__=='__main__':main()
