"""One-shot resource-gated candidate; never signal or restart older jobs."""
from pathlib import Path
import argparse,datetime,json,math,os,subprocess,sys,time,traceback
from runtime import HERE,read,write,sha,verify_bundle

def live(r,pid='pid',ticks='start_ticks'):
    try:
        f=Path('/proc/'+str(r[pid])+'/stat').read_text().split()
        return f[21]==str(r[ticks]) and f[2]!='Z'
    except (FileNotFoundError,ProcessLookupError):return False

def old_paths(plan):
    prefix={1:'/25085400120',2:'/24085403014'}[plan['seed']]
    base=Path(prefix)/'DEIM/experiments'
    return base/'native_stem_scratch12_20261011_v3',base/'stem_component12_20261011_v1'

def coordinator_records(plan):
    reference,component=old_paths(plan)
    return [read(reference/'reference_start.json'),read(component/'parallel_start.json')]

def old_trainers(plan):
    reference,component=old_paths(plan);seed=plan['seed'];paths=[reference/f'seed{seed}_native_stem_scratch/candidate_start.json']
    paths += [component/f'seed{seed}_{arm}/candidate_start.json' for arm in ('detail_lowres','highfreq_rgb','attention_spatial_mean','early_final_downsample')]
    paths += [component.parent/f'original_stem12_20261011_v1/seed{seed}_original_stem/candidate_start.json']
    return [dict(path=str(p),**read(p)) for p in paths if p.exists() and live(read(p),'child_pid','child_start_ticks')]

def resource(screen,plan):
    protocol=read(HERE/'training_protocol.json');c=protocol['concurrency']
    line=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    own=old_trainers(plan);coords=coordinator_records(plan)
    reserve=c['candidate_reserved_budget_MiB']+len(apps)*c['existing_app_reserved_budget_MiB']+c['whole_GPU_headroom_MiB']
    unknown=len(apps)!=len(own)
    closing=not own and any(live(r) for r in coords)
    state=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),UUID=line[0].strip(),
        total_MiB=int(line[1]),used_MiB=int(line[2]),free_MiB=int(line[3]),GPU_compute_apps=apps,
        old_live_trainers=own,old_coordinators=[dict(r,same_live=live(r)) for r in coords],
        proposed_reserved_budget_MiB=reserve,unowned_or_closeout_GPU_app=unknown,old_closeouts_pending_without_trainers=closing)
    assert state['UUID']==protocol['GPU_uuid'][str(plan['seed'])]
    allowed=not unknown and not closing and reserve<=state['total_MiB'] and len(apps)<c['max_training_processes_per_gpu']
    state['eligible_preflight']=allowed and len(apps)<=c['max_existing_apps_before_full_preflight'] and state['free_MiB']>=c['minimum_free_before_full_preflight_MiB']
    state['eligible_training']=allowed and state['free_MiB']>=c['candidate_reserved_budget_MiB']
    return state

def authorize_training(screen,plan,protocol,receipt):
    proof=read(screen/'detector_preflight_v1/report.json')
    assert proof['status']=='PASS_CANDIDATE_FULL_DETECTOR_AMP_LOSS_BACKWARD_NO_UPDATES'
    assert proof['identity']==receipt['identity'] and proof['optimizer_updates']==proof['training_epochs']==0
    assert proof['preflight_source_sha256']==sha(HERE/'preflight.py') and proof['protocol_sha256']==sha(HERE/'training_protocol.json')
    state=resource(screen,plan);assert state['eligible_training'],state
    write(screen/'training_resource_receipt.json',state)

def observe(screen,phase,**data):
    p=screen/'dispatcher_live.tmp';p.write_text(json.dumps(dict(phase=phase,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),**data)));p.replace(screen/'dispatcher_live.json')

def wait_old_closeouts_and_idle(screen,plan,paths):
    started=time.monotonic();owned=coordinator_records(plan)
    while any(live(r) for r in owned) or subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        (screen/'post_training/live.json').write_text(json.dumps(dict(phase='waiting_old_coordinator_handles_and_actual_GPU_idle',
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),old=[dict(r,same_live=live(r)) for r in owned])))
        if time.monotonic()-started>64800:raise TimeoutError('Closeout observation expired; no process is stopped or restarted')
        time.sleep(15)
    write(screen/'old_closeout_handles_stopped.json',dict(handles=owned,all_handles_stopped=True,actual_GPU_idle=True,existing_process_signals=0))

def main(bundle):
    receipt,plan,paths=verify_bundle(bundle);screen=paths['screen_root'];protocol=read(HERE/'training_protocol.json')
    with (screen/'dispatcher_once.lock').open('x') as f:f.write(str(os.getpid()))
    write(screen/'dispatcher_start.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
        identity=receipt['identity'],source_sha256=sha(__file__),automatic_retry=False,BASE_training_epochs=0,BASE_forwards=0))
    try:
        # Both older queued controllers have already been authorized; no new native parent pulse can race this candidate.
        reference,component=old_paths(plan)
        queue=component.parent/'stem_component_queue_pulse_20261011_v2'
        done=read(queue/'completion.json')
        assert set(done['queued_dispatches'])=={'attention_spatial_mean','early_final_downsample'}
        started=time.monotonic()
        while True:
            state=resource(screen,plan);observe(screen,'waiting_full_detector_preflight_resources',resource=state)
            if state['eligible_preflight']:break
            if time.monotonic()-started>64800:raise TimeoutError('Resource observation expired; no trainer started or restarted')
            time.sleep(15)
        command=[sys.executable,'-B',str(HERE/'preflight.py'),'--bundle',str(bundle)]
        with (screen/'detector_preflight_v1.log').open('xb') as log:
            p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'))
            write(screen/'detector_preflight_start.json',dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],command=command))
            observe(screen,'full_detector_preflight',owned_pid=p.pid)
            rc=p.wait()
        write(screen/'detector_preflight_completion.json',dict(exit_code=rc,optimizer_updates=0,training_epochs=0,automatic_retry=False))
        assert rc==0,'Full candidate preflight failed; no training or automatic retry'
        while True:
            state=resource(screen,plan);observe(screen,'waiting_training_resources',resource=state)
            if state['eligible_training']:break
            if time.monotonic()-started>64800:raise TimeoutError('Resource observation expired; no trainer started or restarted')
            time.sleep(15)
        command=[sys.executable,'-B',str(HERE/'supervise.py'),'--bundle',str(bundle)]
        with (screen/'supervisor.log').open('xb') as log:
            controller=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'))
            ticks=Path(f'/proc/{controller.pid}/stat').read_text().split()[21]
            write(screen/'controller_dispatch.json',dict(pid=controller.pid,start_ticks=ticks,command=command))
            deadline=time.monotonic()+120
            while not (screen/'controller_start.json').exists():
                assert controller.poll() is None,'Training controller exited before its launch receipt'
                if time.monotonic()>deadline:raise TimeoutError('Controller receipt observation timed out; no restart')
                time.sleep(.2)
            owned=read(screen/'controller_start.json');assert live(owned)
            args=[sys.executable,'-B',str(HERE/'finish.py'),'--bundle',str(bundle),'--controller-pid',str(owned['pid']),'--controller-start-ticks',owned['start_ticks']]
            with (screen/'finisher.log').open('xb') as f:
                finisher=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'))
                write(screen/'finisher_dispatch.json',dict(pid=finisher.pid,start_ticks=Path(f'/proc/{finisher.pid}/stat').read_text().split()[21],command=args))
                while controller.poll() is None:
                    observe(screen,'training_owned_candidate',controller=dict(owned,same_live=live(owned)),max_epochs=12)
                    time.sleep(15)
                assert controller.returncode==0,'Candidate failed; no automatic retry or extension'
                while finisher.poll() is None:
                    observe(screen,'waiting_owned_closeout',finisher_pid=finisher.pid)
                    time.sleep(15)
                assert finisher.returncode==0,'Saved EMA closeout failed; preserve evidence, no retry'
        write(screen/'dispatcher_completion.json',dict(status='WARM_STEM_FIXED12_TRAIN_RELOAD_COLLECTION_FINISHED',
            identity=receipt['identity'],training_commands=1,max_epochs=12,automatic_retry=False,BASE_forwards=0))
    except BaseException:
        write(screen/'dispatcher_failure.json',dict(status='FAILED_NO_AUTOMATIC_RETRY_OR_EXTENSION',traceback=traceback.format_exc(),BASE_forwards=0))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);a=p.parse_args();main(Path(a.bundle).resolve())
