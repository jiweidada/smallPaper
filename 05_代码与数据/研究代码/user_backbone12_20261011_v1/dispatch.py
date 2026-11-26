"""One queue per GPU, real parallel slots, immutable old jobs and no retries."""
from pathlib import Path
import argparse, datetime, fcntl, json, math, os, subprocess, sys, time, traceback
from runtime import HERE, read, write, sha, verify_bundle

def live(r, pid='pid', ticks='start_ticks'):
    try:
        fields=Path('/proc/'+str(r[pid])+'/stat').read_text().split()
        return fields[21]==str(r[ticks]) and fields[2]!='Z'
    except (FileNotFoundError, ProcessLookupError):
        return False

def experiments(plan):
    return Path({1:'/25085400120',2:'/24085403014'}[plan['seed']])/'DEIM/experiments'

def known_trainers(plan):
    e=experiments(plan);seed=plan['seed'];paths=[]
    for family, arms in [('native_stem_scratch12_20261011_v3',('native_stem_scratch',)),
            ('original_stem12_20261011_v1',('original_stem',)),
            ('stem_component12_20261011_v1',('detail_lowres','highfreq_rgb','attention_spatial_mean','early_final_downsample')),
            ('warm_stem12_20261011_v1',('warm_stem',)),
            ('user_backbone12_20261011_v1',('backbone_silu','backbone_attention','p2_skip'))]:
        for arm in arms:
            path=e/family/f'seed{seed}_{arm}'/'candidate_start.json'
            if path.exists() and live(read(path),'child_pid','child_start_ticks'):
                paths.append(dict(path=str(path),arm=arm,handle=read(path)))
    return paths

def resource(screen,plan):
    protocol=read(HERE/'training_protocol.json');c=protocol['concurrency'];e=experiments(plan)
    gpu=[x.strip() for x in subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')]
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    trainers=known_trainers(plan)
    # nvidia-smi lists host PIDs inside these containers; compare app count to the
    # independently verified container PID/start-tick trainer handles.
    unknown=len(apps)!=len(trainers)
    reserved=sum(8192 if t['arm'] in ('warm_stem','backbone_attention') else 6144 for t in trainers)
    ownbudget=c['reserved_budget_MiB'][plan['arm']]
    total=reserved+ownbudget+c['whole_GPU_headroom_MiB']
    warm=e/'warm_stem12_20261011_v1'/f"seed{plan['seed']}_warm_stem"
    warm_transitional=False
    if (warm/'dispatcher_live.json').exists():
        phase=read(warm/'dispatcher_live.json')['phase']
        warm_transitional=phase in ('full_detector_preflight','waiting_training_resources')
        if phase=='waiting_full_detector_preflight_resources' and len(trainers)<=2:
            warm_transitional=True  # permit the earlier owned candidate to launch first
    state=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),UUID=gpu[0],
        total_MiB=int(gpu[1]),used_MiB=int(gpu[2]),free_MiB=int(gpu[3]),
        apps=apps,known_live_trainers=trainers,unknown_GPU_app=unknown,
        existing_peak_reservations_MiB=reserved,own_reserved_budget_MiB=ownbudget,
        proposed_total_reservations_MiB=total,warm_launch_transition=warm_transitional,
        max_training_processes=4,
        scope='Reservations include growth to earlier observed native peaks; free memory alone is not a capacity proof.')
    assert state['UUID']==protocol['GPU_uuid'][str(plan['seed'])]
    state['eligible']=not unknown and not warm_transitional and len(apps)<4 and total<=state['total_MiB'] and state['free_MiB']>=ownbudget+c['whole_GPU_headroom_MiB']
    return state

def authorize_training(screen,plan,protocol,receipt):
    proof=read(screen/'detector_preflight_v1/report.json')
    assert proof['status']=='PASS_USER_CONFIG_FULL_AMP_LOSS_BACKWARD_NO_UPDATES'
    assert proof['identity']==receipt['identity'] and proof['optimizer_updates']==proof['training_epochs']==0
    assert proof['preflight_source_sha256']==sha(HERE/'preflight.py') and proof['protocol_sha256']==sha(HERE/'training_protocol.json')
    assert resource(screen,plan)['eligible']
    write(screen/'training_resource_receipt.json',resource(screen,plan))

def publish(root,phase,**data):
    p=root/'queue_live.tmp';p.write_text(json.dumps(dict(phase=phase,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),**data)));p.replace(root/'queue_live.json')

def older_closeout_handles(plan):
    e=experiments(plan);warm=e/'warm_stem12_20261011_v1'/f"seed{plan['seed']}_warm_stem"
    return [read(e/'native_stem_scratch12_20261011_v3/reference_start.json'),
        read(e/'stem_component12_20261011_v1/parallel_start.json'),read(warm/'dispatcher_dispatch.json')]

def wait_old_closeouts_and_idle(screen,plan,paths):
    handles=older_closeout_handles(plan);started=time.monotonic()
    while any(live(h) for h in handles) or subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        (screen/'post_training/live.json').write_text(json.dumps(dict(phase='waiting_older_owned_closeouts_and_actual_GPU_idle',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())))
        if time.monotonic()-started>64800:raise TimeoutError('Observation expired; no old job stopped or restarted')
        time.sleep(15)
    write(screen/'old_closeout_handles_stopped.json',dict(handles=handles,all_handles_stopped=True,actual_GPU_idle=True,existing_process_signals=0))

def spawn(screen,name,command):
    log=(screen/(name+'.log')).open('xb')
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'))
    log.close()
    record=dict(pid=process.pid,start_ticks=Path(f'/proc/{process.pid}/stat').read_text().split()[21],command=command)
    write(screen/(name+'_dispatch.json'),record)
    return process,record

def main(prefix,seed):
    root=Path(prefix)/'DEIM/experiments/user_backbone12_20261011_v1';queue=root/f'queue_seed{seed}'
    queue.mkdir(exist_ok=False)
    owner=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],seed=seed,source_sha256=sha(__file__))
    write(queue/'start.json',owner)
    bundles={arm:root/f'seed{seed}_{arm}/bundle' for arm in ('backbone_silu','p2_skip','backbone_attention')}
    jobs={arm:verify_bundle(bundle) for arm,bundle in bundles.items()}
    active={};pending=list(bundles);complete=[];failures=[];started=time.monotonic()
    lock=(root/f'launch_seed{seed}.lock').open('a')
    try:
        while pending or active:
            for arm,(controller,record) in list(active.items()):
                if controller.poll() is not None:
                    screen=jobs[arm][2]['screen_root']
                    if controller.returncode!=0:
                        failures.append(dict(arm=arm,exit_code=controller.returncode))
                    else:
                        assert read(screen/'controller_completion.json')['status']=='COMPLETE_FIXED12'
                        assert read(screen/'run/training_completion.json')['completed_epochs']==12
                        complete.append(arm)
                    del active[arm]
            if failures:
                pending=[]  # preserve every failure; no following new launches or retries
            if pending:
                arm=pending[0];receipt,plan,paths=jobs[arm];screen=paths['screen_root']
                state=resource(screen,plan)
                publish(queue,'training_and_capacity_queue' if active else 'waiting_capacity',pending=pending,completed=complete,
                    active={a:dict(r,same_live=live(r)) for a,(p,r) in active.items()},next_arm=arm,resource=state,failures=failures)
                if state['eligible']:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    try:
                        assert resource(screen,plan)['eligible']
                        process,record=spawn(screen,'detector_preflight',[sys.executable,'-B',str(HERE/'preflight.py'),'--bundle',str(bundles[arm])])
                        publish(queue,'full_detector_preflight',arm=arm,process=record)
                        rc=process.wait();write(screen/'detector_preflight_completion.json',dict(exit_code=rc,training_epochs=0,optimizer_updates=0))
                        assert rc==0,'Preflight failure: no training and no automatic retry'
                        assert resource(screen,plan)['eligible']
                        process,record=spawn(screen,'supervisor',[sys.executable,'-B',str(HERE/'supervise.py'),'--bundle',str(bundles[arm])])
                        active[arm]=(process,record);pending.pop(0)
                        # Observe actual CUDA ownership before trying a second slot.
                        deadline=time.monotonic()+120
                        while not (screen/'candidate_start.json').exists():
                            assert process.poll() is None
                            if time.monotonic()>deadline:raise TimeoutError('Trainer receipt absent; no relaunch')
                            time.sleep(.2)
                        deadline=time.monotonic()+180
                        while len(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines())!=len(known_trainers(plan)):
                            assert process.poll() is None
                            if time.monotonic()>deadline:raise TimeoutError('GPU ownership observation expired; no relaunch')
                            time.sleep(1)
                    finally:
                        fcntl.flock(lock,fcntl.LOCK_UN)
            else:
                publish(queue,'waiting_owned_trainers',completed=complete,active={a:dict(r,same_live=live(r)) for a,(p,r) in active.items()},failures=failures)
            if time.monotonic()-started>172800:raise TimeoutError('Queue observation expired; current trainers keep their own12 epoch limits')
            time.sleep(15)
        assert not failures,failures
        # Finishers are serialized after all own training, so no own closeout races.
        for arm in complete:
            screen=jobs[arm][2]['screen_root'];record=read(screen/'controller_start.json')
            process,_=spawn(screen,'finisher',[sys.executable,'-B',str(HERE/'finish.py'),'--bundle',str(bundles[arm]),'--controller-pid',str(record['pid']),'--controller-start-ticks',str(record['start_ticks'])])
            publish(queue,'serial_owned_saved_EMA_closeout',arm=arm)
            assert process.wait()==0,'Saved EMA closeout failed; no retry'
        write(queue/'completion.json',dict(status='ALL_THREE_USER_CONFIGS_FIXED12_TRAIN_RELOAD_COLLECTION_FINISHED',seed=seed,completed=complete,automatic_retry=False,BASE_forwards=0,BASE_epochs=0))
    except BaseException:
        write(queue/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False,old_process_signals=0,current_owned_trainers_preserved=True))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--seed',type=int,choices=(1,2),required=True)
    a=p.parse_args();main(a.prefix,a.seed)
