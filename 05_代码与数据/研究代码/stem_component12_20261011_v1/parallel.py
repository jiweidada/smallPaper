"""Run four authorized controls, max three trainers/GPU, then serialize EMA replays."""
from pathlib import Path
import argparse, datetime, json, os, signal, subprocess, sys, time, traceback
from runtime import HERE, read, write, sha

def live(record,pid='pid',ticks='start_ticks'):
    try:
        f=Path(f'/proc/{record[pid]}/stat').read_text().split()
        return f[21]==str(record[ticks]) and f[2]!='Z'
    except (FileNotFoundError,ProcessLookupError): return False

def gpu():
    values=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    return dict(uuid=values[0].strip(),total=int(values[1]),used=int(values[2]),free=int(values[3]),host_compute_processes=apps)

def authorize_child(screen,plan,protocol):
    allowance=read(screen/'parallel_allowance.json')
    parent=read(screen.parent/'parallel_start.json')
    assert allowance['parent']==parent and os.getppid()==parent['pid'] and live(parent)
    assert allowance['arm']==plan['arm'] and allowance['seed']==plan['seed']
    assert allowance['protocol_sha256']==sha(HERE/'training_protocol.json')
    assert read(screen/'preflight.json')['status']=='CONTROL_SHAPE_GRADIENT_AND_INITIALIZATION_PASS'
    state=gpu()
    assert state['uuid']==protocol['GPU_uuid'][str(plan['seed'])]
    assert state['free']>=protocol['concurrency']['minimum_free_before_launch_MiB']
    assert len(state['host_compute_processes'])<protocol['concurrency']['max_training_processes_per_gpu']
    write(screen/'parallel_resource_receipt.json',state)

def original_active(root):
    start=root/'candidate_start.json'
    return start.exists() and live(read(start),'child_pid','child_start_ticks')

def invoke(script,args,log):
    with Path(log).open('xb') as f:
        return subprocess.run([sys.executable,'-B',str(HERE/script),*args],stdout=f,stderr=subprocess.STDOUT,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')).returncode

def resume_full_when_idle(original,finisher,root,children):
    # Waiting/closeout orchestration only: training processes are never signalled here.
    while original_active(original) or any(p.poll() is None for p in children): time.sleep(15)
    while gpu()['host_compute_processes']: time.sleep(15)
    if live(finisher):
        os.kill(finisher['pid'],signal.SIGCONT)
        write(root/'original_finisher_resumed.json',dict(finisher=finisher,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        while live(finisher): time.sleep(15)
    write(root/'original_closeout_observation.json',dict(
        success=(original/'post_training/completion.json').exists(),
        failure=read(original/'post_training/failure.json') if (original/'post_training/failure.json').exists() else None))

def main(a):
    root=HERE.parent; protocol=read(HERE/'training_protocol.json')
    assert root==Path(a.prefix)/'DEIM/experiments/stem_component12_20261011_v1'
    with (root/'parallel_once.lock').open('x') as f: f.write(str(os.getpid()))
    parent=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],seed=a.seed,
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(HERE/'training_protocol.json'))
    write(root/'parallel_start.json',parent)
    arms=protocol['concurrency']['queue']; children=[]; jobs=[]; held=False
    original_root=Path(a.prefix)/'DEIM/experiments/original_stem12_20261011_v1'
    original=original_root/f'seed{a.seed}_original_stem'
    for n,d in protocol['original_full_code_manifest'].items(): assert sha(original_root/'code'/n)==d
    expected={1:(13044,'476778560'),2:(14653,'253969229')}[a.seed]
    finisher=read(original/'post_training/start.json')
    assert (finisher['pid'],str(finisher['start_ticks']))==expected and live(finisher)
    assert original_active(original)
    initial_gpu=gpu()
    assert initial_gpu['uuid']==protocol['GPU_uuid'][str(a.seed)] and len(initial_gpu['host_compute_processes'])==1
    write(root/'initial_capacity.json',initial_gpu)
    try:
        for arm in arms:
            assert invoke('prepare_training.py',['--prefix',a.prefix,'--seed',str(a.seed),'--arm',arm],root/f'prepare_{arm}.log')==0,arm
            screen=root/f'seed{a.seed}_{arm}'
            assert invoke('preflight.py',['--bundle',str(screen/'bundle')],root/f'preflight_{arm}.log')==0,arm
        # The existing immutable finisher requires an idle GPU for its independent EMA replay.
        # Suspend only that waiting CPU process; preserve all existing trainer/controller code and PIDs.
        assert live(finisher) and original_active(original)
        os.kill(finisher['pid'],signal.SIGSTOP); held=True
        write(root/'original_finisher_held.json',dict(finisher=finisher,signal='SIGSTOP',
            reason='Serialize its saved-EMA replay after all concurrent training. Training is uninterrupted.',
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        pending=list(arms)
        while pending or any(p.poll() is None for p in children):
            running=sum(p.poll() is None for p in children)+int(original_active(original))
            state=gpu()
            observation=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                live_trainers_or_controllers=running,pending=pending,gpu=state,
                jobs=[dict(arm=j['arm'],controller_pid=p.pid,exit_code=p.poll()) for p,j in zip(children,jobs)])
            temporary=root/'parallel_live.tmp'
            temporary.write_text(json.dumps(observation,indent=2));temporary.replace(root/'parallel_live.json')
            if pending and running<3 and state['free']>=6144:
                arm=pending.pop(0); screen=root/f'seed{a.seed}_{arm}'
                write(screen/'parallel_allowance.json',dict(parent=parent,arm=arm,seed=a.seed,
                    protocol_sha256=sha(HERE/'training_protocol.json')))
                with (screen/'supervisor.log').open('xb') as f:
                    child=subprocess.Popen([sys.executable,'-B',str(HERE/'supervise.py'),'--bundle',str(screen/'bundle')],
                        cwd=screen,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
                        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
                children.append(child);jobs.append(dict(arm=arm,controller_pid=child.pid,
                    controller_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21]))
                write(screen/'parallel_dispatch.json',jobs[-1])
            time.sleep(15)
        resume_full_when_idle(original,finisher,root,children);held=False
        for job,child in zip(jobs,children):
            screen=root/f"seed{a.seed}_{job['arm']}"
            if child.returncode!=0:
                continue  # Failed controls have no replay, extension or automatic retry.
            while gpu()['host_compute_processes']:time.sleep(15)
            rc=invoke('finish.py',['--bundle',str(screen/'bundle'),'--controller-pid',str(job['controller_pid']),
                '--controller-start-ticks',job['controller_start_ticks']],screen/'finish.log')
            write(screen/'parallel_closeout_exit.json',dict(exit_code=rc,automatic_retry=False))
        write(root/'parallel_completion.json',dict(status='AUTHORIZED_QUEUE_FINISHED',
            jobs=[dict(j,controller_exit=p.returncode) for j,p in zip(jobs,children)],
            baseline_forwards=0,automatic_retry=False))
    except BaseException:
        write(root/'parallel_failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False))
        raise
    finally:
        if held: resume_full_when_idle(original,finisher,root,children)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--seed',type=int,choices=(1,2),required=True)
    main(p.parse_args())
