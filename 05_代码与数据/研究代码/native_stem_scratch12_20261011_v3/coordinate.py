"""Insert an authorized scratch reference into the authorized four-job GPU schedule."""
from pathlib import Path
import argparse,datetime,json,os,signal,subprocess,sys,time,traceback
from runtime import HERE,read,write,sha

def live(record):
    try:
        fields=Path(f"/proc/{record['pid']}/stat").read_text().split()
        return str(record['start_ticks'])==fields[21] and fields[2]!='Z'
    except (FileNotFoundError,ProcessLookupError):return False

def gpu():
    fields=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    return dict(uuid=fields[0].strip(),total=int(fields[1]),used=int(fields[2]),free=int(fields[3]),host_compute_processes=apps)

def invoke(script,args,log):
    with Path(log).open('xb') as f:
        return subprocess.run([sys.executable,'-B',str(HERE/script),*args],stdout=f,stderr=subprocess.STDOUT,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')).returncode

def authorize_child(screen,plan,protocol):
    allowance=read(screen/'reference_allowance.json');parent=read(screen.parent/'reference_start.json')
    assert allowance['parent']==parent and os.getppid()==parent['pid'] and live(parent)
    assert allowance['seed']==plan['seed'] and allowance['arm']==plan['arm']
    assert allowance['protocol_sha256']==sha(HERE/'training_protocol.json')
    assert read(screen/'preflight.json')['status']=='NATIVE_STEM_SCRATCH_SHAPE_INITIALIZATION_EQUIVALENCE_PASS'
    held=read(screen.parent/'component_conductor_held.json')['conductor']
    assert live(held) and Path(f"/proc/{held['pid']}/stat").read_text().split()[2]=='T'
    state=gpu()
    assert state['uuid']==protocol['GPU_uuid'][str(plan['seed'])]
    assert state['free']>=protocol['concurrency']['minimum_free_before_launch_MiB']
    assert len(state['host_compute_processes'])<protocol['concurrency']['max_training_processes_per_gpu']
    write(screen/'reference_resource_receipt.json',state)

def main(a):
    root=HERE.parent;protocol=read(HERE/'training_protocol.json')
    assert root==Path(a.prefix)/'DEIM/experiments/native_stem_scratch12_20261011_v3'
    with (root/'reference_once.lock').open('x') as f:f.write(str(os.getpid()))
    parent=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],seed=a.seed,
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(HERE/'training_protocol.json'))
    write(root/'reference_start.json',parent)
    oldroot=Path(a.prefix)/'DEIM/experiments/stem_component12_20261011_v1'
    conductor=protocol['existing_component_conductor'][str(a.seed)]
    for n,d in protocol['existing_component_code_manifest'].items():assert sha(oldroot/'code'/n)==d
    assert live(conductor)
    held=False;child=None
    try:
        assert invoke('prepare_training.py',['--prefix',a.prefix,'--seed',str(a.seed)],root/'prepare.log')==0
        screen=root/f'seed{a.seed}_native_stem_scratch'
        assert invoke('preflight.py',['--bundle',str(screen/'bundle')],root/'preflight.log')==0
        assert live(conductor)
        os.kill(conductor['pid'],signal.SIGSTOP);held=True
        while Path(f"/proc/{conductor['pid']}/stat").read_text().split()[2]!='T':time.sleep(.05)
        write(root/'component_conductor_held.json',dict(conductor=conductor,signal='SIGSTOP',
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            scope='Only waiting component conductor; existing trainers/controllers and full finisher untouched. Insert new authorized reference at next free GPU slot.'))
        while True:
            state=gpu();assert state['uuid']==protocol['GPU_uuid'][str(a.seed)]
            status=dict(phase='waiting_free_training_slot',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu=state)
            temporary=root/'reference_live.tmp';temporary.write_text(json.dumps(status,indent=2));temporary.replace(root/'reference_live.json')
            assert live(conductor)
            if len(state['host_compute_processes'])<protocol['concurrency']['max_training_processes_per_gpu'] and state['free']>=6144:break
            time.sleep(15)
        plan=read(screen/'bundle/plan.json')
        write(screen/'reference_allowance.json',dict(parent=parent,seed=a.seed,arm=plan['arm'],protocol_sha256=sha(HERE/'training_protocol.json')))
        with (screen/'supervisor.log').open('xb') as f:
            child=subprocess.Popen([sys.executable,'-B',str(HERE/'supervise.py'),'--bundle',str(screen/'bundle')],
                cwd=screen,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        owned=dict(pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21])
        write(root/'reference_dispatch.json',owned)
        while child.poll() is None:
            temporary=root/'reference_live.tmp';temporary.write_text(json.dumps(dict(phase='training_reference',
                UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller=owned,gpu=gpu()),indent=2));temporary.replace(root/'reference_live.json')
            time.sleep(15)
        assert live(conductor)
        os.kill(conductor['pid'],signal.SIGCONT);held=False
        write(root/'component_conductor_resumed.json',dict(conductor=conductor,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),reference_controller_exit=child.returncode))
        assert child.returncode==0, 'Reference failed; no retry'
        # Existing immutable closeouts own their GPU-idle replay phase.
        while live(conductor):time.sleep(15)
        assert (oldroot/'parallel_completion.json').exists() or (oldroot/'parallel_failure.json').exists()
        while gpu()['host_compute_processes']:time.sleep(15)
        assert invoke('finish.py',['--bundle',str(screen/'bundle'),'--controller-pid',str(owned['pid']),
            '--controller-start-ticks',owned['start_ticks']],root/'finish.log')==0
        write(root/'reference_completion.json',dict(status='NATIVE_STEM_SCRATCH_FIXED12_FINISHED',seed=a.seed,
            extra_epochs=0,automatic_retry=False,pretrained_BASE_forwards=0))
    except BaseException:
        write(root/'reference_failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False))
        raise
    finally:
        if held:
            if child is not None:
                while child.poll() is None:time.sleep(15)
            if live(conductor):
                os.kill(conductor['pid'],signal.SIGCONT)
                if not (root/'component_conductor_resumed.json').exists():
                    write(root/'component_conductor_resumed.json',dict(conductor=conductor,finally_cleanup=True,
                        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--seed',type=int,choices=(1,2),required=True)
    main(p.parse_args())
