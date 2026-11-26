"""Resume the unchanged owned component conductor for one safe queued dispatch.

This CPU helper never creates a trainer, changes a bundle or skips the native
resource authorization. The existing reference coordinator retains final
ownership of resuming its original held conductor when reference training ends.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,signal,subprocess,time,traceback

HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2)
def proc(record):
    try:
        f=Path('/proc/'+str(record['pid'])+'/stat').read_text().split()
        return f[2] if f[21]==str(record['start_ticks']) and f[2]!='Z' else None
    except (FileNotFoundError,ProcessLookupError):return None
def signal_owned(record,sig):
    assert proc(record) is not None,'Owned process identity no longer live'
    os.kill(record['pid'],sig)
def wait_held(record):
    deadline=time.monotonic()+3
    while proc(record)!='T':
        assert proc(record) is not None and time.monotonic()<deadline
        time.sleep(.02)
def gpu():
    row=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines()
    return dict(uuid=row[0].strip(),free_MiB=int(row[1]),compute_processes=len(apps))

def pulse_once(conductor,dispatch_paths,reference_live,window_seconds=45):
    before={p for p in dispatch_paths if p.exists()}
    if not reference_live():return dict(status='REFERENCE_FINISHED_DEFER_NATIVE_HANDOFF')
    assert proc(conductor)=='T'
    signal_owned(conductor,signal.SIGCONT)
    deadline=time.monotonic()+window_seconds
    try:
        while time.monotonic()<deadline:
            if not reference_live():return dict(status='REFERENCE_FINISHED_DURING_PULSE_NATIVE_CONDUCTOR_RUNNING')
            assert proc(conductor) is not None,'Conductor exited during pulse; inspect native failure'
            new=[p for p in dispatch_paths if p.exists() and p not in before]
            if new:
                assert len(new)==1,'More than one queued dispatch before hold; preserve evidence'
                signal_owned(conductor,signal.SIGSTOP);wait_held(conductor)
                if not reference_live():signal_owned(conductor,signal.SIGCONT)
                return dict(status='ONE_NATIVE_QUEUED_DISPATCH',dispatch=str(new[0]),record=read(new[0]))
            time.sleep(.02)
        if reference_live():
            signal_owned(conductor,signal.SIGSTOP);wait_held(conductor)
        return dict(status='NO_NATIVE_DISPATCH_WITHIN_WINDOW')
    except BaseException:
        if proc(conductor) is not None and reference_live():
            signal_owned(conductor,signal.SIGSTOP);wait_held(conductor)
        raise

def main(a):
    p=read(HERE/'protocol.json');assert sha(HERE/'queue_pulse.py')==p['code_sha256']['queue_pulse.py']
    root=HERE.parent;assert root==Path(a.prefix)/'DEIM/experiments/stem_component_queue_pulse_20261011_v1'
    with (root/'once.lock').open('x') as f:f.write(str(os.getpid()))
    start=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],seed=a.seed,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(root/'start.json',start)
    old=Path(a.prefix)/'DEIM/experiments/stem_component12_20261011_v1';ref=Path(a.prefix)/'DEIM/experiments/native_stem_scratch12_20261011_v3'
    assert {x.name:sha(x) for x in (old/'code').iterdir() if x.is_file()}==p['component_code_sha256']
    conductor=read(old/'parallel_start.json');expected=p['conductor'][str(a.seed)]
    assert (conductor['pid'],str(conductor['start_ticks']))==(expected['pid'],str(expected['start_ticks']))
    held=read(ref/'component_conductor_held.json');assert held['conductor']==expected and proc(conductor)=='T'
    reference=read(ref/f'seed{a.seed}_native_stem_scratch/controller_start.json')
    assert (reference['pid'],str(reference['start_ticks']))==tuple(p['reference_controller'][str(a.seed)])
    ref_live=lambda:proc(reference) is not None
    assert ref_live()
    paths=[]
    for arm in p['queued_arms']:
        screen=old/f'seed{a.seed}_{arm}';assert read(screen/'preflight.json')['status']=='CONTROL_SHAPE_GRADIENT_AND_INITIALIZATION_PASS'
        assert not (screen/'parallel_allowance.json').exists() and not (screen/'candidate_start.json').exists() and not (screen/'run').exists()
        paths.append(screen/'parallel_dispatch.json')
    write(root/'initial_state.json',dict(conductor=conductor,reference_controller=reference,queued_arms=p['queued_arms'],GPU=gpu(),training_code_changed=False))
    pulses=0;deadline=time.monotonic()+p['helper_timeout_seconds']
    try:
        while ref_live() and not all(x.exists() for x in paths):
            assert time.monotonic()<deadline,'CPU scheduling helper deadline reached; no trainers terminated'
            assert proc(conductor)=='T','Native conductor unexpectedly running; defer and inspect'
            state=gpu();assert state['uuid']==p['GPU_uuid'][str(a.seed)]
            status=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),conductor_state=proc(conductor),reference_controller_state=proc(reference),GPU=state,
                remaining_arms=[arm for arm,path in zip(p['queued_arms'],paths) if not path.exists()])
            temporary=root/'live.tmp';temporary.write_text(json.dumps(status,indent=2));temporary.replace(root/'live.json')
            # Native unchanged child authorizer caps actual GPU applications at3.
            # Keep its safety condition; count the reference and foreign apps.
            if state['compute_processes']<=2 and state['free_MiB']>=6144:
                result=pulse_once(conductor,paths,ref_live);pulses+=1
                write(root/f'pulse_{pulses}.json',dict(result,GPU_before=state,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
                if result['status'].startswith('REFERENCE_FINISHED'):break
                assert not (old/'parallel_failure.json').exists(),'Native scheduler failure; no retry'
            time.sleep(10)
        write(root/'completion.json',dict(status='QUEUED_NATIVE_DISPATCHES_COMPLETE_OR_REFERENCE_HANDOFF',pulses=pulses,
            queued_dispatches={arm:read(path) if path.exists() else None for arm,path in zip(p['queued_arms'],paths)},
            conductor_state=proc(conductor),reference_controller_live=ref_live(),training_code_changed=False,helper_created_trainers=0,
            baseline_forwards=0,optimizer_updates=0,automatic_training_retry=False))
    except BaseException:
        write(root/'failure.json',dict(traceback=traceback.format_exc(),training_code_changed=False,automatic_retry=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--seed',type=int,choices=(1,2),required=True);main(p.parse_args())
