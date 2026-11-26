"""Exercise SIGCONT/one-dispatch/SIGSTOP on disposable CPU-only processes."""
from pathlib import Path
import argparse,hashlib,json,os,signal,subprocess,sys,tempfile,time
from queue_pulse import proc,pulse_once,signal_owned,wait_held
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(out):
    with tempfile.TemporaryDirectory(prefix='owned_queue_pulse_fixture_') as temporary:
        root=Path(temporary);child_src="""from pathlib import Path
import json,sys,time
p=Path(sys.argv[1]);(p/'ready').write_text('ready')
for i in (1,2):
 (p/('dispatch'+str(i)+'.json')).write_text(json.dumps(dict(arm=i,pid=0)))
 time.sleep(3)
"""
        child=subprocess.Popen([sys.executable,'-B','-c',child_src,str(root)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        identity=dict(pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21])
        try:
            # Suspend before allowing any fixture dispatch, just like the owned
            # paused native conductor. This fixture has no CUDA/training code.
            signal_owned(identity,signal.SIGSTOP);wait_held(identity)
            assert not (root/'dispatch1.json').exists()
            paths=[root/'dispatch1.json',root/'dispatch2.json']
            first=pulse_once(identity,paths,lambda:True,window_seconds=10)
            assert first['status']=='ONE_NATIVE_QUEUED_DISPATCH' and paths[0].exists() and not paths[1].exists() and proc(identity)=='T'
            time.sleep(.2);assert not paths[1].exists()
            rejected=pulse_once(identity,paths,lambda:False,window_seconds=1)
            assert rejected['status']=='REFERENCE_FINISHED_DEFER_NATIVE_HANDOFF' and proc(identity)=='T' and not paths[1].exists()
            second=pulse_once(identity,paths,lambda:True,window_seconds=10)
            assert second['status']=='ONE_NATIVE_QUEUED_DISPATCH' and paths[1].exists() and proc(identity)=='T'
        finally:
            if proc(identity) is not None:os.kill(child.pid,signal.SIGKILL)
            child.wait(timeout=10)
    report=dict(status='PASS_DISPOSABLE_CPU_ONE_DISPATCH_AND_REFERENCE_HANDOFF_FIXTURE',first_dispatch_single=True,
        no_second_dispatch_while_held=True,finished_reference_does_not_resume=True,second_dispatch_single=True,
        disposable_CPU_processes=1,real_trainer_signals=0,CUDA_calls=0,model_forwards=0,optimizer_updates=0,
        code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        limitation='A disposable process validates signal/dispatch sequencing. Actual native jobs, GPU capacity and reference ownership must still be verified by the helper at runtime.')
    with Path(out).open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(dict(status=report['status'],report_sha256=sha(out))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
