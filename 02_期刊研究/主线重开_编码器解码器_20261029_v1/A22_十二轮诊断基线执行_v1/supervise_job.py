"""Run exactly one frozen training child, preserve exit, enforce48h; never resume/eval."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parent
def utc():return datetime.now(timezone.utc).isoformat()
def wr(p,x):
    tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def identity(pid):
    p=Path('/proc')/str(pid);s=(p/'stat').read_text().split()
    return {'pid':pid,'start_ticks':int(s[21]),'state':s[2],
      'cmdline':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'),'cwd':str((p/'cwd').resolve())}
def main():
    assert not (ROOT/'supervisor_once.json').exists()
    wr(ROOT/'supervisor_once.json',{'utc':utc(),**identity(os.getpid()),'no_followup_job':True})
    manifest=ROOT/'execution_manifest_FROZEN.json';m=json.loads(manifest.read_text())
    assert m['status']=='FROZEN_TRAINING_EXECUTION_ONLY' and m['new_training_complete_forward_cap']==38832
    assert not (ROOT/'training_one_shot.lock').exists() and not (ROOT/'training_child.json').exists()
    for row in m['files']:assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    started=time.monotonic();timeout=False
    with (ROOT/'training_stdout.log').open('w') as out,(ROOT/'training_stderr.log').open('w') as err:
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'train_pilot.py'),'--run'],cwd=str(ROOT),
          env=os.environ.copy(),stdout=out,stderr=err,start_new_session=True)
        child_identity=identity(child.pid);wr(ROOT/'training_child.json',{'utc':utc(),**child_identity,
             'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()})
        try:code=child.wait(timeout=48*3600)
        except subprocess.TimeoutExpired:
            timeout=True
            observed=identity(child.pid)
            assert observed['start_ticks']==child_identity['start_ticks'] and str(ROOT/'train_pilot.py') in observed['cmdline']
            assert os.getpgid(child.pid)==child.pid
            os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                observed=identity(child.pid);assert observed['start_ticks']==child_identity['start_ticks']
                assert os.getpgid(child.pid)==child.pid
                os.killpg(child.pid,signal.SIGKILL);code=child.wait()
    result=json.loads((ROOT/'training_result.json').read_text()) if (ROOT/'training_result.json').exists() else None
    wr(ROOT/'supervisor_result.json',{'utc':utc(),'status':'TRAINING_CHILD_TERMINAL_NO_FOLLOWUP','returncode':code,
      'wall_cap_triggered':timeout,'elapsed_seconds':time.monotonic()-started,'training_result':result,
      'followup_jobs_launched':False,'training_child_identity':child_identity})
    return code
if __name__=='__main__':
    try:raise SystemExit(main())
    except BaseException as e:
        if not isinstance(e,SystemExit):wr(ROOT/'supervisor_failure.json',{'utc':utc(),'traceback':traceback.format_exc(),'no_automatic_retry':True})
        raise
