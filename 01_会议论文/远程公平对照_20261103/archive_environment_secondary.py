"""Archive an existing environment for the authorized second-host execution.

Reads the active environment without modifying it. No model is loaded.
"""
from pathlib import Path
import datetime, hashlib, json, os, subprocess, time

ROOT = Path('/25085400120/mpsma_fair_20261103_v1')
folder = ROOT/'secondary_transfer'
folder.mkdir(exist_ok=True)
receipt = folder/'environment_archive.json'
def save(value):
    tmp = receipt.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(tmp, receipt)
def ticks(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
with (folder/'environment_archive.lock').open('x') as f:
    f.write(json.dumps({'pid':os.getpid(),'start_ticks':ticks(os.getpid())})+'\n')
state = {'status':'RUNNING','pid':os.getpid(),'start_ticks':ticks(os.getpid()),
         'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'source':'/25085400120/envs/deim','archive':str(folder/'deim.tar.gz'),
         'model_executions':0}
save(state)
try:
    # Low CPU/I/O priority; exclude only transient bytecode/cache files.
    command = ['nice','-n','10','tar','-cf',str(folder/'deim.tar.gz.partial'),
               '--use-compress-program=gzip -1','--exclude=__pycache__',
               '--exclude=*.pyc','-C','/25085400120/envs','deim']
    with (folder/'environment_archive.log').open('x') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(child_pid=proc.pid,child_start_ticks=ticks(proc.pid),command=command);save(state)
        code = proc.wait(timeout=3600)
    if code != 0: raise RuntimeError('tar failed: '+str(code))
    source = folder/'deim.tar.gz.partial'
    digest = hashlib.sha256()
    with source.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):digest.update(b)
    source.rename(folder/'deim.tar.gz')
    state.update(status='COMPLETED',bytes=(folder/'deim.tar.gz').stat().st_size,
                 sha256=digest.hexdigest(),ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(state)
except Exception as e:
    state.update(status='FAILED_NO_RETRY',error=repr(e));save(state);raise
