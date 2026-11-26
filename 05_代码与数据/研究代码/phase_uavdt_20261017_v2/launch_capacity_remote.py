"""Persistent disposable capacity tests, after this account's CPU preparation succeeds."""
import datetime,hashlib,json,os,subprocess,textwrap
from pathlib import Path
holder=Path(CONFIG['storage'])/'datasets_verified_20261017/UAVDT_prepare_v2';root=Path(CONFIG['root'])
assert root.is_absolute() and not (holder/'capacity_supervisor_v2.json').exists()
program='CONFIG='+repr(CONFIG)+'\n'+r'''
import datetime,json,os,subprocess,time
from pathlib import Path
holder=Path(CONFIG['storage'])/'datasets_verified_20261017/UAVDT_prepare_v2';root=Path(CONFIG['root'])
try:
 deadline=time.monotonic()+1800
 while True:
  current=json.loads((holder/'CPU_preparation_current.json').read_text())
  if (holder/current['failed']).exists():raise RuntimeError('Current CPU preparation failed; GPU tests not started')
  if (holder/current['complete']).exists():break
  if time.monotonic()>deadline:raise TimeoutError('CPU preparation still incomplete; GPU tests not started')
  time.sleep(3)
 assert json.loads((root/'preparation_receipt.json').read_text())['state']=='CPU_verified_pending_capacity_and_finalization'
 with (root/'capacity_worker_v2.json').open('x') as f:json.dump({'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'script':__file__},f)
 with (root/'logs/capacity_probe.log').open('xb') as log:
  p=subprocess.run([CONFIG['python'],'-u',str(root/'code/capacity_probe.py'),str(root)],cwd=root/'code',
     env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),
     stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 result={'returncode':p.returncode,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'formal_training_started':False}
 with (root/'capacity_worker_result_v2.json').open('x') as f:json.dump(result,f,indent=2)
except BaseException as exc:
 with (holder/'capacity_supervisor_failed_v2.json').open('x') as f:json.dump({'error':type(exc).__name__+': '+str(exc)},f)
 raise
'''
path=holder/'capacity_supervisor_v2.py'
with path.open('x') as f:f.write(program)
with (holder/'capacity_supervisor_v2.log').open('xb') as log:
 p=subprocess.Popen([CONFIG['python'],'-u',str(path)],cwd=holder,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
     env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
result={'account':CONFIG['account'],'pid':p.pid,'start_ticks':Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19],
        'script':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'formal_training_started':False}
with (holder/'capacity_supervisor_v2.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
