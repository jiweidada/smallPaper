"""One-shot persistent CPU preparation; wait for verified data, never launch training."""
import base64,datetime,hashlib,json,os,subprocess,textwrap
from pathlib import Path
PAYLOAD=__PAYLOAD__
holder=Path(CONFIG['storage'])/'datasets_verified_20261017/UAVDT_prepare_v2';holder.mkdir(exist_ok=False)
data=holder.parent/'UAVDT_v1';source=base64.b64decode(PAYLOAD['preparation']).decode()
prefix='''import datetime,json,time\nfrom pathlib import Path\nholder=Path(HOLDER)\ndata=Path(DATA)\ndeadline=time.monotonic()+1800\nwhile not (data/'data_ready.json').exists():\n if (data/'verification_failed_v2.json').exists():raise RuntimeError('Data verifier failed; no CPU deployment or training')\n if time.monotonic()>deadline:raise TimeoutError('Data verification not complete within preparation wait limit')\n time.sleep(5)\n'''.replace('HOLDER',repr(str(holder))).replace('DATA',repr(str(data)))
program='CONFIG='+repr(CONFIG)+'\nimport json\nfrom pathlib import Path\ntry:\n'+textwrap.indent(prefix+source,' ')+'''\nexcept BaseException as exc:\n with (Path(HOLDER)/'CPU_preparation_failed.json').open('x') as f:json.dump({'error':type(exc).__name__+': '+str(exc)},f)\n raise\nelse:\n with (Path(HOLDER)/'CPU_preparation_complete.json').open('x') as f:json.dump({'state':'complete','training_started':False},f)\n'''.replace('HOLDER',repr(str(holder)))
compile(program,'persistent_CPU_preparation','exec')
path=holder/'prepare_CPU_worker_v2.py'
with path.open('x') as f:f.write(program)
with (holder/'CPU_preparation.log').open('xb') as log:
 p=subprocess.Popen([CONFIG['python'],'-u',str(path)],cwd=holder,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
     env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
result={'account':CONFIG['account'],'pid':p.pid,'start_ticks':Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19],
        'script':str(path),'script_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'logs':str(holder/'CPU_preparation.log'),
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training_started':False}
with (holder/'CPU_preparation_worker.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
