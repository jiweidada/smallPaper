"""Preserve partial preparation; replace only the identified interrupted CPU reader."""
import base64,datetime,hashlib,json,os,signal,subprocess,time
from pathlib import Path
PAYLOAD=__PAYLOAD__
base=Path(CONFIG['storage'])/'datasets_verified_20261017';root=base/'UAVDT_v1'
assert root.is_dir() and (root/'data_metadata.json').is_file()
if (root/'data_ready.json').is_file():
 print(json.dumps({'account':CONFIG['account'],'already_ready':True}),flush=True)
else:
 assert not (root/'verification_worker_v2.json').exists(),'Persistent verifier was already attempted; inspect receipt before any recovery.'
 marker=json.loads((root/'preparation_started.json').read_text())
 expected_pid=15981 if CONFIG['account']=='B' else marker['pid']
 proc=Path('/proc')/str(expected_pid);stopped=None
 if proc.exists():
  args=[v.decode() for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
  stat=(proc/'stat').read_text().rsplit(')',1)[1].split();ticks=int(stat[19])
  assert args==[CONFIG['python'],'-u','-'],'Refusing to signal a different process'
  if CONFIG['account']=='B':assert ticks==15592341
  else:
   boot=int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime ')))
   born=boot+ticks/os.sysconf('SC_CLK_TCK');mark=datetime.datetime.fromisoformat(marker['created_utc']).timestamp()
   assert abs(born-mark)<5,'Process no longer matches the preparation start time'
  stopped={'pid':expected_pid,'start_ticks':ticks,'argv':args,'reason':'SSH session reset; CPU-only image verification has no model state'}
  os.kill(expected_pid,signal.SIGTERM)
  deadline=time.monotonic()+20
  while proc.exists():
   if (proc/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':break
   if time.monotonic()>deadline:raise TimeoutError('Old CPU verifier not yet stopped; no duplicate worker launched')
   time.sleep(.1)
 path=root/'verify_data_worker_v2.py';source='CONFIG='+repr(CONFIG)+'\n'+base64.b64decode(PAYLOAD['worker']).decode()
 with path.open('x') as f:f.write(source)
 with (root/'verification_worker_v2.log').open('xb') as log:
  p=subprocess.Popen([CONFIG['python'],'-u',str(path)],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
      env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
 ticks=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
 result={'account':CONFIG['account'],'pid':p.pid,'start_ticks':ticks,'worker':str(path),'worker_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'replaced_interrupted_CPU_reader':stopped,'training_started':False}
 with (root/'verification_worker_v2.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result),flush=True)
