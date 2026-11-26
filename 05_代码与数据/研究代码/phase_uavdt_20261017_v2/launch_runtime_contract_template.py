import base64,hashlib,json,os,subprocess
from pathlib import Path
PAYLOAD=__PAYLOAD__
holder=Path(CONFIG['storage'])/'datasets_verified_20261017/UAVDT_prepare_v2'
assert not (holder/'runtime_contract_supervisor_v2.json').exists()
program='CONFIG='+repr(CONFIG)+'\nPAYLOAD='+repr(PAYLOAD)+'\n'+r'''
import base64,datetime,hashlib,json,os,subprocess,time
from pathlib import Path
holder=Path(CONFIG['storage'])/'datasets_verified_20261017/UAVDT_prepare_v2';root=Path(CONFIG['root'])
try:
 deadline=time.monotonic()+1800
 while True:
  current=json.loads((holder/'CPU_preparation_current.json').read_text())
  if (holder/current['failed']).exists():raise RuntimeError('CPU preparation failed; runtime contract not started')
  if (holder/current['complete']).exists():break
  assert time.monotonic()<deadline;time.sleep(3)
 assert not (root/'final_acceptance.json').exists() and not (root/'launch_receipt.json').exists()
 assert not any((root/'outputs').iterdir()) and not any((root/'jobs').iterdir())
 old=root/'code/train_uavdt.py';before=old.read_bytes()
 assert hashlib.sha256(before).hexdigest()=='8a2095e40e34d71688e1e27453c990e9b072a7f4e32684dfe06a770242a59965'
 with (root/'artifacts/train_uavdt_before_path_guard_v2.py').open('xb') as f:f.write(before)
 raw=base64.b64decode(PAYLOAD['train_uavdt.py']);temp=old.with_suffix('.next');temp.write_bytes(raw);os.replace(temp,old)
 check=root/'code/check_runtime_contract.py'
 with check.open('xb') as f:f.write(base64.b64decode(PAYLOAD['check_runtime_contract.py']))
 proof={'entry_before_sha256':hashlib.sha256(before).hexdigest(),'entry_after_sha256':hashlib.sha256(raw).hexdigest(),
        'runtime_test_sha256':hashlib.sha256(check.read_bytes()).hexdigest(),'reason':'Reject cross-job resume before native solver can override output_dir; no model/recipe change.'}
 with (root/'entry_path_guard_patch_receipt.json').open('x') as f:json.dump(proof,f,indent=2)
 with (root/'logs/runtime_contract.log').open('xb') as log:
  p=subprocess.run([CONFIG['python'],'-u',str(check),str(root)],cwd=root/'code',
       env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'),
       stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 with (root/'runtime_contract_worker_result.json').open('x') as f:json.dump({'returncode':p.returncode},f)
except BaseException as exc:
 with (holder/'runtime_contract_supervisor_failed_v2.json').open('x') as f:json.dump({'error':type(exc).__name__+': '+str(exc)},f)
 raise
'''
path=holder/'runtime_contract_supervisor_v2.py'
with path.open('x') as f:f.write(program)
with (holder/'runtime_contract_supervisor_v2.log').open('xb') as log:
 p=subprocess.Popen([CONFIG['python'],'-u',str(path)],cwd=holder,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result={'account':CONFIG['account'],'pid':p.pid,'start_ticks':Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19],'script':str(path)}
with (holder/'runtime_contract_supervisor_v2.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
