"""Launch each accepted account once; full cross-account proof is supplied by the local verifier."""
import datetime,hashlib,json,os,subprocess
from pathlib import Path
PROOF=__PROOF__
root=Path(CONFIG['root'])
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert set(PROOF)=={'A','B'} and all(p['state']=='ready' for p in PROOF.values())
assert len({p['initialization_sha256'] for p in PROOF.values()})==1
assert len({p['data_metadata_sha256'] for p in PROOF.values()})==1
assert len({json.dumps(p['annotations_sha256'],sort_keys=True) for p in PROOF.values()})==1
assert len({p['model_engine_python_fingerprint'] for p in PROOF.values()})==1
accepted=read(root/'final_acceptance.json');assert accepted==PROOF[CONFIG['account']]
assert sha(root/'plan.json')==accepted['plan_sha256'] and sha(root/'frozen_manifest.json')==accepted['frozen_manifest_sha256']
for item in read(root/'frozen_manifest.json')['files']:
 p=root/item['relative'];assert p.is_relative_to(root) and sha(p)==item['sha256']
stage=Path(CONFIG['old'])/'artifacts/round2_20261014_v1';execution=read(stage/'current_execution.json')['execution_directory']
assert read(stage/execution/'queue_state.json')['state']=='complete'
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
assert not any(s.strip().isdigit() for s in gpu.splitlines()),'GPU must be idle after capacity tests and before formal handoff'
assert not any((root/'outputs').iterdir()) and not any((root/'jobs').iterdir())
assert not (root/'launch.lock').exists() and not (root/'queue.lock').exists()
release={'both_accounts_accepted':True,'plan_sha256':accepted['plan_sha256'],'cross_account_proofs':PROOF,
         'authorization':'User explicitly requested six96-epoch runs and immediate start after preflight.',
         'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (root/'launch.lock').open('x') as f:json.dump(release,f,indent=2)
with (root/'release_after_preflight.json').open('x') as f:json.dump(release,f,indent=2)
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
with (root/'logs/controller.log').open('xb') as log:
 p=subprocess.Popen([CONFIG['python'],'-u',str(root/'code/run_uavdt_queue.py')],cwd=root/'code',env=env,
     stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result={'account':CONFIG['account'],'state':'launched_once','pid':p.pid,
        'start_ticks':Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19],
        'root':str(root),'controller':str(root/'code/run_uavdt_queue.py'),
        'maximum_concurrent_trainings':accepted['maximum_concurrent_trainings'],'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (root/'launch_receipt.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
