"""Verify the copied environment without re-extracting or changing packages.

Installed distributions are collected before torch/torchvision imports. Vendored
setuptools metadata exposed later is checked through direct version resolution,
not a last-entry-wins enumeration. No model is instantiated or executed.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.metadata as metadata,json,os,platform,subprocess,sys,sysconfig
ROOT=Path(__file__).resolve().parent
PROTOCOL_SHA='8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def norm(d):return {k.lower().replace('_','-'):v for k,v in d.items()}
def save(v):
    p=ROOT/'secondary_preparation.json.tmp';p.write_text(json.dumps(v,indent=2)+'\n');os.replace(p,ROOT/'secondary_preparation.json')
with (ROOT/'secondary_runtime_verification_v3.lock').open('x') as f:f.write(str(os.getpid())+'\n')
state=read(ROOT/'secondary_inputs_ready.json')
state.update(status='VERIFYING_EXTRACTED_ENVIRONMENT_NO_MODEL_EXECUTION',phase='verification_v3',pid=os.getpid(),
             start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],started_utc=now())
save(state)
try:
    assert not (ROOT/'runs/FULL').exists() and not (ROOT/'runs/BASE').exists()
    assert sha(ROOT/'execution_protocol.json')==PROTOCOL_SHA
    protocol=read(ROOT/'execution_protocol.json')
    for item in protocol['frozen_files']:assert sha(ROOT/item['path'])==item['sha256'],item['path']
    for item in read(ROOT/'dataset_bundle_manifest.json')['files']:
        assert sha(Path('/tmp/mpsma_fair_20261103_data')/item['path'])==item['sha256'],item['path']
    first=read(ROOT/'secondary_environment_transfer.json');base=read(ROOT/'python_base_archive_completed.json')
    assert sha(ROOT/'deim_environment.tar.gz')==first['sha256']
    assert sha(ROOT/'pybase_environment.tar.gz')==base['sha256']
    resolution=read(ROOT/'metadata_discovery_resolution.json')
    assert resolution['status']=='SAME_INSTALLED_DEPENDENCIES_AND_IMPORT_RESOLUTION'
    expected=protocol['runtime']
    installed={d.metadata['Name']:d.version for d in metadata.distributions(path=[sysconfig.get_path('purelib')])}
    before={d.metadata['Name']:d.version for d in metadata.distributions()}
    assert norm(installed)==norm(before)==norm(expected['packages'])
    import torch,torchvision
    direct={name:metadata.version(name) for name in expected['packages']}
    assert norm(direct)==norm(installed)
    actual={'python':sys.version,'executable':sys.executable,'base_prefix':sys.base_prefix,
      'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda':torch.version.cuda,
      'cudnn':torch.backends.cudnn.version(),'torch_build':torch.__config__.show(),'packages':installed,
      'packages_collection_method':'Installed site-packages metadata before torch/torchvision imports; also checked against direct metadata.version after imports.',
      'post_import_direct_versions':direct,'hostname':platform.node(),
      'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader,nounits'],text=True).strip(),
      'cuda_available':torch.cuda.is_available(),'model_executions':0,
      'metadata_discovery_resolution_sha256':sha(ROOT/'metadata_discovery_resolution.json')}
    for key in ['python','torch','cuda','cudnn','torch_build']:assert actual[key]==expected[key],key
    assert actual['base_prefix']=='/25085400120/envs/pybase'
    assert actual['cuda_available']
    assert actual['gpu']=='NVIDIA GeForce RTX 3090, GPU-ca8d11de-f221-f2b9-82bf-d389523dd71a, 550.100, 24576'
    (ROOT/'secondary_runtime.json').write_text(json.dumps(actual,indent=2)+'\n')
    state.update(status='READY_FOR_AUTHORIZED_FULL_ONLY',runtime_sha256=sha(ROOT/'secondary_runtime.json'),
       gpu=actual['gpu'],runtime_matches_frozen_software=True,completed_utc=now(),
       environment_archive_sha256=first['sha256'],python_base_archive_sha256=base['sha256'],
       runtime_verifier_sha256=sha(Path(__file__)),
       preparation_history={'attempt1':'External base-interpreter dependency detected before extraction.',
                            'attempt2':'Archives extracted; last-entry-wins dependency enumeration rejected vendored metadata.',
                            'resolution':'Same installed packages and import resolution proved on both hosts. This verification changes no package, model, optimizer or hyperparameter.'},
       note='Original frozen environment/preflight files retain first-host provenance; secondary_runtime.json records the actual second host.')
    save(state);print(json.dumps(state))
except Exception as error:
    state.update(status='PREPARATION_FAILED_NO_TRAINING',error=repr(error));save(state);raise
