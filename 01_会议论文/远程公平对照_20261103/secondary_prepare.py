"""Verify and unpack fixed inputs on the authorized second host; no model run."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import hashlib, json, os, posixpath, subprocess, sys, tarfile, zipfile

ROOT = Path('/25085400120/mpsma_fair_20261103_v1')
PROTOCOL_SHA = '8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'
FROZEN_ARCHIVE_SHA = 'e127bf14dacd999ca3794c90dc16d778f34f22ff135c227d599ad43ac243c36d'
DATA_SHA = 'fdde141711867fa1bc2de3a1e6342cca257eb056e2a1700795eb04068aa90266'
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(v):
    tmp = ROOT/'secondary_preparation.json.tmp'
    tmp.write_text(json.dumps(v, indent=2)+'\n')
    os.replace(tmp, ROOT/'secondary_preparation.json')
def extract_tar(path, dest, expected_prefix=None, allowed_external_links=None):
    with tarfile.open(path) as t:
        entries = t.getmembers()
        for m in entries:
            parts = PurePosixPath(m.name).parts
            assert parts and not m.name.startswith('/') and '..' not in parts, m.name
            if expected_prefix: assert parts[0] == expected_prefix, m.name
            assert m.isfile() or m.isdir() or m.issym() or m.islnk(), m.name
            if m.issym() or m.islnk():
                target = posixpath.normpath(posixpath.join(str(dest), posixpath.dirname(m.name) if m.issym() else '', m.linkname))
                limit = str(dest/expected_prefix) if expected_prefix else str(dest)
                explicitly_allowed = (allowed_external_links or {}).get(m.name) == target
                if explicitly_allowed: assert Path(target).is_file(), target
                assert target.startswith(limit+'/') or target==limit or explicitly_allowed, (m.name, m.linkname)
            if not m.isdir(): assert not (dest/m.name).exists() and not (dest/m.name).is_symlink(), m.name
        t.extractall(dest)

phase = sys.argv[1]
assert phase in ['inputs', 'runtime_v2']
state = {} if phase == 'inputs' else read(ROOT/'secondary_inputs_ready.json')
state.update({'status':'PREPARING_NO_MODEL_EXECUTION','pid':os.getpid(),
         'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
         'started_utc':datetime.now(timezone.utc).isoformat(),'model_executions':0,'optimizer_updates':0,'phase':phase})
with (ROOT/('secondary_preparation_'+phase+'.lock')).open('x') as f: f.write(json.dumps(state)+'\n')
save(state)
try:
    if phase == 'inputs':
        assert sha(ROOT/'secondary_frozen_inputs.tar.gz') == FROZEN_ARCHIVE_SHA
        extract_tar(ROOT/'secondary_frozen_inputs.tar.gz', ROOT)
    assert sha(ROOT/'execution_protocol.json') == PROTOCOL_SHA
    protocol = read(ROOT/'execution_protocol.json')
    for item in protocol['frozen_files']: assert sha(ROOT/item['path']) == item['sha256'], item['path']
    state.update(frozen_files_verified=len(protocol['frozen_files']),protocol_sha256=PROTOCOL_SHA);save(state)
    if phase == 'inputs':
        assert sha(ROOT/'verified_dataset.zip') == DATA_SHA
        cache = Path('/tmp/mpsma_fair_20261103_data')
        assert not cache.exists()
        cache.mkdir()
        manifest = read(ROOT/'dataset_bundle_manifest.json')['files']
        with zipfile.ZipFile(ROOT/'verified_dataset.zip') as z:
            assert set(z.namelist()) == {v['path'] for v in manifest}
            for info in z.infolist():
                assert not info.filename.startswith('/') and '..' not in PurePosixPath(info.filename).parts
            z.extractall(cache)
        for item in manifest: assert sha(cache/item['path']) == item['sha256'], item['path']
        assert len(manifest) == 7021
        state.update(dataset_files_verified=len(manifest),dataset_archive_sha256=DATA_SHA,
                     dataset_manifest_sha256=sha(ROOT/'dataset_bundle_manifest.json'),status='INPUTS_READY_WAITING_ENVIRONMENT')
        (ROOT/'secondary_inputs_ready.json').write_text(json.dumps(state,indent=2)+'\n')
        save(state);print(json.dumps(state));sys.exit(0)
    transfer = read(ROOT/'secondary_environment_transfer.json')
    archive = ROOT/'deim_environment.tar.gz'
    assert transfer['status'] == 'TRANSFERRED_DESTINATION_REHASH_REQUIRED'
    assert archive.stat().st_size == transfer['bytes'] and sha(archive) == transfer['sha256']
    env_parent = Path('/25085400120/envs')
    env_parent.mkdir(exist_ok=True)
    base_receipt = read(ROOT/'python_base_archive_completed.json')
    base_archive = ROOT/'pybase_environment.tar.gz'
    assert base_receipt['status']=='COMPLETED' and base_receipt['source']=='/25085400120/envs/pybase'
    assert base_archive.stat().st_size==base_receipt['bytes'] and sha(base_archive)==base_receipt['sha256']
    assert not (env_parent/'pybase').exists()
    extract_tar(base_archive,env_parent,'pybase')
    assert not (env_parent/'deim').exists()
    extract_tar(archive, env_parent, 'deim', {'deim/bin/python':'/25085400120/envs/pybase/bin/python3.10'})
    state.update(environment_archive_sha256=transfer['sha256'],python_base_archive_sha256=base_receipt['sha256'],
                 previous_runtime_attempt='Stopped at external interpreter-link validation before extracting deim; no model execution.',
                 preparation_script_sha256=sha(Path(__file__)),status='INPUTS_EXTRACTED_RUNTIME_CHECK_PENDING');save(state)
    code = """
import importlib.metadata as m,json,platform,subprocess,sys,torch,torchvision
print(json.dumps({'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,
 'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
 'packages':{d.metadata['Name']:d.version for d in m.distributions()},'hostname':platform.node(),
 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader,nounits'],text=True).strip(),
 'torch_build':torch.__config__.show(),'cuda_available':torch.cuda.is_available(),'model_executions':0}))
"""
    result = subprocess.run([str(env_parent/'deim/bin/python'),'-B','-c',code], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr[-2000:]
    actual = json.loads(result.stdout)
    (ROOT/'secondary_runtime.json').write_text(json.dumps(actual,indent=2)+'\n')
    expected = protocol['runtime']
    def normalize(packages): return {k.lower().replace('_','-'):v for k,v in packages.items()}
    assert normalize(actual['packages']) == normalize(expected['packages']), 'Package versions differ'
    for key in ['python','torch','cuda','cudnn','torch_build']:
        assert actual[key] == expected[key], key
    assert actual['cuda_available']
    assert actual['gpu'] == 'NVIDIA GeForce RTX 3090, GPU-ca8d11de-f221-f2b9-82bf-d389523dd71a, 550.100, 24576'
    state.update(status='READY_FOR_AUTHORIZED_FULL_ONLY',runtime_sha256=sha(ROOT/'secondary_runtime.json'),
                 gpu=actual['gpu'],runtime_matches_frozen_software=True,completed_utc=datetime.now(timezone.utc).isoformat(),
                 note='Frozen environment/preflight receipts are preserved first-host provenance; secondary_runtime.json documents the actual second host.')
    save(state);print(json.dumps(state))
except Exception as error:
    state.update(status='PREPARATION_FAILED_NO_TRAINING',error=repr(error));save(state);raise
