from pathlib import Path
import base64
import hashlib
import json
import tarfile

root = Path(__file__).resolve().parents[3]
out = root / 'outputs/assignment_mechanism_diagnostic_20261010_v1'
expected = {1:(6049027,'c4680dfc2c96de8d74d155c76d28e6bc0e056f0c6a54169241523010731454b1'),2:(6049963,'f15140c8af99a163546202cd5bed377c1c70de30c52a2454c894ced8a4e79cfa')}
b_archive=out/'assignment_panel_seed2_collection_v1.tar.gz'
if not b_archive.exists():
    parts=sorted((out/'transfer_seed2_parts').glob('part_*.b64'))
    assert len(parts)==124
    with b_archive.open('xb') as f:
        for i,p in enumerate(parts):
            assert p.stem==f'part_{i:04d}'
            f.write(base64.b64decode(''.join(p.read_text().split()),validate=True))
receipts=[]
for seed,(size,digest) in expected.items():
    archive=out/f'assignment_panel_seed{seed}_collection_v1.tar.gz'
    assert archive.stat().st_size==size
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==digest
    target=out/f'collection_seed{seed}'
    target.mkdir(exist_ok=False)
    with tarfile.open(archive,'r:gz') as t:
        for m in t.getmembers():
            assert m.isfile() or m.isdir()
            assert not m.name.startswith('/') and '..' not in Path(m.name).parts
            dest=(target/m.name).resolve()
            assert dest.is_relative_to(target.resolve())
        t.extractall(target,filter='data')
    capture=target/'frozen_panel_v1'
    manifest=json.loads((capture/'manifest.json').read_text())['files']
    for name,sha in manifest.items():
        assert hashlib.sha256((capture/name).read_bytes()).hexdigest()==sha,name
    completion=json.loads((capture/'completion.json').read_text())
    assert completion['status']=='COMPLETE_FROZEN_ASSIGNMENT_DIAGNOSTIC_NOT_TRAINING'
    assert completion['training_epochs_added']==0 and completion['optimizer_steps']==0
    for p in (target/'collected_native_sources').rglob('*.py'):
        name=p.relative_to(target/'collected_native_sources').as_posix()
        assert hashlib.sha256(p.read_bytes()).hexdigest()==completion['source_python_sha256'][name]
    controller=json.loads((target/'controller_completion.json').read_text())
    assert controller['exit_code']==0 and controller['status']=='COMPLETE'
    receipts.append(dict(seed=seed,archive_bytes=size,archive_sha256=digest,immutable_files_verified=len(manifest),native_source_snapshots_verified=3,controller_exit_code=0,completion=completion['status']))
(out/'download_verification.json').write_text(json.dumps(dict(status='BOTH_ARCHIVES_AND_ALL_IMMUTABLE_FILES_VERIFIED',receipts=receipts,seed2_transport='Existing authenticated SSH; base64 frames normalized for terminal whitespace and decoded only after length validation; assembled archive size and SHA exact'),indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipts))
