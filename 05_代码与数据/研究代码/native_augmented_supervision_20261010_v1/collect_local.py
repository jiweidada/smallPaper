from pathlib import Path
import hashlib
import json
import tarfile

root=Path(__file__).resolve().parents[3]
out=root/'outputs/native_augmented_supervision_20261010_v1'
expected={1:('1aac081aa46e9b8d6e40b686976facde412fb259ddb2b8012f5b953e86e3c791',10999199),2:('36fdd4d40c52dfea8f598db52764b6c274c597fb292d0812c3fa590536f3cb7b',10694783)}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipts=[]
for seed,(digest,size) in expected.items():
    archive=out/f'collection_seed{seed}.tar.gz';assert archive.stat().st_size==size and sha(archive)==digest
    target=out/f'collection_seed{seed}';target.mkdir(exist_ok=False)
    with tarfile.open(archive) as t:t.extractall(target,filter='data')
    capture=target/'capture_v1';manifest=json.loads((capture/'manifest.json').read_text())['files']
    for name,d in manifest.items():assert sha(capture/name)==d,name
    complete=json.loads((capture/'completion.json').read_text());controller=json.loads((target/'controller_completion.json').read_text())
    assert complete['status']=='COMPLETE_NATIVE_AUGMENTED_SUPERVISION_CAPTURE' and complete['model_forward_calls']==34 and complete['optimizer_steps']==0 and controller['exit_code']==0
    protocol=json.loads((out/'protocol.json').read_text())
    assert (out/'protocol.json').read_bytes()==(target/'protocol.json').read_bytes()
    for name,d in protocol['code_sha256'].items():assert sha(target/'code'/name)==d
    receipts.append(dict(seed=seed,archive_sha256=digest,archive_bytes=size,immutable_capture_files_verified=len(manifest),completion=complete,controller=controller,remote_code_identity_exact=True))
(out/'download_verification.json').write_text(json.dumps({'status':'PASS_DUAL_SEED_ARCHIVES_AND_ALL_CAPTURE_FILES','seeds':receipts},indent=2)+'\n',encoding='utf8')
print(json.dumps({'status':'PASS','files':[r['immutable_capture_files_verified'] for r in receipts]}))
