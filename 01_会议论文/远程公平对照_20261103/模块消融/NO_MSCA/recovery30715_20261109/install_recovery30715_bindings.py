"""Link only hash-checked second-recovery files; frozen originals remain unchanged."""
from pathlib import Path, PurePosixPath
from hashlib import sha256
import json
import os
import subprocess
ROOT=Path('/25085400120/mpsma_fair_20261103_v1/module_ablation_20261107_v1/no_msca_20261107_v1')
PREP=ROOT/'recovery30715_preparation'
def digest(path):
    h=sha256()
    with path.open('rb') as f:
        while block:=f.read(4*1024**2): h.update(block)
    return h.hexdigest()
plan_path=PREP/'new_recovery_file_plan.json'
assert digest(plan_path)=='07319ae796b3f0be5c3e7cc5342e65ec15f623f0e1206eb0ef559fbbbb974f19'
plan=json.loads(plan_path.read_text())
assert digest(ROOT/'execution_protocol.json')=='3e5e8937e0889aa360ffe3f7acb722ad1ab5f8e6c66cd6b3991a71516212dcfc'
assert plan['addendum_sha256']=='5b6fec3b7f4fe2f088cd0176b085fa6074731082307e6215d2ff46ce918a71c9'
assert all(not (ROOT/n).exists() for n in ['launch.lock','supervisor.json','supervisor.log','training.log'])
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()=='GPU-ca8d11de-f221-f2b9-82bf-d389523dd71a'
verified=[]
for item in plan['files']:
    rel=PurePosixPath(item['path'])
    assert not rel.is_absolute() and '..' not in rel.parts
    target=ROOT.joinpath(*rel.parts)
    assert target.resolve().is_relative_to(ROOT)
    if not target.exists():
        assert item['path'] in plan['new_flat_names']
        source=PREP.joinpath(*rel.parts)
        assert source.is_file() and digest(source)==item['sha256']
        os.link(source,target)
    assert target.is_file() and digest(target)==item['sha256']
    verified.append({'path':item['path'],'sha256':item['sha256'],'bytes':target.stat().st_size})
report={'status':'ALL_SECOND_RECOVERY_FILES_STAGED_NO_TRAINING_STARTED','files':verified,'addendum_sha256':plan['addendum_sha256'],'staging_plan_sha256':'07319ae796b3f0be5c3e7cc5342e65ec15f623f0e1206eb0ef559fbbbb974f19','training_started':False}
with (PREP/'installed_recovery30715_files.json').open('x') as f: json.dump(report,f,indent=2); f.write('\n')
print(json.dumps(report))
