"""Preserve v1; bind read-only collection to the namespace-repaired v3 producer."""
from pathlib import Path
import ast,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parent.parent
old=ROOT/'英文小论文/code/native_selection_collection12_20261011_v1'
new=ROOT/'英文小论文/code/native_selection_collection12_20261011_v2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
prior=json.loads((old/'protocol.json').read_bytes());assert {n:sha(old/n) for n in prior['files']}==prior['files']
new.mkdir(exist_ok=False)
for p in old.glob('*.py'):
    s=p.read_text(encoding='utf8').replace('native_selection_collection12_v1','native_selection_collection12_v2').replace('native_selection_diagnostic12_20261011_v2','native_selection_diagnostic12_20261011_v3').replace('native_selection_diagnostic12_v2','native_selection_diagnostic12_v3').replace('native_selection_diagnostic_packet_v1','native_selection_diagnostic_packet_v2').replace('_selection_v1','_selection_v2').replace('positive_exact12_v2_binding','positive_exact12_v3_binding').replace('existing producer v2','existing producer v3')
    if p.name=='collector.py':
        s=s.replace("    assert state(owner) is not None, 'Observe the existing diagnostic queue live; never restart it'", "    # A completed positive queue is also collectable; this never launches a replay.\n    qcompletion=qroot/f'seed{seed}_queue/completion.json'\n    assert state(owner) is not None or (qcompletion.exists() and read(qcompletion)['status']=='ALL_THREE_NATIVE_SELECTION_DIAGNOSTICS_FINISHED'), 'Require actual live queue or all-three positive completion; never restart it'")
    if p.name=='verify_local.py':
        s=s.replace('import argparse, hashlib, json, shutil, sys, zipfile','import argparse, hashlib, json, shutil, stat, sys, zipfile')
        s=s.replace('assert all(not x.is_dir() for x in z.infolist()) and z.testzip() is None','assert all(not x.is_dir() and stat.S_ISREG(x.external_attr >> 16) for x in z.infolist()) and z.testzip() is None')
    compile(s,str(new/p.name),'exec');(new/p.name).write_text(s,encoding='utf8',newline='\n')
diag=ROOT/'英文小论文/code/native_selection_diagnostic12_20261011_v3'
source={p.name:sha(p) for p in diag.iterdir() if p.is_file()}
spec=dict(prior,schema='native_selection_collection12_v2',files={p.name:sha(p) for p in new.glob('*.py')},diagnostic_source_sha256=source,diagnostic_protocol_sha256=source['diagnostic_protocol.json'],scope=prior['scope'].replace('producer v2','producer v3'))
(new/'protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
assert {n:sha(old/n) for n in prior['files']}==prior['files']
out=ROOT/'outputs/native_selection_collection12_20261011_v2';out.mkdir(exist_ok=False)
with zipfile.ZipFile(out/'deployment_v2.zip','x',zipfile.ZIP_DEFLATED) as z:
    for p in new.iterdir():
        if p.is_file():z.write(p,'code/'+p.name)
report=dict(status='VERSIONED_COLLECTION_SOURCE_REVIEWED_BINDING_FIXTURES_PENDING',sources={p.name:sha(p) for p in new.iterdir() if p.is_file()},preserved_v1_files_unchanged=True,diagnostic_producer_version=3,archive_sha256=sha(out/'deployment_v2.zip'),archive_bytes=(out/'deployment_v2.zip').stat().st_size,new_model_forwards=0,new_training_epochs=0)
(out/'source_review.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))
