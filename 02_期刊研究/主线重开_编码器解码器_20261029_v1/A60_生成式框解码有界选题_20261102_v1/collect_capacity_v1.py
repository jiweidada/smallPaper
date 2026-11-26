"""Verify the existing, single capacity execution. Never runs a model."""
from pathlib import Path
import hashlib
import json
import zipfile

P=Path(__file__).resolve().parent
D=P.parent/'A54_场景尺度约束候选_20261101_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ready=json.loads((D/'A60_downloads/A60_capacity_ready_v1.json').read_text())
    z=D/'A60_downloads/A60_capacity_bundle_v1.zip'
    assert sha(z)=='cc4fcb4cca7f11448f7163adf812410858a0be7c77f69a8e5ca512a35e2809a8'
    out=P/'capacity_remote_v1';assert not out.exists()
    with zipfile.ZipFile(z) as f:
        assert f.testzip() is None
        manifest=json.loads(f.read('manifest.json'))
        assert len(manifest)==18
        for name,meta in manifest.items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            content=f.read(name)
            assert len(content)==meta['bytes'] and hashlib.sha256(content).hexdigest()==meta['sha256'],name
        out.mkdir();f.extractall(out)
    scope=json.loads((out/'capacity_scope_FROZEN_v1.json').read_text())
    result=json.loads((out/'capacity_v1/result.json').read_text())
    assert result['status']=='PASS_REAL_GPU_CAPACITY_AND_SHARED_INPUTS'
    assert result['counts']==dict(started=9,completed=9,updates=6)
    assert not result['capacity_weights_reused'] and result['formal_training_epochs']==0
    for name,h in scope['source_sha256'].items():assert sha(P/name)==sha(out/name)==h
    events=[json.loads(x) for x in (out/'capacity_v1/actual_events.jsonl').read_text().splitlines()]
    assert sum(x['event']=='forward_start' for x in events)==9
    assert sum(x['event']=='forward_complete' for x in events)==9
    assert sum(x['event']=='optimizer_returned' for x in events)==6
    for case in [0,1]:
        rows={x['arm']:x for x in result['rows'] if x.get('case')==case}
        for key in ['image','GT','labels','prior','u']:
            assert len({x['input_hashes'][key] for x in rows.values()})==1
        assert rows['M']['input_hashes']['noisy']==rows['Q']['input_hashes']['noisy']
        assert rows['M']['input_hashes']['time']!=rows['Q']['input_hashes']['time']
    for name in ['capacity_v1/exit.json','capacity_supervisor_exit_v1.json']:
        assert json.loads((out/name).read_text())['exit']==0
    observe=json.loads((D/'A60_capacity_observe_001.json').read_text())
    verification=dict(status='PASS_EXISTING_CAPACITY_BUNDLE_18_FILES',zip_sha256=sha(z),
        verified_payloads=18,model_forwards=9,discarded_optimizer_updates=6,formal_epochs=0,
        M_Q_identical_noisy_boxes=True,M_Q_different_time_labels=True,source_bindings_verified=True,
        original_process_receipt=str(D/'A60_capacity_observe_001.json'),
        peak_allocated_bytes=max(x.get('peak_allocated_bytes',0) for x in result['rows']),
        new_detector_forwards=0,new_optimizer_updates=0,ready_receipt=ready,
        scope='Checks existing files and journals only; no rerun or efficacy conclusion.')
    with (P/'capacity_local_verification_v1.json').open('x',encoding='utf-8') as f:json.dump(verification,f,indent=2)
    print(json.dumps({k:v for k,v in verification.items() if k!='ready_receipt'}))

if __name__=='__main__':main()
