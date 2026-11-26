"""Export completed immutable arm data after CPU matching passes; no models."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import zipfile

ROOT=Path('/root/uav_a58_20261102_v1/pilot_v1')
INPUTS=Path('/root/uav_a54_20261101_v1/A58_uploads')
EXPORTS=Path('/root/uav_a54_20261101_v1/A58_exports')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def main(arm):
    plan=json.loads((ROOT/'protocol_FROZEN_v1.json').read_text())
    supervisor_bytes=(ROOT/'supervisor_progress.json').read_bytes()
    supervisor=json.loads(supervisor_bytes)
    assert arm in supervisor['completed']
    d=ROOT/('arm_'+arm)
    result=json.loads((d/'result.json').read_text())
    assert result['status']=='COMPLETED' and result['epochs']==12 and result['optimizer_updates']==9708
    assert result['full_model_forwards']==10159 and result['protocol_sha256']==sha(ROOT/'protocol_FROZEN_v1.json')
    assert not Path('/proc/'+str(result['pid'])).exists()
    matching=ROOT/('saved_CPU_matching_'+arm+'_v1')
    assert json.loads((matching/'result.json').read_text())['status']=='PASS_SAVED_PREDICTION_CPU_MATCHING'
    assert json.loads((matching/'exit.json').read_text())['exit']==0
    for n,h in plan['files_sha256'].items():assert sha(ROOT/n)==h,n
    files={};sources={};snapshots={}
    for file in d.rglob('*'):
        if file.is_file() and file.suffix!='.pth':
            assert not file.is_symlink()
            name='arm_'+arm+'/'+file.relative_to(d).as_posix();sources[name]=file
    for file in matching.iterdir():
        if file.is_file():sources['CPU_matching/'+file.name]=file
    for n in plan['files_sha256']:sources['frozen_source/'+n]=ROOT/n
    sources['protocol_FROZEN_v1.json']=ROOT/'protocol_FROZEN_v1.json'
    for n in ['fixed12_audit_scope_FROZEN_v1.json','audit_saved_matching_CPU_v1.py']:
        sources['audit_source/'+n]=INPUTS/n
    snapshots['native_supervisor_snapshot.json']=supervisor_bytes
    snapshots['terminal_evidence.json']=json.dumps(dict(arm=arm,original_worker_pid=result['pid'],
        original_worker_absent=True,supervisor_confirmed_completed=True,source_files_verified=len(plan['files_sha256']),
        supervisor_returncode_scope='Pinned supervisor waits and asserts rc0 before placing arm in completed; no separate per-arm OS exit file was written',
        new_model_forwards=0),indent=2).encode()
    if (ROOT/'B12_learning_gate.json').exists():snapshots['B12_learning_gate_snapshot.json']=(ROOT/'B12_learning_gate.json').read_bytes()
    for name,file in sources.items():files[name]=dict(bytes=file.stat().st_size,sha256=sha(file))
    for name,raw in snapshots.items():files[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    ckpt=d/'epoch12.pth'
    files['arm_'+arm+'/epoch12.pth']=dict(bytes=ckpt.stat().st_size,sha256=sha(ckpt))
    others={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in [d/'epoch04.pth',d/'epoch08.pth']}
    manifest=dict(arm=arm,epoch=12,files=files,preserved_remote_other_checkpoints=others,
        heavy_checkpoint_separate=True,protocol_sha256=result['protocol_sha256'],new_model_forwards=0)
    z=EXPORTS/('fixed12_'+arm+'_small_v1.zip')
    heavy=EXPORTS/('fixed12_'+arm+'_epoch12.pth')
    assert not z.exists() and not heavy.exists()
    os.link(ckpt,heavy)
    with zipfile.ZipFile(z,'x',zipfile.ZIP_DEFLATED) as archive:
        for name,file in sources.items():archive.write(file,name)
        for name,raw in snapshots.items():archive.writestr(name,raw)
        archive.writestr('bundle_manifest.json',json.dumps(manifest,indent=2))
    ready=dict(status='FIXED12_SAVED_DATA_READY',arm=arm,epoch=12,zip_bytes=z.stat().st_size,zip_sha256=sha(z),
        checkpoint_bytes=heavy.stat().st_size,checkpoint_sha256=files['arm_'+arm+'/epoch12.pth']['sha256'],
        payload_files=len(files),endpoint_metrics=result['endpoint_metrics'],CPU_matching_passed=True,
        protocol_sha256=result['protocol_sha256'],new_model_forwards=0)
    with (EXPORTS/('fixed12_'+arm+'_ready_v1.json')).open('x') as f:json.dump(ready,f,indent=2)
    print(json.dumps(ready),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['B','M','Q'],required=True)
    main(parser.parse_args().arm)
