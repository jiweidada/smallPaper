"""Verify and archive existing first-epoch records; never imports a detector."""
from pathlib import Path
import hashlib
import json
import math
import zipfile

P=Path(__file__).resolve().parent
D=P.parent/'A54_场景尺度约束候选_20261101_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    out=P/'epoch1_B_remote_v1';assert not out.exists()
    archive=D/'A60_downloads/epoch1_B_bundle_v1.zip'
    ready=json.loads((D/'A60_downloads/epoch1_B_ready_v1.json').read_text())
    assert archive.stat().st_size==ready['zip_bytes'] and sha(archive)==ready['zip_sha256']
    assert ready['files']==7 and ready['epoch1_forwards']==6471 and ready['epoch1_updates']==809
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        manifest=json.loads(z.read('manifest.json'));assert len(manifest)==7
        for name,m in manifest.items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            data=z.read(name);assert len(data)==m['bytes'] and hashlib.sha256(data).hexdigest()==m['sha256']
        out.mkdir();z.extractall(out)
    assert sha(out/'protocol_FROZEN_v1.json')==sha(P/'protocol_FROZEN_v1.json')
    row=json.loads((out/'B/epoch01.json').read_text())
    flow=json.loads((out/'B/epoch01_flow.json').read_text())
    first=[json.loads(s) for s in (out/'B/first8_input_flow.jsonl').read_text().splitlines()]
    assert len(first)==8 and [v['position'] for v in first]==list(range(8))
    events=[json.loads(s) for s in (out/'B/epoch1_actual_events_prefix.jsonl').read_text().splitlines()]
    assert len(events)==13751
    starts=[x for x in events if x['event']=='forward_start']
    posts=[x for x in events if x['event']=='forward_complete']
    updates=[x for x in events if x['event']=='optimizer_returned']
    assert len(starts)==len(posts)==6471 and len(updates)==809
    assert [x['count'] for x in starts]==[x['count'] for x in posts]==list(range(1,6472))
    assert [x['position'] for x in starts]==list(range(6471))
    assert all(x['phase']=='train' and x['epoch']==1 for x in starts)
    assert [x['group_images'] for x in updates]==[8]*808+[7]
    assert [x['count'] for x in updates]==list(range(1,810))
    assert all(math.isfinite(x[k]) for x in updates for k in ['mean_loss','gradient_norm','time_MLP_gradient_L1','lr_factor'])
    assert min(x['time_MLP_gradient_L1'] for x in updates)>0
    assert row['counts']==dict(started=6471,completed=6471,updates=809,backwards=6471,epochs=1,
        train_forwards=6471,eval_forwards=0,latency_forwards=0)
    assert row['flow']==flow and flow['images']==6471
    assert abs(row['mean_loss']-sum(x['mean_loss']*x['group_images'] for x in updates)/6471)<1e-8
    run=json.loads((out/'B/run_once.json').read_text());init=json.loads((out/'B/initialization.json').read_text())
    assert run['protocol_sha256']==sha(P/'protocol_FROZEN_v1.json')
    scope=json.loads((P/'protocol_FROZEN_v1.json').read_text())
    assert init['initialization_sha256']==scope['initialization_sha256']
    result=dict(status='PASS_EXISTING_B_EPOCH1_BUNDLE_7_PAYLOADS_AND_13751_EVENTS',verified_payloads=7,
        zip_sha256=sha(archive),original_train_forwards=6471,original_backward_calls=6471,original_optimizer_updates=809,
        original_epochs_completed=1,losses_and_gradients_finite=True,time_MLP_gradients_positive_every_update=True,
        flow=flow,mean_loss=row['mean_loss'],minimum_time_MLP_gradient_L1=row['minimum_time_MLP_gradient_L1'],
        input_stream_recorded=True,cross_arm_formal_input_match_verified=False,AP_available=False,
        new_model_forwards=0,new_optimizer_updates=0,new_COCO_evaluations=0,
        scope='Integrity and operational evidence for the original completed first epoch; not AP efficacy, learning-gate completion or mainline validation.')
    with (P/'B_epoch1_bundle_local_verification_v1.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='flow'}))

if __name__=='__main__':main()
