"""Verify transferred terminal artifacts and event flow; no torch imports."""
from pathlib import Path,PurePosixPath
import argparse
import collections
import hashlib
import json
import math
import shutil
import statistics
import zipfile

P=Path(__file__).resolve().parent
R54=P.parent/'A54_场景尺度约束候选_20261101_v1'
def load(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def save_new(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)

def main(arm):
    dl=R54/'A57_downloads'
    ready=load(dl/('A58_fixed12_'+arm+'_ready_v1.json'))
    assert ready['status']=='FIXED12_SAVED_DATA_READY' and ready['arm']==arm and ready['epoch']==12
    assert ready['protocol_sha256']==sha(P/'protocol_FROZEN_v1.json') and ready['CPU_matching_passed']
    z=dl/('A58_fixed12_'+arm+'_small_v1.zip');heavy=dl/('A58_fixed12_'+arm+'_epoch12.pth')
    assert z.stat().st_size==ready['zip_bytes'] and sha(z)==ready['zip_sha256']
    assert heavy.stat().st_size==ready['checkpoint_bytes'] and sha(heavy)==ready['checkpoint_sha256']
    dest=P/('fixed12_'+arm+'_remote_v1');dest.mkdir(exist_ok=False)
    with zipfile.ZipFile(z) as archive:
        manifest=json.loads(archive.read('bundle_manifest.json'))
        assert len(manifest['files'])==ready['payload_files']
        expected=set(manifest['files'])-{'arm_'+arm+'/epoch12.pth'}
        assert set(archive.namelist())==expected|{'bundle_manifest.json'}
        for n in expected:
            rel=PurePosixPath(n);assert not rel.is_absolute() and '..' not in rel.parts
            target=(dest/Path(*rel.parts)).resolve();assert target.is_relative_to(dest.resolve())
            target.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(n) as src,target.open('xb') as out:shutil.copyfileobj(src,out,1<<20)
        save_new(dest/'bundle_manifest.json',manifest)
    shutil.copyfile(heavy,dest/('arm_'+arm)/'epoch12.pth')
    for n,row in manifest['files'].items():
        file=dest/n;assert file.stat().st_size==row['bytes'] and sha(file)==row['sha256'],n
    plan=load(P/'protocol_FROZEN_v1.json')
    for n,h in plan['files_sha256'].items():assert sha(dest/'frozen_source'/n)==sha(P/n)==h,n
    d=dest/('arm_'+arm);result=load(d/'result.json');m=load(d/'epoch12_EMA_metrics.json')
    assert result['status']=='COMPLETED' and result['epochs']==12 and result['optimizer_updates']==9708
    assert result['full_model_forwards']==10159 and result['endpoint_metrics']==m==ready['endpoint_metrics']
    assert result['protocol_sha256']==ready['protocol_sha256']
    assert result['counts']==dict(full_model_forwards_started=10159,full_model_forwards_completed=10159,optimizer_updates=9708,native_counter=10159)
    events=[json.loads(s) for s in (d/'actual_events.jsonl').read_text().splitlines()]
    counts=collections.Counter(e['event'] for e in events)
    assert counts==dict(forward_start=10159,forward_complete=10159,optimizer_returned=9708)
    assert len(events)==30026
    phases=collections.Counter(e['phase'] for e in events if e['event']=='forward_start')
    assert phases==dict(train=9708,validation=411,latency=40)
    for typ,key,end in [('forward_start','forward',10159),('forward_complete','forward',10159),('optimizer_returned','updates',9708)]:
        assert [e[key] for e in events if e['event']==typ]==list(range(1,end+1))
    started=finished=updated=0;phase=None
    for e in events:
        if e['event']=='forward_start':
            assert started==finished;started+=1;phase=e['phase']
        elif e['event']=='forward_complete':
            assert started==finished+1 and e['phase']==phase;finished+=1
        else:
            assert started==finished and phase=='train';updated+=1
    assert updated==9708
    init=load(d/'initialization.json')
    assert init['all_initial_tensors_identical'] and init['common_init_sha256']==plan['augmented_initialization_sha256']
    assert init['arm_mode']==arm and init['model_parameters']==9951721 and init['seed']==20261101
    flow_keys=['order_sha256','first_eight_actual_input_sha256','first_eight_pre_forward_rng_sha256','all_batches_targets_sha256']
    for epoch in range(1,13):
        flow=load(d/('epoch%02d_input_flow.json'%epoch))
        assert flow['epoch']==epoch and flow['optimizer_updates']==809*epoch
        assert len(flow['first_eight_actual_input_sha256'])==len(flow['first_eight_pre_forward_rng_sha256'])==8
        if arm!='B':
            prior=load(P/'fixed12_B_remote_v1/arm_B'/('epoch%02d_input_flow.json'%epoch))
            for key in flow_keys:assert flow[key]==prior[key],(epoch,key)
    cpu=load(dest/'CPU_matching/result.json')
    assert cpu['status']=='PASS_SAVED_PREDICTION_CPU_MATCHING' and cpu['saved_CPU_matching_calls']==1
    assert cpu['model_constructors']==cpu['model_forwards']==cpu['optimizer_updates']==0
    assert load(dest/'CPU_matching/exit.json')['exit']==0 and not cpu['torch_imported']
    assert max(abs(v) for v in cpu['metric_differences_pp'].values())<1e-10
    for n,h in cpu['input_hashes'].items():assert sha(d/n)==h,n
    assert cpu['scope_sha256']==sha(P/'fixed12_audit_scope_FROZEN_v1.json')==sha(dest/'audit_source/fixed12_audit_scope_FROZEN_v1.json')
    assert cpu['script_sha256']==sha(P/'audit_saved_matching_CPU_v1.py')==sha(dest/'audit_source/audit_saved_matching_CPU_v1.py')
    terminal=load(dest/'terminal_evidence.json');assert terminal['original_worker_absent'] and terminal['supervisor_confirmed_completed']
    assert arm in load(dest/'native_supervisor_snapshot.json')['completed']
    gt=load(d/'epoch12_EMA_GT.json');pred=load(d/'epoch12_EMA_predictions.json')
    assert len(gt['images'])==548 and len(gt['annotations'])==38602 and gt['ignored_valid_gt']==157
    assert len(pred)==m['predictions_after_ignore']
    assert sum(a['area']<=1024 for a in gt['annotations'])==26444
    allowed={i['id'] for i in gt['images']};byimage=collections.Counter()
    for item in pred:
        assert item['image_id'] in allowed and item['category_id'] in range(1,11)
        assert all(math.isfinite(v) for v in item['bbox']+[item['score']])
        assert item['bbox'][2]>0 and item['bbox'][3]>0 and 0<=item['score']<=1
        byimage[item['image_id']]+=1
    assert max(byimage.values())<=300
    latency=result['latency'];assert latency['warmup']==10 and latency['repeats']==30 and len(latency['samples_ms'])==30
    assert all(math.isfinite(v) and v>0 for v in latency['samples_ms'])
    assert abs(statistics.median(latency['samples_ms'])-latency['median_ms'])<1e-9
    out=dict(status='PASS_FIXED12_BUNDLE_COUNTERS_INPUT_FLOW_AND_CPU_MATCHING',arm=arm,epochs=12,
        optimizer_updates=9708,full_model_forwards=10159,event_rows=30026,forward_phases=dict(phases),
        payload_files_verified=len(manifest['files']),endpoint_metrics=m,latency=latency,
        input_flow_epochs_verified=12,matched_B_flow=arm!='B',CPU_saved_matching_calls=1,
        native_checkpoint_preserved_locally=True,new_audit_detector_forwards=0,optimizer_updates_added=0,
        pending='CPU checkpoint tensor audit via audit_fixed12_checkpoint_local_v1.py')
    save_new(P/('fixed12_'+arm+'_bundle_local_verification_v1.json'),out)
    print(json.dumps(out),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['B','M','Q'],required=True)
    main(parser.parse_args().arm)
