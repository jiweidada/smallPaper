"""Collect already stopped exact12 primary artifacts; no training or evaluation."""
from pathlib import Path
import argparse,datetime,hashlib,json,sys,zipfile

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def live(record,pid,tick):
    try:
        s=Path('/proc/'+str(record[pid])+'/stat').read_text().split()
        return s[21]==str(record[tick]) and s[2]!='Z'
    except (FileNotFoundError,ProcessLookupError):return False
def collect(prefix,seed,arm):
    assert seed in (1,2) and prefix=={1:'/25085400120',2:'/24085403014'}[seed]
    base=Path(prefix)/'DEIM/experiments'
    assert arm=='warm_stem'
    rootname='warm_stem12_20261011_v1'
    collector_root=Path(__file__).resolve().parent
    protocol=read(collector_root/'protocol.json')
    assert sha(__file__)==protocol['collector_sha256']
    code=base/rootname/'code'
    assert {p.name:sha(p) for p in code.iterdir() if p.is_file()}==protocol['warm_files_sha256']
    sys.path.insert(0,str(code))
    from runtime import verify_bundle
    from stage_trace import verify_trace
    from results import load_endpoint
    bundle=base/rootname/f'seed{seed}_{arm}/bundle'
    receipt,plan,paths=verify_bundle(bundle)
    assert plan['seed']==seed and plan['arm']==arm
    screen,run=paths['screen_root'],paths['output']
    assert not live(read(screen/'controller_start.json'),'pid','start_ticks')
    assert not live(read(screen/'candidate_start.json'),'child_pid','child_start_ticks')
    assert read(screen/'controller_completion.json')['status']=='COMPLETE_FIXED12'
    assert read(screen/'candidate_completion.json')['exit_code']==0
    assert not read(screen/'candidate_completion.json')['timed_out']
    done=read(run/'training_completion.json');identity=receipt['identity']
    assert done['status']=='NATIVE_FIT_FINISHED_EXACT12' and done['completed_epochs']==12
    assert done['entered_epochs']==list(range(12)) and done['identity']==identity
    assert sha(run/'last.pth')==done['last_checkpoint_sha256']
    assert sha(run/'stages_epoch12/manifest.json')==done['stage_manifest_sha256']
    assert sha(run/'endpoint_epoch12/manifest.json')==done['endpoint_manifest_sha256']
    verify_trace(run/'stages_epoch12',identity)
    endpoint,flags,_=load_endpoint(run/'endpoint_epoch12')
    assert endpoint['identity']==identity and endpoint['evaluated_weights']=='EMA' and endpoint['epoch_zero_based']==11
    rows=[json.loads(line) for line in (run/'log.txt').read_text().splitlines() if line]
    assert [r['epoch'] for r in rows]==list(range(12))
    assert rows[-1]['test_coco_eval_bbox']==endpoint['native_stats']
    output=base/'warm_stem_primary_collection12_20261011_v1'/f'seed{seed}_{arm}_v1'
    output.mkdir(parents=True,exist_ok=False)
    pairs={}
    for folder,label in ((code,'code'),(bundle,'bundle'),(run,'run')):
        for p in folder.rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts:continue
            relative=p.relative_to(folder).as_posix()
            if label=='run' and p.suffix=='.pth' and relative!='last.pth':continue
            pairs[label+'/'+relative]=p
    for name in ('candidate_start.json','controller_start.json','candidate_completion.json','controller_completion.json',
                 'baseline_reuse.json','detector_preflight_start.json','detector_preflight_completion.json','training_resource_receipt.json'):
        pairs['screen/'+name]=screen/name
    for p in (screen/'detector_preflight_v1').rglob('*'):
        if p.is_file():pairs['screen/detector_preflight_v1/'+p.relative_to(screen/'detector_preflight_v1').as_posix()]=p
    for key in ('image_manifest','image_verification'):
        pairs['provenance/'+key+'.json']=paths[key]
    for label,p in pairs.items():
        assert p.is_file() and p.resolve().is_relative_to((base/rootname).resolve()),label
    files={name:dict(bytes=p.stat().st_size,sha256=sha(p)) for name,p in sorted(pairs.items())}
    manifest=dict(schema='exact12_warm_primary_artifact_collection_v1',identity=identity,files=files,
        evaluated_weights='EMA',epoch_zero_based=11,completed_epochs=12,
        source_training_completion_sha256=sha(run/'training_completion.json'),
        last_checkpoint_sha256=done['last_checkpoint_sha256'],native_stats=endpoint['native_stats'],
        stage_images=548,saved_EMA_independent_replay_included=False,
        collector_sha256=sha(__file__),new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0,
        scope='Primary native training validation artifacts only. Independent saved-EMA replay and formal terminal reader are still pending; this collection never substitutes for those gates.')
    write(output/'primary_manifest.json',manifest)
    archive=output/'primary_artifacts_v1.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,p in sorted(pairs.items()):z.write(p,name)
        z.write(output/'primary_manifest.json','primary_manifest.json')
    metadata=dict(status='COLLECTED_STOPPED_EXACT12_PRIMARY_ARTIFACTS',seed=seed,arm=arm,
        archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),
        manifest_sha256=sha(output/'primary_manifest.json'),archive_entries=len(pairs)+1,
        last_checkpoint_sha256=done['last_checkpoint_sha256'],native_stats=endpoint['native_stats'],
        native_gate_passed=done['native_gate_passed'],saved_EMA_independent_replay_included=False,
        collected_remote_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        collector_sha256=sha(__file__),collector_protocol_sha256=sha(collector_root/'protocol.json'),
        new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0)
    write(output/'metadata.json',metadata)
    assert 'torch' not in sys.modules and not any(x.startswith('pycocoeval') for x in sys.modules)
    return metadata
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--arm',choices=('warm_stem',),required=True)
    a=p.parse_args()
    assert a.prefix=={1:'/25085400120',2:'/24085403014'}[a.seed]
    collection_root=Path(__file__).resolve().parent.parent
    start=collection_root/f'seed{a.seed}_collect_start.json'
    done=collection_root/f'seed{a.seed}_collect_completion.json'
    failure=collection_root/f'seed{a.seed}_collect_failure.json'
    assert not any(p.exists() for p in (start,done,failure)), 'Inspect existing attempt; no automatic retry'
    import os,traceback
    receipt=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
        seed=a.seed,source_sha256=sha(__file__),protocol_sha256=sha(Path(__file__).parent/'protocol.json'),
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),automatic_retry=False,
        new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0)
    write(start,receipt)
    try:
        metadata=collect(a.prefix,a.seed,a.arm)
        result=dict(status='STOPPED_WARM_PRIMARY12_PACKAGE_COLLECTED',exit_code=0,receipt=receipt,metadata=metadata,
                    automatic_retry=False,new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0)
        write(done,result);print(json.dumps(result),flush=True)
    except BaseException:
        write(failure,dict(status='READ_ONLY_COLLECTION_FAILURE_NO_RETRY',receipt=receipt,traceback=traceback.format_exc()))
        raise

