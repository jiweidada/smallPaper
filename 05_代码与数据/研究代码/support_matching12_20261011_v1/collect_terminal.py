"""Archive actual new FDR endpoint, saved-EMA replay and immutable BASE evidence."""
from pathlib import Path
import datetime
import zipfile
from runtime import HERE,verify_bundle,read,write,sha


def collect(bundle):
    receipt,plan,paths=verify_bundle(bundle)
    screen=paths['screen_root'];run=paths['output']
    assert read(screen/'candidate_completion.json')['exit_code']==0
    assert read(screen/'controller_completion.json')['status']=='COMPLETE_FIXED12'
    reload=screen/'candidate_reload_v1'
    assert read(reload/'completion.json')['exit_code']==0
    assert not read(reload/'completion.json')['timed_out']
    assert read(reload/'verification.json')['status']=='PASS_SAVED_CANDIDATE_RELOAD'
    done=read(run/'training_completion.json')
    assert done['completed_epochs']==12 and done['identity']==receipt['identity']
    assert sha(run/'last.pth')==done['last_checkpoint_sha256']
    pairs={}
    for folder,label in ((HERE,'code'),(screen/'bundle','bundle'),(run,'run'),(reload,'candidate_reload_v1')):
        for file in folder.rglob('*'):
            if file.is_file() and '__pycache__' not in file.parts:
                rel=file.relative_to(folder).as_posix()
                if label=='run' and file.suffix=='.pth' and rel!='last.pth':
                    continue
                pairs[label+'/'+rel]=file
    for file in screen.glob('*'):
        if file.is_file():
            pairs['screen/'+file.name]=file
    baseline=paths['baseline_cached']
    spec=read(HERE/'training_protocol.json')['baseline_reuse'][str(plan['seed'])]
    for name,digest in spec['pinned_files'].items():
        assert sha(baseline/name)==digest,name
    for name in ('completion.json','historical_comparison.json','baseline.yml'):
        pairs['baseline_eval/'+name]=baseline/name
    for name in ('endpoint_epoch12','stages_epoch12'):
        for file in (baseline/name).rglob('*'):
            if file.is_file():
                pairs['baseline_eval/'+name+'/'+file.relative_to(baseline/name).as_posix()]=file
    files={name:dict(bytes=file.stat().st_size,sha256=sha(file)) for name,file in sorted(pairs.items())}
    manifest=dict(schema='support_matching12_terminal_v1',identity=receipt['identity'],files=files,
        baseline_model_forwards_added=0,training_epochs=12,automatic_retry=False)
    write(screen/'terminal_manifest.json',manifest)
    archive=paths['experiment_root']/f"terminal_support_cost_seed{plan['seed']}_v1.zip"
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,file in sorted(pairs.items()):
            z.write(file,name)
        z.write(screen/'terminal_manifest.json','terminal_manifest.json')
    metadata=dict(status='FDR_FIXED12_TRAIN_RELOAD_COLLECTION_FINISHED',seed=plan['seed'],
        bytes=archive.stat().st_size,sha256=sha(archive),manifest_sha256=sha(screen/'terminal_manifest.json'),
        archive_entries=len(pairs)+1,baseline_forwards=0,extra_epochs=0,
        completed_remote_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(paths['experiment_root']/f"terminal_support_cost_seed{plan['seed']}_v1_metadata.json",metadata)
    write(screen/'post_training_completion.json',metadata)
    print(__import__('json').dumps(metadata),flush=True)
