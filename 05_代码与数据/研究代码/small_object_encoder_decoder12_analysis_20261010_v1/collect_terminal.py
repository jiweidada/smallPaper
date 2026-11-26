"""Archive a stopped run and its saved-weight verification. Never starts training."""
from pathlib import Path
import argparse,datetime,hashlib,json,sys,zipfile

read=lambda p:json.loads(Path(p).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def same_live(record,pid_key,tick_key):
    p=Path(f"/proc/{record[pid_key]}/stat")
    if not p.exists():return False
    fields=p.read_text().split()
    return fields[21]==str(record[tick_key]) and fields[2]!='Z'


def terminal_state(screen):
    child=screen/'candidate_start.json';controller=screen/'controller_start.json'
    if child.exists() and same_live(read(child),'child_pid','child_start_ticks'):return 'CANDIDATE_RUNNING'
    if controller.exists() and same_live(read(controller),'pid','start_ticks'):return 'SUPERVISOR_RUNNING'
    if (screen/'controller_failure.json').exists():
        assert (screen/'candidate_completion.json').exists(),'Failure before candidate has a separate engineering archive'
        return 'FAILED_TRAINING_NO_FIXED12'
    if not (screen/'controller_completion.json').exists():return 'TERMINAL_NOT_PROVEN'
    assert read(screen/'controller_completion.json')['status']=='COMPLETE_FIXED12'
    assert read(screen/'candidate_completion.json')['exit_code']==0
    reloaded=screen/'candidate_reload_v1'
    if not (reloaded/'completion.json').exists():return 'FIXED12_COMPLETE_RELOAD_PENDING'
    if same_live(read(reloaded/'start.json'),'pid','start_ticks'):return 'RELOAD_RUNNING'
    result=reloaded/'verification.json'
    if read(reloaded/'completion.json')['exit_code']!=0 or not result.exists():return 'FIXED12_RELOAD_FAILED'
    assert read(result)['status']=='PASS_SAVED_CANDIDATE_RELOAD'
    return 'FIXED12_AND_RELOAD_COMPLETE'


def collect(seed,check_only=False):
    prefix={1:'/25085400120',2:'/24085403014'}[seed];version={1:4,2:3}[seed]
    root=Path(prefix)/f'DEIM/experiments/small_object_encoder_decoder12_20261010_v{version}'
    screen=root/f'seed{seed}_fine';state=terminal_state(screen)
    if check_only:return {'state':state,'archive_created':False,'new_model_forwards':0}
    assert state in ('FAILED_TRAINING_NO_FIXED12','FIXED12_RELOAD_FAILED','FIXED12_AND_RELOAD_COMPLETE'),state
    code=root/'code';bundle=root/f'prepared_seed{seed}';run=screen/'run'
    sys.path.insert(0,str(code))
    from runtime import verify_bundle
    receipt,plan,paths=verify_bundle(bundle)
    assert plan['seed']==seed and plan['variant']=='fine'
    if state!='FAILED_TRAINING_NO_FIXED12':
        done=read(run/'training_completion.json')
        assert done['completed_epochs']==12 and done['entered_epochs']==list(range(12))
        assert done['identity']==receipt['identity'] and sha(run/'last.pth')==done['last_checkpoint_sha256']
        for sub,key in [('endpoint_epoch12','endpoint_manifest_sha256'),('stages_epoch12','stage_manifest_sha256')]:
            assert sha(run/sub/'manifest.json')==done[key]
    archive=root/f'terminal_evidence_seed{seed}.zip';assert not archive.exists()
    files={}
    def add(name,path):
        path=Path(path)
        if path.is_file():
            assert name not in files and not Path(name).is_absolute() and '..' not in Path(name).parts
            files[name]=path
    def tree(label,directory):
        for p in sorted(Path(directory).rglob('*')):
            if '__pycache__' not in p.parts and p.is_file():add(label+'/'+p.relative_to(directory).as_posix(),p)
    tree('code',code);tree('prepared',bundle)
    tree('baseline_eval',screen/'baseline_eval')  # Explicit traversal also follows the v4 baseline-cache root.
    for n in ('controller_start.json','controller_completion.json','controller_failure.json','candidate_start.json',
              'candidate_completion.json','baseline_start.json','baseline_completion.json','baseline_reuse.json',
              'candidate.log','baseline.log','controller_live.json','one_shot_launch.lock'):
        add('screen/'+n,screen/n)
    for p in sorted(run.iterdir()):
        if p.is_file() and (p.suffix in ('.json','.jsonl','.txt','.yml','.yaml') or
                           p.name in ('last.pth','parameters_actual_step8.pth','failed_epoch1_parameters.pth')):
            add('run/'+p.name,p)
    for n in ('endpoint_epoch12','stages_epoch12'):
        if (run/n).exists():tree('run/'+n,run/n)
    if (screen/'candidate_reload_v1').exists():tree('candidate_reload_v1',screen/'candidate_reload_v1')
    source=read(code/'source_binding.json')['native_files_sha256']
    for n,h in source.items():
        p=paths['native_root']/n;assert sha(p)==h;add('native/'+n,p)
    for n,h in receipt['identity']['evaluator_binding']['files'].items():
        p=paths['evaluator_root']/n;assert sha(p)==h;add('evaluator/pycocoeval/'+n,p)
    for n,key in [('baseline_epoch12.pth','baseline_checkpoint'),('baseline_first100.jsonl','baseline_first100'),
                  ('train_annotations.json','train_ann'),('val_annotations.json','val_ann')]:
        add('reference/'+n,paths[key])
    add('reference/baseline_log.txt',paths['baseline_first100'].with_name('log.txt'))
    add('tools/collect_terminal.py',Path(__file__))
    for p in sorted(Path(__file__).parent.glob('*')):
        if p.name!='collect_terminal.py' and p.suffix in ('.py','.json'):add('tools/'+p.name,p)
    manifest={'status':state,'seed':seed,'runtime_version':version,
        'captured_remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'identity':receipt['identity'],'files':{},'new_model_forwards':0,'new_optimizer_updates':0,
        'common_initialization_reused_not_duplicated':{
            'sha256':receipt['identity']['common_init_sha256'],
            'local_relative_path':'数据/03_本轮实验记录_20260928/权重/common_init_seed0.pth',
            'bytes':41486170},
        'checkpoint_policy':'Final or failed-run last.pth plus diagnostic step8 tensors included. All earlier/best checkpoints remain on server; final decisions use epoch12 only.',
        'dataset_policy':'Complete train/val annotations and7019-image manifest included; image bytes remain at original dataset locations.'}
    with zipfile.ZipFile(archive,'x',allowZip64=True) as z:
        for name,p in sorted(files.items()):
            before=p.stat();h=hashlib.sha256();info=zipfile.ZipInfo(name)
            info.compress_type=zipfile.ZIP_STORED if p.suffix in ('.pth','.npz','.gz') else zipfile.ZIP_DEFLATED
            with p.open('rb') as src,z.open(info,'w',force_zip64=True) as dest:
                for b in iter(lambda:src.read(8*1024*1024),b''):h.update(b);dest.write(b)
            after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),name
            manifest['files'][name]={'bytes':before.st_size,'sha256':h.hexdigest(),'source':str(p)}
        z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    metadata={'status':state,'archive':str(archive),'bytes':archive.stat().st_size,
              'sha256':sha(archive),'files':len(files)+1,'seed':seed}
    with (root/f'terminal_evidence_seed{seed}_metadata.json').open('x') as f:json.dump(metadata,f,indent=2)
    return metadata


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,choices=(1,2),required=True)
    p.add_argument('--check-only',action='store_true');a=p.parse_args();print(json.dumps(collect(a.seed,a.check_only)),flush=True)
