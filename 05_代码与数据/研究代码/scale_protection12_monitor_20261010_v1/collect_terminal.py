"""Collect stopped scale-protection screens and immutable saved-weight proofs."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from verify_candidate_reload import verify_tools,same_live,assert_trainers_stopped


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()


def terminal_state(screen):
    for name,pk,tk in [('candidate_start.json','child_pid','child_start_ticks'),
                       ('controller_start.json','pid','start_ticks')]:
        if (screen/name).exists() and same_live(read(screen/name),pk,tk): return 'OWNED_TRAINING_RUNNING'
    if (screen/'controller_failure.json').exists(): return 'FAILED_NO_FIXED12_ACCEPTANCE'
    if not (screen/'controller_completion.json').exists(): return 'TERMINAL_NOT_PROVEN'
    assert read(screen/'controller_completion.json')['status'] == 'COMPLETE_FIXED12'
    replay = screen/'candidate_reload_v1'
    if (replay/'start.json').exists():
        start = read(replay/'start.json')
        if same_live(start,'pid','start_ticks') or same_live(start,'controller_pid','controller_start_ticks'):
            return 'OWNED_RELOAD_RUNNING'
    if not (replay/'completion.json').exists(): return 'FIXED12_RELOAD_PENDING'
    if read(replay/'completion.json')['exit_code'] != 0 or not (replay/'verification.json').exists():
        return 'FIXED12_RELOAD_FAILED'
    assert read(replay/'verification.json')['status'] == 'PASS_SAVED_CANDIDATE_RELOAD'
    return 'FIXED12_AND_RELOAD_COMPLETE'


def collect(bundle, check_only=False):
    verify_tools()
    screen = bundle.parent; root = screen.parent; code = root/'code_v2'
    state = terminal_state(screen)
    if check_only: return dict(state=state,archive_created=False,model_forwards=0)
    assert state in ('FAILED_NO_FIXED12_ACCEPTANCE','FIXED12_RELOAD_FAILED','FIXED12_AND_RELOAD_COMPLETE'),state
    assert_trainers_stopped(screen)
    sys.path.insert(0,str(code))
    from runtime import verify_bundle
    receipt,plan,paths = verify_bundle(bundle)
    ident = receipt['identity']; run = paths['output']; seed,arm = plan['seed'],plan['arm']
    assert screen == paths['screen_root'] and root == paths['experiment_root']
    if state != 'FAILED_NO_FIXED12_ACCEPTANCE':
        done = read(run/'training_completion.json')
        assert done['identity'] == ident and done['completed_epochs'] == 12
        assert done['entered_epochs'] == list(range(12))
        assert sha(run/'last.pth') == done['last_checkpoint_sha256']
        for sub,key in [('endpoint_epoch12','endpoint_manifest_sha256'),('stages_epoch12','stage_manifest_sha256')]:
            assert sha(run/sub/'manifest.json') == done[key]
    archive = root/f'terminal_{arm}_seed{seed}_v1.zip'
    assert not archive.exists()
    files = {}
    def add(name,path):
        path = Path(path)
        if path.is_file():
            assert name not in files and not Path(name).is_absolute() and '..' not in Path(name).parts
            files[name] = path
    def tree(label,path):
        for p in sorted(Path(path).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                add(label+'/'+p.relative_to(path).as_posix(),p)
    tree('code',code); tree('prepared',bundle)
    for name in ('protocol.json','training_profile.json','training_deployment_manifest.json',
                 'training_deployment_manifest_v2.json',f'image_manifest_seed{seed}.json',
                 f'image_verification_seed{seed}.json','dual_seed_native_promotion.json'):
        add('deployment/'+name,root/name)
    for suffix in ('.log','_completion.json'):
        for tag in ('','_v2'): add('deployment/'+f'prepare_seed{seed}{tag}{suffix}',root/f'prepare_seed{seed}{tag}{suffix}')
    add('deployment/data_probe.json',paths['data_only_report'])
    reuse = read(screen/'baseline_reuse.json')
    assert reuse['new_model_forwards'] == reuse['new_optimizer_steps'] == reuse['new_epochs'] == 0
    tree('baseline_eval',screen/'baseline_eval')
    for name in ('launch_receipt.json','controller_start.json','controller_completion.json','controller_failure.json',
                 'candidate_start.json','candidate_completion.json','controller_live.json','one_shot_launch.lock',
                 'baseline_reuse.json','candidate.log','supervisor.log'):
        add('screen/'+name,screen/name)
    if run.exists():
        for p in sorted(run.iterdir()):
            if p.is_file() and (p.suffix in ('.json','.jsonl','.txt','.yaml','.yml') or p.name == 'last.pth'):
                add('run/'+p.name,p)
        for name in ('endpoint_epoch12','stages_epoch12','augmentation_events'):
            if (run/name).exists(): tree('run/'+name,run/name)
    if (screen/'candidate_reload_v1').exists(): tree('candidate_reload_v1',screen/'candidate_reload_v1')
    native = read(code/'source_binding.json')['native_files_sha256']
    for name,digest in native.items():
        p = paths['native_root']/name; assert sha(p) == digest; add('native/'+name,p)
    for name,digest in ident['evaluator_binding']['files'].items():
        p = paths['evaluator_root']/name; assert sha(p) == digest; add('evaluator/pycocoeval/'+name,p)
    for name,key in [('baseline_epoch12.pth','baseline_checkpoint'),('baseline_first100.jsonl','baseline_first100'),
                     ('train_annotations.json','train_ann'),('val_annotations.json','val_ann')]:
        add('reference/'+name,paths[key])
    add('reference/baseline_log.txt',paths['baseline_first100'].with_name('log.txt'))
    for p in Path(__file__).parent.iterdir():
        if p.is_file() and p.name in (*read(Path(__file__).parent/'closeout_protocol.json')['files'],'closeout_protocol.json'):
            add('tools/'+p.name,p)
    manifest = dict(schema='scale_protection12_terminal_v1',status=state,seed=seed,arm=arm,identity=ident,
        captured_remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files={},
        collection_model_forwards=0,collection_optimizer_updates=0,
        common_initialization_reused_not_duplicated=dict(sha256=ident['common_init_sha256'],
            local_relative_path='数据/03_本轮实验记录_20260928/权重/common_init_seed0.pth',bytes=41486170),
        checkpoint_policy='Final last.pth retained; other checkpoints stay on server. Fixed12 EMA decisions only.',
        dataset_policy='Complete annotations and7019 byte manifest included; original image bytes stay in original data paths.',
        scope='Terminal evidence only. Failed or incomplete packets cannot promote a configuration.')
    with zipfile.ZipFile(archive,'x',allowZip64=True) as z:
        for name,p in sorted(files.items()):
            before = p.stat(); h = hashlib.sha256(); info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_STORED if p.suffix in ('.pth','.npz','.gz') else zipfile.ZIP_DEFLATED
            with p.open('rb') as src,z.open(info,'w',force_zip64=True) as dst:
                for b in iter(lambda:src.read(8*1024**2),b''): h.update(b); dst.write(b)
            after = p.stat()
            assert (before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns),name
            manifest['files'][name] = dict(bytes=before.st_size,sha256=h.hexdigest(),source=str(p))
        z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    meta = dict(status=state,seed=seed,arm=arm,archive=str(archive),bytes=archive.stat().st_size,
                sha256=sha(archive),files=len(files)+1)
    with (root/f'terminal_{arm}_seed{seed}_v1_metadata.json').open('x') as f:
        json.dump(meta,f,indent=2)
    return meta


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--bundle',required=True); p.add_argument('--check-only',action='store_true')
    a = p.parse_args(); print(json.dumps(collect(Path(a.bundle).resolve(),a.check_only)),flush=True)
