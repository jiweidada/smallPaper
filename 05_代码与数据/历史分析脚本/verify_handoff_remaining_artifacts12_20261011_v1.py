"""Validate the final downloads and all seven saved12 closeouts; no model run."""
from pathlib import Path, PurePosixPath
import json
import hashlib
import zipfile
import importlib.util
import datetime
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs/handoff_remaining_artifacts12_20261011_v1'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def safe_extract(archive,meta,destination,manifest_name,volatile_allowed=()):
    assert archive.stat().st_size==meta['bytes'] and sha(archive)==meta['sha256']
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        assert len(names)==len(set(names))==len({n.casefold() for n in names})
        if 'archive_entries' in meta:assert len(names)==meta['archive_entries']
        assert z.testzip() is None
        for info in z.infolist():
            n=info.filename;p=PurePosixPath(n)
            assert not info.is_dir() and not p.is_absolute() and '..' not in p.parts and '\\' not in n and ':' not in n
            assert (destination/n).resolve().is_relative_to(destination.resolve())
        raw=z.read(manifest_name)
        assert hashlib.sha256(raw).hexdigest()==meta['manifest_sha256']
        manifest=json.loads(raw)
        assert set(names)==set(manifest['files'])|{manifest_name}
        if destination.exists():
            assert set(names)=={p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()}
        else:
            destination.mkdir(parents=True)
            z.extractall(destination)
    mismatches=[]
    for n,item in manifest['files'].items():
        p=destination/n
        actual=dict(bytes=p.stat().st_size,sha256=sha(p))
        if actual!=item:
            assert n in volatile_allowed,n
            mismatches.append(dict(file=n,expected=item,actual=actual))
    return manifest,mismatches


helper=module('pure_endpoint_handoff_reader',ROOT/'英文小论文/code/p2_lowrank12_local_reader_20261011_v1/verify_terminal_local.py')
user=module('user_saved_EMA_handoff_proof',ROOT/'英文小论文/code/user_backbone_formal_verify12_20261011_v1/verify.py')
stem=module('component_saved_EMA_handoff_reader',ROOT/'英文小论文/code/stem_component12_monitor_20261011_v1/verify_terminal_local.py')
for seed in (1,2):
    capture=read(OUT/f'remote_collection_seed{seed}_v1.json')
    raw=capture['raw_payload'].encode()
    assert hashlib.sha256(raw).hexdigest()==capture['transport']['sha256'] and len(raw)==capture['transport']['bytes']
    assert json.loads(raw)==capture['payload']
    folder=OUT/f'seed{seed}'
    assert sha(folder/'collection_manifest.json')==capture['payload']['collection_manifest_sha256']
    manifest=read(folder/'collection_manifest.json')
    assert manifest==capture['payload']['manifest']
    assert set(manifest['files'])|{'collection_manifest.json'}=={p.name for p in folder.iterdir() if p.is_file()}
    for name,item in manifest['files'].items():
        p=folder/name
        assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],name

results=[]
for arm in ('detail_lowres','attention_spatial_mean','early_final_downsample'):
    folder=OUT/'seed1';name=f'terminal_{arm}_seed1_v1'
    dest=folder/'verified'/arm;report=folder/'verified'/f'{arm}_verification.json'
    if report.exists():
        value=read(report)
        assert value['archive_sha256']==sha(folder/(name+'.zip'))
    else:
        value=stem.verify(folder/(name+'.zip'),folder/(name+'_metadata.json'),dest,report)
    assert value['all548_stage_arrays_reloaded_exact']
    results.append(dict(seed=1,arm=arm,status=value['status'],metrics=value['candidate_overall_percent'],verification=str(report),exact548=True))
for arm in ('backbone_silu','backbone_attention'):
    folder=OUT/'seed1';name=f'terminal_{arm}_seed1_v1'
    dest=folder/'verified'/arm;report=folder/'verified'/f'{arm}_verification.json'
    if not report.exists():
        user.verify(1,arm,folder/(name+'.zip'),folder/(name+'_metadata.json'),dest,report)
    value=read(report)
    assert value['archive_sha256']==sha(folder/(name+'.zip'))
    results.append(dict(seed=1,arm=arm,status=value['status'],metrics=value.get('candidate_overall_percent',value.get('candidate_percent')),verification=str(report),exact548=True))

for seed in (1,2):
    folder=OUT/f'seed{seed}';name=f'terminal_warm_stem_seed{seed}_v1'
    meta=read(folder/(name+'_metadata.json'));destination=folder/'verified/warm_stem'
    # The original full-manifest failure is retained separately. Only this
    # known live dispatcher file can be classified as an archival warning;
    # every model, evaluator, source, data, and stage file remains hash-exact.
    manifest,volatile_mismatches=safe_extract(folder/(name+'.zip'),meta,destination,'terminal_manifest.json',
        volatile_allowed=('screen/dispatcher_live.json',) if seed==1 else ())
    original_archive_check=read(OUT/f'warm_seed{seed}_original_archive_manifest_failure.json')
    assert original_archive_check['mismatches']==volatile_mismatches
    assert manifest['schema']=='warm_stem12_terminal_v1' and manifest['training_epochs']==12 and not manifest['automatic_retry']
    identity=manifest['identity'];assert identity['seed']==seed
    local_code=ROOT/'英文小论文/code/warm_stem12_20261011_v1'
    for name,item in manifest['files'].items():
        if name.startswith('code/'):
            assert sha(local_code/name[5:])==item['sha256']
    receipt=read(destination/'bundle/receipt.json')
    assert receipt['identity']==identity
    runtime=read(destination/'run/runtime_start.json')
    done=read(destination/'run/training_completion.json')
    proof=read(destination/'candidate_reload_v1/verification.json')
    closeout=read(destination/'code/closeout_protocol.json')
    replay_exit=read(destination/'candidate_reload_v1/completion.json')
    user.assert_replay(proof,replay_exit,identity,done,runtime['initialization'],closeout)
    assert sha(destination/'run/last.pth')==done['last_checkpoint_sha256']
    annotations=read(ROOT/f'outputs/scale_protection12_20261010_v1/terminal_collection/seed{seed}_target_protect_v1/extracted/reference/val_annotations.json')
    a,ah,ac=helper.native_endpoint(destination/'run/endpoint_epoch12',annotations)
    b,bh,bc=helper.native_endpoint(destination/'candidate_reload_v1/endpoint_epoch12',annotations)
    assert a['native_stats']==b['native_stats'] and np.array_equal(ah['hits'],bh['hits'])
    helper.stages(destination/'run',identity,annotations)
    helper.stages(destination/'candidate_reload_v1',identity,annotations)
    for i in range(548):
        name=f'image_{i:06d}.npz'
        with np.load(destination/'run/stages_epoch12'/name,allow_pickle=False) as x,np.load(destination/'candidate_reload_v1/stages_epoch12'/name,allow_pickle=False) as y:
            assert set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
    report=folder/'verified/warm_stem_verification.json'
    value=dict(status='IMMUTABLE_WARM_STEM_MODEL_DATA_AND_EXACT548_REPLAY_PASS_WITH_ORIGINAL_ARCHIVE_WARNING' if volatile_mismatches else 'PASS_DOWNLOADED_WARM_STEM_FIXED12_HASHES_NATIVE_PR_AND_EXACT548_SAVED_EMA_REPLAY',seed=seed,identity=identity,
        candidate_overall_percent=a['overall_percent'],all548_stage_arrays_reloaded_exact=True,new_training_epochs=0,new_model_forwards=0,new_evaluator_calls=0,
        checkpoint_sha256=done['last_checkpoint_sha256'],files_verified=len(manifest['files'])+1-len(volatile_mismatches),
        original_archive_full_manifest_passed=not bool(volatile_mismatches),volatile_archive_manifest_mismatches=volatile_mismatches,
        all_model_and_stage_file_hashes_exact=True,original_archive_changed=False,
        claim_boundary='Retained pretrained native Stem path plus random additional branch; do not attribute its raw difference to random native Stem alone.')
    report.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    results.append(dict(seed=seed,arm='warm_stem',status=value['status'],metrics=value['candidate_overall_percent'],verification=str(report),exact548=True))

folder=OUT/'seed1';name='direction_seed1_saved_artifacts_v1'
collection=read(folder/'collection_manifest.json')
meta=dict(collection['files'][name+'.zip'],manifest_sha256=sha(folder/(name+'_manifest.json')))
dest=folder/'verified/direction'
manifest,mismatches=safe_extract(folder/(name+'.zip'),meta,dest,'manifest.json')
assert not mismatches
direction=[]
for mode in ('dc16_norm','svd16_norm'):
    annotations=read(ROOT/'outputs/scale_protection12_20261010_v1/terminal_collection/seed1_target_protect_v1/extracted/reference/val_annotations.json')
    summary,flags,checks=helper.native_endpoint(dest/'GPU'/mode/'endpoint_epoch12',annotations)
    helper.stages(dest/'GPU'/mode,summary['identity'],annotations)
    direction.append(dict(mode=mode,metrics=summary['overall_percent'],checks=checks,stage_images=548,independent_repeat_performed=False))
assert 'torch' not in sys.modules
report=dict(status='ALL_SEVEN_TERMINAL_MODEL_DATA_AND_A_DIRECTION_DATA_LOCAL_VERIFIED_ORIGINAL_WARM_ARCHIVE_WARNING_RETAINED',
    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),terminal_results=results,direction_results=direction,
    original_P2_seed1_failed_replay_status_preserved=True,original_warm_seed1_full_manifest_failure_preserved=True,
    new_training_epochs=0,new_model_forwards=0,new_evaluator_calls=0,
    root_proven=False,novelty_proven=False)
(OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'terminal_packets':len(results),'direction_modes':len(direction)},ensure_ascii=False))
