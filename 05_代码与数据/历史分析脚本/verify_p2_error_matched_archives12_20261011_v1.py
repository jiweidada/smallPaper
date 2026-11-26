"""Verify both existing altered-weight evaluation archives locally; no Torch."""
from pathlib import Path, PurePosixPath
import json
import hashlib
import zipfile
import importlib.util
import datetime
import sys

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs/p2_error_matched_runtime_local_reader12_20261011_v1'
assert not OUT.exists()
OUT.mkdir()
EXPECTED={1:'b03f5643e7f91675850d617fe52b68bed2e4141fd2f2db6d7c48948a64d16eb3',2:'8afa7b512f8f8acb731e3f440a341592c8f0d216f022f629e40560c56d3a68a4'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


helper_path=ROOT/'英文小论文/code/p2_lowrank12_local_reader_20261011_v1/verify_terminal_local.py'
assert sha(helper_path)=='1065525f27f2cf3c7909ce3fa20e510b9c6da334279dd9553708fb564c2c776c'
loader=importlib.util.spec_from_file_location('saved_native_endpoint_reader',helper_path)
helper=importlib.util.module_from_spec(loader)
loader.loader.exec_module(helper)
protocol_path=ROOT/'英文小论文/code/p2_error_matched_runtime12_20261011_v1/protocol.json'
assert sha(protocol_path)=='6c6887d62f634d4fa158a78b473eaa2a376b20a31833eab852d31db8dcacb279'
protocol=read(protocol_path)
records=[]
for seed in (1,2):
    parent=ROOT/'outputs/p2_error_matched_runtime_collection12_20261011_v1'/f'seed{seed}'
    archive=parent/f'seed{seed}_artifacts_v1.zip';metadata=parent/f'seed{seed}_archive_metadata.json'
    meta=read(metadata)
    assert meta['sha256']==EXPECTED[seed]==sha(archive) and archive.stat().st_size==meta['bytes']
    assert meta['seed']==seed and meta['archive_entries']==571 and meta['model_forwards']==137
    assert meta['BASE_forwards']==meta['parent_forwards']==meta['training_epochs']==0
    destination=parent/'extracted'
    assert not destination.exists()
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        assert len(names)==len(set(names))==571 and z.testzip() is None
        assert all(not i.is_dir() for i in z.infolist())
        for name in names:
            p=PurePosixPath(name)
            assert not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name
            assert (destination/name).resolve().is_relative_to(destination.resolve())
        manifest_bytes=z.read('manifest.json')
        assert hashlib.sha256(manifest_bytes).hexdigest()==meta['manifest_sha256']
        manifest=json.loads(manifest_bytes)
        assert set(names)==set(manifest['files'])|{'manifest.json'}
        assert manifest['new_BASE_forwards']==manifest['new_parent_forwards']==manifest['new_training_epochs']==0
        destination.mkdir()
        z.extractall(destination)
    for name,item in manifest['files'].items():
        p=destination/name
        assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],name
    assert sha(destination/'code/protocol.json')==sha(protocol_path)
    for name,h in protocol['remote_sources'].items():
        assert sha(destination/'code'/name)==h
    binding=protocol['bindings'][str(seed)]
    for name,h in binding['files'].items():
        filename='reference_manifest.json' if name=='reference_manifest' else name+'.json'
        assert sha(destination/'provenance'/filename)==h
    assert sha(destination/'provenance/weights.npz')==binding['weights_sha256']
    cpu=read(destination/'provenance/CPU_report.json')
    assert cpu['status']=='PASS_ACTUAL_CPU_RANK16_NORM_DC_ERROR_AND64_BASIS_OPERATOR_CHECK' and cpu['rank']==16
    report=read(destination/'evaluation/report.json')
    done=read(destination/'evaluation/completion.json')
    assert done['exit_code']==0 and done['automatic_retry'] is False and done['report_sha256']==sha(destination/'evaluation/report.json')
    assert report['status']=='ONE_DC_ERROR_MATCHED_RANK16_WEIGHT_NATIVE_EVALUATED'
    assert report['seed']==seed and report['identity']==manifest['identity']
    assert report['mode']==protocol['mode']=='svd16_dc_error_norm'
    assert report['changed_state_tensors']==['backbone.p2_skip.restore.weight'] and report['unchanged_state_tensors']==800
    assert report['parameters']==10249437 and report['candidate_model_forwards']==137
    assert report['new_native_evaluator_runs']==1 and report['new_BASE_forwards']==report['new_parent_forwards']==report['new_training_epochs']==report['new_optimizer_updates']==0
    assert report['checkpoint_unchanged'] is True and report['parent_replay_status_unchanged'] is True
    assert report['parent_replay_status']==binding['reference_replay_status']
    assert report['identity']['runtime_intervention']['parent_checkpoint_sha256']==binding['checkpoint_sha256']
    annotation=ROOT/f'outputs/scale_protection12_20261010_v1/terminal_collection/seed{seed}_target_protect_v1/extracted/reference/val_annotations.json'
    assert sha(annotation)==report['identity']['val_annotation_sha256']
    annotations=read(annotation)
    summary,flags,checks=helper.native_endpoint(destination/'evaluation/endpoint_epoch12',annotations)
    assert summary['identity']==report['identity'] and summary['overall_percent']==report['overall_percent']
    assert summary['categories_percent']==report['categories_percent'] and summary['native_stats']==report['native_stats']
    helper.stages(destination/'evaluation',report['identity'],annotations)
    for name,comparison in report['saved_control_comparisons'].items():
        assert comparison['status']=='SAVED_NATIVE_REFERENCE_BOUND'
        for key,value in comparison['delta_points'].items():
            assert value==summary['overall_percent'][key]-comparison['overall_percent'][key]
    result=dict(seed=seed,status='PASS_LOCAL_ARCHIVE_BINDINGS_NATIVE_PR_GT_FLAGS_AND548_STAGES',archive_sha256=sha(archive),files_verified=571,
        native_checks=checks,stage_images_verified=548,overall_percent=summary['overall_percent'],saved_control_comparisons=report['saved_control_comparisons'],
        original_parent_replay_status=report['parent_replay_status'],original_parent_replay_failure_relabelled=False,
        new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0,archive_evaluation_has_independent_repeat=False,root_proven=False,novelty_proven=False)
    (parent/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    records.append(result)
assert 'torch' not in sys.modules
final=dict(status='BOTH_ERROR_MATCHED_DIAGNOSTIC_ARCHIVES_VERIFIED_LOCALLY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),records=records,
    interpretation='Equal rank, parent norm and DC-matched global coefficient Frobenius error still produce substantially higher APs than DC16 in both seeds. The scalar coefficient approximation error does not determine task quality. Input covariance and direction-specific task responses remain unmatched.',
    new_training_epochs=0,new_model_forwards=0,new_evaluator_calls=0,root_proven=False,novelty_proven=False,training_efficacy_proven=False)
(OUT/'report.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':final['status'],'metrics':[{'seed':x['seed'],'overall':x['overall_percent']} for x in records]},ensure_ascii=False))
