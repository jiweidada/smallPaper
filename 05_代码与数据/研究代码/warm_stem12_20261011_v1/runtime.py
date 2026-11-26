"""Retained native Stem plus zero-initialized user residual; historical baseline evidence."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import re
import sys

HERE = Path(__file__).resolve().parent
COMMON_INIT_SHA = '85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91'
IMPLEMENTATION = ('runtime.py', 'prepare_training.py', 'experiment.py', 'supervise.py', 'reuse_baseline.py', 'training_audit.py', 'geometry.py', 'residual.py', 'original_stem.py', 'warm_stem.py', 'warm_stem_model.py', 'initialization.py', 'original_stem_source.json', 'original_backbone_source.txt', 'results.py', 'stage_trace.py', 'evaluator_binding.py', 'evaluator_sources.json', 'source_binding.json', 'training_protocol.json', 'verify_candidate_reload.py', 'collect_terminal.py', 'finish.py', 'closeout_protocol.json', 'preflight.py', 'dispatch.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def implementation_hashes():
    return {name:sha(HERE/name) for name in IMPLEMENTATION}


def verify_source(root):
    root=Path(root); binding=read(HERE/'source_binding.json')
    actual={p.relative_to(root).as_posix():sha(p) for p in (root/'engine').rglob('*.py')}
    assert len(actual)==468 and actual==binding['native_files_sha256']
    assert all(sha(HERE/name)==digest for name,digest in binding['legacy_constructor_files'].items())
    return actual


def activate(root):
    verify_source(root)
    assert 'engine' not in sys.modules, 'Fresh process required'
    sys.path.insert(0,str(Path(root).resolve()))


def register_model():
    from engine.core import register
    from warm_stem_model import WarmStartEnhancedDEIM
    register()(WarmStartEnhancedDEIM)


def load_common(model, path):
    from initialization import load_warm_common
    assert model.geometry_arm=='baseline'
    proof=load_warm_common(model,path)
    protocol=read(HERE/'training_protocol.json')
    count=sum(p.numel() for p in model.parameters())
    assert count==protocol['expected_parameters']
    assert model.new_stem_parameter_count-model.old_stem_parameter_count==protocol['expected_extra_parameters']
    return dict(proof,state_tensors=len(model.state_dict()),parameters=count,
        extra_parameters=count-10231997,old_stem_parameters=model.old_stem_parameter_count,
        new_stem_parameters=model.new_stem_parameter_count,
        new_stem_initialization='Original EnhancedStem defaults in seed-specific forked RNG;16 gain parameters zero')



def resolved_config(plan, paths):
    cfg=deepcopy(plan['historical_resolved_config'])
    assert cfg['model']=='ResidualGeometryDEIM' and cfg['ResidualGeometryDEIM']['arm']=='baseline'
    assert cfg['epoches']==132 and cfg['flat_epoch']==64 and cfg['no_aug_epoch']==12
    assert cfg['train_dataloader']['total_batch_size']==4
    assert cfg['train_dataloader']['dataset']['transforms']['policy']['epoch']==[4,64,120]
    assert cfg['seed']==plan['seed'] and not cfg.get('use_amp_fallback_fp32',False)
    monitor=cfg.get('baseline_log_monitor')
    assert cfg['use_amp'] and cfg['use_ema']
    assert monitor is None or (isinstance(monitor,dict) and monitor.get('enabled') is False)
    cfg['model']='WarmStartEnhancedDEIM'
    cfg['WarmStartEnhancedDEIM']=cfg.pop('ResidualGeometryDEIM')
    cfg.update(resume=None,tuning=str(paths['common_init']),output_dir=str(paths['output']),
               device='cuda:0',use_swanlab=False)
    for split in ('train','val'):
        cfg[split+'_dataloader']['dataset']['img_folder']=str(paths[split+'_images'])
        cfg[split+'_dataloader']['dataset']['ann_file']=str(paths[split+'_ann'])
    return cfg


def verify_bundle(bundle):
    from evaluator_binding import verify_files
    bundle=Path(bundle); receipt=read(bundle/'receipt.json'); plan=read(bundle/'plan.json')
    paths={k:Path(v) for k,v in receipt['paths'].items()}; ident=receipt['identity']
    protocol=read(HERE/'training_protocol.json')
    assert plan['schema']==protocol['schema']=='warm_stem12_training_v1'
    assert plan['arm']=='warm_stem' and plan['seed'] in protocol['execution_allowlist']['warm_stem']
    assert ident['arm']==plan['arm'] and ident['seed']==plan['seed']
    assert ident['protocol_sha256']==sha(HERE/'training_protocol.json')
    assert sha(bundle/'plan.json')==ident['plan_sha256']
    assert ident['implementation']==implementation_hashes()
    assert sha(HERE/'original_stem.py')==protocol['original_stem_source_sha256']
    assert sha(bundle/'resolved.yml')==receipt['resolved_sha256']
    assert read(bundle/'resolved.yml')==resolved_config(plan,paths)
    assert sha(paths['image_manifest'])==ident['image_manifest_sha256']
    assert sha(paths['image_verification'])==ident['image_verification_sha256']
    image_check=read(paths['image_verification'])
    assert image_check['files_verified']==7019 and image_check['image_manifest_sha256']==ident['image_manifest_sha256']
    for key in ('train_ann','val_ann','common_init','baseline_checkpoint','baseline_first100'):
        assert sha(paths[key])==ident[key+'_sha256'],key
    assert sha(paths['common_init'])==COMMON_INIT_SHA
    verify_source(paths['native_root']); verify_files(paths['evaluator_root'],ident['evaluator_binding'])
    return receipt,plan,paths


def optimizer_coverage(model, optimizer, cfg):
    expected={id(p):n for n,p in model.named_parameters() if p.requires_grad}
    actual=[id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(actual)==len(set(actual)) and set(actual)==set(expected)
    original=cfg['optimizer']; groups={id(p):g for g in optimizer.param_groups for p in g['params']}
    for name,param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr,wd=original['lr'],original['weight_decay']
        for spec in original['params']:
            if re.search(spec['params'],name):
                lr,wd=spec.get('lr',lr),spec.get('weight_decay',wd)
        assert groups[id(param)].get('initial_lr',groups[id(param)]['lr'])==lr
        assert groups[id(param)]['weight_decay']==wd,name
    return dict(trainable_tensors=len(actual),all_native_parameter_groups_unchanged=True,
                all_parameters_exactly_once=True,added_parameters=sum(p.numel() for p in model.parameters())-10231997)
