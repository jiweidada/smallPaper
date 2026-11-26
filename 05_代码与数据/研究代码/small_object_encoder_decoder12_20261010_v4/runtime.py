"""Frozen source/initialization bindings. Import never starts training."""
from pathlib import Path
from copy import deepcopy
import hashlib,json,re,sys
HERE=Path(__file__).resolve().parent
COMMON_INIT_SHA='85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91'
IMPLEMENTATION=('reuse_baseline.py','runtime.py','experiment.py','baseline_eval.py','supervise.py','training_audit.py','results.py','stage_trace.py','small_object_model.py','geometry_memory.py','evaluator_binding.py','evaluator_sources.json','source_binding.json','protocol.json')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def implementation_hashes():return {n:sha(HERE/n) for n in IMPLEMENTATION}
def verify_source(root):
    root=Path(root);expected=read(HERE/'source_binding.json')['native_files_sha256']
    actual={p.relative_to(root).as_posix():sha(p) for p in (root/'engine').rglob('*.py')}
    assert len(actual)==468 and actual==expected,'Native source bytes changed'
    return expected
def activate(root):
    verify_source(root)
    if 'engine' in sys.modules:raise RuntimeError('Fresh interpreter required')
    sys.path.insert(0,str(Path(root).resolve()))
def register_model():
    from engine.core import register
    from small_object_model import SmallObjectDEIM
    register()(SmallObjectDEIM)
def load_common(model,path):
    import torch
    from small_object_model import load_native_initialization
    assert sha(path)==COMMON_INIT_SHA
    payload=torch.load(path,map_location='cpu',weights_only=False)
    state=payload['ema']['module'] if 'ema' in payload else payload['model']
    receipt=load_native_initialization(model,state)
    for name,module in model.named_modules():
        if name.endswith('geometry_proposal') or name.startswith('decoder.decoder.geometry_attention.') and name.count('.')==3:
            assert torch.count_nonzero(module.output.weight).item()==0 and torch.count_nonzero(module.output.bias).item()==0
    return dict(common_init_sha256=COMMON_INIT_SHA,**receipt)
def resolved_config(plan,paths,baseline=False):
    cfg=deepcopy(plan['native_baseline_config']);cfg.pop('__include__',None);cfg.pop('config',None)
    cfg.update(resume=None,tuning=str(paths['common_init']),output_dir=str(paths['output']),device='cuda:0',use_swanlab=False)
    if not baseline:
        cfg['model']='SmallObjectDEIM';cfg['SmallObjectDEIM']=dict(recipe=deepcopy(plan['model_recipe']),variant=plan['variant'],initialization_seed=713)
        # Preserve every native parameter's group; added GN sequential names do
        # not contain norm/bn, so explicitly include those6 learned tensors.
        old='^(?=.*(?:norm|bn)).*$'
        new=old+'|^encoder\\.geometry_field\\.(?:fine_projection|context_projection|local_geometry)\\.1\\.(?:weight|bias)$'
        groups=cfg['optimizer']['params'];assert len(groups)==2 and groups[1]['params']==old
        groups[1]['params']=new
    else:cfg['model']='DEIM';cfg['tuning']=None
    for split in ('train','val'):
        cfg[split+'_dataloader']['dataset']['img_folder']=str(paths[split+'_images'])
        cfg[split+'_dataloader']['dataset']['ann_file']=str(paths[split+'_ann'])
    assert cfg['epoches']==132 and cfg['flat_epoch']==64 and cfg['no_aug_epoch']==12 and cfg['train_dataloader']['total_batch_size']==4
    assert cfg['train_dataloader']['dataset']['transforms']['policy']['epoch']==[4,64,120]
    assert cfg['use_amp'] and cfg['use_ema'] and not cfg.get('use_amp_fallback_fp32',False)
    return cfg
def verify_bundle(bundle):
    from evaluator_binding import verify_files
    bundle=Path(bundle);receipt=read(bundle/'receipt.json');plan=read(bundle/'plan.json');paths={k:Path(v) for k,v in receipt['paths'].items()}
    assert sha(bundle/'plan.json')==receipt['identity']['plan_sha256']
    assert sha(bundle/'plan.json')==sha(HERE/f"configs/{plan['variant']}_seed{plan['seed']}.json")
    assert sha(bundle/'image_manifest.json')==receipt['identity']['image_manifest_sha256']
    assert receipt['identity']['implementation']==implementation_hashes()
    assert sha(bundle/'resolved.yml')==receipt['resolved_sha256']
    assert read(bundle/'resolved.yml')==resolved_config(plan,paths)
    assert receipt['identity']['arm']==plan['variant'] and plan['variant'] in ('fine','degraded') and plan['seed'] in (1,2)
    assert sha(HERE/'protocol.json')==receipt['identity']['protocol_sha256']
    verify_source(paths['native_root']);verify_files(paths['evaluator_root'],receipt['identity']['evaluator_binding'])
    for k in ('train_ann','val_ann','common_init','baseline_checkpoint','baseline_first100'):
        assert sha(paths[k])==receipt['identity'][k+'_sha256'],k
    assert sha(paths['common_init'])==COMMON_INIT_SHA
    return receipt,plan,paths
def optimizer_coverage(model,optimizer,cfg):
    expected={id(p):n for n,p in model.named_parameters() if p.requires_grad}
    actual=[id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(actual)==len(set(actual)) and set(actual)==set(expected)
    native_cfg=deepcopy(cfg);native_cfg['optimizer']['params'][1]['params']='^(?=.*(?:norm|bn)).*$'
    original=native_cfg['optimizer'];group_by_param={id(p):g for g in optimizer.param_groups for p in g['params']}
    from small_object_model import EXTRA_PREFIXES
    checked=0;gn=[]
    for n,p in model.named_parameters():
        if not p.requires_grad:continue
        group=group_by_param[id(p)]
        if not n.startswith(EXTRA_PREFIXES):
            expected_lr=original['lr'];expected_wd=original['weight_decay']
            for spec in original['params']:
                if re.search(spec['params'],n):expected_lr=spec.get('lr',expected_lr);expected_wd=spec.get('weight_decay',expected_wd)
            assert group.get('initial_lr',group['lr'])==expected_lr and group['weight_decay']==expected_wd,n;checked+=1
        if re.search('^encoder\\.geometry_field\\.(?:fine_projection|context_projection|local_geometry)\\.1\\.(?:weight|bias)$',n):
            assert group['weight_decay']==0;gn.append(n)
    assert len(gn)==6
    return {'trainable_tensors':len(actual),'native_tensor_groups_unchanged':checked,'added_GroupNorm_zero_decay':gn,'all_parameters_exactly_once':True}
