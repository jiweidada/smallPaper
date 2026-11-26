"""CPU engineering checks of retained initial weights, zero contribution and gradients."""
from pathlib import Path
import argparse,copy,hashlib,json,os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False)

def run(prefix,seed):
    assert prefix=={1:'/25085400120',2:'/24085403014'}[seed]
    protocol=read(HERE/'protocol.json')
    assert {p.name:sha(p) for p in HERE.iterdir() if p.is_file() and p.name!='protocol.json'}==protocol['files_sha256']
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    original=Path(prefix)/'DEIM/experiments/original_stem12_20261011_v1'
    sys.path.append(str(original/'code'))
    from runtime import verify_bundle,activate
    receipt,plan,paths=verify_bundle(original/f'seed{seed}_original_stem/bundle')
    activate(paths['native_root'])
    from geometry import ResidualGeometryDEIM
    from warm_stem_model import WarmStartEnhancedDEIM
    from initialization import load_warm_common
    recipe=plan['historical_resolved_config']['ResidualGeometryDEIM']['recipe']
    assert recipe['HGNetv2']['pretrained'] is False and recipe['HGNetv2']['use_lab'] is True
    assert plan['seed']==seed
    # Constructors only: neither detector is forwarded.
    torch.manual_seed(seed)
    native_model=ResidualGeometryDEIM(recipe,arm='baseline')
    native_rng=torch.get_rng_state().clone()
    torch.manual_seed(seed)
    model=WarmStartEnhancedDEIM(recipe,arm='baseline')
    assert torch.equal(native_rng,torch.get_rng_state())
    provenance=load_warm_common(model,paths['common_init'])
    payload=torch.load(paths['common_init'],map_location='cpu',weights_only=False)
    common=payload['ema']['module'] if 'ema' in payload else payload['model']
    native_model.load_state_dict(common,strict=True)
    for n,v in native_model.state_dict().items():
        mapped='backbone.stem.native.'+n[len('backbone.stem.'):] if n.startswith('backbone.stem.') else n
        assert torch.equal(v,model.state_dict()[mapped]),n
    assert sum(p.numel() for p in native_model.parameters())==10231997
    assert sum(p.numel() for p in model.parameters())==10243812
    assert sum(p.numel() for p in model.backbone.stem.parameters())==18289
    assert sum(p.numel() for p in model.backbone.stem.enhanced.parameters())==11799
    rng=torch.get_rng_state().clone()
    from original_stem import EnhancedStemBlock
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        same=EnhancedStemBlock(3,16,16,use_lab=True,act_type='relu')
    assert all(torch.equal(v,same.state_dict()[n]) for n,v in model.backbone.stem.enhanced.state_dict().items())
    assert torch.equal(rng,torch.get_rng_state())
    counters=dict(prototype_detector=0,native_detector=0,warm_stem=0,retained_native_stem=0,
                  standalone_native_stem=0,enhanced_stem=0)
    def hook(name):
        def count(module,args):counters[name]+=1
        return count
    handles=[model.register_forward_pre_hook(hook('prototype_detector')),
       native_model.register_forward_pre_hook(hook('native_detector')),
       model.backbone.stem.register_forward_pre_hook(hook('warm_stem')),
       model.backbone.stem.native.register_forward_pre_hook(hook('retained_native_stem')),
       model.backbone.stem.enhanced.register_forward_pre_hook(hook('enhanced_stem')),
       native_model.backbone.stem.register_forward_pre_hook(hook('standalone_native_stem'))]
    stem=model.backbone.stem.eval();reference=native_model.backbone.stem.eval();cases=[]
    for side,autocast in ((64,False),(640,False),(832,False),(64,True)):
        x=torch.linspace(-1,1,3*side*side).reshape(1,3,side,side)
        with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16,enabled=autocast):
            expected=reference(x);actual=stem(x)
        assert actual.shape==expected.shape==(1,16,side//4,side//4)
        assert torch.isfinite(actual).all() and torch.equal(actual,expected)
        cases.append(dict(side=side,CPU_bfloat16_autocast=autocast,output_dtype=str(actual.dtype),
                          shape=list(actual.shape),zero_gain_output_exact=True))
    # Training mode at zero gain preserves native output, BN updates and native gradients.
    stem.train();reference.train();stem.zero_grad(set_to_none=True);reference.zero_grad(set_to_none=True)
    x=torch.linspace(-1,1,3*64*64).reshape(1,3,64,64)
    expected=reference(x);actual=stem(x);assert torch.equal(actual,expected)
    expected.square().mean().backward();actual.square().mean().backward()
    for n,parameter in reference.named_parameters():
        other=dict(stem.native.named_parameters())[n]
        assert parameter.grad is not None and other.grad is not None and torch.equal(parameter.grad,other.grad),n
    assert all(torch.equal(v,reference.state_dict()[n]) for n,v in stem.native.state_dict().items())
    assert stem.residual_gain.grad is not None and torch.isfinite(stem.residual_gain.grad).all()
    gain_nonzero=int((stem.residual_gain.grad!=0).sum())
    assert gain_nonzero>0
    assert all(p.grad is not None and torch.equal(p.grad,torch.zeros_like(p.grad)) for p in stem.enhanced.parameters())
    zero_gain_grad_norm=float(stem.residual_gain.grad.norm())
    # Graph probe only; this explicit assignment is not an optimizer update.
    stem.zero_grad(set_to_none=True)
    with torch.no_grad():stem.residual_gain.fill_(0.1)
    y=stem(x);assert y.shape==actual.shape and torch.isfinite(y).all()
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in stem.parameters())
    active=int(sum(bool((p.grad!=0).any()) for p in stem.enhanced.parameters()))
    assert active>0
    for h in handles:h.remove()
    assert counters['prototype_detector']==counters['native_detector']==0 and not torch.cuda.is_initialized()
    out=HERE.parent/f'seed{seed}_CPU_validation';out.mkdir(exist_ok=False)
    result=dict(status='PASS_CPU_ORIGINAL794_TENSORS_RNG_ZERO_GAIN_OUTPUT_NATIVE_GRADIENTS_AND_ACTIVE_RESIDUAL_GRAPH',
      seed=seed,provenance=provenance,parameters=dict(native_total=10231997,prototype_total=10243812,
        retained_native_Stem=6474,enhanced_branch=11799,gain=16,added_total=11815),
      cases=cases,zero_gain_native_training_output_exact=True,zero_gain_native_BN_updates_exact=True,
      zero_gain_native_gradients_exact=True,zero_gain_enhanced_gradients_all_zero=True,
      zero_gain_nonzero_gate_gradient_channels=gain_nonzero,zero_gain_gate_gradient_norm=zero_gain_grad_norm,
      manually_nonzero_gate_enhanced_parameter_tensors_with_nonzero_gradient=active,
      detector_constructor_CPU_RNG_exact=True,enhanced_seed_initialization_exact=True,
      actual_forward_counts=counters,optimizer_steps=0,training_epochs=0,evaluator_calls=0,CUDA_initialized=False,
      protocol_sha256=sha(HERE/'protocol.json'),files_sha256=protocol['files_sha256'],
      full_detector_forward_validated=False,CUDA_AMP_training_preflight_passed=False,
      initial_full_detector_predictions_or_AP_verified=False,actual_12epoch_training_launched=False,
      causality_or_novelty_proven=False,
      scope='CPU engineering checks only on actual native detector constructors/common initial checkpoint and standalone Stems with synthetic inputs. Full detector predictions, native validation and AP effects are untested. Zero contribution initially blocks enhanced branch gradients; the gate has gradients, then the branch graph is active after a nonzero probe. Extra gate gradients can alter global gradient clipping, so identical native pre-clipping gradients do not imply identical optimizer trajectories. No training job, baseline detector inference or evaluator is run.')
    write(out/'report.json',result)
    print(json.dumps(result))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--prefix',required=True);a.add_argument('--seed',type=int,required=True)
    x=a.parse_args();run(x.prefix,x.seed)

