"""Native architecture/equivalence and scratch initialization check; zero updates."""
import argparse,json,torch
from pathlib import Path
from runtime import verify_bundle,activate,register_model,load_common,write
p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);a=p.parse_args()
receipt,plan,paths=verify_bundle(a.bundle)
activate(paths['native_root']);register_model()
from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.backbone.hgnetv2 import StemBlock as CanonicalNativeStem
dist_utils.setup_distributed(print_rank=0,print_method='builtin',seed=plan['seed'])
cfg=YAMLConfig(str(Path(a.bundle)/'resolved.yml'));model=cfg.model
init=load_common(model,paths['common_init'])
assert init['parameters']==10231997 and init['extra_parameters']==0
assert init['exact_shared_tensors']==754 and init['state_tensors']==794
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(plan['seed'])
    expected=CanonicalNativeStem(3,16,16,use_lab=True)
actual=model.backbone.stem
assert actual.state_dict().keys()==expected.state_dict().keys()
assert all(torch.equal(v,actual.state_dict()[k]) for k,v in expected.state_dict().items())
bn=[m for m in actual.modules() if isinstance(m,torch.nn.BatchNorm2d)]
assert len(bn)==5 and all(torch.equal(m.running_mean,torch.zeros_like(m.running_mean))
    and torch.equal(m.running_var,torch.ones_like(m.running_var)) and m.num_batches_tracked.item()==0 for m in bn)
actual.eval();expected.eval()
x=torch.linspace(-1,1,3*64*64).reshape(1,3,64,64)
with torch.no_grad():assert torch.equal(actual(x),expected(x))
stem=actual.cuda().train();shapes=[]
for side in (640,832):
    x=torch.linspace(-1,1,side*side,device='cuda').reshape(1,1,side,side).expand(4,3,-1,-1).contiguous()
    with torch.autocast(device_type='cuda',dtype=torch.float16):
        y=stem(x);loss=y.float().square().mean()
    assert tuple(y.shape)==(4,16,side//4,side//4) and torch.isfinite(y).all()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in stem.parameters())
    shapes.append(list(y.shape));stem.zero_grad(set_to_none=True)
result=dict(status='NATIVE_STEM_SCRATCH_SHAPE_INITIALIZATION_EQUIVALENCE_PASS',seed=plan['seed'],arm=plan['arm'],
    initialization=init,shapes=shapes,all40_initial_stem_tensors_match_seeded_canonical_native=True,
    native_stem_forward_exact=True,all5_BN_initial_statistics_default=True,
    all_stem_trainable_tensors_receive_finite_gradients=True,full_model_forwards=0,optimizer_updates=0,
    pretrained_BASE_forwards=0,peak_allocated_bytes=torch.cuda.max_memory_allocated())
write(paths['screen_root']/'preflight.json',result);print(json.dumps(result),flush=True)
