"""Test original equivalence, matched initialization and finite controlled gradients; zero updates."""
import argparse,copy,json,torch
from pathlib import Path
from runtime import verify_bundle,activate,register_model,load_common,write
from original_stem import EnhancedStemBlock
from stem_controls import ControlledStemBlock
p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);a=p.parse_args()
receipt,plan,paths=verify_bundle(a.bundle)
activate(paths['native_root']);register_model()
from engine.core import YAMLConfig
from engine.misc import dist_utils
dist_utils.setup_distributed(print_rank=0,print_method='builtin',seed=plan['seed'])
cfg=YAMLConfig(str(Path(a.bundle)/'resolved.yml'));model=cfg.model
init=load_common(model,paths['common_init'])
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(plan['seed'])
    original=EnhancedStemBlock(3,16,16,use_lab=True,act_type='relu')
assert all(torch.equal(v,model.backbone.stem.state_dict()[k]) for k,v in original.state_dict().items())
# The duplicated forward must be exactly the original at the full setting.
full=ControlledStemBlock(3,16,16,use_lab=True,act_type='relu');full.set_control('full')
full.load_state_dict(original.state_dict());full.eval();original.eval()
x=torch.linspace(-1,1,3*64*64).reshape(1,3,64,64)
with torch.no_grad():
    assert torch.equal(full(x),original(x))
stem=model.backbone.stem.cuda().train();shapes=[]
for side in (640,832):
    x=torch.linspace(-1,1,side*side,device='cuda').reshape(1,1,side,side).expand(4,3,-1,-1).contiguous()
    with torch.autocast(device_type='cuda',dtype=torch.float16):
        y=stem(x);loss=y.float().square().mean()
    assert tuple(y.shape)==(4,16,side//4,side//4) and torch.isfinite(y).all()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in stem.parameters())
    shapes.append(list(y.shape));stem.zero_grad(set_to_none=True)
result=dict(status='CONTROL_SHAPE_GRADIENT_AND_INITIALIZATION_PASS',seed=plan['seed'],arm=plan['arm'],
    initialization=init,shapes=shapes,all_initial_stem_tensors_match_full=True,
    full_forward_exactly_original=True,all_stem_trainable_tensors_receive_finite_gradients=True,
    full_model_forwards=0,optimizer_updates=0,baseline_forwards=0,peak_allocated_bytes=torch.cuda.max_memory_allocated())
write(paths['screen_root']/'preflight.json',result);print(json.dumps(result),flush=True)
