"""Candidate-only shape/backward check; no optimizer, no baseline forward."""
import argparse,json,torch
from pathlib import Path
from runtime import verify_bundle,activate,register_model,load_common,write
p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);a=p.parse_args()
receipt,plan,paths=verify_bundle(a.bundle)
activate(paths['native_root']);register_model()
from engine.core import YAMLConfig
from engine.misc import dist_utils
dist_utils.setup_distributed(print_rank=0,print_method='builtin',seed=plan['seed'])
cfg=YAMLConfig(str(Path(a.bundle)/'resolved.yml'));model=cfg.model
init=load_common(model,paths['common_init'])
stem=model.backbone.stem.cuda().train()
shapes=[]
for side in (640,832):
    x=torch.linspace(-1,1,side*side,device='cuda').reshape(1,1,side,side).expand(4,3,-1,-1).contiguous()
    with torch.autocast(device_type='cuda',dtype=torch.float16):
        y=stem(x);loss=y.float().square().mean()
    assert tuple(y.shape)==(4,16,side//4,side//4) and torch.isfinite(y).all()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in stem.parameters() if p.grad is not None)
    shapes.append(list(y.shape));stem.zero_grad(set_to_none=True)
result=dict(status='STEM_SHAPE_BACKWARD_PASS_NO_UPDATES',seed=plan['seed'],initialization=init,shapes=shapes,
    model_forwards=0,stem_forwards=2,optimizer_updates=0,baseline_forwards=0,
    peak_allocated_bytes=torch.cuda.max_memory_allocated())
write(paths['screen_root']/'preflight.json',result);print(json.dumps(result),flush=True)
