"""Reproduce exactly the failed fresh-model synthetic fixture; no updates."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
import hashlib
import torch
from runtime import HERE,activate_snapshot,build_components

torch.set_num_threads(4)
activate_snapshot()
plan=json.loads((HERE/'configs/fused_residual_seed1.json').read_text(encoding='utf-8'))
torch.manual_seed(810210)
model,criterion,_=build_components(plan)
image=torch.rand(1,3,640,640)
target=[{'labels':torch.tensor([3,6]),'boxes':torch.tensor([[.45,.46,.03,.04],[.62,.55,.02,.03]])}]
model.train()
for module in model.modules():
    if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
model.decoder.audit_trace=True
before={k:p.detach().clone() for k,p in model.named_parameters()}
torch.manual_seed(810211)
out=model(image,target)
enc=out['enc_aux_outputs'][0]['pred_boxes'];enc.retain_grad()
losses=criterion(out,target)
sum(losses.values()).backward()
ids=model.decoder.last_trace['ids']
mask=ids<6400
def info(g):
    return None if g is None else dict(shape=list(g.shape),finite=bool(torch.isfinite(g).all()),abs_sum=float(g.abs().sum()),nonzero=int(torch.count_nonzero(g)))
result=dict(status='DIAGNOSIS_ONLY',full_synthetic_detector_forwards=1,
    fixture='Same seed 810210 model/image and seed 810211 training forward as failed test',
    selected_P3=int(mask.sum()),selected_P4=int(((ids>=6400)&(ids<8000)).sum()),selected_P5=int((ids>=8000).sum()),
    encoder_box_gradient=info(enc.grad),encoder_P3_box_gradient=info(enc.grad[mask]),
    parameter_gradients={k:info(p.grad) for k,p in model.decoder.geometry_residual.named_parameters()},
    parameters_unchanged=all(torch.equal(p,before[k]) for k,p in model.named_parameters()),
    optimizer_calls=0,parameter_updates=0,checkpoint_loads=0)
with (HERE/'initial_gradient_diagnosis.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
