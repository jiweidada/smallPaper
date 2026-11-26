"""Deferred disposable GPU checks; never run while the previous round is active."""
import json
from pathlib import Path
import sys
import torch
from tiny_protocol import install_native_adapter

root = Path(sys.argv[1]); plan = json.loads((root/'plan.json').read_text())
assert json.loads((root/'release_after_round2.json').read_text())['both_round2_queues_complete']
install_native_adapter()
from engine.core import YAMLConfig
from engine.solver.det_engine import _build_resize_pad_metadata
checkpoint = torch.load(plan['initialization'],map_location='cpu',weights_only=False)['model']
results = []
for arm in plan['jobs']:
    torch.manual_seed(plan['seed']);cfg=YAMLConfig(plan['configs'][arm]);model=cfg.model
    missing,unexpected=model.load_state_dict(checkpoint,strict=False)
    assert not unexpected and all('p2_skip' in key for key in missing)
    model=model.cuda().train();criterion=cfg.criterion.cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    loader=iter(cfg.train_dataloader);samples,targets=next(loader)
    samples=samples.cuda();targets=[{k:v.cuda() if torch.is_tensor(v) else v for k,v in t.items()} for t in targets]
    for step in range(3):
        optimizer.zero_grad(set_to_none=True);outputs=model(samples,targets)
        loss_dict=criterion(outputs,targets,epoch=0,step=step,global_step=step,epoch_step=len(cfg.train_dataloader));loss=sum(loss_dict.values())
        assert torch.isfinite(loss).all();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.1);optimizer.step()
    results.append({'arm':arm,'three_disposable_updates_finite':True,'loss':float(loss.detach()),
                    'max_allocated_bytes':torch.cuda.max_memory_allocated()})
    del optimizer,criterion,model,samples,targets,loader;torch.cuda.empty_cache()
with (root/'gpu_preflight_receipt.json').open('x') as handle:json.dump({'status':'passed','results':results},handle,indent=2)
print(json.dumps(results),flush=True)
