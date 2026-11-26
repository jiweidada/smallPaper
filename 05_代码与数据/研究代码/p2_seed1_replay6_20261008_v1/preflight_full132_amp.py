"""Bounded AMP development fixture; callable only after the completed24 gate."""
import copy
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
from torch.nn.functional import interpolate
import p2_candidate
from engine.core import YAMLConfig
from screen_runtime import strict_common_init


def main():
    root=Path(sys.argv[1]);init=sys.argv[2]
    gate=json.loads((root/'gate_24_to_132.json').read_text())
    assert gate['allowed'] and gate['completed_epochs']==24
    assert gate['delta_APs_pp']>=.8 and gate['delta_AP_pp']>=-.2
    torch.set_num_threads(6);random.seed(2);np.random.seed(2);torch.manual_seed(2);torch.cuda.manual_seed_all(2)
    cfg=YAMLConfig(str(root/'p2_full132.yml'));model=cfg.model
    strict_common_init(model,init);model.to('cuda:0').train()
    criterion=cfg.criterion.to('cuda:0');criterion.set_model(model)
    optimizer=cfg.optimizer;scaler=cfg.scaler;ema=cfg.ema.to('cuda:0')
    loader=cfg.train_dataloader;loader.set_epoch(0)
    images,targets=next(iter(loader));images=interpolate(images.to('cuda:0'),size=(800,800),mode='bilinear',align_corners=False)
    targets=[{k:v.to('cuda:0') if torch.is_tensor(v) else v for k,v in t.items()} for t in targets]
    counts={'updates':0};optimizer.register_step_post_hook(lambda *_:counts.__setitem__('updates',counts['updates']+1))
    torch.cuda.reset_peak_memory_stats();steps=[]
    for i in range(10):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):outputs=model(images,targets=targets)
        with torch.autocast('cuda',enabled=False):loss=sum(criterion(outputs,targets,epoch=0,step=i,global_step=i,epoch_step=len(loader)).values())
        assert torch.isfinite(loss)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        g=model.decoder.query_recruitment.project.weight.grad
        grad_norm=float(g.float().norm());torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.clip_max_norm)
        inf=sum(float(v.item()) for v in scaler._found_inf_per_device(optimizer).values());before=counts['updates'];scale=scaler.get_scale()
        scaler.step(optimizer);scaler.update();ema.update(model)
        steps.append({'step':i,'loss':float(loss),'scale':scale,'found_inf':inf,'updated':counts['updates']>before,
                      'P2_grad_norm':grad_norm if np.isfinite(grad_norm) else None})
        if i>=2 and counts['updates']>=2:break
    assert counts['updates']>=2 and model.decoder.query_recruitment.project.weight.norm()>0
    receipt={'passed':True,'effective_updates':counts['updates'],'steps':steps,
             'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
             'scope':'fresh development fixture only; full132 trainer restarts from the common initialization'}
    (root/'full132_amp_preflight.json').write_text(json.dumps(receipt,indent=2,allow_nan=False))
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
