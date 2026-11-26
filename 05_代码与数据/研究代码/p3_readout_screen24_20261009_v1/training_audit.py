"""Passive checks on actual training; no added forward/backward or RNG draws."""
import hashlib, json, pickle, random
from pathlib import Path
import numpy as np
import torch

def digest(value):
    h=hashlib.sha256()
    def add(x):
        if isinstance(x,torch.Tensor):
            v=x.detach().as_subclass(torch.Tensor).cpu().contiguous()
            h.update(str((str(v.dtype),tuple(v.shape))).encode()); h.update(v.numpy().tobytes())
        elif isinstance(x,dict):
            for key in sorted(x): h.update(str(key).encode()); add(x[key])
        elif isinstance(x,(tuple,list)):
            for item in x: add(item)
        else: h.update(repr(x).encode())
    add(value)
    return h.hexdigest()

class TrainingAudit:
    def __init__(self,model,optimizer,output):
        self.model=model;self.optimizer=optimizer;self.output=Path(output)
        self.attempts=0;self.steps=0;self.handles=[];self.rows=[]
        self.residual=getattr(model.decoder,'local_readout',None)
        self.stream=(self.output/'first100_training_inputs.jsonl').open('x',encoding='utf-8')
        self.handles.append(model.register_forward_pre_hook(self.forward,with_kwargs=True))
        self.handles.append(optimizer.register_step_pre_hook(self.before_step))
        self.handles.append(optimizer.register_step_post_hook(self.after_step))
    def forward(self,module,args,kwargs):
        if not module.training:return
        self.attempts+=1
        if self.attempts<=100:
            row={'attempt':self.attempts,'inputs_targets_sha256':digest((args,kwargs)),
                'rng_sha256':digest((torch.get_rng_state(),torch.cuda.get_rng_state(),
                    pickle.dumps(random.getstate()),pickle.dumps(np.random.get_state())))}
            self.stream.write(json.dumps(row)+'\n');self.stream.flush()
    def before_step(self,opt,args,kwargs):
        if self.steps<8 and self.residual is not None:
            self.grad={n:None if p.grad is None else {'norm':float(p.grad.detach().float().norm()),
                'finite':bool(torch.isfinite(p.grad).all())} for n,p in self.residual.named_parameters()}
    def after_step(self,opt,args,kwargs):
        self.steps+=1
        if self.steps<=8:
            row={'actual_optimizer_step':self.steps,'attempt':self.attempts}
            if self.residual is not None:
                row.update(gradient=self.grad,output_weight_norm=float(self.residual.output.weight.detach().float().norm()),
                    output_bias_norm=float(self.residual.output.bias.detach().float().norm()))
            self.rows.append(row)
            (self.output/'first8_optimizer_steps.json').write_text(json.dumps(self.rows,indent=2),encoding='utf-8')
        if self.steps%100==0:
            (self.output/'training_audit_live.json').write_text(json.dumps({'attempts':self.attempts,
                'actual_optimizer_steps':self.steps}),encoding='utf-8')
    def epoch_complete(self,epoch,log_stats,ema):
        row={'epoch_zero_based':epoch,'attempts_total':self.attempts,'actual_optimizer_steps_total':self.steps,
             'EMA_updates':int(ema.updates) if ema is not None else None}
        if self.residual is not None:
            row['residual_output_weight_norm']=float(self.residual.output.weight.detach().float().norm())
        with (self.output/'training_audit_epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
    def close(self):
        for handle in self.handles:handle.remove()
        self.stream.close()
