"""Passive original-input parity and real optimizer/EMA/capacity receipts."""
import hashlib,json,pickle,random
from pathlib import Path
import numpy as np
import torch
from small_object_model import EXTRA_PREFIXES
def digest(value):
    h=hashlib.sha256()
    def add(x):
        if isinstance(x,torch.Tensor):
            v=x.detach().as_subclass(torch.Tensor).cpu().contiguous();h.update(str((str(v.dtype),tuple(v.shape))).encode());h.update(v.numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x):h.update(str(k).encode());add(x[k])
        elif isinstance(x,(list,tuple)):
            for t in x:add(t)
        else:h.update(repr(x).encode())
    add(value);return h.hexdigest()
class TrainingAudit:
    def __init__(self,model,optimizer,output,reference,reference_sha):
        self.model=model;self.output=Path(output);self.attempts=0;self.steps=0;self.rows=[]
        assert hashlib.sha256(Path(reference).read_bytes()).hexdigest()==reference_sha
        self.reference=[json.loads(x) for x in Path(reference).read_text().splitlines()];assert len(self.reference)==100
        self.initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n.startswith(EXTRA_PREFIXES)}
        self.stream=(self.output/'first100_training_inputs.jsonl').open('x',encoding='utf8')
        self.handles=[model.register_forward_pre_hook(self.forward,with_kwargs=True),optimizer.register_step_pre_hook(self.before_step),optimizer.register_step_post_hook(self.after_step)]
    def stats(self,grad=False):
        result={}
        for n,p in self.model.named_parameters():
            if not n.startswith(EXTRA_PREFIXES):continue
            value=p.grad if grad else p
            result[n]=None if value is None else {'norm':float(value.detach().float().norm()),'finite':bool(torch.isfinite(value).all())}
        return result
    def forward(self,module,args,kwargs):
        if not module.training:return
        self.attempts+=1
        if self.attempts<=100:
            row={'attempt':self.attempts,'inputs_targets_sha256':digest((args,kwargs)),'rng_sha256':digest((torch.get_rng_state(),torch.cuda.get_rng_state(),pickle.dumps(random.getstate()),pickle.dumps(np.random.get_state())))}
            self.stream.write(json.dumps(row)+'\n');self.stream.flush()
            if row!=self.reference[self.attempts-1]:
                with (self.output/'input_parity_failure.json').open('x') as f:json.dump({'actual':row,'reference':self.reference[self.attempts-1],'training_attempt':self.attempts,'automatic_retry':False},f,indent=2)
                raise RuntimeError('First100 input/RNG parity differs from baseline')
            if self.attempts==100:
                self.stream.flush();actual=hashlib.sha256((self.output/'first100_training_inputs.jsonl').read_bytes()).hexdigest()
                (self.output/'first100_parity.json').write_text(json.dumps({'all100_rows_exact':True,'candidate_raw_sha256':actual,'reference_rows_encoding_same':True,'extra_model_RNG_operations':False,'DN_parity_scope':'Exact pre-forward RNG/targets plus unchanged native DN function and deterministic added modules; not independent historical DN tensor replication.'},indent=2))
    def before_step(self,optimizer,args,kwargs):
        if self.steps<8:
            self.step_gradients=self.stats(grad=True)
            assert all(v is not None and v['finite'] for v in self.step_gradients.values()),'Added gradients nonfinite/disconnected after native unscale/clip'
    def after_step(self,optimizer,args,kwargs):
        self.steps+=1
        if self.steps<=8:
            params=self.stats();deltas={n:float((p.detach().cpu()-self.initial[n]).float().norm()) for n,p in self.model.named_parameters() if n in self.initial}
            self.rows.append({'actual_optimizer_step':self.steps,'attempt':self.attempts,'gradients_after_native_unscale_clip':self.step_gradients,'parameters':params,'delta_from_initial':deltas,'CUDA_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'CUDA_peak_reserved_bytes':torch.cuda.max_memory_reserved()})
            (self.output/'first8_optimizer_steps.json').write_text(json.dumps(self.rows,indent=2))
            if self.steps==8:
                assert all(any(d>0 for n,d in deltas.items() if n.startswith(prefix)) for prefix in EXTRA_PREFIXES),'A new parameter group still has no actual update'
        if self.steps%100==0:(self.output/'training_audit_live.json').write_text(json.dumps({'attempts':self.attempts,'actual_optimizer_steps':self.steps,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}))
    def epoch_complete(self,epoch,log_stats,ema):
        assert ema is not None and int(ema.updates)==self.steps
        row={'epoch_zero_based':epoch,'attempts_total':self.attempts,'actual_optimizer_steps_total':self.steps,'attempts_without_optimizer_step':self.attempts-self.steps,'EMA_updates':int(ema.updates),'CUDA_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'CUDA_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'new_parameters':self.stats(),'no_step_definition':'Forward attempts minus optimizer posthooks; not automatically diagnosed as overflow.'}
        with (self.output/'training_audit_epochs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    def close(self):
        for handle in self.handles:handle.remove()
        self.stream.close()
