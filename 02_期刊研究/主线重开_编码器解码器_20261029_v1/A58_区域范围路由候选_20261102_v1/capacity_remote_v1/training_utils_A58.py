import hashlib, math
import torch

def cpu_state(state):
    return {k:v.detach().cpu().clone() for k,v in state.items()}

class StateEMA:
    def __init__(self, model):
        self.state={k:v.detach().clone() for k,v in model.state_dict().items()}
        self.updates=0

    @torch.no_grad()
    def update(self, model):
        self.updates+=1
        decay=.9999*(1-math.exp(-self.updates/1000))
        dest=[];src=[];current_state=model.state_dict()
        for k,v in self.state.items():
            current=current_state[k].detach()
            if v.is_floating_point() and not k.endswith('.anchors'):
                dest.append(v);src.append(current)
            else:
                v.copy_(current)
        torch._foreach_mul_(dest,decay)
        torch._foreach_add_(dest,src,alpha=1-decay)

def input_digest(images,targets):
    h=hashlib.sha256(images.contiguous().numpy().tobytes())
    for t in targets:
        for k in sorted(t):
            h.update(k.encode());h.update(t[k].contiguous().numpy().tobytes())
    return h.hexdigest()

def rng_digest():
    h=hashlib.sha256(torch.get_rng_state().numpy().tobytes())
    h.update(torch.cuda.get_rng_state().cpu().numpy().tobytes())
    return h.hexdigest()

def assert_finite_output(outputs):
    for key in ['pred_logits','pred_boxes','pred_corners']:
        assert torch.isfinite(outputs[key]).all(),key

def lr_ratio(updates,max_updates=9708):
    if updates<500:
        return (updates+1)/500
    return .1+.9*.5*(1+math.cos(math.pi*(updates-500)/(max_updates-500)))
