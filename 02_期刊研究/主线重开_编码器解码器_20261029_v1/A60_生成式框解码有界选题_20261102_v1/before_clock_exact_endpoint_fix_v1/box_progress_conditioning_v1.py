"""Training conditioning for a single-step, endpoint-predicting box decoder.

B: uniform linear interpolation, condition on physical interpolation time.
M: size-biased interpolation, condition on physical interpolation time.
Q: exactly M's input boxes, condition on normalized geometric progress.

No trainable parameters. Ground truth is used only to construct training inputs.
All modes have the same GT-free single-step inference input and time label.
This is not a claim of exact likelihood training or a new flow-matching theorem.
"""
import torch

def xywh_to_xyxy(boxes):
    return torch.cat((boxes[..., :2]-boxes[..., 2:]/2,
                      boxes[..., :2]+boxes[..., 2:]/2),dim=-1)

@torch.no_grad()
def relative_edge_distance(prior, endpoint):
    p=prior.double();g=endpoint.double()
    assert p.shape==g.shape and p.shape[-1]==4
    assert torch.isfinite(p).all() and torch.isfinite(g).all()
    assert (p[...,2:]>0).all() and (g[...,2:]>0).all()
    scale=torch.cat((g[...,2:],g[...,2:]),dim=-1)
    return ((xywh_to_xyxy(p)-xywh_to_xyxy(g)).abs()/scale).amax(dim=-1)

@torch.no_grad()
def geometric_alpha(progress,distance):
    u=progress.double();d=distance.double()
    assert torch.isfinite(u).all() and torch.isfinite(d).all()
    assert ((u>=0)&(u<=1)).all() and (d>=0).all()
    large=d>1e-10
    denominator=torch.where(large,d,torch.ones_like(d))
    result=1.-torch.expm1((1.-u)*torch.log1p(d))/denominator
    return torch.where(large,result,u)

@torch.no_grad()
def training_inputs(mode,prior,endpoint,progress):
    assert mode in ['B','M','Q'] and prior.ndim==3
    assert prior.shape==endpoint.shape and progress.shape==(prior.shape[0],)
    u=progress[:,None].expand(prior.shape[:2])
    if mode=='B':
        alpha=u
    else:
        d=relative_edge_distance(prior,endpoint)
        alpha=geometric_alpha(u,d).to(prior.dtype)
    noisy=prior+(endpoint-prior)*alpha[...,None]
    condition=alpha if mode in ['B','M'] else u
    # The warm initialization is DiffusionDet: 999 denotes the noisy endpoint.
    time_label=(1.-condition)*999.
    assert torch.isfinite(noisy).all() and (noisy[...,2:]>0).all()
    return noisy,time_label,alpha

@torch.no_grad()
def single_step_inputs(prior):
    """GT-free inference; no scale estimate or endpoint oracle is accepted."""
    assert prior.ndim==3 and prior.shape[-1]==4
    assert torch.isfinite(prior).all() and (prior[...,2:]>0).all()
    return prior,torch.full(prior.shape[:2],999.,dtype=prior.dtype,device=prior.device)

@torch.no_grad()
def sample_prior(batch,queries,device,dtype=torch.float32,generator=None):
    """Independent N(0,1) truncated to [0,1] in normalized cx,cy,w,h."""
    u=torch.rand((batch,queries,4),device=device,dtype=torch.float64,generator=generator)
    cdf0=.5;cdf1=.5*(1.+torch.erf(torch.tensor(1./2**.5,dtype=torch.float64)).item())
    x=2**.5*torch.erfinv(2.*(cdf0+(cdf1-cdf0)*u)-1.)
    x[...,2:]=x[...,2:].clamp_min(1e-8)
    return x.to(dtype)
