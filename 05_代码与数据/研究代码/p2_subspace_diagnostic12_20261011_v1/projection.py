"""Rank16 fixed-weight runtime interventions; no learned parameters or optimizer."""
import torch

def project(weight,mode,rank=16):
    if weight.shape!=(256,64,1,1) or not torch.isfinite(weight).all():
        raise ValueError('Expected finite original256x64 P2 restore weights')
    if mode not in ('dc16_norm','svd16_norm') or rank!=16:
        raise ValueError('Only the preregistered rank16 interventions are allowed')
    original=weight.detach().to(device='cpu',dtype=torch.float64).reshape(256,64).clone()
    if not bool(original.norm()>0):raise ValueError('Zero restore weights would give a vacuous control')
    if mode=='dc16_norm':
        result=original.reshape(256,16,4).mean(2,keepdim=True).expand(-1,-1,4).reshape(256,64).clone()
    else:
        u,s,vh=torch.linalg.svd(original,full_matrices=False)
        result=(u[:,:rank]*s[:rank])@vh[:rank]
    norm=result.norm()
    if not bool(norm>0):raise ValueError('Projection has zero norm')
    result=result*(original.norm()/norm)
    return result.reshape_as(weight).to(dtype=weight.dtype).contiguous()
