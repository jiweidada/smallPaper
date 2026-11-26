"""A51 prototype: cross-resolution paired exchange, not a new diffusion principle.

For D=2x2 average and U=2x nearest, DU=I. With orthogonal channel map A,
Q performs f'=f+UA^T F, c'=c-F, F=alpha*g*(c-ADf).
It preserves f-UDf and ADf+c for each pair, up to floating-point rounding.
The matched control M uses independent directional gates instead of their mean.
"""
import torch
from torch import nn
from torch.nn import functional as F


def down(x):return F.avg_pool2d(x,2,2)
def up(x):return F.interpolate(x,scale_factor=2,mode='nearest')
def channel_map(a,x):return torch.einsum('oc,bchw->bohw',a,x)


class PairExchange(nn.Module):
    def __init__(self,channels,arm):
        super().__init__()
        assert arm in ('M','Q')
        self.arm=arm
        self.frame=nn.Parameter(torch.zeros(channels,channels))
        self.gates=nn.Linear(3,2)
        nn.init.normal_(self.gates.weight,std=.02);nn.init.zeros_(self.gates.bias)
        self.beta=nn.Parameter(torch.zeros(()))

    def orthogonal_frame(self):
        w=self.frame
        k=w-w.T
        eye=torch.eye(w.shape[0],device=w.device,dtype=w.dtype)
        # Cayley parametrization is established; it is not claimed as our invention.
        return torch.linalg.solve(eye+k,eye-k)

    def forward(self,fine,coarse):
        assert fine.shape[:2]==coarse.shape[:2]
        assert fine.shape[2]==2*coarse.shape[2] and fine.shape[3]==2*coarse.shape[3]
        dtype=fine.dtype
        with torch.autocast(device_type=fine.device.type,enabled=False):
            f=fine.float() if dtype!=torch.float64 else fine
            c=coarse.float() if dtype!=torch.float64 else coarse
            a=self.orthogonal_frame()
            source=channel_map(a,down(f))
            rs=(source.square().mean(1,keepdim=True)+1e-12).sqrt()
            rc=(c.square().mean(1,keepdim=True)+1e-12).sqrt()
            denominator=rs+rc+1e-6
            cosine=(source*c).mean(1,keepdim=True)/(rs*rc+1e-6)
            invariant=torch.cat([rs/denominator,rc/denominator,cosine],dim=1).permute(0,2,3,1)
            gate=self.gates(invariant).sigmoid().permute(0,3,1,2)
            if self.arm=='Q':
                shared=gate.mean(1,keepdim=True)
                gu=gd=shared
            else:
                gu,gd=gate[:,0:1],gate[:,1:2]
            alpha=.25*self.beta.tanh()
            difference=c-source
            to_fine=alpha*gu*difference
            to_coarse=alpha*gd*difference
            fout=f+up(channel_map(a.T,to_fine))
            cout=c-to_coarse
        return fout.to(dtype),cout.to(dtype)


class ScaleExchange(nn.Module):
    def __init__(self,channels=192,arm='Q'):
        super().__init__()
        self.arm=arm
        self.pairs=nn.ModuleList([PairExchange(channels,arm),PairExchange(channels,arm)])

    def forward(self,features):
        assert len(features)==3
        result=list(features)
        for i,pair in enumerate(self.pairs):
            result[i],result[i+1]=pair(result[i],result[i+1])
        return result


def install(model,arm):
    assert arm in ('B','M','Q')
    if arm=='B':return None
    encoder=model.encoder
    assert not hasattr(encoder,'a51_exchange')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261102)
        module=ScaleExchange(192,arm)
    encoder.add_module('a51_exchange',module)
    def exchange_hook(current,args,output):
        return current.a51_exchange(output)
    encoder.register_forward_hook(exchange_hook)
    return module
