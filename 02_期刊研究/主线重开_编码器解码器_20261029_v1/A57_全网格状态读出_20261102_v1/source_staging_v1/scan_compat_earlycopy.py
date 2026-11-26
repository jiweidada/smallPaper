"""Unfused legacy Vim/CrossMamba equations using the installed scan ABI.

This changes kernel plumbing, not the published selective recurrence. The
BiMamba-v2 parameter class is vendored unchanged apart from its import path.
"""
import torch
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn,mamba_inner_fn


def _context(xz,conv_w,conv_b,x_proj_w,dt_proj_w,state_size):
    x,z=xz.chunk(2,1)
    x=causal_conv1d_fn(x.contiguous(),conv_w[:,0].contiguous(),conv_b,activation='silu')
    pars=F.linear(x.transpose(1,2),x_proj_w)
    rank=dt_proj_w.shape[1]
    dt=F.linear(pars[:,:,:rank],dt_proj_w).transpose(1,2).contiguous()
    B=pars[:,:,rank:rank+state_size].transpose(1,2).contiguous()
    return x,z,pars,dt,B


def mamba_inner_fn_no_out_proj(xz,conv_w,conv_b,x_proj_w,dt_proj_w,A,B=None,C=None,D=None,
                               delta_bias=None,B_proj_bias=None,C_proj_bias=None,
                               delta_softplus=True,checkpoint_lvl=1):
    assert B is None and C is None and B_proj_bias is None and C_proj_bias is None
    x,z,pars,dt,B=_context(xz,conv_w,conv_b,x_proj_w,dt_proj_w,A.shape[-1])
    C=pars[:,:,dt_proj_w.shape[1]+A.shape[-1]:].transpose(1,2).contiguous()
    return selective_scan_fn(x,dt,A,B,C,D,z=z,delta_bias=delta_bias,delta_softplus=delta_softplus)


def cross_inner_fn(xz,qfc,conv_w,conv_b,x_proj_w,q_proj_w,dt_proj_w,out_proj_w,out_proj_b,
                   A,B=None,C=None,D=None,delta_bias=None,delta_softplus=True,**kwargs):
    assert B is None and C is None
    x,z,pars,dt,B=_context(xz,conv_w,conv_b,x_proj_w,dt_proj_w,A.shape[-1])
    C=F.linear(qfc.transpose(1,2),q_proj_w).transpose(1,2).contiguous()
    y=selective_scan_fn(x,dt,A,B,C,D,z=z,delta_bias=delta_bias,delta_softplus=delta_softplus)
    return F.linear(y.transpose(1,2),out_proj_w,out_proj_b)


def bimamba_inner_fn(*args,**kwargs):
    raise NotImplementedError('Legacy BiMamba-v1 is outside the selected v2 backbone configuration')
