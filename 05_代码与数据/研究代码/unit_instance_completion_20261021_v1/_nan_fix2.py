import os, sys, torch
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OMP_NUM_THREADS"]="2"
sys.dont_write_bytecode=True
BASE=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2"
sys.path.insert(0,BASE+r"\311_unit_instance_core"); sys.path.insert(0,BASE+r"\320_geometry_calibrated_core")
import torch; torch.set_num_threads(2); import torch.nn.functional as F
from core import CLASSES
import gcal as gcalmod
from gcal import GeometryCalibratedHead
torch.manual_seed(2026102101)
B=2; rgb=torch.rand(B,3,128,128); Nc=6
boxes=torch.rand(B,Nc,4); boxes[...,2:]=boxes[...,:2]+0.15*(boxes[...,2:].clone()+0.1)
cls=torch.randint(0,CLASSES,(B,Nc)); sc=torch.rand(B,Nc); valid=torch.ones(B,Nc,dtype=torch.bool)

def cap_fix(mass,cost,null_cost,classes,valid,detach_mass=True):
    m=mass.detach()
    b,c,i=mass.shape; j=cost.shape[2]
    cl=torch.where(valid,classes,torch.zeros_like(classes))
    onehot=F.one_hot(cl,c).to(m.dtype)*valid[...,None]
    sel=m.gather(1,cl[...,None].expand(b,j,i)).transpose(1,2)
    logm=torch.where(sel>0, sel.clamp_min(torch.finfo(m.dtype).tiny).log(), torch.full_like(sel,-torch.inf))
    joined=torch.cat((logm-cost,-null_cost[:,None,:]),dim=1)
    probs=joined.softmax(1)*valid[:,None,:]
    p,p0=probs[:,:-1],probs[:,-1]
    u=torch.einsum("bij,bjc->bci",p,onehot)
    # KEY: denominator = max(u, m) elementwise, so ratio = m/max(u,m) <= 1 always,
    # and where u>=m the ratio is exactly 1 with zero grad (no tiny division ever).
    # where u<m, denominator=m so ratio=m/m=1 too... need care:
    # We want scale=min(1, m/u). Equivalent: m / max(u, m)  (when u>=m -> m/u; when u<m -> m/m=1) ✓
    denom=torch.maximum(u, m)
    scale=torch.where(u>0, m/denom, torch.ones_like(m))
    ss=scale.gather(1,cl[...,None].expand(b,j,i)).transpose(1,2)
    plan=p*ss
    null=p0+(p-plan).sum(1)
    expl=torch.einsum("bij,bjc->bci",plan,onehot)
    resid=(m-expl).clamp_min(0.)
    return dict(p=p,p0=p0,T=plan,z=null,u=u,explained=expl,scale=scale,residual=resid,mass=m)

# frozen for comparison
import core as coremod
def run(fn, seed):
    torch.manual_seed(seed)
    B=2; rgb=torch.rand(B,3,128,128); Nc=6
    boxes=torch.rand(B,Nc,4); boxes[...,2:]=boxes[...,:2]+0.15*(boxes[...,2:].clone()+0.1)
    cls=torch.randint(0,CLASSES,(B,Nc)); sc=torch.rand(B,Nc); valid=torch.ones(B,Nc,dtype=torch.bool)
    g=GeometryCalibratedHead(); g.train(); gcalmod.base.independent_cap=fn
    f=g.forward_features(g.trunk(rgb),boxes,cls,sc,valid)
    resid=f["generation_plan"]["residual"].detach().clone()
    f["generation_plan"]["residual"].sum().backward()
    return g.class_null.grad.detach().clone(), resid

for seed in (2026102101, 2, 3):
    g1,r1=run(coremod.independent_cap, seed)
    g2,r2=run(cap_fix, seed)
    print(f"seed {seed}: frozen nan={int(torch.isnan(g1).sum())} | fixed nan={int(torch.isnan(g2).sum())} finite={bool(torch.isfinite(g2).all())} | resid maxdiff={float((r1-r2).abs().max()):.2e}")
    print("   fixed grads:", [round(float(x),5) for x in g2])
