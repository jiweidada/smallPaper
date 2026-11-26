import os, sys, torch
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OMP_NUM_THREADS"]="2"
sys.dont_write_bytecode=True
BASE=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2"
sys.path.insert(0,BASE+r"\311_unit_instance_core"); sys.path.insert(0,BASE+r"\320_geometry_calibrated_core")
import torch; torch.set_num_threads(2)
from core import CLASSES
import gcal as gcalmod
from gcal import GeometryCalibratedHead
torch.manual_seed(2026102101)
B=2; rgb=torch.rand(B,3,128,128); Nc=6
boxes=torch.rand(B,Nc,4); boxes[...,2:]=boxes[...,:2]+0.15*(boxes[...,2:].clone()+0.1)
cls=torch.randint(0,CLASSES,(B,Nc)); sc=torch.rand(B,Nc); valid=torch.ones(B,Nc,dtype=torch.bool)
g=GeometryCalibratedHead(); g.train()

# Manual re-implementation of the exact core.py lines 57-74 with grad tracking, to see which op yields NaN
orig=gcalmod.base.independent_cap
def cap(mass,cost,null_cost,classes,valid,detach_mass=True):
    m=mass.detach()
    b,c,i=mass.shape; j=cost.shape[2]
    cls=torch.where(valid,classes,torch.zeros_like(classes))
    import torch.nn.functional as F
    onehot=F.one_hot(cls,c).to(m.dtype)*valid[...,None]
    selected_m=m.gather(1,cls[...,None].expand(b,j,i)).transpose(1,2)
    logm=torch.where(selected_m>0, selected_m.clamp_min(torch.finfo(m.dtype).tiny).log(), torch.full_like(selected_m,-torch.inf))
    joined=torch.cat((logm-cost,-null_cost[:,None,:]),dim=1)
    probs=joined.softmax(1)*valid[:,None,:]
    p,p0=probs[:,:-1],probs[:,-1]
    u=torch.einsum("bij,bjc->bci",p,onehot)
    safe_u=torch.where(u>0,u,torch.ones_like(u))
    ratio=(m/safe_u)
    print("  m/safe_u: inf cells", int(torch.isinf(ratio).sum()), "nan", int(torch.isnan(ratio).sum()),
          "max", float(ratio.max()))
    scale=torch.minimum(torch.ones_like(m), ratio)
    scale=torch.where(u>0,scale,torch.ones_like(scale))
    selected_scale=scale.gather(1,cls[...,None].expand(b,j,i)).transpose(1,2)
    plan=p*selected_scale
    null=p0+(p-plan).sum(1)
    explained=torch.einsum("bij,bjc->bci",plan,onehot)
    residual=(m-explained).clamp_min(0.)
    return dict(p=p,p0=p0,T=plan,z=null,u=u,explained=explained,scale=scale,residual=residual,mass=m)
gcalmod.base.independent_cap=cap
f=g.forward_features(g.trunk(rgb),boxes,cls,sc,valid)
f["generation_plan"]["residual"].sum().backward()
gr=g.class_null.grad
print("grad nan:", int(torch.isnan(gr).sum()))
# Also check: does torch.minimum at the 1-branch send grad to ratio? test directly
m_t=torch.tensor([[1.0]],requires_grad=False)
u_t=torch.tensor([[1e-26]],requires_grad=True)
r=m_t/u_t; s=torch.minimum(torch.ones_like(r), r)
s.sum().backward()
print("\ndirect minimum test: r=",float(r),"grad u=",float(u_t.grad),"isnan",bool(torch.isnan(u_t.grad)))
u_t2=torch.tensor([[1e-26]],requires_grad=True); r2=m_t/u_t2
# emulate where(u>0,...) masking after minimum
s2=torch.where(torch.ones_like(r2,dtype=torch.bool), torch.minimum(torch.ones_like(r2), r2), torch.ones_like(r2))
s2.sum().backward()
print("via where+minimum grad:", float(u_t2.grad), "isnan", bool(torch.isnan(u_t2.grad)))
