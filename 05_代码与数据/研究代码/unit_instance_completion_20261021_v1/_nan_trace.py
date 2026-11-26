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
g=GeometryCalibratedHead(); g.train()
# instrument: run with fixed division, and capture tensors with retain_grad
capt={}
def cap_dbg(mass,cost,null_cost,classes,valid,detach_mass=True):
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
    ones=torch.ones_like(m); take=(u>m)
    ratio=torch.where(take, m/u.clamp_min(torch.finfo(m.dtype).tiny), ones)
    scale=torch.where(u>0,ratio,ones)
    ss=scale.gather(1,cl[...,None].expand(b,j,i)).transpose(1,2)
    plan=p*ss
    null=p0+(p-plan).sum(1)
    expl=torch.einsum("bij,bjc->bci",plan,onehot)
    resid=(m-expl).clamp_min(0.)
    for name,t in [("p",p),("p0",p0),("u",u),("scale",scale),("plan",plan),("null",null),("expl",expl),("resid",resid)]:
        t.retain_grad()
    capt.update(dict(p=p,p0=p0,u=u,scale=scale,plan=plan,null=null,expl=expl,resid=resid,mass=m))
    return dict(p=p,p0=p0,T=plan,z=null,u=u,explained=expl,scale=scale,residual=resid,mass=m)
gcalmod.base.independent_cap=cap_dbg
f=g.forward_features(g.trunk(rgb),boxes,cls,sc,valid)
f["generation_plan"]["residual"].sum().backward()
print("class_null grad nan:", int(torch.isnan(g.class_null.grad).sum()))
for name in ["p","p0","u","scale","plan","null","expl","resid"]:
    t=capt[name]; g_=t.grad
    if g_ is None: print(name,"grad None"); continue
    print(name, "shape",tuple(t.shape), "nan", int(torch.isnan(g_).sum()), "inf", int(torch.isinf(g_).sum()), "absmax", float(g_.abs().nan_to_num().max()))
