import os, sys, torch, json
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OMP_NUM_THREADS"]="2"
sys.dont_write_bytecode=True
BASE=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2"
sys.path.insert(0,BASE+r"\311_unit_instance_core"); sys.path.insert(0,BASE+r"\320_geometry_calibrated_core")
import torch; torch.set_num_threads(2)
import torch.nn.functional as F
from core import CLASSES
import gcal as gcalmod
from gcal import GeometryCalibratedHead

def cap_v2(mass,cost,null_cost,classes,valid,detach_mass=True):
    """Fixed: only divide where u>m (min would pick m/u); otherwise exactly 1. No tiny-u division."""
    m=mass.detach() if detach_mass else mass
    b,c,i=mass.shape; j=cost.shape[2]
    cls=torch.where(valid,classes,torch.zeros_like(classes))
    onehot=F.one_hot(cls,c).to(m.dtype)*valid[...,None]
    selected_m=m.gather(1,cls[...,None].expand(b,j,i)).transpose(1,2)
    logm=torch.where(selected_m>0, selected_m.clamp_min(torch.finfo(m.dtype).tiny).log(), torch.full_like(selected_m,-torch.inf))
    joined=torch.cat((logm-cost,-null_cost[:,None,:]),dim=1)
    probs=joined.softmax(1)*valid[:,None,:]
    p,p0=probs[:,:-1],probs[:,-1]
    u=torch.einsum("bij,bjc->bci",p,onehot)
    ones=torch.ones_like(m)
    take = (u>m)                      # only here is m/u < 1
    ratio = torch.where(take, m/u.clamp_min(torch.finfo(m.dtype).tiny), ones)
    scale = torch.where(u>0, ratio, ones)
    selected_scale=scale.gather(1,cls[...,None].expand(b,j,i)).transpose(1,2)
    plan=p*selected_scale
    null=p0+(p-plan).sum(1)
    explained=torch.einsum("bij,bjc->bci",plan,onehot)
    residual=(m-explained).clamp_min(0.)
    return dict(p=p,p0=p0,T=plan,z=null,u=u,explained=explained,scale=scale,residual=residual,mass=m)

def run(capfn, seed=2026102101):
    torch.manual_seed(seed)
    B=2; rgb=torch.rand(B,3,128,128); Nc=6
    boxes=torch.rand(B,Nc,4); boxes[...,2:]=boxes[...,:2]+0.15*(boxes[...,2:].clone()+0.1)
    cls=torch.randint(0,CLASSES,(B,Nc)); sc=torch.rand(B,Nc); valid=torch.ones(B,Nc,dtype=torch.bool)
    g=GeometryCalibratedHead(); g.train()
    gcalmod.base.independent_cap=capfn
    f=g.forward_features(g.trunk(rgb),boxes,cls,sc,valid)
    res=f["generation_plan"]["residual"].detach().clone()
    f["generation_plan"]["residual"].sum().backward()
    return g.class_null.grad.detach().clone(), res

g_old, r_old = run(gcalmod.base.__dict__.get("_orig", None) or None) if False else (None,None)
# frozen run
import core as coremod
frozen_cap = coremod.independent_cap
g1,res1 = run(frozen_cap)
g2,res2 = run(cap_v2)
print("frozen grad nan:", int(torch.isnan(g1).sum()), "v2 grad nan:", int(torch.isnan(g2).sum()))
print("v2 grad all finite:", bool(torch.isfinite(g2).all()))
print("residual allclose atol 1e-7:", bool(torch.allclose(res1,res2,atol=1e-7,rtol=0)))
print("residual max abs diff:", float((res1-res2).abs().max()))
print("v2 grads:", [round(float(x),6) for x in g2])
json.dump({"frozen_nan":int(torch.isnan(g1).sum()),"v2_nan":int(torch.isnan(g2).sum()),
 "v2_all_finite":bool(torch.isfinite(g2).all()),
 "residual_max_abs_diff":float((res1-res2).abs().max()),
 "residual_allclose_atol_1e-7":bool(torch.allclose(res1,res2,atol=1e-7,rtol=0)),
 "v2_grads":[float(x) for x in g2]},
 open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\_v2_mem_test.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
