import os, sys, json, torch
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"; os.environ["MKL_THREADING_LAYER"]="SEQUENTIAL"
sys.dont_write_bytecode=True
BASE=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2"
sys.path.insert(0,BASE+r"\311_unit_instance_core"); sys.path.insert(0,BASE+r"\320_geometry_calibrated_core")
import torch; torch.set_num_threads(2)
from core import CLASSES
import core as coremod
from gcal import GeometryCalibratedHead

torch.manual_seed(2026102101)
B=2; rgb=torch.rand(B,3,128,128); Nc=6
boxes=torch.rand(B,Nc,4); boxes[...,2:]=boxes[...,:2]+0.15*(boxes[...,2:].clone()+0.1)
cls=torch.randint(0,CLASSES,(B,Nc)); sc=torch.rand(B,Nc); valid=torch.ones(B,Nc,dtype=torch.bool)
g=GeometryCalibratedHead(); g.train()

# instrument independent_cap by monkeypatching a wrapper that records u and scale internals
orig_cap = coremod.independent_cap
captured={}
def cap_wrap(mass, cost, null_cost, classes, valid, detach_mass=True):
    res = orig_cap(mass,cost,null_cost,classes,valid,detach_mass)
    captured["u"]=res["u"].detach().clone()
    captured["scale"]=res["scale"].detach().clone()
    captured["residual"]=res["residual"].detach().clone()
    captured["mass"]=res["mass"].detach().clone()
    return res
import gcal as gcalmod
coremod.independent_cap = cap_wrap
gcalmod.base.independent_cap = cap_wrap
f = g.forward_features(g.trunk(rgb),boxes,cls,sc,valid)
f["generation_plan"]["residual"].sum().backward()
gr = g.class_null.grad
u=captured["u"]; m=captured["mass"]; scale=captured["scale"]

eps=torch.finfo(torch.float32).tiny
mask_pos = u>0
vals = u[mask_pos]
print("=== NaN gradient reproduction ===")
print("class_null grad:", [float(x) for x in gr])
print("nan count:", int(torch.isnan(gr).sum()), "/ 10")
print("u>0 cells:", int(mask_pos.sum()), "total cells:", u.numel())
print("min positive u:", float(vals.min()))
print("min positive u^2:", float((vals.min()**2)), " (fp32 tiny:", eps, ")")
# count u where u^2 underflows to 0
u2 = vals*vals
print("u>0 cells whose u^2 == 0 in fp32:", int((u2==0).sum()))
# count m/safe_u inf
safe_u=torch.where(u>0,u,torch.ones_like(u))
ratio=m/safe_u
print("m/u inf count:", int(torch.isinf(ratio).sum()))
print("cells with u>0 and u<m (would take min=1 branch):", int(((u>0)&(u<m)).sum()))
print("all u>0 cells are u<m ?", bool(((u>0)&(u>=m)).sum()==0))
# show relation between NaN classes and classes present
present=sorted(set(cls.flatten().tolist()))
print("classes present:", present)
nan_classes=[i for i,x in enumerate(gr.tolist()) if x!=x]
print("nan classes:", nan_classes)
json.dump({"grad":[float(x) for x in gr],"nan_count":int(torch.isnan(gr).sum()),
  "u_positive_cells":int(mask_pos.sum()),"min_positive_u":float(vals.min()),
  "min_u_squared":float((vals.min()**2)),"u2_underflow_cells":int((u2==0).sum()),
  "ratio_inf_cells":int(torch.isinf(ratio).sum()),
  "cells_u_gt0_and_u_lt_m":int(((u>0)&(u<m)).sum()),
  "nan_classes":nan_classes,"classes_present":present},
  open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\_nan_repro.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)