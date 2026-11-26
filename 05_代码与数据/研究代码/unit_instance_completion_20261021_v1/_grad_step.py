import os, sys, torch
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OMP_NUM_THREADS"]="2"
sys.dont_write_bytecode=True
BASE=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2"
sys.path.insert(0,BASE+r"\311_unit_instance_core"); sys.path.insert(0,BASE+r"\320_geometry_calibrated_core")
import torch; torch.set_num_threads(2)
from core import CLASSES
from gcal import GeometryCalibratedHead
torch.manual_seed(2)
B=2; rgb=torch.rand(B,3,128,128); Nc=6
boxes=torch.rand(B,Nc,4); boxes[...,2:]=boxes[...,:2]+0.15*(boxes[...,2:].clone()+0.1)
cls=torch.randint(0,CLASSES,(B,Nc)); sc=torch.rand(B,Nc); valid=torch.ones(B,Nc,dtype=torch.bool)
g=GeometryCalibratedHead(); g.train()
with torch.no_grad(): g.mass_head.bias.add_(10.0)
f=g.forward_features(g.trunk(rgb),boxes,cls,sc,valid)
gp=f["generation_plan"]
print("mass min", float(f["mass"].min()), "zeros", int((f["mass"]==0).sum()))
print("u zeros", int((gp["u"]==0).sum()), "scale finite", torch.isfinite(gp["scale"]).all().item())
print("p finite", torch.isfinite(gp["p"]).all().item(), "p0 finite", torch.isfinite(gp["p0"]).all().item())
print("p0 min", float(gp["p0"].min()))
# backward on residual only, check grad on p/p0
gp["residual"].sum().backward(retain_graph=True)
print("class_null grad", g.class_null.grad)
# check if p0 has nan grad
print("p0 grad nan", torch.isnan(gp["p0"].grad).any().item() if gp["p0"].grad is not None else None)
