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
# capture cost / geometry / null_cost as tensors
cap={}
orig=gcalmod.base.independent_cap
def wrapped(mass,cost,null_cost,classes,valid,detach_mass=True):
    cap["cost"]=cost.detach().clone(); cap["null_cost"]=null_cost.detach().clone()
    return orig(mass,cost,null_cost,classes,valid,detach_mass)
gcalmod.base.independent_cap=wrapped
f=g.forward_features(g.trunk(rgb),boxes,cls,sc,valid)
f["generation_plan"]["residual"].sum().backward()
c=cap["cost"]; nc=cap["null_cost"]
print("cost shape",tuple(c.shape),"finite",bool(torch.isfinite(c).all()),"min",float(c.min()),"max",float(c.max()))
print("null_cost finite",bool(torch.isfinite(nc).all()),"max",float(nc.max()))
print("grad nan classes", int(torch.isnan(g.class_null.grad).sum()))
# Now reproduce softmax grad NaN minimal: joined with -inf in first rows
import torch
x=torch.tensor([[-float('inf'), 0.0, -1.0],[ -2.0, -3.0, -4.0]],requires_grad=True)
p=torch.softmax(x,1); p.sum().backward(); print("softmax with -inf grad:",x.grad)
# with -1e30
y=torch.tensor([[-1e30,0.,-1.],[-2.,-3.,-4.]],requires_grad=True)
q=torch.softmax(y,1); q.sum().backward(); print("softmax with -1e30 grad:",y.grad)
