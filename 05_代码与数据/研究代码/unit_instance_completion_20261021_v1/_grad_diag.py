import os, sys, json, torch
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"; os.environ["MKL_THREADING_LAYER"]="SEQUENTIAL"
sys.dont_write_bytecode=True
BASE=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2"
sys.path.insert(0,BASE+r"\311_unit_instance_core"); sys.path.insert(0,BASE+r"\320_geometry_calibrated_core")
import torch; torch.set_num_threads(2)
from core import CLASSES, DIM, SIDE
from gcal import GeometryCalibratedHead
torch.manual_seed(2026102101)
B=2; rgb=torch.rand(B,3,128,128); Nc=6
boxes=torch.rand(B,Nc,4); boxes[...,2:]=boxes[...,:2]+0.15*(boxes[...,2:].clone()+0.1)
cls=torch.randint(0,CLASSES,(B,Nc)); sc=torch.rand(B,Nc); valid=torch.ones(B,Nc,dtype=torch.bool)
g=GeometryCalibratedHead(); g.train()
f=g.forward_features(g.trunk(rgb),boxes,cls,sc,valid)
f["generation_plan"]["residual"].sum().backward()
gr=g.class_null.grad
print("grad:",gr)
print("any nan:", torch.isnan(gr).any().item())
# per class
print("per class:", [float(x) for x in gr])
# residual itself finite?
print("residual finite:", torch.isfinite(f["generation_plan"]["residual"]).all().item())
# find where nan arises: check intermediate 'u','T','r'
for k in ("residual","z"):
    v=f["generation_plan"].get(k)
    if v is not None: print(k,"finite",torch.isfinite(v).all().item(), "min",float(v.min()),"max",float(v.max()))
