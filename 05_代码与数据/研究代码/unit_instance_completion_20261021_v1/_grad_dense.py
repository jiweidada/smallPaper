import os, sys, torch
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OMP_NUM_THREADS"]="2"
sys.dont_write_bytecode=True
BASE=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2"
sys.path.insert(0,BASE+r"\311_unit_instance_core"); sys.path.insert(0,BASE+r"\320_geometry_calibrated_core")
import torch; torch.set_num_threads(2)
from core import CLASSES
from gcal import GeometryCalibratedHead
def run(seed, dense):
    torch.manual_seed(seed)
    B=2; rgb=torch.rand(B,3,128,128); Nc=6
    boxes=torch.rand(B,Nc,4); boxes[...,2:]=boxes[...,:2]+0.15*(boxes[...,2:].clone()+0.1)
    cls=torch.randint(0,CLASSES,(B,Nc)); sc=torch.rand(B,Nc); valid=torch.ones(B,Nc,dtype=torch.bool)
    g=GeometryCalibratedHead(); g.train()
    if dense:
        # make mass strictly positive everywhere by biasing mass head
        with torch.no_grad(): g.mass_head.bias.add_(10.0)
    f=g.forward_features(g.trunk(rgb),boxes,cls,sc,valid)
    f["generation_plan"]["residual"].sum().backward()
    gr=g.class_null.grad
    return float(gr.abs().sum()), bool(torch.isnan(gr).any())
for dense in (False,True):
    for seed in (1,2,3):
        tot,nan=run(seed,dense)
        print("dense" if dense else "sparse","seed",seed,"gradabs",tot,"nan",nan)
