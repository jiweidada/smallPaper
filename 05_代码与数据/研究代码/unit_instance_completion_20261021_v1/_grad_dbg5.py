import os, sys, json, torch
sys.dont_write_bytecode=True
BASE=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2"
sys.path.insert(0,BASE+chr(92)+'311_unit_instance_core_v2'); sys.path.insert(0,BASE+chr(92)+'320_geometry_calibrated_core_v2')
import torch; torch.set_num_threads(2)
from core import CLASSES
from gcal import GeometryCalibratedHead
torch.manual_seed(1)
B=2; rgb=torch.rand(B,3,128,128); Nc=6
boxes=torch.rand(B,Nc,4); boxes[...,2:]=boxes[...,:2]+0.15
cls=torch.randint(0,CLASSES,(B,Nc)); sc=torch.rand(B,Nc); valid=torch.ones(B,Nc,dtype=torch.bool)
g=GeometryCalibratedHead(); g.train()
o=g.forward_features(g.trunk(rgb),boxes,cls,sc,valid)
print('box_head last w abs:', float(g.box_head[-1].weight.abs().sum()))
o['pred_boxes'].sum().backward()
print('kappa grad via pred_boxes (zero-init head):', g.class_null.grad)
g.zero_grad()
with torch.no_grad(): g.box_head[-1].weight.normal_(0,0.01)
o2=g.forward_features(g.trunk(rgb),boxes,cls,sc,valid)
o2['pred_boxes'].sum().backward()
print('kappa grad after unzero:', [round(float(x),10) for x in g.class_null.grad])