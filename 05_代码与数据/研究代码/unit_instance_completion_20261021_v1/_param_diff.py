import sys, os, json, torch
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OMP_NUM_THREADS"]="2"
sys.dont_write_bytecode=True
NATIVE=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\dn_pre_frozen_diagnostic_20261020_v1\native"
EXP=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\fine_residual_screen24_20261009_v2"
sys.path.insert(0,NATIVE); sys.path.insert(0,EXP)
import torch; torch.set_num_threads(2)
from geometry import ResidualGeometryDEIM
base=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1"
cfg=json.load(open(os.path.join(base,"seed1","bundle","resolved.yml"),encoding="utf-8"))
m=ResidualGeometryDEIM(cfg["ResidualGeometryDEIM"]["recipe"],arm="baseline")
tot=sum(p.numel() for p in m.parameters())
print("total params:",tot)
# count buffers
buf=sum(b.numel() for b in m.buffers())
print("buffers:",buf)
# 1-element params
ones=[(n,p.numel()) for n,p in m.named_parameters() if p.numel()<=2]
print("params with numel<=2:",ones)
# per top module
from collections import defaultdict
d=defaultdict(int)
for n,p in m.named_parameters(): d[n.split('.')[0]]+=p.numel()
print(dict(d))
# check the log: n_parameters maybe counts trainable only?
print("trainable:", sum(p.numel() for p in m.parameters() if p.requires_grad))
