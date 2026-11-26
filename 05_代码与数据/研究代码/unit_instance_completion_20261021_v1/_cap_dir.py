import sys, torch
sys.dont_write_bytecode=True
BASE=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2"
sys.path.insert(0, BASE+chr(92)+'311_unit_instance_core_v2')
import torch; torch.set_num_threads(2)
import core as C
# single candidate, class 0, mass 1/256 everywhere
mass=torch.full((1,10,256),1.0/256)
cost=torch.zeros(1,256,1)     # one candidate, zero geometry cost
classes=torch.zeros(1,1,dtype=torch.long)
valid=torch.ones(1,1,dtype=torch.bool)
for k in [-6.7,-0.67,0.0,0.67,6.7]:
    nc=torch.full((1,1),float(k))
    plan=C.independent_cap(mass,cost,nc,classes,valid)
    print('kappa=%.2f residual/mass=%.4f p0=%.6f' % (k, float(plan['residual'].sum()/mass.sum()), float(plan['p0'].mean())))