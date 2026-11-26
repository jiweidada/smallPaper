import os, sys, json, torch
sys.dont_write_bytecode=True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import train340 as T
torch.set_num_threads(2)
IDS = json.load(open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\08_匹配聚合覆盖探针_20261019_v1\sample_plan.json", encoding='utf-8'))['source_pools']['development'][:2]
es=[T.load_entry(1,i) for i in IDS]; b=T.collate(es)
m=T.make_common_init('G',1); m.train()
f=m.trunk(b['rgb'])
tokens, sb, sc, counts = T.core311.candidate_inputs(b['P_boxes'], b['P_cls'], b['P_scores'], b['P_valid'])
import torch.nn.functional as F
mass=F.softplus(m.mass_head(f)).flatten(2)
gp=m.generation_plan(mass, sb, sc, b['P_valid'])
print('residual requires_grad', gp['residual'].requires_grad)
print('class_null requires_grad', m.class_null.requires_grad)
gp['residual'].sum().backward()
print('grad after direct gen backward:', m.class_null.grad)
print('mass requires_grad', mass.requires_grad)