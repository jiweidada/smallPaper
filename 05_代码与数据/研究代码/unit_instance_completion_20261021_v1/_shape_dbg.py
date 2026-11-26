import os, sys, json, torch
sys.dont_write_bytecode=True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import train340 as T
torch.set_num_threads(2)
IDS = json.load(open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\08_匹配聚合覆盖探针_20261019_v1\sample_plan.json", encoding='utf-8'))['source_pools']['development'][:2]
es = [T.load_entry(1,i) for i in IDS]
b = T.collate(es)
print('P_boxes', b['P_boxes'].shape, 'P_valid', b['P_valid'].shape)
m = T.make_common_init('R',1)
out = m.forward_features(m.trunk(b['rgb']), b['P_boxes'], b['P_cls'], b['P_scores'], b['P_valid'])
print('mass', out['mass'].shape)
print('T', out['plan']['T'].shape, 'z', out['plan']['z'].shape)