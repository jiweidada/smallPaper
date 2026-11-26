import os, sys, json, torch
sys.dont_write_bytecode=True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import train340 as T
torch.set_num_threads(2)
IDS = json.load(open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\08_匹配聚合覆盖探针_20261019_v1\sample_plan.json", encoding='utf-8'))['source_pools']['development'][:4]
es=[T.load_entry(1,i) for i in IDS]; b=T.collate(es)
m=T.make_common_init('G',1); m.train()
out=m.forward_features(m.trunk(b['rgb']), b['P_boxes'], b['P_cls'], b['P_scores'], b['P_valid'])
print('keys', sorted(out.keys()))
print('has generation_plan:', 'generation_plan' in out)
losses,_=T.compute_losses_340('G',m,b)
print('losses', {k:float(v) for k,v in losses.items() if torch.is_tensor(v)})
losses['loss_total'].backward()
g=m.class_null.grad
print('class_null grad:', [round(float(x),12) for x in g])
print('max abs:', float(g.abs().max()))
m2=T.make_common_init('G',1); m2.train()
out2=m2.forward_features(m2.trunk(b['rgb']), b['P_boxes'], b['P_cls'], b['P_scores'], b['P_valid'])
print('density_field equals generation residual:', torch.equal(out2['density_field'], out2['generation_plan']['residual']))
print('kappa_c values:', m2.class_null.detach().tolist())