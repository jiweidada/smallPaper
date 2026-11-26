import os, sys, json, torch
sys.dont_write_bytecode=True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import train340 as T
torch.set_num_threads(2)
IDS = json.load(open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\08_匹配聚合覆盖探针_20261019_v1\sample_plan.json", encoding='utf-8'))['source_pools']['development'][:2]
es=[T.load_entry(1,i) for i in IDS]; b=T.collate(es)
m=T.make_common_init('G',1); m.train()
print('box_head last weight abs sum:', float(m.box_head[-1].weight.abs().sum()))
print('classifier weight abs sum:', float(m.classifier.weight.abs().sum()))
losses,_=T.compute_losses_340('G',m,b)
losses['loss_total'].backward()
print('grad before unzero:', m.class_null.grad)
m.zero_grad()
with torch.no_grad(): m.box_head[-1].weight.normal_(0,0.01); m.classifier.weight.normal_(0,0.01)
losses2,_=T.compute_losses_340('G',m,b)
losses2['loss_total'].backward()
print('grad after unzero:', [round(float(x),8) for x in m.class_null.grad])
print('nonzero classes:', int((m.class_null.grad.abs()>0).sum()))