import os, sys, json, torch
sys.dont_write_bytecode=True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import train340 as T
torch.set_num_threads(2)
OUT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
IDS = json.load(open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\08_匹配聚合覆盖探针_20261019_v1\sample_plan.json", encoding='utf-8'))['source_pools']['development'][:32]
es=[T.load_entry(1,i) for i in IDS]; bs=[T.collate(es[0:16]), T.collate(es[16:32])]
torch.manual_seed(2026102101)
m=T.make_common_init('R',1); m.train()
a,_=T.compute_losses_340('R',m,bs[0],denom_mode='batch_all_uw')
b,_=T.compute_losses_340('R',m,bs[0],denom_mode='per_image')
res={'item':'340 分母口径对比（只报告）','primary':'batch内全部完整U_W目标数之和 clamp>=1（主控解读）',
 'batch_all_uw':{k:float(v) for k,v in a.items() if torch.is_tensor(v)},'per_image_mean':{k:float(v) for k,v in b.items() if torch.is_tensor(v)}}
res['diff']={k: round(res['batch_all_uw'][k]-res['per_image_mean'][k],9) for k in res['batch_all_uw']}
print(json.dumps(res,indent=1))
json.dump(res,open(os.path.join(OUT,'340_result_denominator.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=2)