import os, sys, json, torch
sys.dont_write_bytecode=True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import train340 as T
import stage1_cache_v3 as S3
torch.set_num_threads(2)
OUT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
CKPT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1\seed1\run\last.pth"
m=S3.load_real_model(CKPT)
n=sum(p.numel() for p in m.parameters())
req=all(not p.requires_grad for p in m.parameters())
before={k:v.detach().clone() for k,v in m.state_dict().items()}
x=torch.rand(1,3,640,640)
with torch.no_grad(): o=m(x)
after=m.state_dict()
res={'item':'340 global detector zero-grad/state','params':n,'all_requires_grad_false':bool(req),
 'state_unchanged':all(torch.equal(before[k],after[k]) for k in before) and set(before)==set(after),
 'all_grad_none':all(p.grad is None for p in m.parameters()),
 'forward_ok':isinstance(o,dict),'keys':sorted(o.keys()) if isinstance(o,dict) else None}
json.dump(res,open(os.path.join(OUT,'340_result_global.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=2)
print(json.dumps(res,indent=1))