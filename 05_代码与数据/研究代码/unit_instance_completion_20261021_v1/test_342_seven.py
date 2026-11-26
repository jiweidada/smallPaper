import os, sys, json, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import geom342 as G
import core as core311
torch.set_num_threads(2)
OUT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
BASE=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2"
ORDER=json.load(open(BASE+chr(92)+'327_训练实施输入索引.json',encoding='utf-8'))['sample_plan']['development_ids'][:32]
KS=[-6.7,-0.67,-0.067,0.0,0.067,0.67,6.7]
TARGET={-6.7:0.9989,-0.67:0.7920,-0.067:0.7489,0.0:0.7445,0.067:0.7402,0.67:0.7067,6.7:0.6315}
acc={k:[] for k in KS}
for iid in ORDER:
    e=torch.load(os.path.join(OUT,'340_cache_seed1',f'{iid:06d}.pt'),map_location='cpu',weights_only=True)
    W=e['window_xyxy_640']; b=e['fullp_boxes']
    keep=G.intersecting_candidates(b,W)
    n=int(keep.sum())
    if n==0: continue
    bb=b[keep]; cls=e['fullp_labels'][keep]
    win_xyxy=G.win_norm_xyxy(bb,W)
    geom=G.geometry_cost(win_xyxy)
    mass=torch.full((1,10,256),1.0/256)
    cls2=cls.unsqueeze(0); valid=torch.ones(1,n,dtype=torch.bool)
    for k in KS:
        kap=torch.full((10,),float(k))
        plan=core311.independent_cap(mass,geom,kap[cls2],cls2,valid)
        acc[k].append(float(plan['residual'].sum()/mass.sum()))
got={k: round(sum(v)/len(v),4) for k,v in acc.items()}
res={'item':'342 阶段1 验收(a) 复现主控七个数','got':{str(k):got[k] for k in KS},
     'target':{str(k):TARGET[k] for k in KS},
     'all_match_4dp':all(abs(got[k]-TARGET[k])<5e-5 for k in KS),
     'max_abs_diff':max(abs(got[k]-TARGET[k]) for k in KS)}
print(json.dumps(res,indent=1))
json.dump(res,open(os.path.join(OUT,'342_result_geom_seven.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=2)