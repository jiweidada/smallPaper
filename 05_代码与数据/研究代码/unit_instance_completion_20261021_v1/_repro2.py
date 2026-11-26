import os, sys, json, torch
sys.dont_write_bytecode=True
BASE=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2"
sys.path.insert(0, BASE+chr(92)+'311_unit_instance_core_v2')
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import torch; torch.set_num_threads(2)
import core as C
OUT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
P327=BASE+chr(92)+'327_训练实施输入索引.json'
ORDER=json.load(open(P327,encoding='utf-8'))['sample_plan']['development_ids'][:32]
# Try variants of normalization to find which reproduces 0.7445 at kappa=0
variants={'all_mass':[], 'class_mass':[], 'only_cand_classes':[]}
ks=[-6.7,-0.67,-0.067,0.0,0.067,0.67,6.7]
res={str(k):{'all_mass':[], 'class_mass':[]} for k in ks}
for iid in ORDER:
    e=torch.load(os.path.join(OUT,'340_cache_seed1',f'{iid:06d}.pt'),map_location='cpu',weights_only=True)
    W=e['window_xyxy_640']; b=e['fullp_boxes']
    ix=torch.clamp(torch.minimum(b[:,2],W[2])-torch.maximum(b[:,0],W[0]),min=0)
    iy=torch.clamp(torch.minimum(b[:,3],W[3])-torch.maximum(b[:,1],W[1]),min=0)
    keep=(ix*iy)>0
    n=int(keep.sum())
    if n==0: continue
    bb=b[keep]; cls=e['fullp_labels'][keep]
    mass=torch.full((1,10,256),1.0/256)
    loc=torch.stack([(bb[:,0]+bb[:,2])/2-W[0],(bb[:,1]+bb[:,3])/2-W[1],bb[:,2]-bb[:,0],bb[:,3]-bb[:,1]],-1).unsqueeze(0)
    xy=C.grid_positions()[None,:,None,:]
    outside=(loc[:,None,:,:2]-xy).clamp_min(0.)+(xy-loc[:,None,:,2:]).clamp_min(0.)
    wh=loc[...,2:]-loc[...,:2]
    geom=(outside/wh[:,None]).sum(-1)
    cls2=cls.unsqueeze(0); valid=torch.ones(1,n,dtype=torch.bool)
    for k in ks:
        kap=torch.full((10,),float(k))
        plan=C.independent_cap(mass,geom,kap[cls2],cls2,valid)
        allm=float(plan['residual'].sum()/mass.sum())
        # class-restricted: residual only over classes present in candidates
        present=torch.unique(cls)
        rm=plan['residual'][0,present].sum(); mm=mass[0,present].sum()
        res[str(k)]['all_mass'].append(allm); res[str(k)]['class_mass'].append(float(rm/mm))
out={k:{kk: round(sum(vv)/len(vv),4) for kk,vv in d.items()} for k,d in res.items()}
print(json.dumps(out,indent=1))
print('TARGET (317v4): -6.7:0.9989 -0.67:0.7920 -0.067:0.7489 0:0.7445 0.067:0.7402 0.67:0.7067 6.7:0.6315')