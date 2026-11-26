import os, sys, json, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
# compare 339 subset (v3, seed1) against independent recompute on the 338 cache for SAME images
sub=json.load(open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\339_result_subset.json",encoding="utf-8"))
# independent recompute for the same 32 ids (seed1) using 338 cache
P327=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2\327_训练实施输入索引.json"
def complete_in(gt,W):
    return (gt[:,0]>=W[0]-1e-6)&(gt[:,1]>=W[1]-1e-6)&(gt[:,2]<=W[2]+1e-6)&(gt[:,3]<=W[3]+1e-6)
def sel(b):
    cx=((b[:,0]+b[:,2])/2).numpy(); cy=((b[:,1]+b[:,3])/2).numpy()
    best=None;bn=-1
    for y in range(0,640,64):
        for x in range(0,640,64):
            n=int(((cx>=x)&(cx<x+64)&(cy>=y)&(cy<y+64)).sum())
            if n>bn: bn=n;best=(y//64,x//64)
    return best
def norm(b):
    b=b.float()*640.0; cx,cy,w,h=b.unbind(-1); return torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],-1)
rows=[]
for r in sub["tests"]["per_image"]:
    iid=r["image_id"]; p=os.path.join(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1","338_cache_seed1",f"{iid:06d}.pt")
    e=torch.load(p,map_location="cpu",weights_only=True)
    rc=sel(norm(e["final300_boxes"])); W=[rc[1]*64,rc[0]*64,rc[1]*64+64,rc[0]*64+64]
    gt=e["gt_boxes_640"]; ex=e["m0_gt_explained"]
    GW=complete_in(gt,W) if gt.numel() else torch.zeros(0,dtype=torch.bool)
    UW=GW&(~ex) if gt.numel() else torch.zeros(0,dtype=torch.bool)
    rows.append({"image_id":iid,"subset_W":r["W_cell"],"recomputed_W":list(rc),
      "subset_GW":r["n_GW"],"recomp_GW":int(GW.sum()),
      "subset_UW":r["n_UW"],"recomp_UW":int(UW.sum()),
      "match": list(r["W_cell"])==list(rc) and r["n_GW"]==int(GW.sum()) and r["n_UW"]==int(UW.sum())})
ok=sum(1 for x in rows if x["match"])
out={"item":"339 32张子集 vs 主控式独立重算","checked":len(rows),"match":ok,"mismatch":[x for x in rows if not x["match"]]}
json.dump(out,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\339_subset_vs_recompute.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False,indent=1)[:1500])
