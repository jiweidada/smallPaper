import os, sys, json, math
import numpy as np, torch
sys.dont_write_bytecode=True
OUT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
P327=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2\327_训练实施输入索引.json"
ORDER=json.load(open(P327,encoding="utf-8"))["sample_plan"]["development_ids"]
res=json.load(open(os.path.join(OUT,"339_recompute_from_cache.json"),encoding="utf-8"))
# distribution + diagnostics under corrected window
def complete_in(gt,W):
    return (gt[:,0]>=W[0]-1e-6)&(gt[:,1]>=W[1]-1e-6)&(gt[:,2]<=W[2]+1e-6)&(gt[:,3]<=W[3]+1e-6)
def norm(b):
    b=b.float()*640.0; cx,cy,w,h=b.unbind(-1); return torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],-1)
def selw(b):
    cx=((b[:,0]+b[:,2])/2).numpy(); cy=((b[:,1]+b[:,3])/2).numpy()
    cnt=np.zeros((10,10),dtype=np.int64)
    for x,y in zip(cx,cy):
        if 0<=x<=640 and 0<=y<=640: cnt[min(int(y//64),9),min(int(x//64),9)]+=1
    idx=int(np.argmax(cnt.reshape(-1))) if cnt.max()>0 else 0
    return (idx//10,idx%10),cnt
def win(rc):
    r,c=rc; x=c*64; y=r*64; return [x,y,min(x+64,640),min(y+64,640)]
def dist(vals):
    return {"0":sum(1 for v in vals if v==0),"1":sum(1 for v in vals if v==1),
            "2":sum(1 for v in vals if v==2),"3-5":sum(1 for v in vals if 3<=v<=5),
            "6+":sum(1 for v in vals if v>=6)}
for seed in (1,2):
    d=os.path.join(OUT,f"338_cache_seed{seed}")
    gws=[];uws=[];gtwin=[]
    for iid in ORDER:
        p=os.path.join(d,f"{iid:06d}.pt")
        if not os.path.exists(p): continue
        e=torch.load(p,map_location="cpu",weights_only=True)
        b=norm(e["final300_boxes"]); rc,cnt=selw(b); W=win(rc)
        gt=e["gt_boxes_640"]; ex=e["m0_gt_explained"]
        GW=complete_in(gt,W) if gt.numel() else torch.zeros(0,dtype=torch.bool)
        UW=GW&(~ex) if gt.numel() else torch.zeros(0,dtype=torch.bool)
        gws.append(int(GW.sum())); uws.append(int(UW.sum()))
        # GT-density counterfactual (diagnostic only)
        if gt.numel():
            g=gt.numpy(); gx=(g[:,0]+g[:,2])/2; gy=(g[:,1]+g[:,3])/2
            gc=np.zeros((10,10),dtype=np.int64)
            for x,y in zip(gx,gy):
                if 0<=x<=640 and 0<=y<=640: gc[min(int(y//64),9),min(int(x//64),9)]+=1
            r2,c2=np.unravel_index(int(np.argmax(gc)),(10,10))
            W2=[c2*64,r2*64,min(c2*64+64,640),min(r2*64+64,640)]
            gtwin.append(int(complete_in(gt,W2).sum()))
        else: gtwin.append(0)
    res["seeds"][f"seed{seed}"]["G_W_per_image_dist"]=dist(gws)
    res["seeds"][f"seed{seed}"]["U_W_per_image_dist"]=dist(uws)
    res["seeds"][f"seed{seed}"]["gt_density_window_diagnostic"]={
        "G_W_positive_imgs":sum(1 for v in gtwin if v>0),"of":len(gtwin),
        "rate":round(sum(1 for v in gtwin if v>0)/len(gtwin),6)}
json.dump(res,open(os.path.join(OUT,"339_recompute_from_cache.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps({s:{k:res["seeds"][s][k] for k in ["G_W_per_image_dist","U_W_per_image_dist","gt_density_window_diagnostic"]} for s in ["seed1","seed2"]},ensure_ascii=False,indent=1))
