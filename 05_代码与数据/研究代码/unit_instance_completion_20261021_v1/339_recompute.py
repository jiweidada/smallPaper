import os, sys, json, math
import numpy as np, torch
sys.dont_write_bytecode=True
OUT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
P327=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2\327_训练实施索引.json"
P327=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2\327_训练实施输入索引.json"
ORDER=json.load(open(P327,encoding="utf-8"))["sample_plan"]["development_ids"]

def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return [round((c-r)/d,6),round((c+r)/d,6)]

def norm_cxcywh_to_640(b):
    b=b.float()*640.0; cx,cy,w,h=b.unbind(-1)
    return torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],-1)

def select_window_correct(boxes640):
    """262 semantics: grid = [(x,y,min(x+64,w),min(y+64,h)) for y in range(0,h,64) for x in range(0,w,64)]
    => iterate y (rows) then x (cols); pick first max in row-major (y then x)."""
    cx=((boxes640[:,0]+boxes640[:,2])/2).numpy()
    cy=((boxes640[:,1]+boxes640[:,3])/2).numpy()
    cnt=np.zeros((10,10),dtype=np.int64)   # [row=y][col=x]  <-- CORRECT
    for x,y in zip(cx,cy):
        if 0<=x<=640 and 0<=y<=640:
            cnt[min(int(y//64),9), min(int(x//64),9)]+=1
    flat=cnt.reshape(-1)
    idx=int(np.argmax(flat)) if flat.max()>0 else 0
    return (idx//10, idx%10), cnt   # (row=y, col=x)

def win_xyxy(rc):
    r,c=rc; x=c*64; y=r*64
    return [x,y,min(x+64,640),min(y+64,640)]

def complete_in(gt,W):
    return (gt[:,0]>=W[0]-1e-6)&(gt[:,1]>=W[1]-1e-6)&(gt[:,2]<=W[2]+1e-6)&(gt[:,3]<=W[3]+1e-6)

res={"item":"339 用落盘缓存独立重算（修正窗）","seeds":{}}
allw={}
for seed in (1,2):
    d=os.path.join(OUT,f"338_cache_seed{seed}")
    rows_gw=0; rows_uw=0; gw_tot=0; uw_tot=0; ov_imgs=0; ov_gt=0
    mism=0; n=0
    for iid in ORDER:
        p=os.path.join(d,f"{iid:06d}.pt")
        if not os.path.exists(p): continue
        e=torch.load(p,map_location="cpu",weights_only=True); n+=1
        b640=norm_cxcywh_to_640(e["final300_boxes"])
        rc,cnt=select_window_correct(b640)
        W=win_xyxy(rc)
        gt=e["gt_boxes_640"]
        ex=e["m0_gt_explained"]
        if gt.numel():
            GW=complete_in(gt,W)
            UW=GW & (~ex)
        else:
            GW=torch.zeros(0,dtype=torch.bool); UW=torch.zeros(0,dtype=torch.bool)
        ng=int(GW.sum()); nu=int(UW.sum())
        gw_tot+=ng; uw_tot+=nu
        if ng>0: rows_gw+=1
        if nu>0: rows_uw+=1
        if nu>4: ov_imgs+=1; ov_gt+=nu-4
        if list(rc)!=list(e["window_cell"]): mism+=1
        allw[(seed,iid)]=rc
    res["seeds"][f"seed{seed}"]={
      "entries":n,
      "G_W_positive":{"count":rows_gw,"rate":round(rows_gw/n,6),"wilson95":wilson(rows_gw,n)},
      "U_W_positive":{"count":rows_uw,"rate":round(rows_uw/n,6),"wilson95":wilson(rows_uw,n)},
      "G_W_total":gw_tot,"U_W_total":uw_tot,
      "U_W_over_G_W":round(uw_tot/gw_tot,6) if gw_tot else None,
      "U_W_gt4":{"images":ov_imgs,"overflow_gt_total":ov_gt},
      "stored_window_differs_from_correct":mism,
      "note":"存储的 window_cell 与本重算不同的数量"}
# cross-seed same window (corrected)
same=sum(1 for iid in ORDER if allw.get((1,iid))==allw.get((2,iid)))
res["cross_seed_same_window"]={"same":same,"n":len(ORDER),"rate":round(same/len(ORDER),6),
  "wilson95":wilson(same,len(ORDER))}
json.dump(res,open(os.path.join(OUT,"339_recompute_from_cache.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(res,ensure_ascii=False,indent=2)[:3500])
