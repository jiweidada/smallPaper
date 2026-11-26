import sys, json, os, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache_v2 as S
ANN=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\annotations\train.json"
ann=json.load(open(ANN,encoding="utf-8")); imgs={im["id"]:im for im in ann["images"]}
by={}
for a in ann["annotations"]: by.setdefault(a["image_id"],[]).append(a)
# For image 4113 selected window (4,0) -> x 0..64,y 0..64(rc means r=4,c=0 -> x=0..64, y=256..320)
pid=4113; im=imgs[pid]; ow,oh=im["width"],im["height"]
items=by[pid]
xywh=torch.tensor([a["bbox"] for a in items],dtype=torch.float32)
gt=S.orig_xywh_to_640_xyxy(xywh,ow,oh)
W=S.window_xyxy((4,0))
print("W",W.tolist())
inside=0; reasons={}
for i in range(gt.shape[0]):
    g=gt[i].tolist()
    ok=S.gt_complete_in_window(g,W)
    if ok: inside+=1
print("GT fully inside selected W:",inside,"of",gt.shape[0])
# how many GT intersect but not fully
inter=0
for i in range(gt.shape[0]):
    g=gt[i]
    ix=min(g[2],W[2])-max(g[0],W[0]); iy=min(g[3],W[3])-max(g[1],W[1])
    if ix>0 and iy>0: inter+=1
print("GT intersecting W:",inter)
