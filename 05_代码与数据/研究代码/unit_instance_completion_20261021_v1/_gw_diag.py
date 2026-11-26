import sys, json, os, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache_v2 as S
ANN=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\annotations\train.json"
ann=json.load(open(ANN,encoding="utf-8")); imgs={im["id"]:im for im in ann["images"]}
by={}
for a in ann["annotations"]: by.setdefault(a["image_id"],[]).append(a)
d=json.load(open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\337_result_stage1_subset.json",encoding="utf-8"))
rows=[]
for e in d["tests"]["per_image"]:
    pid=e["image_id"]; im=imgs[pid]; ow,oh=im["width"],im["height"]
    items=by.get(pid,[])
    xywh=torch.tensor([a["bbox"] for a in items],dtype=torch.float32)
    gt=S.orig_xywh_to_640_xyxy(xywh,ow,oh)
    W=S.window_xyxy(tuple(e["W_cell"]))
    # how many GT intersect W (not just complete)
    n_inter=0
    for i in range(gt.shape[0]):
        g=gt[i]
        ix=min(float(g[2]),float(W[2]))-max(float(g[0]),float(W[0]))
        iy=min(float(g[3]),float(W[3]))-max(float(g[1]),float(W[1]))
        if ix>0 and iy>0: n_inter+=1
    rows.append({"id":pid,"n_gt":len(items),"n_inter_W":n_inter,"n_GW":e["n_GW"],"n_UW":e["n_UW"]})
json.dump(rows,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\_gw_diag.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("images with GW=0 but GT intersecting W:", sum(1 for r in rows if r["n_GW"]==0 and r["n_inter_W"]>0))
print("images with no GT intersecting W:", sum(1 for r in rows if r["n_inter_W"]==0))
for r in rows:
    if r["n_GW"]==0 and r["n_inter_W"]>0: print("  intersect-but-none-complete:",r)
