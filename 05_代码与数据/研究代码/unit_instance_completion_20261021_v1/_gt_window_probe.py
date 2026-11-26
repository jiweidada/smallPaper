import sys, json, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache_v2 as S
ANN=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\annotations\train.json"
IMGDIR=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\images"
ann=json.load(open(ANN,encoding="utf-8")); imgs={im["id"]:im for im in ann["images"]}
by={}
for a in ann["annotations"]: by.setdefault(a["image_id"],[]).append(a)
SP=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\08_匹配聚合覆盖探针_20261019_v1\sample_plan.json"
dev=json.load(open(SP,encoding="utf-8"))["source_pools"]["development"][:32]
# For each image: count GT fully inside EACH of the 100 windows; report max over windows
print("id   n_gt   maxGTfull-per-window   gt640_bbox_range")
for pid in dev[:12]:
    im=imgs[pid]; ow,oh=im["width"],im["height"]
    items=by.get(pid,[])
    if not items: print(pid,0,"no gt"); continue
    xywh=torch.tensor([a["bbox"] for a in items],dtype=torch.float32)
    gt=S.orig_xywh_to_640_xyxy(xywh,ow,oh)
    counts=[]
    for r in range(10):
        for c in range(10):
            W=S.window_xyxy((r,c))
            n=sum(1 for i in range(gt.shape[0]) if S.gt_complete_in_window(gt[i].tolist(),W))
            counts.append(n)
    print(pid, len(items), max(counts), "x[%.0f,%.0f] y[%.0f,%.0f]"%(gt[:,0].min(),gt[:,2].max(),gt[:,1].min(),gt[:,3].max()))
