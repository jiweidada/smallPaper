import sys, json, os, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache_v2 as S
from PIL import Image
torch.set_num_threads(2)
ANN=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\annotations\train.json"
IMGDIR=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\images"
CKPT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1\seed1\run\last.pth"
SP=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\08_匹配聚合覆盖探针_20261019_v1\sample_plan.json"
ann=json.load(open(ANN,encoding="utf-8")); imgs={im["id"]:im for im in ann["images"]}
by={}
for a in ann["annotations"]: by.setdefault(a["image_id"],[]).append(a)
dev=json.load(open(SP,encoding="utf-8"))["source_pools"]["development"][:32]
model=S.load_real_model(CKPT)
# Compare: window chosen by prediction density vs window chosen by GT density (diagnostic only!)
stats=[]
for pid in dev:
    im=imgs[pid]; pil=Image.open(os.path.join(IMGDIR,im["file_name"])).convert("RGB")
    pb,l=S.global_forward_640(model,pil); p300=S.norm_cxcywh_to_640_xyxy(pb)
    rc,cnt=S.select_window(p300)
    xywh=torch.tensor([a["bbox"] for a in by[pid]],dtype=torch.float32) if by.get(pid) else torch.zeros(0,4)
    gt=S.orig_xywh_to_640_xyxy(xywh,im["width"],im["height"]) if xywh.numel() else torch.zeros(0,4)
    # GT density per 64px window
    gcnt=np.zeros((10,10),dtype=np.int64)
    for i in range(gt.shape[0]):
        cx=float((gt[i,0]+gt[i,2])/2); cy=float((gt[i,1]+gt[i,3])/2)
        if 0<=cx<=640 and 0<=cy<=640:
            gcnt[min(int(cy//64),9), min(int(cx//64),9)]+=1
    best_gt_cell=tuple(int(v) for v in np.unravel_index(int(np.argmax(gcnt)),(10,10)))
    W=S.window_xyxy(rc)
    nGW=sum(1 for i in range(gt.shape[0]) if S.gt_complete_in_window(gt[i].tolist(),W))
    Wg=S.window_xyxy(best_gt_cell)
    nGWg=sum(1 for i in range(gt.shape[0]) if S.gt_complete_in_window(gt[i].tolist(),Wg))
    stats.append({"id":pid,"pred_density_cell":list(rc),"pred_density":int(cnt[rc]),
      "nGW_at_pred_cell":nGW,"gt_density_cell":list(best_gt_cell),
      "gt_center_count":int(gcnt[best_gt_cell]),"nGW_at_gt_cell":nGWg})
nz=[s for s in stats if s["nGW_at_pred_cell"]>0]
print("pred-window has GW>0:",len(nz),"/",len(stats))
print("GT-window would have GW>0:",sum(1 for s in stats if s["nGW_at_gt_cell"]>0),"/",len(stats))
print("mean nGW pred-window:",round(np.mean([s["nGW_at_pred_cell"] for s in stats]),2),
      "| GT-window:",round(np.mean([s["nGW_at_gt_cell"] for s in stats]),2))
json.dump(stats,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\_window_diag.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
