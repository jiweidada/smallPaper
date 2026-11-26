import os, sys, json, time, hashlib
import numpy as np, torch
sys.dont_write_bytecode=True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache_v2 as S
from PIL import Image

torch.set_num_threads(2)
OUT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
SP=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\08_匹配聚合覆盖探针_20261019_v1\sample_plan.json"
ANN=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\annotations\train.json"
IMGDIR=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\images"
CKPT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1\seed1\run\last.pth"

plan=json.load(open(SP,encoding="utf-8")); dev=plan["source_pools"]["development"]
SUBSET=dev[:32]                     # 337 D: 336-mandated first 32
ann=json.load(open(ANN,encoding="utf-8"))
imgs={im["id"]:im for im in ann["images"]}
by_img={}
for a in ann["annotations"]: by_img.setdefault(a["image_id"],[]).append(a)

res={"item":"337 阶段1 子集重做（坐标修正版）","subset_ids":SUBSET,
     "subset_source":"sample_plan.source_pools.development[:32]（336 要求的前32）",
     "coord_contract":{
       "global_input":"natural RGB -> PIL Resize(640,640) stretch -> float32/255 (296 L45)",
       "pred_boxes":"model pred_boxes = normalized cxcywh in 640 plane -> *640 -> xyxy",
       "gt":"annotation pixel xywh -> xyxy -> *(640/orig_w, 640/orig_h) -> 640 plane",
       "window":"64x64 grid on 640 plane; density=count of ORIGINAL final300 box centers (296 L11)",
       "crop":"W_640 -> orig px by *orig_w/640,*orig_h/640 (296 L45)"},
     "executed":[],"tests":{},"assertions":{}}
model=S.load_real_model(CKPT)
res["executed"].append("load real model seed1")

def get_gt(pid):
    items=by_img.get(pid,[])
    if not items: return torch.zeros(0,4), torch.zeros(0,dtype=torch.long)
    xywh=torch.tensor([a["bbox"] for a in items],dtype=torch.float32)
    labs=torch.tensor([a["category_id"] for a in items],dtype=torch.long)
    return xywh, labs

entries=[]; keep={}
t0=time.time()
for pid in SUBSET:
    im=imgs[pid]
    pil=Image.open(os.path.join(IMGDIR,im["file_name"])).convert("RGB")
    ow,oh=im["width"],im["height"]
    tt=time.time(); pb_norm,logits=S.global_forward_640(model,pil); dt=time.time()-tt
    p300_640=S.norm_cxcywh_to_640_xyxy(pb_norm)           # ALL 300 final boxes in 640 plane
    fullp=S.make_fullp(pb_norm,logits)
    rc,cnt=S.select_window(p300_640)                      # density from ORIGINAL final300
    W=S.window_xyxy(rc)
    xywh,labs=get_gt(pid)
    gt640=S.orig_xywh_to_640_xyxy(xywh,ow,oh) if xywh.numel() else torch.zeros(0,4)
    tg=S.build_targets_v2(fullp,gt640,labs,W)
    # assertions (337 C)
    b=fullp["boxes"]
    xyxy_ok=float(((b[:,2]>b[:,0])&(b[:,3]>b[:,1])).float().mean()) if b.numel() else 1.0
    gt_in=bool(((gt640>= -1e-3)&(gt640<=640+1e-3)).all()) if gt640.numel() else True
    entries.append({"image_id":pid,"orig_wh":[ow,oh],"W_cell":list(rc),
      "W_xyxy_640":[float(x) for x in W.tolist()],
      "W_xyxy_orig":[float(x) for x in S.window_640_to_orig_xyxy(W,ow,oh).tolist()],
      "n_final300":300,"n_fullp":int(b.shape[0]),
      "score_min":float(fullp["scores"].min()) if fullp["scores"].numel() else None,
      "score_max":float(fullp["scores"].max()) if fullp["scores"].numel() else None,
      "n_gt":int(gt640.shape[0]),"n_GW":int(tg["G_W"].sum()),"n_UW":int(tg["U_W"].sum()),
      "m0_coverage":int(tg["m0"]["coverage"]),"xyxy_valid_frac":xyxy_ok,"gt_in_640":gt_in,
      "secs":round(dt,3),"sha_fullp_boxes":S.sha_tensor(b),"sha_logits":S.sha_tensor(logits)})
    keep[pid]=(pb_norm,logits,p300_640,fullp,tg,gt640,labs,W)
res["executed"].append(f"global forward {len(SUBSET)} images (1 each)")
res["assertions"]["all_fullp_xyxy_valid"]=all(e["xyxy_valid_frac"]==1.0 for e in entries)
res["assertions"]["all_gt_in_640"]=all(e["gt_in_640"] for e in entries)
gw_positive=[e["image_id"] for e in entries if e["n_GW"]>0]
uw_positive=[e["image_id"] for e in entries if e["n_UW"]>0]
res["assertions"]["images_with_GW_gt0"]={"count":len(gw_positive),"of":len(entries),"ids":gw_positive}
res["assertions"]["images_with_UW_gt0"]={"count":len(uw_positive),"of":len(entries),"ids":uw_positive}
res["assertions"]["UW_not_all_zero"]=len(uw_positive)>0
res["tests"]["per_image"]=entries
res["timing"]={"mean_secs":round(float(np.mean([e["secs"] for e in entries])),3),
  "total_secs":round(time.time()-t0,2)}
json.dump(res,open(os.path.join(OUT,"337_result_stage1_subset.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps({"assertions":res["assertions"],"timing":res["timing"],
  "sample":[e for e in entries[:4]]},ensure_ascii=False,indent=2)[:3500])
