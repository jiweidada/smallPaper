import os, sys, json, time, hashlib
import numpy as np, torch
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache_v3 as S3
import stage1_cache_v4 as S4
from PIL import Image
torch.set_num_threads(2)
OUT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
SP=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\08_匹配聚合覆盖探针_20261019_v1\sample_plan.json"
ANN=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\annotations\train.json"
IMGDIR=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\images"
CKPT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1\seed1\run\last.pth"
dev=json.load(open(SP,encoding="utf-8"))["source_pools"]["development"][:32]
ann=json.load(open(ANN,encoding="utf-8")); imgs={im["id"]:im for im in ann["images"]}
by={}
for a in ann["annotations"]: by.setdefault(a["image_id"],[]).append(a)
model=S3.load_real_model(CKPT)
def sha_t(t): return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()
rows=[]; mism=[]; t0=time.time()
d_out=os.path.join(OUT,"340_cache_seed1"); os.makedirs(d_out,exist_ok=True)
for pid in dev:
    im=imgs[pid]; pil=Image.open(os.path.join(IMGDIR,im["file_name"])).convert("RGB")
    pb,l=S3.global_forward_640(model,pil)
    p300=S3.norm_cxcywh_to_640_xyxy(pb); fullp=S3.make_fullp(pb,l)
    rc,cnt=S3.select_window(p300); W=S3.window_xyxy(rc)
    xywh=torch.tensor([a["bbox"] for a in by[pid]],dtype=torch.float32) if by.get(pid) else torch.zeros(0,4)
    labs=torch.tensor([a["category_id"] for a in by[pid]],dtype=torch.long) if by.get(pid) else torch.zeros(0,dtype=torch.long)
    gt=S3.orig_xywh_to_640_xyxy(xywh,im["width"],im["height"]) if xywh.numel() else torch.zeros(0,4)
    tg=S3.build_targets_v2(fullp,gt,labs,W)   # gives G_W/U_W/M0 (unchanged)
    # NEW: window-local K_g
    K=torch.zeros(gt.shape[0],256)
    for i in range(gt.shape[0]):
        if bool(tg["G_W"][i]):
            x0,y0,x1,y1=[float(v) for v in gt[i]]
            K[i]=torch.from_numpy(S4.bilinear_scatter_window([(x0+x1)/2,(y0+y1)/2,1,1],W)).float()
    crop=S4.crop128_from_window(pil,W,(im["width"],im["height"]))
    e={"image_id":pid,"orig_wh":[im["width"],im["height"]],"file_name":im["file_name"],
       "final300_boxes":pb,"final300_logits":l,
       "fullp_boxes":fullp["boxes"],"fullp_scores":fullp["scores"],"fullp_labels":fullp["labels"],
       "fullp_query_index":fullp["query_index"],"fullp_logits_row":fullp["logits_row"],
       "window_cell":list(rc),"window_xyxy_640":W,
       "G_W":tg["G_W"],"U_W":tg["U_W"],"m0_gt_explained":tg["m0"]["gt_explained"],
       "m0_cand_owner":tg["m0"]["cand_owner"],"m0_coverage":int(tg["m0"]["coverage"]),
       "K_g":K,"gt_boxes_640":gt,"gt_labels":labs,
       "n_gt":int(gt.shape[0]),"n_G_W":int(tg["G_W"].sum()),"n_U_W":int(tg["U_W"].sum()),
       "crop128":crop,"crop_sha":sha_t(crop)}
    torch.save(e,os.path.join(d_out,f"{pid:06d}.pt"))
    # parity vs 339 for non-K_g fields
    old=torch.load(os.path.join(OUT,"339_cache_seed1",f"{pid:06d}.pt"),map_location="cpu",weights_only=True)
    same=all(torch.equal(e[k],old[k]) for k in ["fullp_boxes","fullp_scores","G_W","U_W","gt_boxes_640","final300_boxes","final300_logits"])
    same=same and e["window_cell"]==old["window_cell"] and e["n_G_W"]==old["n_G_W"] and e["n_U_W"]==old["n_U_W"]
    if not same: mism.append(pid)
    rows.append({"image_id":pid,"n_GW":e["n_G_W"],"n_UW":e["n_U_W"],
      "kg_nz":int((K>0).sum()),"kg_sum_ok":bool(torch.allclose(K[tg["G_W"]].sum(1),torch.ones(int(tg["G_W"].sum()))) if tg["G_W"].any() else True),
      "crop_shape":list(crop.shape),"crop_sha":e["crop_sha"][:16]})
res={"item":"340 第一批：32张子集缓存（K_g窗内+crop）","n":len(rows),
 "per_image":rows,"parity_vs_339":{"checked":len(rows),"mismatch":mism,"all_same":len(mism)==0},
 "secs":round(time.time()-t0,2)}
json.dump(res,open(os.path.join(OUT,"340_result_cache_subset.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps({k:res[k] for k in ["n","parity_vs_339","secs"]},ensure_ascii=False,indent=1))
print("kg_sum_all_ok:",all(r["kg_sum_ok"] for r in rows))
print("sample:",rows[:3])
