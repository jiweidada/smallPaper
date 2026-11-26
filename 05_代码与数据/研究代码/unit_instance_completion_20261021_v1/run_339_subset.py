import os, sys, json, time, hashlib
import numpy as np, torch
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache_v3 as S
from PIL import Image
torch.set_num_threads(2)
OUT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
SP=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\08_匹配聚合覆盖探针_20261019_v1\sample_plan.json"
ANN=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\annotations\train.json"
IMGDIR=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\images"
CKPT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1\seed1\run\last.pth"
dev=json.load(open(SP,encoding="utf-8"))["source_pools"]["development"][:32]  # 337 same order
ann=json.load(open(ANN,encoding="utf-8")); imgs={im["id"]:im for im in ann["images"]}
by={}
for a in ann["annotations"]: by.setdefault(a["image_id"],[]).append(a)
model=S.load_real_model(CKPT)
res={"item":"339 32张子集重做（v3修正窗）","subset_ids":dev,"tests":{}}
rows=[]
t0=time.time()
for pid in dev:
    im=imgs[pid]; pil=Image.open(os.path.join(IMGDIR,im["file_name"])).convert("RGB")
    pb,l=S.global_forward_640(model,pil)
    p300=S.norm_cxcywh_to_640_xyxy(pb)
    fullp=S.make_fullp(pb,l)
    rc,cnt=S.select_window(p300)
    W=S.window_xyxy(rc)
    xywh=torch.tensor([a["bbox"] for a in by[pid]],dtype=torch.float32) if by.get(pid) else torch.zeros(0,4)
    labs=torch.tensor([a["category_id"] for a in by[pid]],dtype=torch.long) if by.get(pid) else torch.zeros(0,dtype=torch.long)
    gt=S.orig_xywh_to_640_xyxy(xywh,im["width"],im["height"]) if xywh.numel() else torch.zeros(0,4)
    tg=S.build_targets_v2(fullp,gt,labs,W)
    rows.append({"image_id":pid,"W_cell":list(rc),"W_xyxy":[float(x) for x in W.tolist()],
      "n_gt":int(gt.shape[0]),"n_GW":int(tg["G_W"].sum()),"n_UW":int(tg["U_W"].sum()),
      "m0_coverage":int(tg["m0"]["coverage"]),
      "K_g_sum_ok":bool(torch.allclose(tg["K_g"][tg["G_W"]].sum(1),torch.ones(int(tg["G_W"].sum()))) if tg["G_W"].any() else True)})
gw_pos=sum(1 for r in rows if r["n_GW"]>0); uw_pos=sum(1 for r in rows if r["n_UW"]>0)
res["tests"]["per_image"]=rows
res["tests"]["summary"]={"n":len(rows),"G_W_positive":gw_pos,"U_W_positive":uw_pos,
  "G_W_total":sum(r["n_GW"] for r in rows),"U_W_total":sum(r["n_UW"] for r in rows)}
res["tests"]["assertions"]={"GW_positive_gt_4_of_32":gw_pos>4,"UW_not_all_zero":uw_pos>0,
  "all_Kg_unit":all(r["K_g_sum_ok"] for r in rows)}
res["timing_secs"]=round(time.time()-t0,2)
json.dump(res,open(os.path.join(OUT,"339_result_subset.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(res["tests"]["summary"],ensure_ascii=False,indent=1))
print("assertions:",res["tests"]["assertions"])
print("sample:",json.dumps(rows[:5],ensure_ascii=False))
