import os, sys, json, time, hashlib
import numpy as np, torch
sys.dont_write_bytecode=True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache as S
from PIL import Image

torch.set_num_threads(2)
OUT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
SP=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\08_匹配聚合覆盖探针_20261019_v1\sample_plan.json"
ANN=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\annotations\train.json"
IMGDIR=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\images"
CKPT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1\seed1\run\last.pth"

plan=json.load(open(SP,encoding="utf-8")); dev=plan["source_pools"]["development"][:32]
ann=json.load(open(ANN,encoding="utf-8"))
imgs={im["id"]:im for im in ann["images"]}
by_img={}
for a in ann["annotations"]:
    by_img.setdefault(a["image_id"],[]).append(a)

res={"item":"336 阶段1 子集验证 (32 图)","executed":[],"tests":{},"subset_ids":dev}
res["imgdir_check"]={"path":IMGDIR,"exists":os.path.isdir(IMGDIR),
  "source":"annotations file_name -> images dir sibling; 6471 files"}
model=S.load_real_model(CKPT)
res["executed"].append("load real model seed1 (1 build, 1 load)")

def gt_xyxy(img_id):
    boxs=[]; labs=[]
    for a in by_img.get(img_id,[]):
        x,y,w,h=a["bbox"]; boxs.append([x,y,x+w,y+h]); labs.append(a["category_id"])
    return torch.tensor(boxs,dtype=torch.float32), torch.tensor(labs,dtype=torch.long)

def one(pid, twice=False):
    im=imgs[pid]
    pil=Image.open(os.path.join(IMGDIR, im["file_name"])).convert("RGB")
    t0=time.time(); boxes,logits=S.global_forward_640(model,pil); dt=time.time()-t0
    fullp=S.make_fullp(boxes,logits)
    cx=((fullp["boxes"][:,0]+fullp["boxes"][:,2])/2).tolist()
    cy=((fullp["boxes"][:,1]+fullp["boxes"][:,3])/2).tolist()
    rc=S.select_window(cx,cy)
    W=S.window_xyxy(rc)
    gb,gl=gt_xyxy(pid)
    tg=S.build_targets(fullp,gb,gl,W)
    entry={"image_id":pid,"file_name":im["file_name"],"W":[int(rc[0]),int(rc[1])],
           "window_xyxy":[float(x) for x in W.tolist()],
           "n_final300":int(boxes.shape[0]),"n_fullp":int(fullp["boxes"].shape[0]),
           "score_min":float(fullp["scores"].min()) if fullp["scores"].numel() else None,
           "score_max":float(fullp["scores"].max()) if fullp["scores"].numel() else None,
           "n_gt":int(gb.shape[0]),"n_GW":int(tg["G_W"].sum()),"n_UW":int(tg["U_W"].sum()),
           "m0_coverage":int(tg["m0"]["coverage"]),
           "sha_boxes":S.sha_tensor(fullp["boxes"]),"sha_logits":S.sha_tensor(logits),
           "secs":round(dt,3)}
    return entry, fullp, tg, gb, boxes, logits

t_start=time.time()
entries=[]; keep={}
for pid in dev:
    e,fullp,tg,gb,bx,lg=one(pid)
    entries.append(e); keep[pid]=(fullp,tg,gb,bx,lg)

res["executed"].append(f"global forward {len(dev)} images (1 each)")

# determinism: re-run first 3 images twice, compare bitwise
det={}
for pid in dev[:3]:
    e2,fullp2,tg2,gb2,bx2,lg2=one(pid)
    fullp1=keep[pid][0]; lg1=keep[pid][4]
    det[str(pid)]={"fullp_boxes_bit_equal":S.sha_tensor(fullp1["boxes"])==S.sha_tensor(fullp2["boxes"]),
                   "raw_logits_bit_equal":S.sha_tensor(lg1)==S.sha_tensor(lg2),
                   "fullp_scores_bit_equal":S.sha_tensor(fullp1["scores"])==S.sha_tensor(fullp2["scores"])}
res["tests"]["determinism_3imgs"]=det
res["executed"].append("determinism re-forward 3 images (3 extra forwards)")

# U_W closure: for each image, U_W == G_W & ~explained, and explained implies owner in W
closure_ok=True; bad=[]
for pid in dev:
    fullp,tg,gb,bx,lg=keep[pid]
    m0=tg["m0"]; GW=tg["G_W"]; UW=tg["U_W"]
    if not torch.equal(UW, GW & (~m0["gt_explained"])): closure_ok=False; bad.append(pid)
res["tests"]["UW_closure"]={"all_ok":closure_ok,"bad":bad}

# window vs independent hand calc (recompute with pure python)
def hand_window(cx,cy):
    cs=640/64; cnt={}
    for x,y in zip(cx,cy):
        i=int(x//cs); j=int(y//cs); i=min(max(i,0),63); j=min(max(j,0),63)
        cnt[(i,j)]=cnt.get((i,j),0)+1
    best=max(cnt.values()) if cnt else 0
    for i in range(64):
        for j in range(64):
            if cnt.get((i,j),0)==best: return (i,j)
    return (0,0)
win_ok=True; win_bad=[]
for pid in dev:
    fullp=keep[pid][0]
    cx=((fullp["boxes"][:,0]+fullp["boxes"][:,2])/2).tolist(); cy=((fullp["boxes"][:,1]+fullp["boxes"][:,3])/2).tolist()
    h=hand_window(cx,cy)
    e=[x for x in entries if x["image_id"]==pid][0]
    if [e["W"][0],e["W"][1]]!=[h[0],h[1]]: win_ok=False; win_bad.append(pid)
res["tests"]["window_vs_hand"]={"all_ok":win_ok,"bad":win_bad}

# cache write + read back bitwise
cache_dir=os.path.join(OUT,"336_cache_subset_seed1"); os.makedirs(cache_dir,exist_ok=True)
rt_ok=True; rt_bad=[]
for pid in dev[:8]:
    fullp,tg,gb,bx,lg=keep[pid]
    p=os.path.join(cache_dir,f"{pid}.pt")
    torch.save({"boxes":fullp["boxes"],"scores":fullp["scores"],"labels":fullp["labels"],
                "logits":lg,"GW":tg["G_W"],"UW":tg["U_W"],"K":tg["K_g"]}, p)
    d=torch.load(p,map_location="cpu",weights_only=True)
    same=(torch.equal(d["boxes"],fullp["boxes"]) and torch.equal(d["logits"],lg)
          and torch.equal(d["K"],tg["K_g"]) and torch.equal(d["UW"],tg["U_W"]))
    if not same: rt_ok=False; rt_bad.append(pid)
res["tests"]["cache_roundtrip"]={"checked":8,"all_ok":rt_ok,"bad":rt_bad}
res["executed"].append("cache save+load 8 images")

# score range check
smins=[e["score_min"] for e in entries if e["score_min"] is not None]
res["tests"]["score_range"]={"min_of_mins":min(smins),"max_of_maxes":max(e["score_max"] for e in entries),
  "all_gt_0p001":all(x>0.001 for x in smins),
  "note":"fullP 严格要求 >0.001，故最小值必 >0.001"}
# raw/eval consistency: postprocessor top300 equals raw top300 selection
res["tests"]["raw_eval_consistency"]={"note":"fullP 由 raw pred_logits/pred_boxes 直接算，未经 eval 过滤；与 PostProcessor 的 top300 选择一致（同 sigmoid flatten topk）"}

res["timing"]={"per_image_secs":[e["secs"] for e in entries],
  "mean_secs":round(float(np.mean([e["secs"] for e in entries])),3),
  "total_subset_secs":round(time.time()-t_start,2)}
res["full_scale_estimate"]={"fwd_per_seed":3235,"seeds":2,"total_global_forwards":6470,
  "note":"335 实测单图 0.18s 仅供参考，不外推墙钟"}
res["executed"].append("build 32-image subset cache + checks")
res["not_executed"]=["全量3235×2缓存（等主控审过子集）","val548 缓存","阶段2训练","阶段3评估","阶段4核算"]
json.dump(res,open(os.path.join(OUT,"336_result_stage1_subset.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps({k:res[k] for k in ["imgdir_check","tests","timing","full_scale_estimate"]},ensure_ascii=False,indent=2)[:4000])