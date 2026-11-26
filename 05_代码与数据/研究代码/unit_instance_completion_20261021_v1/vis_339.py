import os, sys, json, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache_v3 as S
from PIL import Image, ImageDraw
torch.set_num_threads(2)
OUT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
ANN=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\annotations\train.json"
IMGDIR=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\images"
CKPT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1\seed1\run\last.pth"
P327=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2\327_训练实施输入索引.json"
ann=json.load(open(ANN,encoding="utf-8")); imgs={im["id"]:im for im in ann["images"]}
by={}
for a in ann["annotations"]: by.setdefault(a["image_id"],[]).append(a)
order=json.load(open(P327,encoding="utf-8"))["sample_plan"]["development_ids"]
# fixed seed pick 6 from those with G_W>0 in 339 cache seed1
import random
random.seed(2026102101)
cands=[]
d=os.path.join(OUT,"339_cache_seed1")
for iid in order:
    p=os.path.join(d,f"{iid:06d}.pt")
    e=torch.load(p,map_location="cpu",weights_only=True)
    if e["n_G_W"]>0 and e["n_U_W"]>0: cands.append(iid)
pick=random.sample(cands,6)
os.makedirs(os.path.join(OUT,"339_vis"),exist_ok=True)
model=S.load_real_model(CKPT)
info=[]
for pid in pick:
    im=imgs[pid]; pil=Image.open(os.path.join(IMGDIR,im["file_name"])).convert("RGB")
    pb,l=S.global_forward_640(model,pil)
    fullp=S.make_fullp(pb,l); p300=S.norm_cxcywh_to_640_xyxy(pb)
    rc,cnt=S.select_window(p300); W=S.window_xyxy(rc)
    xywh=torch.tensor([a["bbox"] for a in by[pid]],dtype=torch.float32)
    labs=torch.tensor([a["category_id"] for a in by[pid]],dtype=torch.long)
    gt=S.orig_xywh_to_640_xyxy(xywh,im["width"],im["height"])
    tg=S.build_targets_v2(fullp,gt,labs,W)
    canvas=pil.resize((640,640)); dr=ImageDraw.Draw(canvas)
    dr.rectangle([float(W[0]),float(W[1]),float(W[2]),float(W[3])],outline=(0,255,255),width=3)
    for i in range(min(50,fullp["boxes"].shape[0])):
        dr.rectangle([float(v) for v in fullp["boxes"][i].tolist()],outline=(0,200,0),width=1)
    g=gt.tolist(); ex=tg["m0"]["gt_explained"].tolist(); GW=tg["G_W"].tolist()
    for i in range(len(g)):
        col=(160,160,160)
        if GW[i] and ex[i]: col=(0,80,255)
        elif GW[i] and not ex[i]: col=(255,0,0)
        dr.rectangle(g[i],outline=col,width=2)
    pth=os.path.join(OUT,"339_vis",f"339_vis_{pid}.png"); canvas.save(pth)
    info.append({"image_id":pid,"png":pth,"W_cell":list(rc),"n_GW":int(tg["G_W"].sum()),"n_UW":int(tg["U_W"].sum())})
json.dump({"item":"339 目视核对","seed":2026102101,"images":info},open(os.path.join(OUT,"339_result_vis.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(info,ensure_ascii=False,indent=1))
