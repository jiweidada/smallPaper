import sys, json, os, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache_v2 as S
from PIL import Image, ImageDraw
torch.set_num_threads(2)
ANN=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\annotations\train.json"
IMGDIR=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\images"
CKPT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1\seed1\run\last.pth"
ann=json.load(open(ANN,encoding="utf-8")); imgs={im["id"]:im for im in ann["images"]}
by={}
for a in ann["annotations"]: by.setdefault(a["image_id"],[]).append(a)
model=S.load_real_model(CKPT)
PICK=[491,1573,1983]
outdir=os.path.join(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1","337_vis"); os.makedirs(outdir,exist_ok=True)
info=[]
for pid in PICK:
    im=imgs[pid]; ow,oh=im["width"],im["height"]
    pil=Image.open(os.path.join(IMGDIR,im["file_name"])).convert("RGB")
    pb,l=S.global_forward_640(model,pil)
    fullp=S.make_fullp(pb,l)
    p300=S.norm_cxcywh_to_640_xyxy(pb)
    rc,cnt=S.select_window(p300); W=S.window_xyxy(rc)
    xywh=torch.tensor([a["bbox"] for a in by[pid]],dtype=torch.float32)
    labs=torch.tensor([a["category_id"] for a in by[pid]],dtype=torch.long)
    gt=S.orig_xywh_to_640_xyxy(xywh,ow,oh)
    tg=S.build_targets_v2(fullp,gt,labs,W)
    # draw on 640x640 rendered image (stretch to match native input)
    canvas=pil.resize((640,640))
    dr=ImageDraw.Draw(canvas)
    # window
    dr.rectangle([float(W[0]),float(W[1]),float(W[2]),float(W[3])],outline=(0,255,255),width=3)
    # fullP top50 boxes (thin green)
    for i in range(min(50,fullp["boxes"].shape[0])):
        b=fullp["boxes"][i].tolist()
        dr.rectangle(b,outline=(0,200,0),width=1)
    # GT: complete-in-W explained (blue), complete-in-W unexplained (red), other GT (gray)
    g=gt.tolist(); expl=tg["m0"]["gt_explained"].tolist(); GW=tg["G_W"].tolist()
    for i in range(len(g)):
        b=g[i]
        col=(160,160,160)
        if GW[i] and expl[i]: col=(0,80,255)
        elif GW[i] and not expl[i]: col=(255,0,0)
        dr.rectangle(b,outline=col,width=2)
    p=os.path.join(outdir,f"337_vis_{pid}.png"); canvas.save(p)
    info.append({"image_id":pid,"png":p,"W_cell":list(rc),"n_fullp":int(fullp["boxes"].shape[0]),
      "n_gt":int(gt.shape[0]),"n_GW":int(tg["G_W"].sum()),"n_UW":int(tg["U_W"].sum()),
      "colors":"cyan=W, green=fullP top50, blue=GW explained, red=GW unexplained(U_W), gray=other GT"})
res={"item":"337 可视化核对","images":info}
json.dump(res,open(os.path.join(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1","337_result_vis.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(res,ensure_ascii=False,indent=2))
