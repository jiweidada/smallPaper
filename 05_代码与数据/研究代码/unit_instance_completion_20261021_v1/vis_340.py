import os, sys, json, random, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
ANN=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\annotations\train.json"
IMGDIR=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\images"
ann=json.load(open(ANN,encoding="utf-8")); imgs={im["id"]:im for im in ann["images"]}
by={}
for a in ann["annotations"]: by.setdefault(a["image_id"],[]).append(a)
d=os.path.join(OUT,"340_cache_seed1")
files=sorted(os.listdir(d))
cands=[]
for f in files:
    e=torch.load(os.path.join(d,f),map_location="cpu",weights_only=True)
    if e["n_G_W"]>0: cands.append(f)
random.seed(2026102101)
pick=random.sample(cands,min(6,len(cands)))
os.makedirs(os.path.join(OUT,"340_vis"),exist_ok=True)
info=[]
for f in pick:
    e=torch.load(os.path.join(d,f),map_location="cpu",weights_only=True)
    pid=e["image_id"]; im=imgs[pid]
    full=Image.open(os.path.join(IMGDIR,im["file_name"])).convert("RGB")
    crop=e["crop128"].permute(1,2,0).numpy()
    K=e["K_g"]; GW=e["G_W"]; W=[float(v) for v in e["window_xyxy_640"]]
    fig,axes=plt.subplots(1,2,figsize=(11,5.2),dpi=110)
    ax=axes[0]; ax.imshow(crop)
    # K_g heatmap overlaid (sum over G_W targets, reshape 16x16)
    Ksum=K[GW].sum(0).reshape(16,16).numpy() if GW.any() else np.zeros((16,16))
    ax.imshow(Ksum,cmap="jet",alpha=0.45,extent=[0,128,128,0],vmin=0,vmax=max(1e-6,Ksum.max()))
    # GT boxes in crop coords: (gt640 - W0)/Wwidth*128
    ww=W[2]-W[0]; wh=W[3]-W[1]
    for i in range(e["gt_boxes_640"].shape[0]):
        if not bool(GW[i]): continue
        g=e["gt_boxes_640"][i]
        x0=(float(g[0])-W[0])/ww*128; x1=(float(g[2])-W[0])/ww*128
        y0=(float(g[1])-W[1])/wh*128; y1=(float(g[3])-W[1])/wh*128
        ax.add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor="lime",lw=1.5))
    ax.set_title(f"crop128 + K_g heat (GW={int(GW.sum())})"); ax.axis("off")
    ax2=axes[1]; ax2.imshow(full.resize((640,640)))
    ax2.add_patch(plt.Rectangle((W[0],W[1]),W[2]-W[0],W[3]-W[1],fill=False,edgecolor="cyan",lw=2))
    ax2.set_title(f"image {pid}, window {W}"); ax2.axis("off")
    p=os.path.join(OUT,"340_vis",f"340_vis_{pid}.png"); fig.tight_layout(); fig.savefig(p); plt.close(fig)
    info.append({"image_id":pid,"png":p,"n_GW":int(GW.sum()),"n_UW":int(e["n_U_W"]),
                 "Ksum_max":round(float(Ksum.max()),4)})
json.dump({"item":"340 目视核对","seed":2026102101,"images":info},open(os.path.join(OUT,"340_result_vis.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(info,ensure_ascii=False,indent=1))
