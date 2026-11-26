import sys, json, os, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache_v2 as S
from PIL import Image
torch.set_num_threads(2)
ANN=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\annotations\train.json"
IMGDIR=r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\images"
CKPT=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1\seed1\run\last.pth"
ann=json.load(open(ANN,encoding="utf-8")); imgs={im["id"]:im for im in ann["images"]}
by={}
for a in ann["annotations"]: by.setdefault(a["image_id"],[]).append(a)
model=S.load_real_model(CKPT)
def iou(pred, gt):  # pred[N,4], gt[M,4] -> [N,M]
    lt=np.maximum(pred[:,None,:2], gt[None,:,:2]); rb=np.minimum(pred[:,None,2:], gt[None,:,2:])
    wh=np.clip(rb-lt,0,None); inter=wh[...,0]*wh[...,1]
    ap=np.clip(pred[:,2]-pred[:,0],0,None)*np.clip(pred[:,3]-pred[:,1],0,None)
    ag=np.clip(gt[:,2]-gt[:,0],0,None)*np.clip(gt[:,3]-gt[:,1],0,None)
    return inter/np.maximum(ap[:,None]+ag[None,:]-inter,1e-12)
for pid in [4113,269,5429]:
    im=imgs[pid]; pil=Image.open(os.path.join(IMGDIR,im["file_name"])).convert("RGB")
    pb,l=S.global_forward_640(model,pil)
    b=S.norm_cxcywh_to_640_xyxy(pb).numpy()
    xywh=torch.tensor([a["bbox"] for a in by[pid]],dtype=torch.float32)
    gt=S.orig_xywh_to_640_xyxy(xywh,im["width"],im["height"]).numpy()
    I=iou(b,gt); mx=I.max(0)
    print(pid,"GT maxIoU mean",round(float(mx.mean()),3),"median",round(float(np.median(mx)),3),
          "frac>=0.5",round(float((mx>=0.5).mean()),3),"best",round(float(mx.max()),3))
