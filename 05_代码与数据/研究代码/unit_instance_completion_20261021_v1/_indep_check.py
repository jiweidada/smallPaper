import sys, json, os, torch, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache_v2 as S
from PIL import Image, ImageDraw
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
res=json.load(open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\337_result_stage1_subset.json",encoding="utf-8"))
res.setdefault("tests",{})
# independent numpy legal-edge / coverage check for each image
def iou(pred,gt):
    lt=np.maximum(pred[:,None,:2],gt[None,:,:2]); rb=np.minimum(pred[:,None,2:],gt[None,:,2:])
    wh=np.clip(rb-lt,0,None); inter=wh[...,0]*wh[...,1]
    ap=np.clip(pred[:,2]-pred[:,0],0,None)*np.clip(pred[:,3]-pred[:,1],0,None)
    ag=np.clip(gt[:,2]-gt[:,0],0,None)*np.clip(gt[:,3]-gt[:,1],0,None)
    return inter/np.maximum(ap[:,None]+ag[None,:]-inter,1e-12)
cmp_rows=[]
for e in res["tests"]["per_image"]:
    pid=e["image_id"]; im=imgs[pid]; ow,oh=im["width"],im["height"]
    pil=Image.open(os.path.join(IMGDIR,im["file_name"])).convert("RGB")
    pb,l=S.global_forward_640(model,pil)
    fullp=S.make_fullp(pb,l)
    xywh=torch.tensor([a["bbox"] for a in by[pid]],dtype=torch.float32) if by.get(pid) else torch.zeros(0,4)
    labs=torch.tensor([a["category_id"] for a in by[pid]],dtype=torch.long) if by.get(pid) else torch.zeros(0,dtype=torch.long)
    gt=S.orig_xywh_to_640_xyxy(xywh,ow,oh) if xywh.numel() else torch.zeros(0,4)
    W=S.window_xyxy(tuple(e["W_cell"]))
    tg=S.build_targets_v2(fullp,gt,labs,W)
    # independent legal edges
    if fullp["boxes"].shape[0] and gt.shape[0]:
        I=iou(fullp["boxes"].numpy(),gt.numpy())
        same=fullp["labels"].numpy()[:,None]==labs.numpy()[None,:]
        legal=same&(I>=0.5)
        n_legal=int(legal.sum())
    else:
        n_legal=0
    cmp_rows.append({"image_id":pid,"m0_coverage":int(tg["m0"]["coverage"]),
                     "independent_legal_edges":n_legal,
                     "coverage_le_legal_edges":int(tg["m0"]["coverage"])<=n_legal,
                     "n_GW":int(tg["G_W"].sum()),"n_UW":int(tg["U_W"].sum())})
res["tests"]["independent_numpy_check"]={"rows":cmp_rows,
  "all_coverage_le_legal":all(r["coverage_le_legal_edges"] for r in cmp_rows),
  "note":"独立 numpy 计算同类 IoU>=0.5 合法边总数，M0 覆盖不得超过它（必要条件）；M0 的 lex 最优性由 uic_m0 单测保证"}
json.dump(res,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\337_result_stage1_subset.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps({k:res["tests"]["independent_numpy_check"][k] for k in ["all_coverage_le_legal"]},indent=1))
for r in cmp_rows[:8]: print(r)
