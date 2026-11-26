import os, sys, json, torch
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OMP_NUM_THREADS"]="2"
sys.dont_write_bytecode=True
PKG=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\单一候选评价依赖补齐与构建核验_20261008_v1\isolated_runtime"
sys.path.insert(0,PKG)
import torch, numpy as np; torch.set_num_threads(2)
from pycocoeval.coco import COCO
from pycocoeval.cocoeval import COCOeval
from uic_fusion import fuse_and_postprocess, torchvision_box_convert
sys.path.insert(0,r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import uic_fusion
torch.manual_seed(2026102104)
C=10
B=2
gl=torch.randn(B,300,C)*3; gb=torch.rand(B,300,4)*0.5+0.25
nl=torch.full((B,4,C),-12.0); nb=torch.rand(B,4,4)*0.5+0.25  # new4 all <0.001
orig=torch.tensor([[800.,1200.],[640.,480.]])
out,_=uic_fusion.fuse_and_postprocess(gl,gb,nl,nb,orig)
# PostProcessor reference
def manual(pl,pb,orig,C=10,topq=300):
    cx,cy,w,h=pb.unbind(-1); b=torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],-1)*orig.repeat(1,2).unsqueeze(1)
    s=torch.sigmoid(pl); sv,idx=torch.topk(s.flatten(1),topq,dim=-1); lab=idx%C; bi=idx//C
    return lab,b.gather(1,bi.unsqueeze(-1).repeat(1,1,4)),sv
lab,box,sco=manual(gl,gb,orig)

def to_xywh(b):
    x0,y0,x1,y1=b.unbind(-1); return torch.stack([x0,y0,x1-x0,y1-y0],-1)

img_ids=[1,2]
gt_anns=[]; aid=1
for b in range(B):
    bx=box[b][:30]; lb=lab[b][:30]
    xywh=to_xywh(bx)
    for k in range(30):
        x,y,w,h=xywh[k].tolist()
        gt_anns.append({"id":aid,"image_id":img_ids[b],"category_id":int(lb[k])+1,"bbox":[x,y,w,h],"area":w*h,"iscrowd":0}); aid+=1
gt={"images":[{"id":i,"width":int(orig[b,1]),"height":int(orig[b,0])} for b,i in enumerate(img_ids)],
    "categories":[{"id":c+1,"name":str(c)} for c in range(C)],"annotations":gt_anns}
coco=COCO(); coco.dataset=gt; coco.createIndex()

def dets_from(res_list):
    rows=[]
    for b,i in enumerate(img_ids):
        o=res_list[b]; xywh=to_xywh(o["boxes"])
        for k in range(o["labels"].shape[0]):
            x,y,w,h=xywh[k].tolist()
            rows.append({"image_id":i,"category_id":int(o["labels"][k])+1,"bbox":[x,y,w,h],"score":float(o["scores"][k])})
    return rows

def run_ap(dets, tiny=False):
    ev=COCOeval(coco, coco.loadRes(dets), 'bbox', using_tiny_metrice=tiny, max_dets=100)
    ev.params.imgIds=img_ids
    ev.evaluate(); ev.accumulate()
    old=sys.stdout
    import io; sys.stdout=io.StringIO()
    ev.summarize()
    sys.stdout=old
    return [float(x) for x in ev.stats[:12]]

d_self=dets_from(out)
d_ref=[{"image_id":img_ids[b],"category_id":int(lab[b][k])+1,"bbox":to_xywh(box[b][k:k+1])[0].tolist(),"score":float(sco[b][k])} for b in range(B) for k in range(300)]
ap_self=run_ap(d_self); ap_ref=run_ap(d_ref)
res={"item":"补做2 任务3② 绑定评估库对照","evaluator":{"package_root":PKG,"cocoeval_sha":"288eab6c...","backend":"_mask.cp310-win_amd64.pyd (5a49c132...)"},
 "max_dets_consumed_at":["pycocoeval/cocoeval.py L60 __init__(max_dets=100)","L666 self.maxDets=[1,10,self.max_dets]","L221-222 computeIoU truncation"],
 "score_threshold":"绑定库内无 score_threshold 键；评估不过滤 score",
 "ap_selfbuilt":ap_self,"ap_reference":ap_ref,
 "max_abs_diff":max(abs(a-b) for a,b in zip(ap_self,ap_ref)),
 "all_close_1e-9":all(abs(a-b)<1e-9 for a,b in zip(ap_self,ap_ref)),
 "note":"new4 全 <0.001；自建后处理 vs PostProcessor(top300) 喂同一 pycocoeval.COCOeval"}
json.dump(res,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_redo2_bound_evaluator.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(res,ensure_ascii=False,indent=2)[:4000])