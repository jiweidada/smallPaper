import sys, json, torch
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"); sys.path.insert(0, r"C:\Users\jiwei\Desktop\DEIM")
torch.set_num_threads(2)
import numpy as np
from uic_fusion import fuse_and_postprocess, native_postprocess, torchvision_box_convert
from engine.deim.postprocessor import PostProcessor

torch.manual_seed(2026102103)
res={"item":"任务3 global300+new4 融合 parity","executed":[],"tests":{}}

B,Q,C=2,300,10
gl=torch.randn(B,Q,C)*3
gb=torch.rand(B,Q,4)*0.5+0.25
nl=torch.randn(B,4,C)*3
nb=torch.rand(B,4,4)*0.5+0.25
orig=torch.tensor([[800.,1200.],[640.,480.]])  # (h,w)

# --- layer1: no new4 (new4 scores all <0.001) -> top300 selection identical to PostProcessor ---
# make new4 logits very negative so sigmoid<0.001
nl_neg=torch.full((B,4,C),-12.0)  # sigmoid(-12)=6e-6 <0.001
out,pre=fuse_and_postprocess(gl,gb,nl_neg,nb,orig,return_pretop=True)
pp=PostProcessor(num_classes=C,num_top_queries=300,use_focal_loss=True)
pp_out=pp({"pred_logits":gl,"pred_boxes":gb}, orig)

# compare pre-top (top300) selection
pl=pp_out  # list of dict
label_match=True; box_match=True; score_match=True; maxdiff_s=0; maxdiff_b=0
for b in range(B):
    if not torch.equal(pre["labels"][b], pp_out[b]["labels"]): label_match=False
    if not torch.equal(pre["scores"][b], pp_out[b]["scores"]): score_match=False
    maxdiff_s=max(maxdiff_s,float((pre["scores"][b]-pp_out[b]["scores"]).abs().max()))
    maxdiff_b=max(maxdiff_b,float((pre["boxes"][b]-pp_out[b]["boxes"]).abs().max()))
    if not torch.allclose(pre["boxes"][b],pp_out[b]["boxes"],atol=1e-6): box_match=False
res["tests"]["layer1_no_new4_vs_PostProcessor"]={"label_identical":label_match,"score_identical":score_match,
  "box_allclose":box_match,"max_score_diff":maxdiff_s,"max_box_diff":maxdiff_b,
  "note":"pre-top(融合后 top300)与 PostProcessor 输出逐项比较"}
# and final self-built output when new4<0.001 equals PostProcessor after 0.001+max100 post? must be subset; check count
cnt_final=sum(x["labels"].shape[0] for x in out)
cnt_pp=sum(pp_out[b]["labels"].shape[0] for b in range(B))
res["tests"]["layer1_final_count"]={"selfbuilt_final_total":cnt_final,"postprocessor_top300_total":cnt_pp,
  "note":"final 经 >0.001+逐类max100 后应 <= top300（部分低分框被滤）"}

# --- layer2: AP parity: self-built postprocess vs PostProcessor, both into faster_coco_eval ---
from faster_coco_eval import COCO, COCOeval_faster
# build tiny GT: use orig image ids, a few gt boxes in orig xywh
def to_xywh(b):
    x0,y0,x1,y1=b.unbind(-1); return torch.stack([x0,y0,x1-x0,y1-y0],-1)
# Take B=2 images as separate image_ids 1,2
img_ids=[1,2]
# GT: for each image, convert PostProcessor boxes top-30 as "pseudo GT" (so predictions overlap heavily)
gt_anns=[]; ann_id=1
gt_boxes_per={}
for b in range(B):
    boxes=pp_out[b]["boxes"][:30]; labs=pp_out[b]["labels"][:30]
    lst=[]
    for k in range(30):
        x,y,w,h=to_xywh(boxes[k]).tolist()
        gt_anns.append({"id":ann_id,"image_id":img_ids[b],"category_id":int(labs[k])+1,"bbox":[x,y,w,h],"area":w*h,"iscrowd":0})
        ann_id+=1
    gt_boxes_per[img_ids[b]]=None
gt={"images":[{"id":i,"width":int(orig[b,1]),"height":int(orig[b,0])} for b,i in enumerate(img_ids)],
    "categories":[{"id":c+1} for c in range(C)],"annotations":gt_anns}
coco_gt=COCO(); coco_gt.dataset=gt; coco_gt.createIndex()

def dt_from_selfbuilt():
    rows=[]
    for b,i in enumerate(img_ids):
        o=out[b]
        xywh=to_xywh(o["boxes"])
        for k in range(o["labels"].shape[0]):
            x,y,w,h=xywh[k].tolist()
            rows.append({"image_id":i,"category_id":int(o["labels"][k])+1,"bbox":[x,y,w,h],"score":float(o["scores"][k])})
    return rows
def dt_from_pp():
    rows=[]
    for b,i in enumerate(img_ids):
        xywh=to_xywh(pp_out[b]["boxes"])
        for k in range(pp_out[b]["labels"].shape[0]):
            x,y,w,h=xywh[k].tolist()
            rows.append({"image_id":i,"category_id":int(pp_out[b]["labels"][k])+1,"bbox":[x,y,w,h],"score":float(pp_out[b]["scores"][k])})
    return rows

def run_ap(dets):
    ev=COCOeval_faster(coco_gt,iouType='bbox',print_function=lambda *a,**k:None,separate_eval=True)
    dt=coco_gt.loadRes(dets)
    ev.cocoDt=dt; ev.params.imgIds=img_ids
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return [float(x) for x in ev.stats[:12]]

ap_self=run_ap(dt_from_selfbuilt())
ap_pp=run_ap(dt_from_pp())
res["tests"]["layer2_AP_selfbuilt_vs_pp"]={"selfbuilt_12":ap_self,"pp_12":ap_pp,
  "max_abs_diff":max(abs(a-b) for a,b in zip(ap_self,ap_pp)),
  "all_close_1e-9":all(abs(a-b)<1e-9 for a,b in zip(ap_self,ap_pp))}

# --- same query multiple classes kept ---
# construct one query with two classes both >0.001 and high
g2=torch.zeros(1,300,10)-12; b2=torch.rand(1,300,4)*0.4+0.3
g2[0,0,3]=5; g2[0,0,7]=5  # same query q0 two classes high
n2=torch.zeros(1,4,10)-12; nb2=torch.rand(1,4,4)*0.4+0.3
o2,_=fuse_and_postprocess(g2,b2,n2,nb2,torch.tensor([[640.,640.]]))
labs=o2[0]["labels"].tolist()
res["tests"]["same_query_multi_class"]={"classes_present":sorted(set(labs)),"has_3_and_7":(3 in labs and 7 in labs)}

json.dump(res,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_task3_fusion.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(res,ensure_ascii=False,indent=2))
