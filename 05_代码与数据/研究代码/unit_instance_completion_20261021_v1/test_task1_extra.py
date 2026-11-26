import sys, json, torch
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"); sys.path.insert(0, r"C:\Users\jiwei\Desktop\DEIM")
torch.set_num_threads(2)
from uic_m0 import m0_lex, iou_xyxy
from engine.deim.box_ops import box_cxcywh_to_xyxy
res=json.load(open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_task1_m0.json",encoding="utf-8"))

# genuine cost tie: two candidates mirrored around GT, same class, same IoU -> equal matcher cost
# GT at center 0.5,0.5 size .2. cand A at .45,.5 size .2 ; cand B at .55,.5 size .2 ; symmetric -> same IoU & L1
gb=torch.tensor([[0.5,0.5,0.2,0.2]]); gl=torch.tensor([0])
cb=torch.tensor([[0.45,0.5,0.2,0.2],[0.55,0.5,0.2,0.2]])
cl=torch.zeros(2,10); cl[:,0]=3.0
rA=m0_lex(cb,cl,gb,gl)
# reorder: now B first
perm=torch.tensor([1,0])
rB=m0_lex(cb[perm],cl[perm],gb,gl)
# map back: rB owner index0 corresponds to original index1; index1->orig0
back=torch.full((2,),-1); back[1]=rB["cand_owner"][0]; back[0]=rB["cand_owner"][1]
iouA=iou_xyxy(box_cxcywh_to_xyxy(cb),box_cxcywh_to_xyxy(gb)).squeeze(1)
res["tests"]["genuine_cost_tie"]={
 "iou_A":float(iouA[0]),"iou_B":float(iouA[1]),
 "owner_order1":rA["cand_owner"].tolist(),"owner_order2_mapped_back":back.tolist(),
 "same_choice_under_reorder": bool(torch.equal(rA["cand_owner"],back)),
 "note":"两个镜像候选等代价；重排后仍选同一物理候选(低原索引)"}

# IoU==0.5 boundary inclusive check
# construct two boxes with exact IoU=0.5: A size 1x1 at (0,0)-(1,1); B offset so IoU=0.5 -> B=(0,0)-(1,0.5)?? use xyxy
a=torch.tensor([[0.,0.,2.,2.]])   # area4
b=torch.tensor([[0.,0.,2.,1.]])   # overlap 2 -> IoU 2/(4+2-2)=0.5
i=iou_xyxy(a,b).item()
# feed via cxcywh
def xyxy2cxcywh(x):
  x0,y0,x1,y1=x.unbind(-1); return torch.stack([(x0+x1)/2,(y0+y1)/2,x1-x0,y1-y0],-1)
cbc=xyxy2cxcywh(b); gbc=xyxy2cxcywh(a)
clb=torch.zeros(1,10); clb[:,0]=3.0; glb=torch.tensor([0])
rb=m0_lex(cbc,clb,gbc,glb, iou_thr=0.5)
res["tests"]["iou_boundary_inclusive"]={"iou":i,"explained":int(rb["gt_explained"][0]),
 "included_at_exactly_0p5": bool(rb["gt_explained"][0])}

# efficiency/coverage priority: one candidate could cover GT-A cheaply OR GT-B; but another cand covers GT-A; max coverage must=2
# cands: c0 near A, c1 near A(exact), c2 near B. max coverage 2 (A by c1, B by c2) chosen even if c0-covers-A cheaper
gb2=torch.tensor([[0.2,0.2,0.1,0.1],[0.8,0.8,0.1,0.1]]); gl2=torch.tensor([0,0])
cb2=torch.tensor([[0.2,0.2,0.1,0.1],[0.21,0.2,0.1,0.1],[0.8,0.8,0.1,0.1]])
cl2=torch.zeros(3,10); cl2[:,0]=3.0
r2b=m0_lex(cb2,cl2,gb2,gl2)
res["tests"]["max_coverage_priority"]={"coverage":r2b["coverage"],"gt_explained":r2b["gt_explained"].tolist(),
 "expect_coverage_2": r2b["coverage"]==2}

json.dump(res, open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_task1_m0.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps({k:res["tests"][k] for k in ["genuine_cost_tie","iou_boundary_inclusive","max_coverage_priority"]}, ensure_ascii=False, indent=2))
