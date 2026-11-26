import sys, json, torch
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
sys.path.insert(0, r"C:\Users\jiwei\Desktop\DEIM")
torch.set_num_threads(2)
import numpy as np
from uic_m0 import m0_lex, native_matcher_cost, iou_xyxy
from engine.deim.box_ops import box_cxcywh_to_xyxy
torch.manual_seed(2026102101)
res = {"item":"任务1 M0三级lex","executed":[],"tests":{}}

# --- cross-check my cost vs real HungarianMatcher ---
from engine.deim.matcher import HungarianMatcher
wd = {"cost_class":2.0,"cost_bbox":5.0,"cost_giou":2.0}
matcher = HungarianMatcher(wd, use_focal_loss=True, alpha=0.25, gamma=2.0)
N,M=7,4
cb = torch.rand(N,4)*0.6+0.2
gl = torch.randint(0,10,(M,))
gb = torch.rand(M,4)*0.6+0.2
cl = torch.randn(N,10)*2
# real matcher cost matrix
import torch.nn.functional as F
out_prob = torch.sigmoid(cl)
tp = out_prob[:, gl]
neg=(1-0.25)*(tp**2.0)*(-(1-tp+1e-8).log()); pos=0.25*((1-tp)**2.0)*(-(tp+1e-8).log())
cc=pos-neg
cbld=torch.cdist(cb, gb, p=1)
from engine.deim.box_ops import generalized_box_iou
cg=-generalized_box_iou(box_cxcywh_to_xyxy(cb), box_cxcywh_to_xyxy(gb))
Cref=5.0*cbld + 2.0*cc + 2.0*cg
Cmine,_=native_matcher_cost(cb,cl,gb,gl,10)
res["tests"]["cost_parity_vs_matcher_formula"]={"max_abs_diff":float((Cref-Cmine).abs().max())}
res["executed"].append("cost matrix cross-check")

# Build a controlled scenario: 3 GT same class, 4 cands, one GT far (no legal edge)
# class 3 for all
gb = torch.tensor([[0.3,0.3,0.1,0.1],[0.7,0.7,0.1,0.1],[0.5,0.9,0.05,0.05]])
gl = torch.tensor([3,3,3])
# cands: two near g0, one near g1, one random (no edge to g2)
cb = torch.tensor([[0.30,0.30,0.10,0.10],[0.31,0.29,0.10,0.10],[0.70,0.70,0.10,0.10],[0.2,0.2,0.05,0.05]])
cl = torch.full((4,10), -5.0); cl[:,3]=5.0  # confident class 3
r = m0_lex(cb, cl, gb, gl)
res["tests"]["basic"]={"coverage":r["coverage"],"gt_explained":r["gt_explained"].tolist(),
  "cand_owner":r["cand_owner"].tolist(),"pairs":r["matched_pairs"]}

# (1) tie stability + reorder invariance: permute input candidates, M0 owner must map consistently
perm = torch.tensor([3,1,0,2])
r2 = m0_lex(cb[perm], cl[perm], gb, gl)
# map owner back
owner2_back = torch.full((4,),-1)
for newpos,oldpos in enumerate(perm.tolist()):
    owner2_back[oldpos]=r2["cand_owner"][newpos]
res["tests"]["reorder_invariance"]={
  "same_gt_explained":bool(torch.equal(r["gt_explained"], r2["gt_explained"])),
  "same_coverage":r["coverage"]==r2["coverage"],
  "same_cost_close":abs(r["total_cost"]-r2["total_cost"])<1e-6,
  "owner_maps_back":bool(torch.equal(r["cand_owner"], owner2_back))
}

# (2) tie case: two identical candidates equally good for one GT -> fixed index picks lower index
cb_t = torch.tensor([[0.5,0.5,0.2,0.2],[0.5,0.5,0.2,0.2]])
cl_t = torch.zeros(2,10); cl_t[:,0]=3.0
gb_t=torch.tensor([[0.5,0.5,0.2,0.2]]); gl_t=torch.tensor([0])
rt=m0_lex(cb_t,cl_t,gb_t,gl_t)
# reorder them (identical) -> still lower original index owns
res["tests"]["tie_fixed_index"]={"owner":rt["cand_owner"].tolist(),"pairs":rt["matched_pairs"],
  "note":"identical cands; deterministic lower-index chosen"}

# (3) U_W closure: G_W - M0. Build W hypothesis: all 3 GT complete; M0 explains g0,g1 not g2 => U_W={g2}
G_W = torch.tensor([True,True,True])
U_W = G_W & (~r["gt_explained"])
res["tests"]["UW_closure"]={"G_W":G_W.tolist(),"M0_explained":r["gt_explained"].tolist(),
  "U_W":U_W.tolist(),"closure_ok":bool(torch.equal(U_W, torch.tensor([False,False,True])))}

# (4a) empty P
r_e = m0_lex(torch.zeros(0,4), torch.zeros(0,10), gb, gl)
res["tests"]["empty_P"]={"coverage":r_e["coverage"],"matches":r_e["matched_pairs"],
  "all_gt_unexplained":bool((~r_e["gt_explained"]).all())}
# (4b) empty GT
r_g = m0_lex(cb, cl, torch.zeros(0,4), torch.zeros(0,dtype=torch.long))
res["tests"]["empty_GT"]={"coverage":r_g["coverage"],"matches":r_g["matched_pairs"],
  "owner_all_none":bool((r_g["cand_owner"]==-1).all())}
# (4c) no legal edge (all wrong class)
cl_w = torch.full((4,10),-5.0); cl_w[:,7]=5.0  # class 7 != 3
r_n = m0_lex(cb, cl_w, gb, gl)
res["tests"]["no_legal_edge"]={"coverage":r_n["coverage"],"all_gt_unexplained":bool((~r_n["gt_explained"]).all())}

# (5) any-class最大IoU<.5 GT must be in U_W. Construct GT with no cand of ANY class IoU>=.5
gb5 = torch.tensor([[0.05,0.05,0.02,0.02],[0.5,0.5,0.1,0.1]])
gl5 = torch.tensor([1,2])
cb5 = torch.tensor([[0.9,0.9,0.1,0.1],[0.3,0.3,0.1,0.1]])  # far from both
cl5 = torch.zeros(2,10); cl5[:,1]=3.0; cl5[:,2]=3.0
r5 = m0_lex(cb5, cl5, gb5, gl5)
# compute any-class max IoU per GT
iou_any = iou_xyxy(box_cxcywh_to_xyxy(cb5), box_cxcywh_to_xyxy(gb5)).max(0).values
neverM0 = iou_any < 0.5
U_W5 = ~r5["gt_explained"]
res["tests"]["never_m0_gt_in_UW"]={
  "any_class_max_iou":iou_any.tolist(),"neverM0":neverM0.tolist(),
  "m0_explained":r5["gt_explained"].tolist(),
  "gt_with_maxiou_lt05_all_unexplained":bool((neverM0 <= U_W5).all())
}
res["executed"] += ["basic M0","reorder invariance","tie fixed index","UW closure","empty P","empty GT","no legal edge","never-M0 GT in UW"]
json.dump(res, open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_task1_m0.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(res, ensure_ascii=False, indent=2))
