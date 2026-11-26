import os, sys, json, torch, hashlib
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"; os.environ["MKL_THREADING_LAYER"]="SEQUENTIAL"
sys.dont_write_bytecode=True
BASE=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2"
sys.path.insert(0, BASE+r"\311_unit_instance_core")
sys.path.insert(0, BASE+r"\320_geometry_calibrated_core")
import torch; torch.set_num_threads(2)
import core, gcal
from core import CompletionHead, copy_common, DIM, SIDE, CLASSES, QUERIES
from gcal import GeometryCalibratedHead, copy_R_common

out={"item":"补做3 用冻结311/320重做G/R合同","executed":[],"tests":{},"frozen_sha":{
  "core_py":hashlib.sha256(open(BASE+r"\311_unit_instance_core\core.py","rb").read()).hexdigest(),
  "expected":"8cd5dbc61052c7d3c3b4058b4d25b150b906ecc06c6eb005793c737cef3f17ef",
  "gcal_py":hashlib.sha256(open(BASE+r"\320_geometry_calibrated_core\gcal.py","rb").read()).hexdigest()}}
out["frozen_sha"]["core_match"]=out["frozen_sha"]["core_py"]==out["frozen_sha"]["expected"]

torch.manual_seed(2026102101)
B=2
rgb=torch.rand(B,3,128,128)
Nc=6
boxes=torch.rand(B,Nc,4); boxes[...,2:]=boxes[...,:2]+0.15*(boxes[...,2:].clone()+0.1)
cls=torch.randint(0,CLASSES,(B,Nc)); sc=torch.rand(B,Nc); valid=torch.ones(B,Nc,dtype=torch.bool)

# --- G/R initial equality via frozen copy_R_common ---
r_head=CompletionHead("R"); g_head=GeometryCalibratedHead()
copy_R_common(r_head,g_head)
r_head.eval(); g_head.eval()
with torch.no_grad():
    orr=r_head.forward_features(r_head.trunk(rgb),boxes,cls,sc,valid)
    og=g_head.forward_features(g_head.trunk(rgb),boxes,cls,sc,valid)
out["tests"]["GR_initial_equality"]={
 "pred_logits_equal":bool(torch.equal(orr["pred_logits"],og["pred_logits"])),
 "pred_boxes_equal":bool(torch.equal(orr["pred_boxes"],og["pred_boxes"])),
 "aux_plan_equal":bool(torch.equal(orr["plan"]["residual"],og["aux_plan"]["residual"])),
 "generation_vs_R_residual_equal_at_kappa0":bool(torch.equal(og["generation_plan"]["residual"],orr["plan"]["residual"])),
 "class_null_all_zero":bool((g_head.class_null==0).all())}
out["executed"].append("G/R initial equality on frozen 311/320 (2 forwards)")

# --- theta(aux) change affects G aux plan but not G generation plan ---
with torch.no_grad():
    gen_before=og["generation_plan"]["residual"].clone()
    aux_before=og["aux_plan"]["residual"].clone()
    g_head.association.query.weight.add_(0.01)   # thetaC-ish
    og2=g_head.forward_features(g_head.trunk(rgb),boxes,cls,sc,valid)
out["tests"]["theta_aux_only_changes_aux_not_gen"]={
 "gen_unchanged":bool(torch.equal(gen_before,og2["generation_plan"]["residual"])),
 "aux_changed":bool(not torch.equal(aux_before,og2["aux_plan"]["residual"]))}
out["executed"].append("aux-only perturbation (1 forward)")

# --- class_null gradient only via generation plan; not mass/assoc ---
g2=GeometryCalibratedHead(); g2.train()
def feats(h):
    return h.forward_features(h.trunk(rgb),boxes,cls,sc,valid)
# (a) generation residual sum -> class_null grad nonzero
g2.zero_grad(); f=feats(g2); f["generation_plan"]["residual"].sum().backward()
g1=g2.class_null.grad.clone(); nz=int((g1.abs()>0).sum())
# (b) aux/assoc plan loss -> no class_null grad
g2.zero_grad(); f=feats(g2); (-torch.log(f["aux_plan"]["z"]+1e-8)).sum().backward()
g2b=g2.class_null.grad
# (c) mass -> no class_null
g2.zero_grad(); f=feats(g2); f["mass"].sum().backward()
g2c=g2.class_null.grad
out["tests"]["class_null_grad_scope"]={
 "gen_grad_nonzero_classes":nz,"gen_grad_total":float(g1.abs().sum()),
 "aux_grad_zero":(g2b is None) or float(g2b.abs().sum())==0.0,
 "mass_grad_zero":(g2c is None) or float(g2c.abs().sum())==0.0}
out["executed"].append("class_null gradient scope (3 backward passes)")

# --- saturation: class_null very negative -> generation field -> mass (ignore geometry) ---
vals={}
for k in (0.0,-5.0,-10.0):
    h=GeometryCalibratedHead()
    with torch.no_grad(): h.class_null.fill_(k)
    h.eval()
    with torch.no_grad(): ff=feats(h)
    vals[str(k)]={"residual_sum":float(ff["generation_plan"]["residual"].sum()),
                  "mass_sum":float(ff["mass"].sum()),
                  "residual_over_mass":float(ff["generation_plan"]["residual"].sum()/ff["mass"].sum())}
out["tests"]["class_null_saturation"]=vals
out["executed"].append("saturation sweep k=0,-5,-10 (3 forwards)")
out["not_executed"]=["kappa_c 端到端 det-loss→class_null 链路（320 原型只到生成计划，未接检测损失；见317§8与320文档）"]
json.dump(out,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_redo3_frozen311_320.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False,indent=2)[:4500])