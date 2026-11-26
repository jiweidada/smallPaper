import os, sys, json, torch, hashlib
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"; os.environ["MKL_THREADING_LAYER"]="SEQUENTIAL"
sys.dont_write_bytecode=True
BASE=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2"
sys.path.insert(0,BASE+r"\311_unit_instance_core_v2")
sys.path.insert(0,BASE+r"\320_geometry_calibrated_core_v2")
import torch; torch.set_num_threads(2)
import core, gcal
from core import CompletionHead, copy_common, CLASSES, DIM, SIDE
from gcal import GeometryCalibratedHead, copy_R_common

out={"item":"补做3(v2) 冻结311v2/320v2 G-R合同重做","tests":{},"frozen_sha":{
 "core_v1":hashlib.sha256(open(BASE+r"\311_unit_instance_core\core.py","rb").read()).hexdigest(),
 "core_v2":hashlib.sha256(open(BASE+r"\311_unit_instance_core_v2\core.py","rb").read()).hexdigest(),
 "gcal_v2":hashlib.sha256(open(BASE+r"\320_geometry_calibrated_core_v2\gcal.py","rb").read()).hexdigest()}}
out["frozen_sha"]["v1_unchanged"]=out["frozen_sha"]["core_v1"]=="8cd5dbc61052c7d3c3b4058b4d25b150b906ecc06c6eb005793c737cef3f17ef"

torch.manual_seed(2026102101)
B=2; rgb=torch.rand(B,3,128,128); Nc=6
boxes=torch.rand(B,Nc,4); boxes[...,2:]=boxes[...,:2]+0.15*(boxes[...,2:].clone()+0.1)
cls=torch.randint(0,CLASSES,(B,Nc)); sc=torch.rand(B,Nc); valid=torch.ones(B,Nc,dtype=torch.bool)

r_head=CompletionHead("R"); g_head=GeometryCalibratedHead(); copy_R_common(r_head,g_head)
r_head.eval(); g_head.eval()
with torch.no_grad():
    orr=r_head.forward_features(r_head.trunk(rgb),boxes,cls,sc,valid)
    og=g_head.forward_features(g_head.trunk(rgb),boxes,cls,sc,valid)
out["tests"]["GR_initial_equality"]={"pred_logits_equal":bool(torch.equal(orr["pred_logits"],og["pred_logits"])),
 "pred_boxes_equal":bool(torch.equal(orr["pred_boxes"],og["pred_boxes"])),
 "aux_plan_equal":bool(torch.equal(orr["plan"]["residual"],og["aux_plan"]["residual"])),
 "gen_vs_R_equal":bool(torch.equal(og["generation_plan"]["residual"],orr["plan"]["residual"])),
 "class_null_zero":bool((g_head.class_null==0).all())}

with torch.no_grad():
    gb_=og["generation_plan"]["residual"].clone(); ab_=og["aux_plan"]["residual"].clone()
    g_head.association.query.weight.add_(0.01)
    og2=g_head.forward_features(g_head.trunk(rgb),boxes,cls,sc,valid)
out["tests"]["theta_aux_only_changes_aux"]={"gen_unchanged":bool(torch.equal(gb_,og2["generation_plan"]["residual"])),
 "aux_changed":bool(not torch.equal(ab_,og2["aux_plan"]["residual"]))}

# grad scope on v2
g2=GeometryCalibratedHead(); g2.train()
g2.zero_grad(); f=g2.forward_features(g2.trunk(rgb),boxes,cls,sc,valid); f["generation_plan"]["residual"].sum().backward()
gr=g2.class_null.grad
out["tests"]["class_null_grad_scope"]={"gen_nan":int(torch.isnan(gr).sum()),"gen_all_finite":bool(torch.isfinite(gr).all()),
 "gen_grads":[float(x) for x in gr],"gen_nonzero_classes":int((gr.abs()>0).sum())}
g2.zero_grad(); f=g2.forward_features(g2.trunk(rgb),boxes,cls,sc,valid); (-torch.log(f["aux_plan"]["z"]+1e-8)).sum().backward()
gb2=g2.class_null.grad
g2.zero_grad(); f=g2.forward_features(g2.trunk(rgb),boxes,cls,sc,valid); f["mass"].sum().backward()
gc2=g2.class_null.grad
out["tests"]["class_null_grad_scope"]["aux_zero"]=(gb2 is None) or float(gb2.abs().sum())==0
out["tests"]["class_null_grad_scope"]["mass_zero"]=(gc2 is None) or float(gc2.abs().sum())==0

# saturation
sat={}
for k in (0.0,-5.0,-10.0):
    h=GeometryCalibratedHead()
    with torch.no_grad(): h.class_null.fill_(k)
    h.eval()
    with torch.no_grad(): ff=h.forward_features(h.trunk(rgb),boxes,cls,sc,valid)
    sat[str(k)]={"resid_over_mass":float(ff["generation_plan"]["residual"].sum()/ff["mass"].sum())}
out["tests"]["saturation"]=sat
json.dump(out,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_redo3v2_frozen311v2.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False,indent=2)[:4000])
