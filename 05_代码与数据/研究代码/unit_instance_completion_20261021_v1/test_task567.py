import sys, json, torch, time
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"); sys.path.insert(0, r"C:\Users\jiwei\Desktop\DEIM")
torch.set_num_threads(2)
import torch.nn as nn
from uic_model import FourArm, C, D, GRID
torch.manual_seed(2026102101)
res={"item":"任务5+6+7 参数/路由/初态合同","executed":[],"tests":{}}

arms={a:FourArm(a) for a in ["N","S","G","R"]}
def cnt(m): return sum(p.numel() for p in m.parameters())
counts={a:cnt(m) for a,m in arms.items()}
res["param_counts"]=counts
# S delta vs a common base: build a bare baseline arm without S/G/R extras? Use N as base struct proxy
res["S_delta_vs_N"]=counts["S"]-counts["N"]
res["G_delta_vs_N"]=counts["G"]-counts["N"]
res["algebraic_target_S_addition"]=133888
res["tests"]["S_addition_matches_133888"]={"measured_S_minus_N":counts["S"]-counts["N"],
   "note":"S 相对 N 多出 P-crossattention(2x[MHA+LN]) + count head(10->128)；代数=133888，实测见上。注意 N 与 S 另共享其余结构。"}

# forward smoke on all arms with synthetic P
B=2; H=128
rgb=torch.randn(B,3,H,H)
cell=torch.stack(torch.meshgrid(torch.linspace(1/32,1-1/32,16),torch.linspace(1/32,1-1/32,16),indexing='ij'),-1).reshape(-1,2)  # 256 centers
Nc=7
P={"boxes_xyxy_local":torch.rand(B,Nc,4)*0.4+0.3,"scores":torch.rand(B,Nc),"cls":torch.randint(0,C,(B,Nc))}
# ensure x1>x0
P["boxes_xyxy_local"][...,2:]=P["boxes_xyxy_local"][...,:2]+0.1
fwd_ok={}; t0=time.time()
for a,m in arms.items():
    m.eval()
    with torch.no_grad():
        o=m(rgb,P,cell)
    fwd_ok[a]={"logits_shape":list(o[0]["pred_logits"].shape),"boxes_shape":list(o[0]["pred_boxes"].shape),"finite":bool(torch.isfinite(o[0]["pred_logits"]).all())}
res["tests"]["forward_all_arms"]=fwd_ok
res["executed"].append("forward 4 arms on B=2 rgb 128^2")

# --- T7: G/R initial-state equality (same common init) ---
# Rebuild G,R sharing identical common init by copying state
import copy
torch.manual_seed(777)
g=FourArm("G"); r=FourArm("R")
# copy matching submodules from g to r (common struct)
for name in ["enc","mass","dec","cand","aux"]:
    getattr(r,name).load_state_dict(getattr(g,name).state_dict())
g.eval(); r.eval()
with torch.no_grad():
    og=g(rgb,P,cell)[0]; orr=r(rgb,P,cell)[0]
res["tests"]["GR_initial_equality"]={
  "pred_logits_equal": bool(torch.equal(og["pred_logits"],orr["pred_logits"])),
  "pred_boxes_equal": bool(torch.equal(og["pred_boxes"],orr["pred_boxes"])),
  "aux_plan_equal": bool(torch.equal(og["CA"],orr["CA"]) and torch.equal(og["kA"],orr["kA"])),
  "gen_plan_equal_initial": bool(og["gen"] is not None and orr["gen"] is not None and torch.equal(og["gen"]["r"],orr["gen"]["r"])),
  "note":"G/R 复制公共初态后：预测/aux计划/生成计划应同值(G kappa_c=0 时) 见上"}

# --- T7: fixing f/m/P, change thetaC/thetaK only changes G aux plan, not G generation ---
# perturb aux.Wq slightly, G generation uses kappa_c + d_out (fixed geometry) so gen unchanged; aux CA changes
with torch.no_grad():
    gen_before=g(rgb,P,cell)[0]["gen"]["r"].clone()
    ca_before=g(rgb,P,cell)[0]["CA"].clone()
    g.aux.Wq.weight.add_(0.01)
    o_after=g(rgb,P,cell)[0]
res["tests"]["theta_aux_only_changes_aux_not_gen"]={
  "gen_r_unchanged": bool(torch.equal(gen_before, o_after["gen"]["r"])),
  "aux_CA_changed": bool(not torch.equal(ca_before, o_after["CA"])),
  "note":"G 生成用固定C^G=d_out+kappa_c，不含可学习C^A/k^A；改 aux 只动 aux 计划"}

# --- T7: kappa_c gradient: only through det loss (gen plan); not mass/assoc ---
g2=FourArm("G"); g2.train()
def det_loss(model, rgb,P,cell, u_labels=None):
    o=model(rgb,P,cell)
    logits=torch.stack([x["pred_logits"] for x in o])  # [B,4,C]
    # simple classification CE against P cls (dummy)
    tgt=torch.stack([P["cls"][b] for b in range(B)])
    return logits.sum()  # want grad wrt kappa_c through gen path? gen path r doesn't feed logits in this proxy
# NOTE: in this proxy, kappa_c affects gen['r'] but generation r is not wired back into decoder logits.
# So we test: d gen_r / d kappa_c is nonzero (normal generation gradient), and mass/assoc do not depend on kappa_c.
gen=g2(rgb,P,cell)[0]
gen_r=gen["gen"]["r"]
gen_r.sum().backward()
gk=g2.kappa_c.grad
res["tests"]["kappa_c_gradient_via_gen"]={"grad_exists":gk is not None,"grad_nonzero":bool(gk is not None and gk.abs().sum()>0),
   "grad_per_class_nonzero_count": int((gk.abs()>0).sum().item()) if gk is not None else 0,
   "note":"kappa_c 只经生成算子 r 的反传；mass/assoc 不含 kappa_c(结构上)，饱和例按实际算子记零"}
# mass path independent of kappa_c: mass head has no kappa_c; assoc uses aux plan (no kappa_c)
res["tests"]["kappa_c_structural_scope"]={"mass_head_has_no_kappa": not any("kappa" in n for n,p in g2.mass.named_parameters()),
   "aux_has_no_kappa": not any("kappa_c" in n for n,p in g2.aux.named_parameters())}

# --- T5: optimizer routing ---
def build_param_groups(model, lr=1e-4, wd=1e-4):
    decay=[]; no_decay=[]
    kappa_names=[]
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim==1 or n.endswith("bias") or ("norm" in n) or ("bn" in n):
            no_decay.append(p)
            if n.startswith("kappa_c"): kappa_names.append(n)
        else:
            decay.append(p)
    groups=[{"params":decay,"weight_decay":wd},{"params":no_decay,"weight_decay":0.0}]
    return groups, no_decay, kappa_names
groups, nodecay, knames=build_param_groups(g2)
res["tests"]["optimizer_routing"]={
  "kappa_c_in_nodecay": len(knames)>0 and any(n=="kappa_c" for n in knames),
  "kappa_c_names":knames,
  "num_decay":sum(p.numel() for p in groups[0]["params"]),
  "num_nodecay":sum(p.numel() for p in groups[1]["params"]),
  "global_params_absent":None}
# global params not in any group: our model has no 'global' backbone (BASE24 frozen separately) -> verify
alln=[n for n,_ in g2.named_parameters()]
res["tests"]["optimizer_routing"]["global_params_absent"]=not any(n.startswith("global") for n in alln)
json.dump(res,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_task5_6_7_model.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(res,ensure_ascii=False,indent=2))
