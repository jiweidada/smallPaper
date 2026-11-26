import sys, json, torch
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
torch.set_num_threads(2)
from uic_model import FourArm, C, D
torch.manual_seed(5)
res=json.load(open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_task5_6_7_model.json",encoding="utf-8"))
rgb=torch.randn(1,3,128,128)
cell=torch.stack(torch.meshgrid(torch.linspace(1/32,1-1/32,16),torch.linspace(1/32,1-1/32,16),indexing='ij'),-1).reshape(-1,2)
Nc=6
P={"boxes_xyxy_local":torch.rand(1,Nc,4)*0.4+0.3,"scores":torch.rand(1,Nc),"cls":torch.randint(0,C,(1,Nc))}
P["boxes_xyxy_local"][...,2:]=P["boxes_xyxy_local"][...,:2]+0.1
def grad_at(kval):
    m=FourArm("G")
    with torch.no_grad(): m.kappa_c.fill_(kval)
    m.zero_grad(); o=m(rgb,P,cell)[0]; o["gen"]["r"].sum().backward()
    g=m.kappa_c.grad
    # also measure how close r is to mbar (saturation: r->mbar)
    dev=float((o["gen"]["r"].sum(0)-o["mbar"].sum(0)[0:0] if False else (o["gen"]["r"].sum()- (o["mbar"].sum()*0)) ).abs())
    return None if g is None else float(g.abs().sum())
res["tests"]["kappa_c_saturation_sweep"]={
  "grad_k0":grad_at(0.0),"grad_k_-2":grad_at(-2.0),"grad_k_-5":grad_at(-5.0),"grad_k_-10":grad_at(-10.0),
  "grad_k_-60_nan_note":"kappa_c=-60 时 exp(-kappa_c)=e^60 溢出使 Z=inf -> r 出现 NaN；该实现需 log-sum-exp/夹取才能处理极端饱和"}
# check r->mbar as kappa -> -inf (saturation = ignore geometry, r趋mbar)
for k in [0.0,-5.0,-10.0]:
    m=FourArm("G")
    with torch.no_grad(): m.kappa_c.fill_(k)
    o=m(rgb,P,cell)[0]
    rb=o["gen"]["r"]; mb=o["mbar"]
    # r_ci=mbar - sum_j T ; as kappa->-inf, T->0 -> r->mbar
    diff=float((rb-mb).abs().sum()/mb.abs().sum())
    res["tests"].setdefault("r_approaches_mbar",{})[f"k={k}"]={"rel_diff_r_vs_mbar":round(diff,5)}
json.dump(res,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_task5_6_7_model.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps({"sweep":res["tests"]["kappa_c_saturation_sweep"],"r_approaches_mbar":res["tests"]["r_approaches_mbar"]},ensure_ascii=False,indent=2))
