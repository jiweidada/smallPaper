import sys, json, torch
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
torch.set_num_threads(2)
from uic_model import FourArm, shared_operator, C, D
torch.manual_seed(5)
res=json.load(open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_task5_6_7_model.json",encoding="utf-8"))

# structural proof: kappa_c only used inside G generation operator; mass head & aux plan modules have no kappa param.
m=FourArm("G")
kp=[n for n,_ in m.named_parameters() if "kappa_c" in n]
# reachability: which params get grad from (a) mass loss (b) assoc loss (c) gen/det
# (a) mass loss = sum(m) -> grads only mass head + encoder
rgb=torch.randn(1,3,128,128)
cell=torch.stack(torch.meshgrid(torch.linspace(1/32,1-1/32,16),torch.linspace(1/32,1-1/32,16),indexing='ij'),-1).reshape(-1,2)
Nc=6
P={"boxes_xyxy_local":torch.rand(1,Nc,4)*0.4+0.3,"scores":torch.rand(1,Nc),"cls":torch.randint(0,C,(1,Nc))}
P["boxes_xyxy_local"][...,2:]=P["boxes_xyxy_local"][...,:2]+0.1
m.zero_grad()
o=m(rgb,P,cell)[0]
mass_loss=o["m"].sum()
mass_loss.backward()
res["tests"]["kappa_c_grad_from_mass"]={"kappa_grad_after_mass_loss": None if m.kappa_c.grad is None else float(m.kappa_c.grad.abs().sum())}
res["tests"]["kappa_c_grad_from_mass"]["zero"]= (m.kappa_c.grad is None) or float(m.kappa_c.grad.abs().sum())==0.0

# (b) assoc loss = -(sum log(z_A) ...) uses aux plan (CA,kA) not kappa_c
m.zero_grad()
o=m(rgb,P,cell)[0]
assoc_loss=-(torch.log(o["gen"]["z"]+1e-8)).sum()  # z from G gen uses kappa? G gen z uses kappa_c!
# NOTE G's gen z DOES use kappa_c (Z^G). But 317 says kappa_c only via det loss, NOT L_assoc.
# G's L_assoc uses plan_aux (learnable), NOT gen plan. So assoc should use aux operator, not gen.
# Our proxy gen for G returns gen from d_out+kappa_c; assoc for G should use shared_operator(CA,kA,...).
m.zero_grad()
# compute aux assoc independent of kappa_c
auxm=m.aux
a_j=o["a_j"]
Wkf=auxm.Wk(m.enc(rgb)[0].reshape(D,-1).transpose(0,1))
from uic_model import d_out
x0,y0,x1,y1=P["boxes_xyxy_local"][0].unbind(-1)
pb=torch.stack([(x0+x1)/2,(y0+y1)/2,(x1-x0).clamp(min=1e-6),(y1-y0).clamp(min=1e-6)],-1)
dO=d_out(pb,cell)
dot=(auxm.Wq(a_j)@Wkf.t())/ (D**0.5)
CA=dO-dot
kA=auxm.kap(torch.cat([a_j, m.enc(rgb)[0].reshape(D,-1).mean(1).unsqueeze(0).expand(a_j.shape[0],-1)],-1)).squeeze(-1)
genA=shared_operator(CA,kA,o["mbar"],P["cls"][0])
assocA=-(torch.log(genA["z"]+1e-8)).sum()
assocA.backward()
res["tests"]["kappa_c_grad_from_assoc_aux"]={"kappa_grad_after_assoc":None if m.kappa_c.grad is None else float(m.kappa_c.grad.abs().sum()),
  "zero": (m.kappa_c.grad is None) or float(m.kappa_c.grad.abs().sum())==0.0,
  "note":"G 的 L_assoc 走 plan_aux(C^A/k^A)，不含 kappa_c"}

# (c) det/gen loss -> kappa_c grad (normal, unsaturated)
m.zero_grad()
o=m(rgb,P,cell)[0]
gen_r=o["gen"]["r"]
gen_r.sum().backward()
gk=m.kappa_c.grad
res["tests"]["kappa_c_grad_from_gen_det"]={"grad_nonzero_count":int((gk.abs()>0).sum().item()),
  "grad_total_abs":float(gk.abs().sum()),
  "note":"未饱和例：kappa_c 经 G 生成算子 r 有非零梯度（det 路径）"}
# saturated: kappa_c very negative -> gen r -> mbar (ignore geometry) -> gradient wrt kappa_c should approach 0
with torch.no_grad(): m.kappa_c.fill_(-60.0)
m.zero_grad()
o2=m(rgb,P,cell)[0]
o2["gen"]["r"].sum().backward()
gk2=m.kappa_c.grad
res["tests"]["kappa_c_saturated_grad"]={"grad_total_abs_kappa=-60":float(gk2.abs().sum()) if gk2 is not None else None,
  "near_zero": bool(gk2 is not None and float(gk2.abs().sum())<1e-6)}

res["task5_6_7_caveat"]={
  "end_to_end_warning":"代理模型中 G 的生成计划 gen['r'] 未实际回灌 decoder logits（生成→解码接线为设计态，见task4）。故 'kappa_c 经检测损失端到端获得梯度' 仅验证算子层 d r/d kappa_c；完整 det-loss→kappa_c 链路待真实模型接线后复核。",
  "optimizer":"kappa_c(10标量,1维) 落入 no-decay 组且 bias/norm 无decay；global 参数不在任何组（BASE24 全局冻结、不参与本优化器）。"}
json.dump(res,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_task5_6_7_model.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps({k:res["tests"][k] for k in ["kappa_c_grad_from_mass","kappa_c_grad_from_assoc_aux","kappa_c_grad_from_gen_det","kappa_c_saturated_grad"]},ensure_ascii=False,indent=2))
