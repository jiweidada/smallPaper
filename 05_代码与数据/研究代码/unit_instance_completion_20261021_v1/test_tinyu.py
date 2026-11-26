import os, sys, json, torch, importlib.util
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OMP_NUM_THREADS"]="2"
sys.dont_write_bytecode=True
BASE=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2"
sys.path.insert(0,BASE+r"\311_unit_instance_core_v2"); sys.path.insert(0,BASE+r"\320_geometry_calibrated_core_v2")
import torch; torch.set_num_threads(2); import torch.nn.functional as F
import core as core_v2
# load v1 standalone under different module name
spec=importlib.util.spec_from_file_location("core_v1", BASE+r"\311_unit_instance_core\core.py")
core_v1=importlib.util.module_from_spec(spec); spec.loader.exec_module(core_v1)

res={"item":"极小 u 梯度用例（335 补做要求3）","tests":{}}
C=core_v2.CLASSES
# construct u tiny by making mass just above 0 in one cell and huge null probability
# Build directly: mass with one cell m=1e-23, all others 0; cost 0; null_cost 0
for tag,mod in (("v1",core_v1),("v2",core_v2)):
    m=torch.zeros(1,C,256); m[:, :, 0]=1e-23
    cost=torch.zeros(1,256,2)
    nc=torch.zeros(1,2)
    classes=torch.tensor([[0,1]]); valid=torch.ones(1,2,dtype=torch.bool)
    o=mod.independent_cap(m,cost,nc,classes,valid)
    u=o["u"]
    print(tag,"min positive u:",float(u[u>0].min()) if (u>0).any() else None,
          "scale finite:",bool(torch.isfinite(o["scale"]).all()))
    # backward through residual to u-like leaf: make cost require grad
    cost2=cost.clone().requires_grad_(True)
    o2=mod.independent_cap(m,cost2,nc,classes,valid)
    o2["residual"].sum().backward()
    g=cost2.grad
    res["tests"][tag]={"min_u":float(u[u>0].min()) if (u>0).any() else None,
      "residual_finite":bool(torch.isfinite(o2["residual"]).all()),
      "grad_finite":bool(torch.isfinite(g).all()),"grad_nan":int(torch.isnan(g).sum()),
      "grad_absmax":float(g.abs().max()) if g is not None else None}
# also mass-grad path (m is detached -> no grad, expected)
# direct unit test of the two formulas
m_t=torch.tensor([1e-23]); u_t=torch.tensor([4.7e-26],requires_grad=True)
s1=torch.minimum(torch.ones_like(m_t), m_t/u_t); s1.sum().backward()
g1=float(u_t.grad)
u_t2=torch.tensor([4.7e-26],requires_grad=True)
s2=m_t/torch.maximum(u_t2,m_t); s2.sum().backward()
g2=float(u_t2.grad)
res["tests"]["formula_micro"]={"v1_minimum_grad":g1,"v1_is_nan":g1!=g1,"v2_maxdenom_grad":g2,"v2_is_nan":g2!=g2}
json.dump(res,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_redo3v2_tinyu.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(res,ensure_ascii=False,indent=2))
