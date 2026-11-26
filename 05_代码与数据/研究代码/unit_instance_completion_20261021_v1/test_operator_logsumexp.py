import sys, json, torch, hashlib
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
torch.set_num_threads(2)
from uic_model import shared_operator, C
torch.manual_seed(3)
res={"item":"生成算子 logsumexp 稳定化（335 要求）","tests":{},"file_sha256":hashlib.sha256(open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\uic_model.py","rb").read()).hexdigest()}
N,L=5,16
Cmat=torch.rand(N,L)*3
mbar=torch.rand(C,L)*0.01
cand_cls=torch.randint(0,C,(N,))
ok={}
for k in [0.0,-5.0,-10.0,-60.0,-200.0]:
    kap=torch.full((C,),k)
    o=shared_operator(Cmat,kap,mbar,cand_cls)
    finite=all(torch.isfinite(o[x]).all().item() for x in ["u","a","T","z","r","p0","p_ij"])
    ok[str(k)]={"finite":bool(finite),
      "p0":round(float(o["p0"].mean()),8),
      "z_min":round(float(o["z"].min()),8),
      "r_over_mbar":round(float(o["r"].sum()/mbar.sum()),8)}
res["tests"]["extreme_kappa"]=ok
# monotonic behavior: as kappa -> -inf, p0 -> 1 (all mass to null => r ~ mbar)
res["tests"]["p0_monotone"]=[ok[str(k)]["p0"] for k in [0.0,-5.0,-10.0,-60.0,-200.0]]
json.dump(res,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_redo_operator_logsumexp.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(res,ensure_ascii=False,indent=2))
