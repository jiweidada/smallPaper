import sys, json, torch
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"); sys.path.insert(0, r"C:\Users\jiwei\Desktop\DEIM")
torch.set_num_threads(2)
from uic_m0 import m0_lex, native_matcher_cost, iou_xyxy
from engine.deim.box_ops import box_cxcywh_to_xyxy
import hashlib
res={"item":"M0 修正后复核（335 要求）","executed":[],"tests":{}}
res["file_sha256"]=hashlib.sha256(open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\uic_m0.py","rb").read()).hexdigest()

# 1) baseline correctness
gb=torch.tensor([[0.3,0.3,0.1,0.1],[0.7,0.7,0.1,0.1],[0.5,0.9,0.05,0.05]]); gl=torch.tensor([3,3,3])
cb=torch.tensor([[0.30,0.30,0.10,0.10],[0.31,0.29,0.10,0.10],[0.70,0.70,0.10,0.10],[0.2,0.2,0.05,0.05]])
cl=torch.full((4,10),-5.0); cl[:,3]=5.0
r=m0_lex(cb,cl,gb,gl)
res["tests"]["baseline"]={"coverage":r["coverage"],"pairs":r["matched_pairs"],"max_card":r["max_cardinality"],"eps_changed":r["eps_changed_optimum"]}

# 2) tie case: mirrored equal-cost candidates -> fixed intrinsic index, reorder invariant
gb2=torch.tensor([[0.5,0.5,0.2,0.2]]); gl2=torch.tensor([0]); cb2=torch.tensor([[0.45,0.5,0.2,0.2],[0.55,0.5,0.2,0.2]])
cl2=torch.zeros(2,10); cl2[:,0]=3.0
rA=m0_lex(cb2,cl2,gb2,gl2,cand_index=[0,1])
rB=m0_lex(cb2[[1,0]],cl2[[1,0]],gb2,gl2,cand_index=[1,0])
ownB=torch.full((2,),-1); ownB[1]=rB["cand_owner"][0]; ownB[0]=rB["cand_owner"][1]
res["tests"]["tie_fixed_index"]={"owner_A":rA["cand_owner"].tolist(),"owner_B_mapped":ownB.tolist(),
  "same":bool(torch.equal(rA["cand_owner"],ownB)),"eps_changed":rA["eps_changed_optimum"]}

# 3) LARGE COST MAGNITUDE test: scale logits/boxes so costs are huge; eps must not change optimum
torch.manual_seed(7)
gb3=torch.rand(4,4); gb3[:,2:]=gb3[:,:2]+0.1
gl3=torch.randint(0,10,(4,))
cb3=torch.rand(6,4); cb3[:,2:]=cb3[:,:2]+0.1
cl3=torch.randn(6,10)*80   # huge logits -> huge focal costs
r3=m0_lex(cb3,cl3,gb3,gl3)
res["tests"]["large_cost_eps_check"]={"max_abs_cost":float(r3["C"].abs().max()),
  "eps_changed_optimum":r3["eps_changed_optimum"],"coverage":r3["coverage"],
  "note":"logits*80 令代价量级极大；若 epsilon 改变最优解则会置 eps_changed_optimum=True"}

# 4) empty cases
r4=m0_lex(torch.zeros(0,4),torch.zeros(0,10),gb,gl)
r5=m0_lex(cb,cl,torch.zeros(0,4),torch.zeros(0,dtype=torch.long))
res["tests"]["empty"]={"emptyP_cov":r4["coverage"],"emptyGT_cov":r5["coverage"]}
json.dump(res,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_redo_M0_fix.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(res,ensure_ascii=False,indent=2))
