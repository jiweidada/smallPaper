import sys, json, torch
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"); sys.path.insert(0, r"C:\Users\jiwei\Desktop\DEIM")
torch.set_num_threads(2)
from uic_fusion import fuse_and_postprocess
torch.manual_seed(2026102104)
res=json.load(open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_task3_fusion.json",encoding="utf-8"))

B,Q,C=2,300,10
gl=torch.randn(B,Q,C)*3; gb=torch.rand(B,Q,4)*0.5+0.25
# new4 scores strictly <0.001: logits <= log(0.001/(1-0.001)) ~ -6.906
thr_logit=torch.log(torch.tensor(0.001/(1-0.001)))
nl=torch.full((B,4,C),float(thr_logit)-0.5)  # below threshold
nb=torch.rand(B,4,4)*0.5+0.25
orig=torch.tensor([[800.,1200.],[640.,480.]])

# new4 present but <0.001
out_with,_=fuse_and_postprocess(gl,gb,nl,nb,orig)
# global only (new4 = -inf-ish, must not add anything)
nl_inf=torch.full((B,4,C),-1e9); nb0=torch.zeros(B,4,4)
out_global_only,_=fuse_and_postprocess(gl,gb,nl_inf,nb0,orig)

bit_id=True; counts_same=True; maxbd=0
for b in range(B):
    a=out_with[b]; g=out_global_only[b]
    if a["labels"].shape!=g["labels"].shape: counts_same=False; bit_id=False; continue
    if not torch.equal(a["labels"],g["labels"]): bit_id=False
    if not torch.equal(a["scores"],g["scores"]): bit_id=False
    maxbd=max(maxbd,float((a["boxes"]-g["boxes"]).abs().max()))
    if not torch.equal(a["boxes"],g["boxes"]): bit_id=False
res["tests"]["initial_parity_new4_lt_0p001_equals_global_only"]={
  "bit_identical":bit_id,"counts_same":counts_same,"max_box_diff":maxbd,
  "note":"317初态parity：new4全<0.001时融合输出必须与仅global逐位相同"}
# also verify sigmoid threshold actually <0.001
mx=float(torch.sigmoid(nl).max())
res["tests"]["initial_parity_new4_lt_0p001_equals_global_only"]["new4_max_score"]=mx

# per-class max100 real trim: make class 0 dominate with 150 boxes >0.001
B2=1
g3=torch.full((1,Q,C),-9.0)
# set 200 entries of class0 high across queries*class slots: set logits for class0 on first 200 (q,class) slots
g3[0,:,0]=6.0  # 300 queries class0 high, plus new4
n3=torch.full((1,4,C),-9.0)
o3,_=fuse_and_postprocess(g3,gb[:1],n3,nb[:1],orig[:1])
n0=int((o3[0]["labels"]==0).sum())
res["tests"]["per_class_max100_trim"]={"class0_count":n0,"trimmed_to_100":n0<=100}
json.dump(res,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_task3_fusion.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps({k:res["tests"][k] for k in ["initial_parity_new4_lt_0p001_equals_global_only","per_class_max100_trim"]},ensure_ascii=False,indent=2))
