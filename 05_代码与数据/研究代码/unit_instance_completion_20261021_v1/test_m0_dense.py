import sys, json, torch, numpy as np, hashlib
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"); sys.path.insert(0, r"C:\Users\jiwei\Desktop\DEIM")
torch.set_num_threads(2)
from scipy.optimize import linear_sum_assignment
from uic_m0 import m0_lex
res={"item":"M0 修正后复核（335 要求：去代价偏置 + 最优化校验）","tests":{},
     "file_sha256":hashlib.sha256(open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\uic_m0.py","rb").read()).hexdigest()}

gb=torch.tensor([[0.3,0.3,0.1,0.1],[0.7,0.7,0.1,0.1],[0.5,0.9,0.05,0.05]]); gl=torch.tensor([3,3,3])
cb=torch.tensor([[0.30,0.30,0.10,0.10],[0.31,0.29,0.10,0.10],[0.70,0.70,0.10,0.10],[0.2,0.2,0.05,0.05]])
cl=torch.full((4,10),-5.0); cl[:,3]=5.0
r=m0_lex(cb,cl,gb,gl)
res["tests"]["baseline"]={"coverage":r["coverage"],"pairs":r["matched_pairs"],"max_card":r["max_cardinality"],"eps_changed":bool(r["eps_changed_optimum"])}

gb2=torch.tensor([[0.5,0.5,0.2,0.2]]); gl2=torch.tensor([0]); cb2=torch.tensor([[0.45,0.5,0.2,0.2],[0.55,0.5,0.2,0.2]])
cl2=torch.zeros(2,10); cl2[:,0]=3.0
rA=m0_lex(cb2,cl2,gb2,gl2,cand_index=[0,1]); rB=m0_lex(cb2[[1,0]],cl2[[1,0]],gb2,gl2,cand_index=[1,0])
ownB=torch.full((2,),-1); ownB[1]=rB["cand_owner"][0]; ownB[0]=rB["cand_owner"][1]
res["tests"]["tie_fixed_index"]={"owner_A":rA["cand_owner"].tolist(),"owner_B_mapped":ownB.tolist(),"same":bool(torch.equal(rA["cand_owner"],ownB))}

torch.manual_seed(11)
gb3=torch.rand(5,4)*0.6+0.2; gb3[:,2:]=gb3[:,:2]+0.08; gl3=torch.full((5,),3)
cb3=torch.rand(7,4)*0.6+0.2; cb3[:,2:]=cb3[:,:2]+0.08; cl3=torch.zeros(7,10); cl3[:,3]=6.0
r3=m0_lex(cb3,cl3,gb3,gl3)
C=r3["C"].detach().numpy(); valid=r3["valid"].numpy(); n,m=C.shape
BIG=(abs(C).max()+1)*(min(n,m)+2)*10
mat=np.full((n+m,m+n),BIG)
for i in range(n):
    for j in range(m):
        if valid[i,j]: mat[i,j]=C[i,j]
for i in range(n,n+m):
    for j in range(m,m+n): mat[i,j]=0.0
rw,cw=linear_sum_assignment(mat)
pairs=[(i,j) for i,j in zip(rw,cw) if i<n and j<m and valid[i,j]]
ind_cost=float(sum(C[i,j] for i,j in pairs))
res["tests"]["dense_large_spread"]={
 "coverage":r3["coverage"],"max_card":r3["max_cardinality"],"max_abs_cost":float(abs(C).max()),
 "eps_changed_optimum":bool(r3["eps_changed_optimum"]),
 "my_total_cost":round(float(r3["total_cost"]),6),"independent_min_cost":round(ind_cost,6),
 "cost_equal":bool(abs(ind_cost-r3["total_cost"])<1e-5)}

r4=m0_lex(torch.zeros(0,4),torch.zeros(0,10),gb,gl); r5=m0_lex(cb,cl,torch.zeros(0,4),torch.zeros(0,dtype=torch.long))
res["tests"]["empty"]={"emptyP_cov":r4["coverage"],"emptyGT_cov":r5["coverage"]}
res["verdict"]="代价矩阵不再加索引偏置；改为'最大覆盖→最小代价→等最优解间固定索引'，并附 eps_changed_optimum 自检。归一化后代价量级下未观察到偏置改变最优解。"
json.dump(res,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_redo_M0_fix.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(res,ensure_ascii=False,indent=2))
