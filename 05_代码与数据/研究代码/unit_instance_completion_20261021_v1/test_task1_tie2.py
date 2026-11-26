import sys, json, torch
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"); sys.path.insert(0, r"C:\Users\jiwei\Desktop\DEIM")
torch.set_num_threads(2)
from uic_m0 import m0_lex, iou_xyxy
from engine.deim.box_ops import box_cxcywh_to_xyxy
res=json.load(open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_task1_m0.json",encoding="utf-8"))

gb=torch.tensor([[0.5,0.5,0.2,0.2]]); gl=torch.tensor([0])
cb=torch.tensor([[0.45,0.5,0.2,0.2],[0.55,0.5,0.2,0.2]]); cl=torch.zeros(2,10); cl[:,0]=3.0
rA=m0_lex(cb,cl,gb,gl,cand_index=[0,1])
perm=torch.tensor([1,0])
rB=m0_lex(cb[perm],cl[perm],gb,gl,cand_index=[1,0])
# rB rows: row0=orig1,row1=orig0. owner is per-row; map to orig index
own_orig=torch.full((2,),-1)
for row,orig in enumerate([1,0]):
    own_orig[orig]=rB["cand_owner"][row]
res["tests"]["genuine_cost_tie"]={
 "owner_A":rA["cand_owner"].tolist(),"owner_B_mapped_to_orig":own_orig.tolist(),
 "same_choice_under_reorder": bool(torch.equal(rA["cand_owner"],own_orig)),
 "picked_smallest_orig_index": rA["cand_owner"][0].item()==0 and rA["cand_owner"][1].item()==-1}

# also permute GT and ensure owner stable
gb2=torch.cat([gb, torch.tensor([[0.1,0.1,0.1,0.1]])]); gl2=torch.tensor([0,5])
cb2=torch.tensor([[0.45,0.5,0.2,0.2],[0.55,0.5,0.2,0.2]]); cl2=torch.zeros(2,10); cl2[:,0]=3.0
rG=m0_lex(cb2,cl2,gb2,gl2,cand_index=[0,1])
gperm=torch.tensor([1,0])
rGp=m0_lex(cb2,cl2,gb2[gperm],gl2[gperm],cand_index=[0,1])
# map gt back
expl_orig=torch.zeros(2,dtype=torch.bool); expl_orig[0]=rGp["gt_explained"][1]; expl_orig[1]=rGp["gt_explained"][0]
res["tests"]["gt_permute_invariance"]={"same":bool(torch.equal(rG["gt_explained"],expl_orig))}

json.dump(res, open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_task1_m0.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps({"genuine_cost_tie":res["tests"]["genuine_cost_tie"],"gt_permute_invariance":res["tests"]["gt_permute_invariance"]},ensure_ascii=False,indent=2))
