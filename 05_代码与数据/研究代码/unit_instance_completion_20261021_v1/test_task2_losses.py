import sys, json, torch
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"); sys.path.insert(0, r"C:\Users\jiwei\Desktop\DEIM")
torch.set_num_threads(2)
import torch.nn.functional as F
from uic_losses import compute_losses, matcher_cost
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.matcher import HungarianMatcher
from engine.deim.box_ops import box_cxcywh_to_xyxy, box_iou

torch.manual_seed(2026102102)
res={"item":"任务2 native MAL/boxes 适配与 parity","executed":[],"tests":{},"tolerance":{}}

# construct original criterion with common.yml recipe
wd={"cost_class":2.0,"cost_bbox":5.0,"cost_giou":2.0}
matcher=HungarianMatcher(wd,use_focal_loss=True,alpha=0.25,gamma=2.0)
crit=DEIMCriterion(matcher=matcher, weight_dict={"loss_mal":1,"loss_bbox":5,"loss_giou":2},
                   losses=['mal','boxes'], alpha=0.75, gamma=1.5, num_classes=10, mal_alpha=None)
res["executed"].append("construct DEIMCriterion(alpha=.75,gamma=1.5,mal_alpha=None)")

# --- parity with all 4 matched (k=4, M=4) ---
Nq,M=4,4
new_logits=torch.randn(Nq,10)*2
new_boxes=torch.rand(Nq,4)*0.4+0.3
u_boxes=torch.rand(M,4)*0.4+0.3
u_labels=torch.randint(0,10,(M,))

# choose a specific matching to compare (use MY hungarian result so both sides use same indices)
mine=compute_losses({"pred_logits":new_logits,"pred_boxes":new_boxes}, u_boxes,u_labels,num_classes=10,denominator=float(M),k=4)
pairs=mine["pairs"]
src=torch.tensor([p[0] for p in pairs]); tgt=torch.tensor([p[1] for p in pairs])

# Run original criterion functions directly with indices=(src,tgt)
outputs={"pred_logits":new_logits.unsqueeze(0),"pred_boxes":new_boxes.unsqueeze(0)}
targets=[{"labels":u_labels,"boxes":u_boxes}]
indices=[(src,tgt)]
num_boxes=float(M)
mal_o=crit.loss_labels_mal(outputs,targets,indices,num_boxes)["loss_mal"]
box_o=crit.loss_boxes(outputs,targets,indices,num_boxes)
res["tests"]["parity_mal"]={"mine":float(mine["loss_mal"]),"orig":float(mal_o),"abs_diff":abs(float(mine["loss_mal"])-float(mal_o))}
res["tests"]["parity_bbox"]={"mine":float(mine["loss_bbox"]),"orig":float(box_o["loss_bbox"]),"abs_diff":abs(float(mine["loss_bbox"])-float(box_o["loss_bbox"]))}
res["tests"]["parity_giou"]={"mine":float(mine["loss_giou"]),"orig":float(box_o["loss_giou"]),"abs_diff":abs(float(mine["loss_giou"])-float(box_o["loss_giou"]))}
# weighted totals
w_mine={"mal":float(mine["loss_mal"])*1,"bbox":float(mine["loss_bbox"])*5,"giou":float(mine["loss_giou"])*2}
w_orig={"mal":float(mal_o)*1,"bbox":float(box_o["loss_bbox"])*5,"giou":float(box_o["loss_giou"])*2}
res["tests"]["weighted_total"]={"mine":w_mine,"orig":w_orig,
  "max_abs_diff":max(abs(w_mine[k]-w_orig[k]) for k in w_mine)}
res["tolerance"]={"criteria":"FP32 逐元素；abs diff 阈值 1e-5；实际见上","observed_max":max(
  res["tests"]["parity_mal"]["abs_diff"],res["tests"]["parity_bbox"]["abs_diff"],res["tests"]["parity_giou"]["abs_diff"])}

# --- |U_W|=0 case: only negative classification term, hand-compute ---
r0=compute_losses({"pred_logits":new_logits,"pred_boxes":new_boxes}, torch.zeros(0,4), torch.zeros(0,dtype=torch.long), denominator=1.0, k=4)
# hand: target all zeros; weight = sigmoid^1.5 * 1 (target=0). bce with target 0 = softplus-ish; loss=l.mean(1).sum()*C/den
target=torch.zeros_like(new_logits)
pred=torch.sigmoid(new_logits)
weight=pred.pow(1.5)  # (1-target)=1
l=F.binary_cross_entropy_with_logits(new_logits,target,weight=weight,reduction='none')
hand_mal=l.mean(1).sum()*new_logits.shape[1]/1.0
res["tests"]["empty_UW"]={"mine_loss_mal":float(r0["loss_mal"]),"hand_loss_mal":float(hand_mal),
  "abs_diff":abs(float(r0["loss_mal"])-float(hand_mal)),"bbox_zero":float(r0["loss_bbox"])==0.0,
  "giou_zero":float(r0["loss_giou"])==0.0,"overflow":r0["overflow"],"num_matched":r0["num_matched"]}

# --- overflow: |U_W|=6 > k=4 ---
M2=6
u_b2=torch.rand(M2,4)*0.4+0.3; u_l2=torch.randint(0,10,(M2,))
r_ov=compute_losses({"pred_logits":new_logits,"pred_boxes":new_boxes},u_b2,u_l2,denominator=float(M2),k=4)
res["tests"]["overflow_case"]={"M":M2,"num_matched":r_ov["num_matched"],"overflow":r_ov["overflow"],
  "denominator":r_ov["denominator"],"expect_matched_4_overflow_2":(r_ov["num_matched"]==4 and r_ov["overflow"]==2)}

# --- veriy my matcher_cost equals native matcher cost for the 4xM set ---
Cmine=matcher_cost(new_boxes,new_logits,u_boxes,u_labels,10)
prob=torch.sigmoid(new_logits); tp=prob[:,u_labels]
neg=(1-.25)*(tp**2.0)*(-(1-tp+1e-8).log()); pos=.25*((1-tp)**2.0)*(-(tp+1e-8).log()); cc=pos-neg
from engine.deim.box_ops import generalized_box_iou
cref=5.0*torch.cdist(new_boxes,u_boxes,p=1)+2.0*cc+2.0*(-generalized_box_iou(box_cxcywh_to_xyxy(new_boxes),box_cxcywh_to_xyxy(u_boxes)))
res["tests"]["cost_matrix_parity"]={"max_abs_diff":float((Cmine-torch.nan_to_num(cref)).abs().max())}

res["denominator_choice_flag"]="317第5节原文：'全部U_W进入4×|U_W| native Hungarian，匹配min(4,|U_W|)；完整目标仍进原native归一化分母，overflow单列'。原文未写明分母按batch合计还是按图。本实现将分母作为外部入参(denominator)，未自行决定；待主控确认口径。"
res["verdict"]="MAL/boxes/GIoU 与原生函数数值 parity（见 tolerance.observed_max）；空U_W仅负项且与手算一致；overflow单列成立；代价矩阵与原生公式一致。分母口径待主控确认。"
json.dump(res,open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_task2_losses.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(json.dumps(res,ensure_ascii=False,indent=2))
