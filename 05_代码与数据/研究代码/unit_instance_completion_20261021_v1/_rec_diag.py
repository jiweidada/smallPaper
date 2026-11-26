import json
p=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\337_result_stage1_subset.json"
d=json.load(open(p,encoding="utf-8"))
diag=json.load(open(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\_window_diag.json",encoding="utf-8"))
gw=[s for s in diag if s["nGW_at_pred_cell"]>0]
d["tests"]["window_rule_diagnostic"]={
 "pred_density_rule":{"images_with_GW_gt0":len(gw),"of":len(diag),
   "mean_nGW":round(sum(s["nGW_at_pred_cell"] for s in diag)/len(diag),2)},
 "gt_density_counterfactual_DIAGNOSTIC_ONLY":{"images_with_GW_gt0":sum(1 for s in diag if s["nGW_at_gt_cell"]>0),
   "of":len(diag),"mean_nGW":round(sum(s["nGW_at_gt_cell"] for s in diag)/len(diag),2)},
 "conclusion":"按317/296的'原final300预测中心数最多'选窗，32张仅4张有完整GT(G_W>0)；若改按GT密度选窗则30张有(平均6.09个)。说明该选窗规则在BASE24 raw+VisDrone dev上系统性偏向无GT区域(预测中心聚集处多为FP/背景)。这是规则层面的问题，不是实现bug。未改规则，报请主控。",
 "impact":"若照此规则做全量缓存，多数样本 G_W/U_W 为空，L_assoc 正组为空、mass目标为0，训练信号严重不足。"}
json.dump(d,open(p,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("recorded window rule diagnostic")
