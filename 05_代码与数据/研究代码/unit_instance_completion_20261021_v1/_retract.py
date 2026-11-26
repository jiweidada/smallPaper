import json
p=r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\result_redo3_frozen311_320.json"
d=json.load(open(p,encoding="utf-8"))
d["superseded_by"]="result_redo3v2_frozen311v2.json（在 311 v2 / 320 v2 上重跑）"
d["original_nan_diagnosis_RETRACTED"]={
 "wrong_claim":"上一轮写『NaN 来自 logm=where(m>0,log,-inf) 的 -inf 分支在反传泄漏』——错误。",
 "evidence_against_it":"实测 torch.softmax 输入含 -inf 时 gradient 全 0 无 NaN；把 logm 换成 -1e30 仍有 5 个 NaN。",
 "correct_cause":"independent_cap 的 m/safe_u：u 极小（最小 4.718e-26），u*u 在 fp32 下溢为 0，d(m/u)/du=-m/u^2=-inf；torch.minimum(1,m/u) 在 1 分支给上游 0，0*(-inf)=NaN 传给 class_null。",
 "confirming_measurements":{"min_positive_u":4.7181376310816106e-26,"u_squared_underflows_cells":25,
   "u_gt0_cells":2816,"all_u_gt0_are_u_lt_m":True,"direct_divide_grad":"-inf","minimum_grad":"NaN"},
 "credit":"该定位由主控独立复现并指出；本会话原 -inf 解释作废。"}
json.dump(d,open(p,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("ok")
