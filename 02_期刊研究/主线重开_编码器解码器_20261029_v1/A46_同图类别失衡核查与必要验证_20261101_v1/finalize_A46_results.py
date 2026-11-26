from pathlib import Path
import json,hashlib,shutil,copy
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
OLD=BASE/"A45_旧路线之外重新选题_20261031_v1/04_历史校验与零实验回执.json"
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for z in iter(lambda:f.read(1<<20),b""):h.update(z)
 return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
old=read(OLD);old_hash=sha(OLD)
assert old["status"]=="PASS"
for path,expected in old["current_files_sha256"].items():assert sha(path)==expected,path
# Retain exact versions named by older signed receipts; never rewrite those receipts.
for k in ["signed_107_all12_inputs_resolved","signed_105_all53_inputs_resolved"]:
 for original,v in old[k].items():assert sha(v["preserved_at"])==v["sha256"],original
for prefix in ["signed_107","signed_105"]:
 assert sha(old[prefix+"_receipt"])==old[prefix+"_receipt_sha256"]

d0=read(ROOT/"D0_results_v1/D0_complete.json")
d1=read(ROOT/"D1_downloaded/D1_results_v1/D1_complete.json")
assert d0["status"]==d1["status"]=="COMPLETE"
assert not d0["prespecified_practical_screen_pass"]
assert d1["head_dominant_tail_classes"]==[5]
assert d1["prespecified_head_specific_screen_pass"] is False
assert d1["counters"]=={"model_forwards":16,"logit_gradient_calls":16,"model_backward_calls":0,"optimizer_updates":0}
assert d1["runtime_counter"]["forwards"]==16 and d1["runtime_counter"]["teacher_forwards"]==0
assert d1["parameters_sha256_before"]==d1["parameters_sha256_after"]
assert d1["state_sha256_before"]==d1["state_sha256_after"]
assert sha(ROOT/"01_D0诊断协议_冻结.json")==d0["protocol_sha256"]
assert sha(ROOT/"run_D0_saved_prediction_diagnostic.py")==d0["script_sha256"]
assert sha(ROOT/"02_D1梯度来源诊断协议_冻结.json")==d1["protocol_sha256"]
assert sha(ROOT/"run_D1_native_logit_gradients.py")==d1["script_sha256"]
assert read(ROOT/"D1_independent_reaggregation_check.json")["status"]=="PASS"
assert read(ROOT/"D1_download_verified.json")["status"]=="PASS"
terminal_bridge=read(ROOT/"D1_observation_01.json")
assert terminal_bridge["exit"]==0
terminal=json.loads(terminal_bridge["stdout"])
assert terminal["identity_alive"] is False and terminal["exit"]["exit_code"]==0
assert terminal["progress"]["counters"]==d1["counters"]

# Independent per-image/class aggregation and fixed rule evaluation.
rows=read(ROOT/"D1_downloaded/D1_results_v1/per_image_class_gradients.json")
assert len(rows)==448
for c,t in d1["classes"].items():
 rr=[r for r in rows if r["category_id"]==int(c)]
 assert len(rr)==64
 for key,v in t.items():
  if (key.endswith("_grad") or key.endswith("_count")) and key in rr[0]:
   assert abs(sum(r[key] for r in rr)-v)<1e-12,(c,key)
 for r in rr:
  assert sum(r[k] for k in ["positive_count","negative_head_count","negative_other_tail_count","negative_unmatched_count"])==300
 neg=sum(t[k] for k in ["negative_head_grad","negative_other_tail_grad","negative_unmatched_grad"])
 ratio=t["negative_head_grad"]/neg
 hm=t["negative_head_grad"]/t["negative_head_count"]
 um=t["negative_unmatched_grad"]/t["negative_unmatched_count"]
 assert abs(ratio-t["negative_head_fraction"])<1e-12
 assert t["head_specific_dominance"]==(t["small_positive_count"]>=8 and ratio>=.5 and hm/um>=2)

snapshot=ROOT/"before_A46";snapshot.mkdir(exist_ok=False)
remap={}
touch=[BASE/"01_候选判别表.json",BASE/"15_当前状态与结果校验.json"]
for path,expected in old["current_files_sha256"].items():
 p=Path(path)
 dest=snapshot/p.name if p in touch else p
 if p in touch:shutil.copy2(p,dest)
 assert sha(dest)==expected
 remap[path]={"preserved_at":str(dest),"sha256":expected}
save(ROOT/"before_A46_manifest.json",remap)
board=read(touch[0]);state=read(touch[1])
a43_before=copy.deepcopy(board["A43_terminal_result"])
a44_before=copy.deepcopy(board["A44_recommendation_withdrawal"])
state_a43_before=copy.deepcopy(state["A43_terminal_verification"])
assert state["model_forwards"]==83915
assert state["training_updates"]==58793
assert state["formal_training_epochs"]==72
now=datetime.now(timezone.utc).isoformat()
receipt_path=ROOT/"06_A46结果与历史保留校验回执.json"
phase={
 "phase":"A46_METHOD_COMPARISON_AND_NECESSARY_DIAGNOSTICS",
 "client_date":"2026-11-01","recorded_utc":now,
 "status":"HEAD_SPECIFIC_NEGATIVE_SUPPRESSION_NOT_ADVANCED_AFTER_D0_D1",
 "mainline_validated":False,"current_priority_candidate":None,"pilot_recommended":False,
 "stage_complete":"current bounded question/diagnostics only; broader mainline search remains incomplete",
 "D0":"saved predictions of two original fixed12 seeds; observational common support inadequate",
 "D1":"64 train images,16 frozen model calls,16 final-logit gradient calls,0 optimizer steps; only van met predeclared head-dominance rule",
 "new_model_forwards":16,"new_logit_gradient_calls":16,"new_model_backward_calls":0,
 "new_optimizer_updates":0,"new_training_epochs":0,"new_training_jobs":0,
 "remote_access":True,"old_routes_restarted":False,"new_forward_cap":16,"new_forward_cap_closed":True,
 "ordinary_EFL_EQFL_sufficiency_empirically_proven":False,
 "all_long_tail_research_rejected":False,"van_head_gradient_source_proven_to_be_car":False,
 "report":str(ROOT/"05_方法级裁决与必要验证结果.txt"),"receipt":str(receipt_path),
 "D0_result":str(ROOT/"D0_results_v1/D0_complete.json"),
 "D1_result":str(ROOT/"D1_downloaded/D1_results_v1/D1_complete.json"),
 "literature":str(ROOT/"04_方法对照与检索阅读记录.json"),
 "goal_completion_tool_receipt":str(ROOT/"goal_completion_tool_receipt.json")
}
for item in [board,state]:
 item["A46_method_and_diagnostics"]=phase
 item["latest_report"]=phase["report"]
 item["mainline_validated"]=False
 item["mainline_selected_for_pilot"]=False
 item["current_priority_question"]="A46头类前景负梯度特异抑制改法暂不推进：D0支持不足、D1仅van满足预设条件；未选择新主线、无新训练队列。"
 item["goal_status"]="complete"
 item["goal_status_scope"]="A46 bounded method-level followup and necessary diagnostics complete; broader paper/mainline objective remains incomplete."
board["status"]="NO_NEW_MAINLINE_A46_HEAD_SPECIFIC_RATIONALE_NOT_ADVANCED"
board["current_priority_candidate"]=None
board["latest_update_utc"]=now
state.setdefault("authoritative_phase_history",[]).append(state["latest_authoritative_phase"])
state["latest_authoritative_phase"]=phase
state["current_user_task_phase"]=phase
state["current_user_task_status"]=phase["status"]
state["recorded_utc"]=now
state["goal_completion_tool_receipt"]=phase["goal_completion_tool_receipt"]
state["model_forwards"]=83931
state["frozen_diagnostic_model_forwards_A46"]=16
state["final_logit_gradient_calls_A46"]=16
state["counts_terminal"]=True
state["cap_scope_A46"]="Fresh explicit user authorization;16 frozen-model forward cap fulfilled and closed. Previous cap fields retain their historical scopes."
assert board["A43_terminal_result"]==a43_before and board["A44_recommendation_withdrawal"]==a44_before
assert state["A43_terminal_verification"]==state_a43_before
assert state["training_updates"]==58793 and state["formal_detector_training_updates"]==58774
assert state["capacity_only_optimizer_updates"]==19 and state["formal_training_epochs"]==72
for p,x in zip(touch,[board,state]):save(p,x)
assert sha(OLD)==old_hash
for original,v in remap.items():assert sha(v["preserved_at"])==v["sha256"]

paths=[]
for p in ROOT.rglob("*"):
 if not p.is_file():continue
 if p==receipt_path or p.name=="goal_completion_tool_receipt.json" or "__pycache__" in p.parts:continue
 if "before_A46" in p.parts or "sources" in p.parts:continue
 paths.append(p)
paths+=touch
signed={str(p):sha(p) for p in sorted(set(paths))}
receipt={
 "status":"PASS","phase":"A46_METHOD_COMPARISON_AND_NECESSARY_DIAGNOSTICS","recorded_utc":now,
 "current_stage_complete":True,"broader_mainline_found":False,"mainline_validated":False,
 "specific_route_not_advanced":"head-matched foreground negative-gradient suppression",
 "new_model_forwards":16,"new_logit_gradient_calls":16,"new_optimizer_updates":0,
 "new_training_epochs":0,"new_training_jobs":0,"remote_access":True,"training_queue_created":False,
 "model_state_unchanged_in_D1":True,"D0_original_aggregate_AP_reconstructed":True,
 "D0_scope":"candidate-coverage association, not TP/AP benefit;14 low-exposure common-support images",
 "D1_scope":"one fixed model state, final classification logits only; no full model backward/causal/AP test",
 "D1_prespecified_supported_classes":7,"D1_head_dominant_classes":[5],
 "D1_execution_exit_code":0,"D1_worker_identity_absent_at_recorded_observation":True,
 "post_download_GPU_query_succeeded":False,
 "total_model_forwards":83931,"unchanged_optimizer_updates":58793,"unchanged_formal_training_epochs_sum":72,
 "A42_A43_results_and_A44_withdrawal_preserved":True,
 "A45_signed_receipt":str(OLD),"A45_signed_receipt_sha256":old_hash,
 "A45_all8_signed_inputs_resolved":remap,
 "older_107_signed_inputs_resolved":old["signed_107_all12_inputs_resolved"],
 "older_105_signed_inputs_resolved":old["signed_105_all53_inputs_resolved"],
 "current_files_sha256":signed,
 "goal_completion_tool_receipt":phase["goal_completion_tool_receipt"]
}
save(receipt_path,receipt)
print(json.dumps({"status":"PASS","D0":"complete; support inadequate","D1":"complete; screen not passed",
 "new_model_forwards":16,"new_optimizer_updates":0,"total_model_forwards":83931,
 "old_A45_inputs_preserved":len(remap),"old_A107_verified":12,"old_A105_verified":53,
 "signed_current_files":len(signed),"report":phase["report"]},ensure_ascii=False))

