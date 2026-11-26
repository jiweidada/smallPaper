from pathlib import Path
import json,hashlib,shutil,copy
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
BOARD=BASE/"01_候选判别表.json";STATE=BASE/"15_当前状态与结果校验.json"
PREVIOUS=BASE/"A46_同图类别失衡核查与必要验证_20261101_v1/06_A46结果与历史保留校验回执.json"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
prior=load(PREVIOUS);prior_sha=sha(PREVIOUS)
assert prior["status"]=="PASS"
for p,h in prior["current_files_sha256"].items():assert sha(p)==h,p
older_checked=0
for field in ["A45_all8_signed_inputs_resolved","older_107_signed_inputs_resolved","older_105_signed_inputs_resolved"]:
    for p,row in prior[field].items():
        assert sha(row["preserved_at"])==row["sha256"],p
        older_checked+=1
assert older_checked==73
assert sha(prior["A45_signed_receipt"])==prior["A45_signed_receipt_sha256"]
old=load(ROOT/"offline_complete.json")
new=load(ROOT/"remote_downloaded_30172/D0_results_v1/complete.json")
assert old["status"]==new["status"]=="COMPLETE"
assert new["counters"]=={"model_forwards":32,"model_backward_calls":0,"optimizer_updates":0}
assert new["parameter_hash_before"]==new["parameter_hash_after"] and new["state_hash_before"]==new["state_hash_after"]
assert load(ROOT/"remote_download_verified.json")["status"]=="PASS"
assert load(ROOT/"remote_independent_verification.json")["status"]=="PASS"
assert load(ROOT/"offline_independent_verification_v2.json")["status"]=="PASS"
assert not new["screen"]["prominent_native_primary_sensitivity"]
assert all(not s["prespecified_screen"]["prominent_native_primary_sensitivity"] for s in old["summaries"].values())
for p,h in old["source_sha256"].items():assert sha(p)==h,p
term=load(ROOT/"remote_terminal_observation.json")
assert term["status"]=="EXECUTED" and term["exit"]==0
term_data=json.loads(term["stdout"])
assert not term_data["owned_worker_identity_alive"] and term_data["nvidia_smi_exit"]==0
snapshot=ROOT/"before_A47";snapshot.mkdir(exist_ok=False)
resolved={}
for p,h in prior["current_files_sha256"].items():
    src=Path(p);preserved=src
    if src.resolve() in [BOARD.resolve(),STATE.resolve()]:
        preserved=snapshot/src.name;shutil.copyfile(src,preserved)
    assert sha(preserved)==h
    resolved[p]={"preserved_at":str(preserved),"sha256":h}
save(ROOT/"before_A47_manifest.json",{"previous_receipt":str(PREVIOUS),"previous_receipt_sha256":prior_sha,"all49_inputs":resolved})
board=load(BOARD);state=load(STATE)
assert state["model_forwards"]==83931 and state["training_updates"]==58793 and state["formal_training_epochs"]==72
protected={k:copy.deepcopy(state.get(k)) for k in ["A42_terminal_result","A43_terminal_result","A44_recommendation_withdrawal","A46_method_and_diagnostics"]}
phase={
 "phase":"A47_MATCHING_SENSITIVITY_CANDIDATE_AND_NECESSARY_DIAGNOSTICS",
 "client_date":"2026-11-01","recorded_utc":datetime.now(timezone.utc).isoformat(),
 "status":"NOT_ADVANCED_AFTER_PREDECLARED_OFFLINE_AND_ECDET_SCREENS","stage_complete":True,
 "stage_scope":"This bounded candidate only; broader mainline search remains incomplete.",
 "mainline_validated":False,"current_priority_candidate":None,"pilot_recommended":False,
 "question":"Bounded original-coordinate annotation edge jitter and tiny-object assignment/positive-class identity sensitivity.",
 "old_route_ids":["D035","D036","D062","D073"],"old_stopped_routes_restarted":False,
 "offline":{"models":"two existing BASE24 baseline seeds, same val128","GT":7931,"tiny_GT":2372,"new_model_forwards":0,"seconds":old["elapsed_seconds"]},
 "remote":{"model":"ECDet-S A43 B epoch12 EMA","split":"train128","GT":7306,"tiny_GT":1909,
           "new_model_forwards":32,"seconds":new["elapsed_seconds"],"exit_code":0,"weights_unchanged":True,
           "data_root":new["data_root"],"isolated_root":"/root/uav_a47_20261101_v1","source_identity_verified":True},
 "new_model_forwards":32,"new_model_backward_calls":0,"new_optimizer_updates":0,"new_training_epochs":0,"new_training_jobs":0,
 "new_forward_cap":32,"new_forward_cap_closed":True,"remote_access":True,
 "ordinary_NWD_training_sufficiency_empirically_proven":False,"all_matching_or_label_noise_research_rejected":False,
 "new_algorithm_training_failure_claimed":False,"natural_annotation_noise_identified":False,
 "report":str(ROOT/"05_下一候选方法核查与验证结论.txt"),
 "literature":str(ROOT/"01_核心文献矩阵与逐项对比.json"),"receipt":str(ROOT/"06_A47结果与历史保留校验回执.json"),
 "offline_verification":str(ROOT/"offline_independent_verification_v2.json"),
 "remote_verification":str(ROOT/"remote_independent_verification.json"),
 "goal_completion_tool_receipt":str(ROOT/"goal_completion_tool_receipt.json")}
for obj in [board,state]:
    obj["A47_method_and_diagnostics"]=phase
    obj["latest_report"]=phase["report"]
    obj["mainline_validated"]=False;obj["mainline_selected_for_pilot"]=False
    obj["current_priority_candidate"]=None
    obj["current_priority_question"]="当前无已确认主线。A47半像素标注扰动候选未达预设投入门槛；不调幅度、C或子集补救，不启动新训练。"
    obj["goal_status"]="complete"
    obj["goal_status_scope"]="A47 bounded candidate investigation and necessary diagnostics complete; broader paper/mainline objective remains incomplete."
board["status"]="NO_NEW_MAINLINE_A47_MATCH_SENSITIVITY_NOT_ADVANCED"
board["new_empirical_evidence"]="Two old BASE24 cache seeds + one genuine ECDet-S fixed-state run;32 new detector forwards,0updates;all prespecified primary practical gates failed."
candidate_id="A47_BOUNDED_ANNOTATION_MATCH_SENSITIVITY"
assert all(x.get("id")!=candidate_id for x in board["candidates"])
board["candidates"].append({"id":candidate_id,"question":phase["question"],"status":phase["status"],
     "old_ids":phase["old_route_ids"],"increment_approved":False,"new_training_jobs":0,
     "result_scope":"Synthetic bounded annotation jitter at fixed outputs, not actual noise, causal training harm or AP.",
     "next":"Stop the current half-pixel candidate; no automatic amplitude/C/subgroup rescue, module implementation or training."})
state.setdefault("authoritative_history",[]).append({"phase":"before_A47","preserved_state":str(snapshot/STATE.name),"previous_receipt":str(PREVIOUS),"previous_receipt_sha256":prior_sha})
state["latest_authoritative_phase"]="A47_MATCHING_SENSITIVITY_CANDIDATE_AND_NECESSARY_DIAGNOSTICS"
state["current_phase"]="A47"
state["current_user_task_status"]=phase["status"]
state["model_forwards"]=83963
state["frozen_diagnostic_model_forwards_A47"]=32
state["counts_terminal"]=True
state["full_forward_cap_closed"]=True
state["cap_scope_A47"]={"previous_total_model_forwards":83931,"new_phase_cap":32,"new_phase_actual":32,"closed":True,"prior_caps_are_historical":True}
state["formal_training_epochs_scope"]="Sum across historical A42/A43 arms is72; A47 adds0 and has no training arms."
for k,v in protected.items():assert state.get(k)==v,k
assert state["training_updates"]==58793 and state["formal_training_epochs"]==72
save(BOARD,board);save(STATE,state)
assert sha(PREVIOUS)==prior_sha
for p,row in resolved.items():assert sha(row["preserved_at"])==row["sha256"],p
files={}
excluded={"pip_cache","deploy_wheels","deploy_wheels_pypi_v1","before_A47","__pycache__"}
for p in ROOT.rglob("*"):
    if p.is_file() and not any(x in excluded for x in p.relative_to(ROOT).parts) and p.name not in {"06_A47结果与历史保留校验回执.json","goal_completion_tool_receipt.json"}:
        files[str(p)]=sha(p)
for p in [BOARD,STATE]:files[str(p)]=sha(p)
receipt={"status":"PASS","phase":phase["phase"],"current_stage_complete":True,"broader_mainline_found":False,
 "mainline_validated":False,"specific_candidate_advanced":False,"new_model_forwards":32,"new_optimizer_updates":0,
 "new_training_epochs":0,"remote_exit_code":0,"remote_worker_identity_absent":True,"remote_data_and_modern_env_reused":True,
 "total_model_forwards":83963,"unchanged_optimizer_updates":58793,"unchanged_formal_training_epochs":72,
 "old_results_and_stopping_decisions_preserved":True,"previous_A46_receipt":str(PREVIOUS),"previous_A46_receipt_sha256":prior_sha,
 "previous_A46_all49_inputs_resolved":resolved,"older_A45_A107_A105_preserved_inputs_checked":73,
 "older_input_resolution":{k:prior[k] for k in ["A45_all8_signed_inputs_resolved","older_107_signed_inputs_resolved","older_105_signed_inputs_resolved"]},
 "current_files_sha256":files,"goal_completion_tool_receipt":str(ROOT/"goal_completion_tool_receipt.json")}
save(ROOT/"06_A47结果与历史保留校验回执.json",receipt)
print(json.dumps({"status":"PASS","current_files_signed":len(files),"A46_inputs_preserved":len(resolved),
       "older_inputs_checked":73,"new_forwards":32,"new_training_updates":0,"total_forwards":83963,"mainline_found":False}))

