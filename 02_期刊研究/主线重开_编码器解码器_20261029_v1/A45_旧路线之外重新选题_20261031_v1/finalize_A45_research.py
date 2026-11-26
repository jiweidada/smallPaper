from pathlib import Path
import json, hashlib, shutil
from datetime import datetime, timezone

BASE = Path(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\主线重开_编码器解码器_20261029_v1")
OUT = BASE / "A45_旧路线之外重新选题_20261031_v1"
RECEIPT = BASE / "107_A44撤回与历史记录校验.json"

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def write(p, value):
    Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

assert (OUT/"01_定向调研报告.txt").is_file()
records=read(OUT/"02_检索记录与来源.json")
screen=read(OUT/"03_候选去留.json")
assert records["query_count"]==len(records["queries"])==47
assert len(records["sources"])==25
assert len(screen["candidates"])==6
assert sum(c["batch"]==1 for c in screen["candidates"])==3
assert sum(c["batch"]==2 for c in screen["candidates"])==3
assert all(not c["selected_for_pilot"] and c["training_allocation"]==0 for c in screen["candidates"])
old=read(RECEIPT)
old_receipt_hash=sha(RECEIPT)
assert sha(old["signed_old_receipt"])==old["signed_old_receipt_sha256"]

resolved105={}
for original,m in old["signed_105_all53_inputs_resolved"].items():
    p=Path(m["preserved_at"])
    actual=sha(p)
    assert actual==m["sha256"], ("old105 changed",str(p))
    resolved105[original]={"preserved_at":str(p),"sha256":actual}
assert len(resolved105)==53

# Validate the former current files BEFORE preserving and editing any of them.
for name,expected in old["current_files_sha256"].items():
    assert sha(name)==expected, ("old107 current file changed",name)
assert len(old["current_files_sha256"])==12

names=["01_候选判别表.json","15_当前状态与结果校验.json"]
snapshot=OUT/"before_A45"
snapshot.mkdir(exist_ok=True)
remap107={}
for name,expected in old["current_files_sha256"].items():
    p=Path(name)
    if p.name in names and p.parent==BASE:
        dest=snapshot/p.name
        if dest.exists():
            assert sha(dest)==expected
        else:
            shutil.copy2(p,dest)
        assert sha(dest)==expected
    else:
        dest=p
    remap107[name]={"preserved_at":str(dest),"sha256":expected}
write(OUT/"before_A45_manifest.json",remap107)

board=read(BASE/names[0])
state=read(BASE/names[1])
old_a43_board=board["A43_terminal_result"]
old_a43_state=state.get("A43_terminal_verification")
expected_counts=old["unchanged_model_counters"]
for k,v in expected_counts.items():
    assert state[k]==v,(k,state[k],v)
now=datetime.now(timezone.utc).isoformat()
phase={
 "phase":"A45_BOUNDED_RESEARCH_SCREEN",
 "recorded_utc":now,
 "client_date":"2026-10-31",
 "status":"TWO_BATCHES_SIX_QUESTIONS_SCREENED_NO_NEW_MAINLINE_SELECTED",
 "stage_complete":"bounded research delivery only; broader mainline search remains incomplete",
 "mainline_validated":False,
 "current_priority_candidate":None,
 "pilot_recommended":False,
 "training_allocation":0,
 "screened_question_count":6,
 "selected_candidate_count":0,
 "new_model_forwards":0,
 "new_optimizer_updates":0,
 "new_training_jobs":0,
 "remote_access":False,
 "report":str(OUT/"01_定向调研报告.txt"),
 "search_and_reading_record":str(OUT/"02_检索记录与来源.json"),
 "decisions":str(OUT/"03_候选去留.json"),
 "receipt":str(OUT/"04_历史校验与零实验回执.json"),
 "goal_completion_tool_receipt":str(OUT/"goal_completion_tool_receipt.json")
}
for item in [board,state]:
    item["A45_research_screen"]=phase
    item["latest_report"]=phase["report"]
    item["mainline_validated"]=False
    item["mainline_selected_for_pilot"]=False
    item["current_priority_question"]="A45两批六个问题已筛查，尚无可推荐的新主线；未分配新训练。"
    item["goal_status"]="complete"
    item["goal_status_scope"]="A45 bounded research deliverable complete; broader paper/mainline objective remains incomplete and unvalidated."
board["status"]="NO_NEW_MAINLINE_AFTER_A45_TWO_BATCH_RESEARCH_SCREEN"
board["latest_update_utc"]=now
board["current_priority_candidate"]=None
state["recorded_utc"]=now
state.setdefault("authoritative_phase_history",[]).append(state["latest_authoritative_phase"])
state["latest_authoritative_phase"]=phase
state["current_user_task_phase"]=phase
state["current_user_task_status"]=phase["status"]
state["goal_completion_tool_receipt"]=phase["goal_completion_tool_receipt"]
assert board["A43_terminal_result"]==old_a43_board
assert state.get("A43_terminal_verification")==old_a43_state
for k,v in expected_counts.items():
    assert state[k]==v
write(BASE/names[0],board)
write(BASE/names[1],state)

for original,m in remap107.items():
    assert sha(m["preserved_at"])==m["sha256"]
for original,m in resolved105.items():
    assert sha(m["preserved_at"])==m["sha256"]
assert sha(old["signed_old_receipt"])==old["signed_old_receipt_sha256"]
assert sha(RECEIPT)==old_receipt_hash
new_signed=[OUT/"00_目标与检索边界.json",OUT/"01_定向调研报告.txt",
            OUT/"02_检索记录与来源.json",OUT/"03_候选去留.json",
            OUT/"before_A45_manifest.json",OUT/"finalize_A45_research.py",
            BASE/names[0],BASE/names[1]]
receipt={
 "status":"PASS",
 "phase":"A45_BOUNDED_RESEARCH_SCREEN",
 "recorded_utc":now,
 "client_date":"2026-10-31",
 "task_scope_complete":True,
 "broader_mainline_search_complete":False,
 "mainline_selected":False,
 "mainline_validated":False,
 "recommendations":[],
 "screened_questions":6,
 "research_batches":2,
 "search_query_entries":47,
 "primary_source_or_resource_records":25,
 "fulltext_read_all_records":False,
 "experimental_failures_of_new_questions_claimed":False,
 "no_field_exhaustion_or_publishability_claim":True,
 "new_model_forwards":0,
 "new_optimizer_updates":0,
 "new_training_jobs":0,
 "new_probe_fits":0,
 "new_model_re_evaluations":0,
 "remote_access":False,
 "training_queue_created":False,
 "unchanged_model_counters":expected_counts,
 "A43_terminal_values_unchanged_in_current_records":True,
 "A44_withdrawal_retained":board["A44_recommendation_withdrawal"]["recommendation_withdrawn"] if "recommendation_withdrawn" in board["A44_recommendation_withdrawal"] else board["A44_recommendation_withdrawal"]["status"].startswith("WITHDRAWN"),
 "signed_107_receipt":str(RECEIPT),
 "signed_107_receipt_sha256":sha(RECEIPT),
 "signed_107_all12_inputs_resolved":remap107,
 "signed_105_receipt":old["signed_old_receipt"],
 "signed_105_receipt_sha256":old["signed_old_receipt_sha256"],
 "signed_105_all53_inputs_resolved":resolved105,
 "current_files_sha256":{str(p):sha(p) for p in new_signed},
 "goal_completion_tool_receipt":phase["goal_completion_tool_receipt"],
 "audit_limit":"Local provenance and no-model execution of this stage; not a new audit of training validity, source methods or empirical effectiveness."
}
assert receipt["A44_withdrawal_retained"] is True
write(OUT/"04_历史校验与零实验回执.json",receipt)
print(json.dumps({
 "status":"PASS","report_characters":len((OUT/"01_定向调研报告.txt").read_text(encoding="utf-8")),
 "queries":47,"questions":6,"sources":25,"selected":0,
 "old107_inputs_verified":len(remap107),"old105_inputs_verified":len(resolved105),
 "new_model_forwards":0,"new_optimizer_updates":0,"remote_access":False,
 "unchanged_counts":expected_counts
},ensure_ascii=False))

