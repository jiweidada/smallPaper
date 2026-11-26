from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil

ROOT=Path(r"C:\Users\jiwei\Desktop\小论文投稿\01_会议论文")
HERE=Path(__file__).resolve().parent
NOW=dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def rec(p):return {"file":p.relative_to(ROOT).as_posix(),"sha256":sha(p)}
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");return rec(p)

def main():
    proposal=HERE/"proposed_BASE_FULL_FRESH_protocol.json"
    assert sha(proposal)=="fa4a0f72bec33b54218ba0948908f087f790c69b061108feff61e4bf3a6b33f7"
    decision_path=HERE/"用户决定_保留方案暂不训练_20261112.json"
    decision=write(decision_path,{
        "status":"DEFERRED_BY_USER_NO_NEW_TRAINING","recorded_utc":NOW,
        "question_id":"call_8db17ef45bd84b5d8a7f96e506c4dbee",
        "user_answer":"先保留方案，暂不训练","proposal":rec(proposal),
        "scope":"BASE_FRESH and FULL_FRESH,320 additional rounds each, remain a saved proposal. No new model preflight, training or complete validation is authorized by this answer.",
        "training_authorized":False,"response_received":True,"reask_only_if_user_changes_direction":True,
        "paper_goal_paused_by_user":False,"manuscript_claims_or_results_changed":False,
    })
    original_plan=HERE/"最小主对照补强方案_待授权.txt"
    held_plan=HERE/"最小主对照补强方案_已保留暂不训练.txt"
    old=original_plan.read_text(encoding="utf-8-sig")
    held_plan.write_text("当前决定：用户于"+NOW+"明确回复“先保留方案，暂不训练”。方案保留，不启动新实验；下面是原提案供以后复用，不能当作已获授权。\n\n"+old,encoding="utf-8")
    progress_path=HERE/"active_goal_progress_user_deferral_20261112.json"
    progress=write(progress_path,{
        "goal_turn_classification":"progress","observed_utc":NOW,"goal_status":"active",
        "user_decision":decision,"saved_plan":rec(held_plan),"training_authorized":False,
        "next_action":"Complete the existing-material source audit and record the empirical limits; do not ask again or start the deferred pair unless the user changes the decision.",
        "new_training_runs":0,"model_forwards":0,"evaluator_runs":0,
    })
    state_path=ROOT/"远程公平对照_20261103"/"模块消融"/"current_execution_state.json"
    shutil.copyfile(state_path,HERE/"current_execution_state_before_user_deferral.json")
    state=read(state_path)
    state.update({"observed_local_utc":NOW,"goal_status":"active","operational_work_status":"REV40_RETAINED; USER_DEFERRED_NEW_FRESH_PAIR; SAVED_PROPOSAL_ONLY; EXISTING_MATERIAL_AUDIT_CONTINUES","last_goal_turn_classification":"progress","latest_active_goal_progress":str(progress_path),"latest_active_goal_progress_sha256":progress["sha256"],"next_action":"Audit remaining existing source rows; keep empirical limits explicit. No new training or repeated authorization request while the user's deferral stands."})
    state["pending_additional_training_proposal"].update({"status":"DEFERRED_BY_USER_NO_NEW_TRAINING","response_received":True,"approval_received":False,"training_authorized":False,"decision":str(decision_path),"decision_sha256":decision["sha256"],"human_readable_plan":str(held_plan)})
    if "current_pending_decision_episode" in state:
        state["current_pending_decision_episode"].update({"status":"ANSWERED_DEFER_TRAINING","answered_utc":NOW,"unanswered":False})
    write(state_path,state)
    current=ROOT/"当前修订入口.txt";previous=current.read_text(encoding="utf-8-sig")
    shutil.copyfile(current,HERE/"当前修订入口_before_user_deferral.txt")
    anchor="当前第40轮维护稿";assert anchor in previous
    body=previous[previous.index(anchor):]
    note=("【用户已决定：保留补强方案，暂不训练】\n"
          "新增BASE_FRESH/FULL_FRESH的提案保留。用户明确回复“先保留方案，暂不训练”，当前不启动新预检、训练或额外完整验证，也不重复催问；只在用户改变决定后重新处理。论文目标未被要求暂停，稿件仍为rev40，现有重要实验限制继续保留。\n"
          f"决定记录：{rec(decision_path)['file']}\n保留方案：{rec(held_plan)['file']}\n\n")
    current.write_text(note+body,encoding="utf-8")
    (HERE/"proposal_status_note.txt").write_text(note,encoding="utf-8")
    for p in [ROOT/"目标自主执行入口_20261109.txt",ROOT/"修订记录_20261103"/"修订说明与未解决问题.txt",ROOT/"远程公平对照_20261103"/"结果并入与验收清单.txt"]:
        p.write_text(note+"以下为历史记录。\n\n"+p.read_text(encoding="utf-8-sig"),encoding="utf-8")
    print(json.dumps({"decision":decision,"held_plan":rec(held_plan),"current_state":rec(state_path),"training_authorized":False,"goal_status":"active"},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
