from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import shutil

ROOT=Path(r"C:\Users\jiwei\Desktop\小论文投稿\01_会议论文")
HERE=Path(__file__).resolve().parent
EPISODE="IMPORTANT_COMPARISON_EVIDENCE_MISSING_AND_NEW_RUNS_DEFERRED_BY_USER"

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def rec(p):return {"file":p.relative_to(ROOT).as_posix(),"sha256":sha(p)}
def write(p,d):
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--expected-prior-count",type=int,required=True)
    args=parser.parse_args()
    now=dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")
    state_path=ROOT/"远程公平对照_20261103"/"模块消融"/"current_execution_state.json"
    state=read(state_path)
    manuscript=ROOT/"稿件"/"MPSMA-DEIM_conference_EN_revised.docx"
    decision_path=ROOT/"修订记录_20261103"/"剩余证据与最小补强方案_20261112"/"用户决定_保留方案暂不训练_20261112.json"
    assert sha(manuscript)==state["paper_sha256"]=="0e76fa742521cd1cdcacf9e610347af84e14186d8529816304f302db6d6dbe4c"
    assert sha(decision_path)=="b2603b1e62a1a0ba7818e34edc39049ccf9d43f9e400afabb90eabcf889e2d6f"
    assert read(decision_path)["user_answer"]=="先保留方案，暂不训练"
    assert state["pending_additional_training_proposal"]["training_authorized"] is False
    assert state["current_pending_decision_episode"]["unanswered"] is False
    assert state["goal_status"]=="active" and state["paper_submission_ready"] is False
    acceptance_path=Path(state["current_goal_acceptance_audit"])
    assert sha(acceptance_path)==state["current_goal_acceptance_audit_sha256"]
    acceptance=read(acceptance_path)
    assert not acceptance["submission_ready"] and len(acceptance["important_evidence_limits_still_present"])>=4
    previous=state.get("current_deferral_blocking_episode",{})
    count=previous.get("consecutive_goal_turns",0) if previous.get("id")==EPISODE else 0
    assert count==args.expected_prior_count and count<3
    next_count=count+1
    out=HERE/f"deferred_training_blocking_turn{next_count}.json"
    backup=HERE/f"current_execution_state_before_deferral_blocking_turn{next_count}.json"
    assert not out.exists() and not backup.exists()
    shutil.copyfile(state_path,backup)
    report={
        "observed_utc":now,"classification":"no_progress","blocking_episode":EPISODE,
        "consecutive_goal_turns":next_count,"previous_goal_turn_classification":"progress" if count==0 else "no_progress",
        "manuscript":rec(manuscript),"user_decision":rec(decision_path),"current_acceptance":rec(acceptance_path),
        "original_scope_retained":True,"overall_goal_achieved":False,
        "blocking_condition":"The primary configuration comparison retains differing continuation histories and early missing main-process random states. Existing records cannot reconstruct the missing states or create matched-history evidence. The user explicitly deferred the proposed new pair. No new training is authorized, and no available source-only action has been identified that resolves this important evidence gap.",
        "other_limits":"Capacity differences, a single seed, validation selection, unquantified unretained work and unmeasured deployment latency remain disclosed. Venue selection/template adaptation is deferred and is not the sole reason for the present content-evidence gap.",
        "current_manuscript_revalidated_by_sha":True,"prior_content_and_visual_checks_carried_forward_for_identical_bytes":True,"completion_requirements":[
            {"item":1,"source_identity":"verified","comparison_design":"important gap remains"},
            {"item":2,"method_figure_parameter_consistency":"prior static and visual checks retained; no new conflict found"},
            {"item":3,"content_language_citations":"revised within documented access scope; formal full texts3 and10 remain unavailable"},
            {"item":4,"original_backup_maintained_draft_and_logs":"retained"},
        ],
        "read_only_literature_followup_in_first_turn":{
            "performed_this_turn":count==0,
            "IEEE_conference_paper":{"url":"https://xplorestaging.ieee.org/document/10640832/","observed_access":"Public indexed abstract supports the existing narrow LCE/Swin summary; direct open returned418. No formal full text obtained."},
            "Elsevier_paper":{"url":"https://www.sciencedirect.com/science/article/pii/S026288562600003X","observed_access":"Public indexed publisher abstract/highlights support the existing Mamba/attention/feature-pyramid summary; direct open returned403. No formal full text obtained."},
            "related_author_thesis":{"url":"https://uwspace.uwaterloo.ca/items/ff40f405-df90-4c0c-98af-9bcdf2e1cae0","date":"2023-05-23","author":"Weidong Liang","type":"Master thesis","inspection_scope":"Institutional repository metadata and abstract only. PDF link could not be opened by the web tool; no alternative fetch attempted.","disposition":"Related primary source, not the2024 conference article. Not substituted for it; no claim of having read either full text."},
            "changes_to_manuscript_or_submission_evidence_gap":False,
        } if count==0 else {"performed_this_turn":False,"first_turn_record":"deferred_training_blocking_turn1.json"},
        "new_training_runs":0,"model_forwards":0,"evaluator_runs":0,"new_remote_observation":False,
        "live_task_handle_confirmed":False,"verified_wait":False,
        "user_requested_pause":False,"goal_tool_status_before_turn":"active",
        "goal_tool_action":"Leave active: blocked threshold not yet met." if next_count<3 else "Blocked threshold met if independent audit confirms no meaningful safe action; call update_goal blocked and record its returned status.",
        "no_repeat_training_question":True,"new_experiment_authorized":False,
    }
    write(out,report)
    state.update({
        "observed_local_utc":now,"last_goal_turn_classification":"no_progress",
        "goal_blocking_audit_consecutive_turns":next_count,
        "current_deferral_blocking_episode":{"id":EPISODE,"first_observed_utc":previous.get("first_observed_utc",now),"last_observed_utc":now,"consecutive_goal_turns":next_count,"latest_audit":str(out),"latest_audit_sha256":sha(out),"different_from_historical_server_access_blocks":True},
        "operational_work_status":"REV40_RETAINED; EXISTING_MATERIAL_AUDIT_COMPLETE; IMPORTANT_COMPARISON_GAP_UNRESOLVED; USER_DEFERRED_NEW_TRAINING",
        "next_action":"Revalidate this same blocker on a later goal turn. Do not start the deferred pair or repeat the authorization question. Keep the goal active until the blocked-audit threshold is met; do not certify submission readiness.",
    })
    write(state_path,state)
    assert sha(manuscript)=="0e76fa742521cd1cdcacf9e610347af84e14186d8529816304f302db6d6dbe4c"
    assert read(state_path)["goal_status"]=="active"
    print(json.dumps({"classification":"no_progress","consecutive_goal_turns":next_count,"goal_status":"active","manuscript_unchanged":True,"training_authorized":False,"audit":rec(out)},ensure_ascii=False))

if __name__=="__main__":main()
