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
    master=ROOT/"稿件"/"MPSMA-DEIM_conference_EN_revised.docx"
    assert sha(master)=="0e76fa742521cd1cdcacf9e610347af84e14186d8529816304f302db6d6dbe4c"
    proposal=HERE/"proposed_BASE_FULL_FRESH_protocol.json";p=read(proposal)
    assert p["status"]=="PROPOSED_NOT_AUTHORIZED_NOT_STARTED"
    assert p["budget"]["max_full_validation_passes_total"]==640
    assert p["budget"]["max_optimizer_update_attempts_total"]==1034880
    audit=HERE/"frozen_inputs_and_resource_audit.json";a=read(audit)
    assert len(a["frozen_files_verified"])==247 and a["common_initial_tensor_key_count"]==554
    assert p["local_input_audit"]["sha256"]==sha(audit)
    plan_text=HERE/"最小主对照补强方案_待授权.txt"
    index40=ROOT/"修订记录_20261103"/"主要结论证据索引"/"主要结论证据索引_第四十轮.json"
    acceptance40=ROOT/"修订记录_20261103"/"第四十轮训练分支来源核对_20261112"/"current_goal_acceptance_audit_rev40.json"
    progress_path=HERE/"active_goal_progress_proposal_20261112.json"
    progress=write(progress_path,{
        "goal_turn_classification":"progress","previous_goal_turn_classification":"progress","observed_utc":NOW,"goal_status":"active",
        "manuscript_revision":40,"manuscript":rec(master),"current_evidence_index":rec(index40),"current_acceptance_audit":rec(acceptance40),
        "proposal":rec(proposal),"human_readable_plan":rec(plan_text),"input_and_resource_audit":rec(audit),
        "concrete_progress":["Converted the remaining history-confounding question into an exact bounded two-arm proposal.","Rehashed both initial checkpoints,247 frozen inputs, the1.74GB dataset archive and its manifest.","Verified the two frozen configurations differ only in backbone declaration/settings and output path.","Replaced the old two-day estimate with recorded40.5/59.2h round-time references and a120GiB D-drive reservation.","Asked for new explicit authorization; prior budgets are not reused."],
        "pending_user_decision":{"status":"AWAITING_EXPLICIT_REPLY","options":["Authorize BASE_FRESH and FULL_FRESH,320 rounds each, within the saved proposal","Keep the proposal without new training"],"scope":"The additional main pair only; no extra seed, module arm, capacity-control arm, tuning or speed benchmark.","user_requirement":"The user explicitly required new experiments to wait for clear authorization.","dependent_training_started":False},
        "new_model_forwards":0,"new_evaluator_runs":0,"new_training_runs":0,"scientific_manuscript_claims_changed":False,
        "no_completion_claim":True,
    })
    state_path=ROOT/"远程公平对照_20261103"/"模块消融"/"current_execution_state.json"
    shutil.copyfile(state_path,HERE/"current_execution_state_before_proposal.json")
    state=read(state_path)
    state.update({"observed_local_utc":NOW,"operational_work_status":"REV40_RETAINED; BOUNDED_FRESH_PAIR_PROPOSAL_AND_INPUTS_PREPARED; AWAITING_NEW_EXPLICIT_AUTHORIZATION; NO_NEW_TRAINING_STARTED","goal_status":"active","last_goal_turn_classification":"progress","latest_active_goal_progress":str(progress_path),"latest_active_goal_progress_sha256":progress["sha256"],"next_action":"Wait for the user's explicit decision on the new bounded BASE_FRESH/FULL_FRESH proposal. No dependent model execution or server launch before authorization.","pending_additional_training_proposal":{"file":str(proposal),"sha256":sha(proposal),"status":p["status"],"approval_received":False,"human_readable_plan":str(plan_text)},"current_pending_decision_episode":{"id":"FRESH_PAIR_AUTHORIZATION_20261112","first_observed_utc":NOW,"goal_turns_observing_this_pending_decision":1,"different_from_historical_server_access_blocks":True},"paper_submission_ready":False})
    write(state_path,state)
    current=ROOT/"当前修订入口.txt";shutil.copyfile(current,HERE/"当前修订入口_before_proposal.txt")
    note=(
        "【新增主对照补强提案，待授权；稿件仍为rev40】\n"
        "现有恢复历史无法仅靠重读数据消除。已准备BASE_FRESH/FULL_FRESH各320轮，统一seed0、batch4、FP32、数据/配方/评价；最多640次548图验证，额外总更新尝试1034880。只用仍有效的已提供服务器，双卡与持久化预检通过后方可运行，不增组、不调参、不购买资源。\n"
        "历史轮内耗时BASE40.5h/FULL59.2h，共99.7GPUh；并行应按3天以上预留，另有备份停顿，单臂硬超时7天。新备份D盘预留120GiB，当前约495GiB可用。两份初始权重、247份冻结文件、1.74GB数据归档及清单SHA已核对。\n"
        "该提案只补完整配置主对照，仍不隔离MSCA容量效应。用户此前明确要求新实验另行授权，旧预算已经完成；已发出授权选择题，尚无明确答复，未启动任何训练/评测。\n"
        f"可审阅方案：{rec(plan_text)['file']}\n机器协议：{rec(proposal)['file']}\n协议SHA256：{sha(proposal)}\n\n"
    )
    current.write_text(note+current.read_text(encoding="utf-8-sig"),encoding="utf-8")
    for f in [ROOT/"目标自主执行入口_20261109.txt",ROOT/"修订记录_20261103"/"修订说明与未解决问题.txt",ROOT/"远程公平对照_20261103"/"结果并入与验收清单.txt"]:
        f.write_text(note+"以下为既有修订记录。\n\n"+f.read_text(encoding="utf-8-sig"),encoding="utf-8")
    (HERE/"proposal_status_note.txt").write_text(note,encoding="utf-8")
    verify={"status":"PROPOSAL_READY_FOR_USER_DECISION_NO_RUN_STARTED","checked_utc":NOW,"proposal":rec(proposal),"plan":rec(plan_text),"inputs_audit":rec(audit),"progress":rec(progress_path),"current_state":rec(state_path),"unchanged_rev40_manuscript":rec(master),"authorization_received":False,"training_started":False,"goal_status":"active"}
    assert read(state_path)["pending_additional_training_proposal"]["sha256"]==sha(proposal)
    write(HERE/"proposal_readback.json",verify)
    print(json.dumps(verify,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
