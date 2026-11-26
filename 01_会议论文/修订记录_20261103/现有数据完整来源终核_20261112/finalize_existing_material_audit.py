from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil

ROOT=Path(r"C:\Users\jiwei\Desktop\小论文投稿\01_会议论文")
HERE=Path(__file__).resolve().parent
RECORDS=ROOT/"修订记录_20261103"
NOW=dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def rec(p):return {"file":p.relative_to(ROOT).as_posix(),"sha256":sha(p)}
def write(p,data):
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return rec(p)
def backup(p):
    out=HERE/(p.name+".before_full_source_finalization")
    assert not out.exists(),f"Refusing to overwrite prior backup: {out}"
    shutil.copyfile(p,out)
    assert sha(out)==sha(p)
    return {"source":rec(p),"backup":rec(out)}

def main():
    audit_path=HERE/"all_native_rows_readback.json"
    audit=read(audit_path)
    assert audit["status"]=="ALL1280_NATIVE_ROWS_AND8_ENDPOINTS_READBACK_PASSED"
    assert audit["rows_per_arm"]==dict.fromkeys(["BASE","FULL","NO_MPS","NO_MSCA"],320)
    assert audit["maximum_metric_readback_difference"]==0 and len(audit["endpoint_checks"])==8
    assert audit["new_training_runs"]==audit["new_model_forwards"]==audit["new_evaluator_runs"]==0
    manuscript=ROOT/"稿件"/"MPSMA-DEIM_conference_EN_revised.docx"
    original=ROOT/"稿件"/"MPSMA-DEIM_conference_EN.docx"
    assert sha(manuscript)==audit["manuscript"]["sha256"]=="0e76fa742521cd1cdcacf9e610347af84e14186d8529816304f302db6d6dbe4c"
    assert sha(original)=="ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323"
    decision_path=RECORDS/"剩余证据与最小补强方案_20261112"/"用户决定_保留方案暂不训练_20261112.json"
    assert sha(decision_path)=="b2603b1e62a1a0ba7818e34edc39049ccf9d43f9e400afabb90eabcf889e2d6f"
    decision=read(decision_path)
    assert decision["user_answer"]=="先保留方案，暂不训练" and not decision["training_authorized"]
    base_acceptance=RECORDS/"第四十轮训练分支来源核对_20261112"/"current_goal_acceptance_audit_rev40.json"
    index_path=RECORDS/"主要结论证据索引"/"主要结论证据索引_第四十轮.json"
    assert sha(index_path)=="f8bbefb20952c669f43d3b870532b27fd41df86a563ca28055581db60a6be7ee"

    status_path=HERE/"逐轮数据状态解释.json"
    status=write(status_path,{
        "recorded_utc":NOW,"source_audit":rec(audit_path),
        "current_completion_status":{"BASE":320,"FULL":320,"NO_MPS":320,"NO_MSCA":320},
        "NO_MSCA_native_status":audit["NO_MSCA_completion_readback"],
        "legacy_csv_status_label":"accepted_prefix_only",
        "legacy_label_affected_NO_MSCA_rows":320,
        "interpretation":"逐轮CSV的NO_MSCA全部320行沿用导出阶段的accepted_prefix_only标签，包括最终追加的281–320行。完整轮次、终点元数据及原生完成/备份凭据共同确认已完成320轮；该旧字段不能作实时任务状态使用。",
        "immutable_snapshot_modified":False,"manuscript_numerics_modified":False,
        "source_rows_and_metric_values_verified":1280,
        "remote_rechecked":False,
    })
    supplement_path=RECORDS/"主要结论证据索引"/"主要结论证据索引_第四十轮_全源终核补充_20261112.json"
    supplement=write(supplement_path,{
        "revision":40,"recorded_utc":NOW,"manuscript":rec(manuscript),
        "base_evidence_index":rec(index_path),"supplement_scope":"Existing archived sources only; manuscript revision and scientific numbers unchanged.",
        "all_native_rows_readback":rec(audit_path),"data_status_interpretation":status,
        "user_decision":rec(decision_path),
        "verified":{"native_rows":1280,"native_source_slices":len(audit["source_slice_counts"]),"metrics_per_row":12,"endpoints":8,"endpoint_metric_entries":96,"both_manuscript_tables_match":True,"maximum_metric_readback_difference":0},
        "unchanged_limits":[
            "BASE/FULL continuation histories differ and early anchors lack main-process random states; observed differences cannot be attributed solely to architecture.",
            "FULL and NO_MSCA differ by3841280 trainable parameters; no capacity-matched MSCA mechanism attribution.",
            "Single-seed validation-selected trajectories do not establish variance or independent test generalization.",
            "Unretained physical work is unquantified; no equal physical-compute claim.",
            "Current n-model latency and onboard performance remain unmeasured.",
        ],
        "readback_scope_limit":audit["scope_limit"],
        "new_training_runs":0,"model_forwards":0,"evaluator_runs":0,"new_remote_observation":False,
        "submission_ready":False,
    })
    acceptance=read(base_acceptance)
    acceptance.pop("new_experiment_authorization_requested_or_used",None)
    acceptance.update({
        "observed_utc":NOW,"review_stage":"Rev40 retained after complete native-row readback and user's deferral of additional training",
        "prior_acceptance_audit":rec(base_acceptance),"evidence_index_supplement":supplement,
        "full_source_readback":rec(audit_path),"user_decision":rec(decision_path),
        "new_training_proposal_status":"DEFERRED_BY_USER_NO_NEW_TRAINING",
        "new_training_authorized":False,"new_authorization_request_in_this_audit":False,
        "manuscript_changed_in_this_audit":False,"remote_rechecked_in_this_audit":False,
        "submission_status":"Content and existing-material review recorded. Important comparison-design gaps remain; target venue/template is unselected. Overall submission acceptance is not complete.",
    })
    acceptance["original_requirements"][0]["evidence"].append(rec(audit_path))
    acceptance["original_requirements"][0]["state"]="All1280 rows and8 selected endpoints traced to archived native sources; manuscript tables match. Comparison-design limitations remain explicit."
    acceptance_path=HERE/"current_goal_acceptance_after_user_deferral.json"
    acceptance_record=write(acceptance_path,acceptance)

    memo_path=HERE/"现有材料核对与剩余问题.txt"
    memo=(
        f"现有材料核对记录（{NOW}）\n\n"
        "用户决定已执行：保留BASE_FRESH/FULL_FRESH补强方案，暂不训练。没有启动新模型预检、训练、完整验证或远程任务，也不重复询问这项授权。\n\n"
        "现有数据：BASE、FULL、NO_MPS、NO_MSCA各保留320轮，共1280条逐轮记录。每行12项指标、轮次/更新/EMA计数、原始文件及记录哈希、预测/评价数组标识、学习率和协议/配置标识均与已归档原生源一致，指标差值为0。8个端点的96项指标逐项一致。主端点均为固定320轮，次端点为各轨迹AP最大值所在轮：BASE303、FULL305、NO_MPS320、NO_MSCA302，均整行取值。稿件两张表与对应记录一致。\n\n"
        "数据状态解释：NO_MSCA逐轮CSV全部320行仍沿用accepted_prefix_only旧入库标签，包括最后追加的40行。完整轮次与终点元数据一致；冻结的原生凭据显示COMPLETED_AND_BACKED_UP、完成320轮、监督进程exit0。冻结快照不改写，分析时用当前完成元数据解释旧字段，不能把旧字段当作实时服务器状态。\n\n"
        "稿件：继续维护第40轮英文稿，未因此次只读核对另增版本。原稿和上一版备份保留。摘要、方法、图式、表格、引用与9页视觉核查结果沿用rev40记录。\n\n"
        "仍缺的重要证据：BASE/FULL恢复历史不同，早期锚点缺失主随机状态；现有来源核对无法消除该设计混杂。FULL/NO_MSCA容量不同，不能把差值全部归因于MSCA门控机制。仍是单seed和同一验证集选点，没有独立测试、多seed方差、当前n模型时延或机载性能证据。未保留物理计算量未知。诊断未复现骤降，也不能确定根因。\n\n"
        "引用与排版：现有13条引用的核查范围沿用rev40，[3]、[10]全文未获取，当前狭义方法概述以已核查元数据/摘要为依据。会议尚未选择，模板、页数限制与会议投稿清单尚未适配。\n\n"
        "验收判断：现有材料的内容修订与来源核对已记录完成，重要实验缺口仍在，尚不能宣布整篇稿件达到投稿验收要求。用户暂缓训练不代表论文目标已完成或已要求暂停。\n\n"
        "核对范围：本次校验记录及哈希字段的身份和数值，未重新计算全部权重、预测和评价数组载荷哈希，未运行推理或评价器。载荷核验范围见此前各端点审计。本次没有重新查询远程，旧点时观测不代表实时状态。\n\n"
        f"详细来源：{rec(audit_path)['file']}\n证据补充：{supplement['file']}\n"
    )
    memo_path.write_text(memo,encoding="utf-8")
    progress_path=HERE/"active_goal_progress_existing_material_readback_20261112.json"
    progress=write(progress_path,{
        "goal_turn_classification":"progress","observed_utc":NOW,"goal_status":"active",
        "manuscript":rec(manuscript),"revision":40,"manuscript_changed":False,
        "all_native_rows_readback":rec(audit_path),"evidence_supplement":supplement,
        "current_acceptance":acceptance_record,"user_decision":rec(decision_path),"readable_status":rec(memo_path),
        "additional_training_authorized":False,"submission_ready":False,
        "next_action":"Preserve rev40 and the deferred proposal; continue content work when new existing material or user feedback supplies a concrete change. Adapt template after venue selection. Do not repeat training authorization or launch the deferred runs.",
        "new_training_runs":0,"model_forwards":0,"evaluator_runs":0,
        "remote_rechecked":False,
    })
    note=(
        f"【现有材料全源终核完成，新增训练仍暂缓；{NOW}】\n"
        "四组各320轮、合计1280条记录已逐项对回17段原生来源；8个主/次端点共96项指标及稿件两张表一致，指标读回差值为0。NO_MSCA旧入库状态标签已另行解释，冻结数据不改写。稿件仍为rev40。\n"
        "用户决定继续有效：保留补强方案，暂不训练。本次无模型前向、评价器重跑或新远程任务。来源核验不能消除恢复历史与容量混杂，尚不能宣布达到投稿要求。\n"
        f"核对与问题：{rec(memo_path)['file']}\n证据补充：{supplement['file']}\n\n"
    )
    changed=[]
    current_entry=ROOT/"当前修订入口.txt"
    previous=current_entry.read_text(encoding="utf-8-sig")
    assert "【用户已决定：保留补强方案，暂不训练】" in previous
    current_backup=backup(current_entry);changed.append(current_backup)
    anchor="当前第40轮维护稿";assert anchor in previous
    position=previous.index(anchor)
    current_entry.write_text(previous[:position]+note+previous[position:],encoding="utf-8")
    logs=[ROOT/"目标自主执行入口_20261109.txt",RECORDS/"修订说明与未解决问题.txt",ROOT/"远程公平对照_20261103"/"结果并入与验收清单.txt"]
    for p in logs:
        old=p.read_text(encoding="utf-8-sig")
        changed.append(backup(p))
        p.write_text(note+"以下为历史记录。\n\n"+old,encoding="utf-8")
    state_path=ROOT/"远程公平对照_20261103"/"模块消融"/"current_execution_state.json"
    changed.append(backup(state_path));state=read(state_path)
    assert state["pending_additional_training_proposal"]["status"]=="DEFERRED_BY_USER_NO_NEW_TRAINING"
    previous_supplement=state.get("latest_evidence_index_supplement")
    if previous_supplement:
        state.setdefault("evidence_index_supplement_history",[]).append({"file":previous_supplement,"sha256":state.get("latest_evidence_index_supplement_sha256"),"superseded_pointer_utc":NOW})
    state.update({
        "observed_local_utc":NOW,"paper_revision":40,"paper_sha256":sha(manuscript),"goal_status":"active",
        "operational_work_status":"REV40_RETAINED; EXISTING1280_NATIVE_ROWS_AND8_ENDPOINTS_READBACK_COMPLETE; USER_DEFERRED_NEW_TRAINING; IMPORTANT_COMPARISON_GAPS_REMAIN",
        "latest_evidence_index_supplement":str(supplement_path),"latest_evidence_index_supplement_sha256":supplement["sha256"],
        "current_goal_acceptance_audit":str(acceptance_path),"current_goal_acceptance_audit_sha256":acceptance_record["sha256"],
        "latest_active_goal_progress":str(progress_path),"latest_active_goal_progress_sha256":progress["sha256"],
        "latest_all_native_rows_readback":str(audit_path),"latest_all_native_rows_readback_sha256":sha(audit_path),
        "latest_data_status_interpretation":str(status_path),"latest_data_status_interpretation_sha256":status["sha256"],
        "acceptance_checklist_sha256":sha(logs[-1]),"paper_submission_ready":False,
        "last_goal_turn_classification":"progress","goal_blocking_audit_consecutive_turns":0,
        "next_action":read(progress_path)["next_action"],
    })
    write(state_path,state)
    assert not read(state_path)["pending_additional_training_proposal"]["training_authorized"]
    assert sha(manuscript)=="0e76fa742521cd1cdcacf9e610347af84e14186d8529816304f302db6d6dbe4c"
    assert sha(original)=="ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323"
    outputs=[audit_path,status_path,supplement_path,acceptance_path,memo_path,progress_path,current_entry,*logs,state_path]
    result=write(HERE/"finalization_readback.json",{
        "observed_utc":NOW,"status":"EXISTING_MATERIAL_AUDIT_AND_USER_DEFERRAL_RECORDED",
        "manuscript_unchanged":rec(manuscript),"original_unchanged":rec(original),
        "outputs":[rec(p) for p in outputs],"backups":changed,
        "training_authorized":False,"submission_ready":False,"new_training_runs":0,"model_forwards":0,"evaluator_runs":0,
    })
    print(json.dumps({"status":"RECORDED","audit":rec(audit_path),"summary":rec(memo_path),"acceptance":acceptance_record,"finalization":result,"manuscript_unchanged":True,"training_authorized":False,"submission_ready":False},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
