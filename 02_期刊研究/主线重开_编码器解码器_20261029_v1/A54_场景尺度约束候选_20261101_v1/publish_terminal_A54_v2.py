"""Publish the fixed endpoint only after both archives and cached analysis pass."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

P = Path(__file__).resolve().parent
BASE = P.parent


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    receipt = P / "terminal_A54_preservation_receipt.json"
    assert not receipt.exists()
    current = load(P / "CURRENT_A54.json")
    assert not current.get("do_not_run_live_updaters_after_terminal", False)
    previous = Path(current["receipt"])
    assert sha(previous) == current["receipt_sha256"]
    prior = load(previous)
    assert all(sha(Path(n)) == h for n, h in prior["current_files_sha256"].items())
    protocol = load(P / "protocol_FROZEN_v1.json")
    assert sha(P / "protocol_FROZEN_v1.json") == "20d73d282fa24357257fcd0a136d90346b414c44ffe9cb7973a4c4160e6a2166"
    final = load(P / "three_arm_result.json")
    audit = load(P / "terminal_frozen_input_audit.json")
    assert final["status"] == "COMPLETED" and audit["status"] == "PASS" and not audit["owned_jobs_live"]
    assert all(audit["local_frozen_files"].values()) and all(audit["external_frozen_inputs"].values())
    assert all(not p["owned_live"] for p in audit["owned_processes"].values())
    checks = {a: load(P / (a + "_completed_local_verification.json")) for a in ["M", "Q"]}
    for a, v in checks.items():
        assert v["status"] == "PASS" and v["epochs"] == 12 and v["optimizer_updates"] == 9708
        assert v["full_model_forwards"] == 10159 and v["optimizer_step_histogram"] == {"9708": 487}
        assert v["verifier_version"] == 2 and v["endpoint_metrics"] == final["metrics"][a]
        assert all(all(x.values()) for x in v["recorded_flow_checks_against_B"].values())
    bfolder = BASE / "A49_下一主线有界筛选_20261101_v1/completed_B_v1/arm_B"
    assert final["metrics"]["B"] == load(bfolder / "epoch12_EMA_metrics.json")
    secondary_file = P / "secondary_cached_analysis_v2/result.json"
    secondary = load(secondary_file)
    assert secondary["status"] == "DESCRIPTIVE_SECONDARY_COMPLETE" and secondary["COCO_evaluate_calls"] == 4 and secondary["successful_version_COCO_evaluate_calls"] == 3
    assert secondary["new_detector_forwards"] == secondary["new_optimizer_updates"] == 0
    assert secondary["scope_protocol_sha256"] == sha(P / "secondary_analysis_scope_v2.json")
    for a in ["B", "M", "Q"]:
        assert max(abs(x) for x in secondary["arms"][a]["metric_minus_original_pp"].values()) <= 1e-8
    metrics = final["metrics"]
    qb = metrics["Q"]["APs"] - metrics["B"]["APs"]
    qm = metrics["Q"]["APs"] - metrics["M"]["APs"]
    qa = metrics["Q"]["AP"] - metrics["B"]["AP"]
    gates = {"Q_minus_B_APs_ge_0_5pp": qb >= protocol["primary_gates_pp"]["Q_minus_B_APs_min"],
             "Q_minus_M_APs_ge_0_3pp": qm >= protocol["primary_gates_pp"]["Q_minus_M_APs_min"],
             "Q_minus_B_AP_ge_minus_0_2pp": qa >= protocol["primary_gates_pp"]["Q_minus_B_AP_min"]}
    passed = all(gates.values())
    assert gates == final["gates"] and passed == final["pilot_pass"]
    capacity = load(P / "capacity_v2/capacity_result.json")
    failed_capacity = load(P / "capacity_v1_result.json")
    assert failed_capacity["whole_model_forwards"] == 1 and failed_capacity["capacity_only_optimizer_updates"] == 0
    assert capacity["whole_model_forwards"] == 8 and capacity["capacity_only_optimizer_updates"] == 6
    assert final["new_formal_model_forwards"] == 20318 and final["new_formal_optimizer_updates"] == 19416
    judgment = {"status": "PILOT_PASS_RESEARCH_ASSESSMENT_PENDING" if passed else "RETIRE_EXACT_RECIPE_NO_RESCUE_SWEEP",
        "phase": "A54_FIXED12_TERMINAL", "goal_status": "active", "mainline_confirmed": False,
        "dataset": "VisDrone2019-DET train6471/val548", "host": "ECDet-S", "candidate": "Shared affine scene-scale readout for reference WH",
        "metrics": metrics, "gates": gates, "pilot_pass": passed,
        "Q_minus_B_APs_pp": qb, "Q_minus_M_APs_pp": qm, "Q_minus_B_AP_pp": qa,
        "secondary_cached_analysis": str(secondary_file), "archives": {a: {"bundle_sha256": v["bundle_sha256"], "payload_files_verified": v["payload_files_verified"]} for a, v in checks.items()},
        "owned_jobs_live": False, "new_training_queued": False,
        "interpretation": "One-seed fixed12 allocation evidence. No independent-test, significance, causal-mechanism or complete novelty claim."}
    judgment_file = P / "fixed_endpoint_judgment_A54.json"
    report = P / "13_固定12轮结果与主线裁决.txt"
    save(judgment_file, judgment)
    lines = ["A54固定12轮比较完成；总目标继续active，主线仍待研究综合判断。",
        "已核实事实：VisDrone train6471/val548、640stretch、batch8、共同原生及新增初始化、训练seed20261101。B复用A49，不重复计入计算；M/Q各12轮。"]
    for a in ["B", "M", "Q"]:
        m = metrics[a]
        lines.append(f"{a}：AP {m['AP']:.9f}，APs {m['APs']:.9f}，small AP75 {m['APs75']:.9f}。")
    lines += [f"数据计算：Q−B APs {qb:+.9f}、Q−M APs {qm:+.9f}、Q−B AP {qa:+.9f}，单位均为绝对百分点。",
        "冻结门槛逐项：" + json.dumps(gates, ensure_ascii=False),
        "全部通过，仅作为有短程效果依据的候选继续价值判断，不等于论文创新已成立。" if passed else "门槛未全部通过，淘汰这项共享仿射场精确配方；不改阶数、ridge、位置、损失权重、门、seed或轮数救援。", ""]
    for a in ["B", "M", "Q"]:
        g = secondary["arms"][a]["groups"]["small_4to8"]
        lines.append(f"补充描述 {a}，原图small且输入短边[4,8)像素：GT={g['GT']}，R50={g['recall50_percent']:.6f}%，R75={g['recall75_percent']:.6f}%，own-TP50尺寸绝对对数误差={g['mean_absolute_log_geomean_size_error_on_own_TP50']:.9f}，TP数={g['size_error_TP50_count']}。")
    lines += ["全部尺寸组、共同TP50样本的尺寸误差及逐GT记录见secondary_cached_analysis_v2。召回是按GT微平均，不是COCO类平均AR/AP。TP尺寸误差排除漏检；共同TP也依赖模型结果，均不能单独作因果归因。",
        "缓存分析首版在完成B一次计算后，因COCO内存标注被原地修改而错误比较M标注；三组磁盘GT逐字节一致。纠正版只为COCO对象复制标注，不改变匹配、分组、预测或门槛；失败保留，累计CPU评估4次。",
        "完整性：M/Q均exit0、各9708次正式更新、10159次整网前向、487份优化器状态step9708；raw/EMA状态和归档SHA已核验。12轮顺序、每轮前8批像素/RNG及全轮目标/忽略区元数据同B。",
        "固定类别偏移raw逐位未变；EMA浮点缓存存在最大5.364418e−7舍入变化，CPU乘法/FMA式递推逐位复现。原checker错误已保存，v2精确核验不放宽容差、不改权重。",
        "容量初版检查误用eval缺少的pred_corners字段，1次整网前向/0更新后停止；纠正版8前向/6临时更新通过，状态全部丢弃。失败与原源码保留。",
        "本阶段新增20327整网前向（容量9、正式20318）、19422次含容量更新（正式19416、容量6）、24正式轮。累计205939前向、175270正式更新、55容量更新，含容量175325更新，216正式轮。缓存COCO累计4次（首版1、纠正版3），无新模型推理。",
        "计时范围：时延包含诊断开销，不能据此作部署提速结论；仅报告训练耗时和显存。最终场系数是最后计时图的观测，不是验证集均值。",
        "文献事实：尺度场、场景/物理尺度先验和局部/全局表示比较均已有研究，不能包装为首次。最近原文及GSDet/GSDDet等全文缺口见06及supplemental_literature/method_scope.json。",
        "待验证假设：GT尺度关系能否由图像稳健估计、主路径宽高修正和辅助任务正则化的贡献，以及跨场景/模型推广，均未由这一次短测完全确立。",
        "评价范围：已有忽略区过滤COCO式small≤1024、maxDets300，非VisDrone官方挑战计分。val548已多次参与选题，一次seed12不支持独立测试或统计显著。",
        "所属监督器与M/Q均已退出，无额外自动队列。旧数据、源码、权重、所有历史受限/阴性/未训练裁决均保留。"]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    canonical = [BASE / "01_候选判别表.json", BASE / "15_当前状态与结果校验.json"]
    before = P / "before_A54_terminal"
    before.mkdir(exist_ok=False)
    resolved = {}
    for f in canonical:
        copied = before / f.name
        shutil.copyfile(f, copied)
        resolved[str(f)] = {"preserved_path": str(copied), "sha256": sha(copied)}
    shutil.copyfile(P / "CURRENT_A54.json", before / "CURRENT_A54.json")
    now = datetime.now(timezone.utc).isoformat()
    status = "A54_PILOT_PASS_RESEARCH_ASSESSMENT_PENDING" if passed else "A54_RETIRED_FIXED12_GATES_FAILED"
    board = load(canonical[0])
    entry = next(x for x in board["candidates"] if x["id"] == "A54_SCENE_SCALE_FIELD")
    entry.update(status=status, terminal_result=judgment, mainline_confirmed=False)
    board.update(status=status, goal_status="active", mainline_validated=False, candidate_selected_for_pilot=passed,
        mainline_selected_for_pilot=False, latest_A54=entry, A54_terminal_result=judgment, latest_report=str(report), latest_update_utc=now)
    if not passed:
        board.update(current_priority_candidate=None, current_priority_question="继续筛选实质不同的算法机制；不救援A54精确配方。")
    save(canonical[0], board)
    state = load(canonical[1])
    state.update(recorded_utc=now, status=status, goal_status="active", mainline_validated=False, current_phase="A54_TERMINAL_RESEARCH_GOAL_ACTIVE",
        candidate_selected_for_pilot=passed, mainline_selected_for_pilot=False, selection_stage_complete=False,
        current_user_task_phase="MAINLINE_RESEARCH_AFTER_A54_FIXED12", current_user_task_status="A54_PHASE_COMPLETED_GOAL_ACTIVE",
        latest_authoritative_phase="A54_TERMINAL_VERIFIED", latest_report=str(report), A54_terminal_result=judgment,
        current_priority_candidate=board.get("current_priority_candidate"), current_priority_question=board.get("current_priority_question"),
        model_forwards=205939, cap=205939, formal_detector_training_updates=175270, capacity_only_optimizer_updates=55,
        detector_optimizer_updates_including_capacity=175325, training_updates=175325, formal_training_epochs=216,
        counts_terminal=True, training_counts_terminal=True, current_native_job_terminal=True, full_forward_cap_closed=True,
        counters_observed_lower_bounds_while_job_live=False, CPU_saved_prediction_COCO_evaluate_calls_A54=4,
        training_update_scope="A54 exact terminal cumulative totals, including all55 capacity updates; reused B not recounted.",
        full_model_forwards_scope="A53 exact185612 + A54 capacity9 + A54 formal20318 =205939; excludes standalone and CPU arithmetic.",
        formal_training_epochs_scope="A53 exact192 + A54 M12/Q12 =216; capacity steps are not epochs.", cap_scope="A54 fixed12 phase completed; prior caps retain historical scopes.")
    state["A54_execution"].update(status="TERMINAL_NO_OWNED_JOBS", terminal_result=judgment, counts_are_observed_lower_bounds_while_live=False)
    save(canonical[1], state)
    tracked = canonical + [report, judgment_file, P / "three_arm_result.json", P / "terminal_frozen_input_audit.json",
        secondary_file, P / "secondary_analysis_scope_v2.json", P / "analyze_cached_endpoints_A54_v2.py", P / "verify_completed_arm_A54_local_v2.py",
        P / "M_completed_local_verification.json", P / "Q_completed_local_verification.json", P / "supplemental_literature/method_scope.json",
        P / "secondary_cached_analysis_v1/failure.json", P / "terminal_audit_path_correction.json", P / "audit_terminal_inputs_A54_v2.py", P / "publish_terminal_A54_v2.py"]
    save(receipt, {"status": "PASS", "phase": "A54_TERMINAL", "goal_status": "active", "mainline_confirmed": False,
        "previous_receipt": str(previous), "previous_receipt_sha256": sha(previous), "previous_signed_current_inputs_resolved": resolved,
        "current_files_sha256": {str(f): sha(f) for f in tracked}, "pilot_pass": passed, "owned_jobs_live": False,
        "new_training_queued": False, "publication_added_model_operations": 0,
        "cumulative_counts": {"whole_model_forwards": 205939, "formal_optimizer_updates": 175270, "capacity_updates": 55, "all_optimizer_updates": 175325, "formal_epochs": 216}})
    current.update(status=status, receipt=str(receipt), receipt_sha256=sha(receipt), latest_report=str(report), goal_status="active",
        mainline_confirmed=False, owned_jobs_live=False, owned_jobs_live_at_observation=False, Q_terminal_verified=True,
        current_priority_candidate=board.get("current_priority_candidate"),
        last_observed_Q_completed_epochs=12, last_observed_Q_optimizer_updates=9708, do_not_run_live_updaters_after_terminal=True,
        next="Assess positive pilot against closest methods and causal limits." if passed else "Choose a substantively different mechanism; do not rescue A54. Goal remains active.")
    save(P / "CURRENT_A54.json", current)
    print(json.dumps({"status": status, "pilot_pass": passed, "deltas_pp": {"Q_B_APs": qb, "Q_M_APs": qm, "Q_B_AP": qa},
        "goal_status": "active", "receipt_sha256": sha(receipt)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
