"""Publish verified M12 and an observed Q start; never starts or changes jobs."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import shutil

P = Path(__file__).resolve().parent
BASE = P.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main(observation):
    receipt = P / "M12_终点与Q启动保留回执.json"
    assert not receipt.exists()
    current = load(P / "CURRENT_A54.json")
    previous = Path(current["receipt"])
    assert sha(previous) == current["receipt_sha256"]
    prev = load(previous)
    for name, expected in prev["current_files_sha256"].items():
        assert sha(Path(name)) == expected, name
    verification = load(P / "M_completed_local_verification.json")
    assert verification["status"] == "PASS" and verification["epochs"] == 12
    assert verification["optimizer_step_histogram"] == {"9708": 487}
    mf = P / "completed_M_v1/arm_M"
    bf = BASE / "A49_下一主线有界筛选_20261101_v1/completed_B_v1/arm_B"
    m, b = load(mf / "result.json"), load(bf / "result.json")
    assert m["endpoint_metrics"] == verification["endpoint_metrics"] and m["full_model_forwards"] == 10159
    mi, qi = load(mf / "initialization.json"), load(P / "Q_startup/initialization.json")
    assert mi == qi
    assert mi["new_module_initialization_sha256"] == "91a952243abbe8446b5efaed054dc1ec2a076d593b900e981a2d80b6dbc710a8"
    assert mi["all_native_initial_tensors_identical"] and mi["scratch_no_capacity_weights"]
    q1, b1, m1 = [load(f / "epoch01_input_flow.json") for f in [P / "Q_startup", bf, mf]]
    flow_keys = ["order_sha256", "first_eight_actual_input_sha256", "first_eight_pre_forward_rng_sha256", "all_batches_targets_including_ignore_tiles_sha256"]
    qchecks = {k: q1[k] == b1[k] == m1[k] for k in flow_keys}
    assert all(qchecks.values()) and q1["optimizer_updates"] == 809
    wire = load(P / observation)
    assert wire["exit"] == 0
    obs = json.loads(wire["stdout"])
    assert all(obs["frozen_files_unchanged"].values())
    assert obs["supervisor_process"]["live"] and obs["supervisor_process"]["identity_matches"]
    assert obs["arms"]["M"]["exit"]["exit_code"] == 0 and not obs["arms"]["M"]["process"]["live"]
    assert obs["arms"]["Q"]["process"]["live"] and obs["arms"]["Q"]["process"]["identity_matches"]
    qp = obs["arms"]["Q"]["progress"]
    assert qp["status"] in ["TRAINING", "VALIDATING"] and qp["optimizer_updates"] > 0
    mb, bb = m["endpoint_metrics"], b["endpoint_metrics"]
    assert all(mb[k] == bb[k] for k in ["evaluated_GT", "ignored_valid_GT", "max_detections", "area_convention", "scope"])
    delta = {k: mb[k] - bb[k] for k in ["AP", "APs", "APs75", "ARs"]}
    requirements = {"Q_APs_at_least": max(bb["APs"] + .5, mb["APs"] + .3), "Q_AP_at_least": bb["AP"] - .2}
    record = {"status": "M12_VERIFIED_Q12_RUNNING", "goal_status": "active", "mainline_confirmed": False,
        "metrics": {"B12": bb, "M12": mb}, "M_minus_B_pp": delta, "M12_verification": verification,
        "Q_initialization_identical_to_M": True, "Q_first_epoch_recorded_flow_checks": qchecks,
        "Q_input_flow_scope": "First epoch independently matches B/M recorded flow keys. All12 independently verified at terminal; no all12 Q claim yet.",
        "Q_fixed_endpoint_requirements": requirements, "snapshot": obs, "observation": str(P / observation),
        "secondary_scope": str(P / "secondary_analysis_scope_v1.json"), "new_model_operations_for_publication": 0,
        "checker_v1_failure_and_numerical_reproduction": str(P / "verification_v1_failure"),
        "fixed_buffer_scope": "Raw class offsets remain bitwise unchanged; EMA floating-buffer roundoff matches the explicit FP32 mul/FMA recurrence exactly. No checkpoint, training or scoring change."}
    recordfile = P / "M12_终点与Q初始化核验.json"
    report = P / "12_M12结果与Q启动.txt"
    save(recordfile, record)
    report.write_text(
        "A54普通尺度辅助头M完成固定12，候选Q按冻结顺序实际运行。主线尚未成立。\n"
        f"固定12 EMA：B AP={bb['AP']:.9f}、APs={bb['APs']:.9f}；M AP={mb['AP']:.9f}、APs={mb['APs']:.9f}。\n"
        f"M−B：AP {delta['AP']:+.9f}、APs {delta['APs']:+.9f}、APs75 {delta['APs75']:+.9f}，单位均为绝对百分点。\n"
        f"M exit0、9708正式更新、10159整网前向；已下载并核验{verification['payload_files_verified']}个归档文件、raw/EMA/487份优化器状态和全部12轮记录流。\n"
        "M的记录顺序、每轮前8批实际像素/RNG、全轮标签与忽略区元数据同B。不能把前8批核查写成每轮每批像素均有记录。\n"
        "归档checker v1误要求EMA中的固定浮点缓存也逐位不变；raw类别偏移确实未变，EMA最大舍入差5.364418e−7。已在CPU用FP32乘法/FMA式递推逐位复现保存值，v2维持精确比较，不改权重/算法/评分。原checker和失败保留于verification_v1_failure。\n"
        f"Q从与M相同的原生和新增模块初值开训，截至本次观测已更新{qp['optimizer_updates']}次；未使用容量检查权重。\n"
        "Q首轮记录的顺序、前8批实际像素/RNG、全轮标签与忽略区元数据已独立比对，与B/M一致。尚不称Q全部12轮已核对。\n"
        f"Q固定12需同时APs≥{requirements['Q_APs_at_least']:.12f}、AP≥{requirements['Q_AP_at_least']:.12f}，对应Q−B APs≥0.5、Q−M APs≥0.3、Q−B AP≥−0.2个百分点。\n"
        "不按4/8轮挑最优，不追加轮次/seed/投影阶数/ridge/注入位置/辅助权重救援。\n"
        "剩余4–8输入像素组与尺寸误差仅做既定缓存预测的CPU描述分析，不改主终点。辅助监督和几何读出贡献尚未完全分开。\n"
        "最终计时包含诊断开销，不能用来作部署提速结论。使用忽略区过滤COCO式评分，非VisDrone官方挑战分数。\n"
        "尺度场等已有工作与未取得全文范围见supplemental_literature/method_scope.json；一次seed12不证明统计显著、独立泛化或全球新颖性。\n", encoding="utf-8")
    canonical = [BASE / "01_候选判别表.json", BASE / "15_当前状态与结果校验.json"]
    before = P / "before_A54_M12"
    before.mkdir(exist_ok=False)
    resolved = {}
    for path in canonical:
        copied = before / path.name
        shutil.copyfile(path, copied)
        resolved[str(path)] = {"preserved_path": str(copied), "sha256": sha(copied)}
    shutil.copyfile(P / "CURRENT_A54.json", before / "CURRENT_A54.json")
    qup, qfw = qp["optimizer_updates"], qp["full_model_forwards"]
    counts = {"model_forwards": 185612 + 9 + 10159 + qfw, "formal_detector_training_updates": 155854 + 9708 + qup,
        "capacity_only_optimizer_updates": 55, "detector_optimizer_updates_including_capacity": 155903 + 6 + 9708 + qup,
        "training_updates": 155903 + 6 + 9708 + qup, "formal_training_epochs": 192 + 12 + qup // 809}
    now, status = datetime.now(timezone.utc).isoformat(), "A54_M12_VERIFIED_Q12_RUNNING"
    board = load(canonical[0])
    entry = next(x for x in board["candidates"] if x["id"] == "A54_SCENE_SCALE_FIELD")
    entry.update(status=status, latest_milestone=record, evidence_scope="M fixed12 verified; Q running; no candidate Q endpoint or efficacy claim.")
    board.update(status=status, latest_A54=entry, latest_report=str(report), latest_update_utc=now,
                 A54_M_terminal_verification=record, mainline_validated=False)
    save(canonical[0], board)
    state = load(canonical[1])
    state.update(**counts, recorded_utc=now, status=status, latest_report=str(report), A54_M_terminal_verification=record,
        counts_terminal=False, training_counts_terminal=False, current_native_job_terminal=False,
        counters_observed_lower_bounds_while_job_live=True,
        training_update_scope="A53 exact155903 incl49 capacity + A54 capacity6 + M9708 + observed Q updates. B reused, not recounted; live lower bound.",
        full_model_forwards_scope="A53 exact185612 + A54 capacity9 + M10159 + observed Q whole-model forwards; live lower bound.",
        formal_training_epochs_scope="A53 exact192 + M12 + complete Q training passes at snapshot; partial pass excluded.")
    state["A54_execution"].update(status=status, snapshot=obs, snapshot_client_receipt=str(P / observation),
        observed_M_progress=m, observed_Q_state=qp, report=str(report))
    save(canonical[1], state)
    tracked = canonical + [recordfile, report, P / observation, P / "M_completed_local_verification.json",
        P / "Q_startup/initialization.json", P / "Q_startup/epoch01_input_flow.json", P / "secondary_analysis_scope_v1.json", P / "analyze_cached_endpoints_A54.py",
        P / "supplemental_literature/method_scope.json", P / "supplemental_literature/acquisition_v1.json",
        P / "verify_completed_arm_A54_local_v2.py", P / "verification_v1_failure/verify_completed_arm_A54_local.py",
        P / "verification_v1_failure/failure.json", P / "verification_v1_failure/CPU_constant_buffer_EMA_replay.json",
        P / "verification_v1_failure/CPU_FMA_constant_buffer_EMA_replay.json"]
    save(receipt, {"status": "PASS", "phase": "A54_M12_Q_START", "goal_status": "active", "mainline_confirmed": False,
        "previous_receipt": str(previous), "previous_receipt_sha256": sha(previous), "previous_signed_current_inputs_resolved": resolved,
        "current_files_sha256": {str(f): sha(f) for f in tracked}, "counts_observed_lower_bound": counts, "counts_terminal": False,
        "publication_added_model_operations": 0})
    current.update(status=status, receipt=str(receipt), receipt_sha256=sha(receipt), latest_report=str(report),
        latest_observation=str(P / observation), M_terminal_verified=True, last_observed_M_completed_epochs=12,
        last_observed_M_optimizer_updates=9708, Q_pid=obs["arms"]["Q"]["process"]["pid"],
        Q_start_ticks=obs["arms"]["Q"]["process"]["start_ticks"], last_observed_Q_optimizer_updates=qup,
        next="Observe the same Q process; fixed12 and cached secondary analysis. No restart or rescue sweep.")
    save(P / "CURRENT_A54.json", current)
    print(json.dumps({"status": status, "M_minus_B_pp": delta, "Q_requirements": requirements,
                      "counts_lower_bound": counts, "receipt_sha256": sha(receipt)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", required=True)
    main(parser.parse_args().observation)
