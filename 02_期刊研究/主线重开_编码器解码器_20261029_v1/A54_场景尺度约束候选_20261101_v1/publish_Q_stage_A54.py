"""Descriptive Q4/Q8 publication. Does not run or alter training/evaluation."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import shutil

P = Path(__file__).resolve().parent
BASE = P.parent


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main(epoch, observation):
    current = load(P / "CURRENT_A54.json")
    assert not current.get("do_not_run_live_updaters_after_terminal", False)
    previous = Path(current["receipt"])
    assert sha(previous) == current["receipt_sha256"]
    old = load(previous)
    assert all(sha(Path(n)) == h for n, h in old["current_files_sha256"].items())
    receipt = P / ("Q%d_阶段保留回执.json" % epoch)
    assert not receipt.exists()
    obsfile = P / observation
    wire = load(obsfile)
    assert wire["exit"] == 0
    obs = json.loads(wire["stdout"])
    assert all(obs["frozen_files_unchanged"].values())
    assert obs["supervisor_process"]["live"] and obs["supervisor_process"]["identity_matches"]
    qp = obs["arms"]["Q"]["process"]
    assert qp["live"] and qp["identity_matches"] and qp["pid"] == current["Q_pid"] and qp["start_ticks"] == current["Q_start_ticks"]
    progress = obs["arms"]["Q"]["progress"]
    bf = BASE / "A49_下一主线有界筛选_20261101_v1/completed_B_v1/arm_B"
    mf, qf = P / "completed_M_v1/arm_M", P / ("Q%d" % epoch)
    metrics_name, flow_name = "epoch%02d_EMA_metrics.json" % epoch, "epoch%02d_input_flow.json" % epoch
    metrics = {a: load(f / metrics_name) for a, f in [("B", bf), ("M", mf), ("Q", qf)]}
    assert metrics["Q"] == obs["arms"]["Q"]["metrics"][str(epoch)]
    for key in ["evaluated_GT", "ignored_valid_GT", "max_detections", "area_convention", "scope"]:
        assert metrics["B"][key] == metrics["M"][key] == metrics["Q"][key]
    flows = {a: load(f / flow_name) for a, f in [("B", bf), ("M", mf), ("Q", qf)]}
    keys = ["order_sha256", "first_eight_actual_input_sha256", "first_eight_pre_forward_rng_sha256", "all_batches_targets_including_ignore_tiles_sha256"]
    checks = {k: flows["B"][k] == flows["M"][k] == flows["Q"][k] for k in keys}
    assert all(checks.values()) and all(x["optimizer_updates"] == 809 * epoch for x in flows.values())
    deltas = {a: {k: metrics["Q"][k] - metrics[a][k] for k in ["AP", "APs", "APs75", "ARs"]} for a in ["B", "M"]}
    verified = {"status": "PASS_DESCRIPTIVE_ONLY", "epoch": epoch, "metrics": metrics, "Q_minus_reference_pp": deltas,
        "recorded_flow_checks": checks, "recorded_flow_scope": "Whole-epoch order/targets/ignore metadata; first8 batches actual pixel/RNG records.",
        "source_files_sha256": {str(f / n): sha(f / n) for f in [bf, mf, qf] for n in [metrics_name, flow_name]},
        "observation": str(obsfile), "snapshot": obs, "new_detector_forwards": 0, "new_optimizer_updates": 0,
        "interpretation": "No early pass/fail, endpoint picking, recipe change or claim of confirmed mainline. Fixed12 pending."}
    verification = qf / "local_independent_verification.json"
    save(verification, verified)
    report = P / ("Q%d_阶段结果.txt" % epoch)
    lines = [f"A54第{epoch}轮三组同轮EMA结果，仅描述，不替代固定12轮终点。"]
    for a, m in metrics.items():
        lines.append(f"{a}：AP {m['AP']:.9f}，APs {m['APs']:.9f}，APs75 {m['APs75']:.9f}。")
    for a, delta in deltas.items():
        lines.append(f"Q−{a}：AP {delta['AP']:+.9f}，APs {delta['APs']:+.9f}，APs75 {delta['APs75']:+.9f}，单位为绝对百分点。")
    lines += ["同轮记录的输入顺序、首8批实际像素/RNG、全轮标签与忽略区元数据三组一致。不是全轮每批像素均有留存。",
        f"观测unix={obs['observed_server_unix']}：Q更新{progress['optimizer_updates']}次，完整训练轮数下界{progress['optimizer_updates']//809}；所属监督器与Q进程身份一致，冻结输入未变。",
        "M已完成固定12并归档。Q仍按固定12继续，不追加救援队列，不称当前微小差值为稳定收益。",
        "总目标active，未确认主线。旧方法、数据、源码、权重与裁决均保留。"]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    canonical = [BASE / "01_候选判别表.json", BASE / "15_当前状态与结果校验.json"]
    before = P / ("before_A54_Q%d" % epoch)
    before.mkdir(exist_ok=False)
    resolved = {}
    for f in canonical:
        copied = before / f.name; shutil.copyfile(f, copied)
        resolved[str(f)] = {"preserved_path": str(copied), "sha256": sha(copied)}
    shutil.copyfile(P / "CURRENT_A54.json", before / "CURRENT_A54.json")
    milestone = {"epoch": epoch, "scope": "DESCRIPTIVE_ONLY_FIXED12_PENDING", "verification": str(verification), "report": str(report),
        "metrics": metrics, "Q_minus_reference_pp": deltas, "observation": str(obsfile), "mainline_confirmed": False}
    board = load(canonical[0])
    entry = next(x for x in board["candidates"] if x["id"] == "A54_SCENE_SCALE_FIELD")
    entry["latest_milestone"] = milestone
    status = "A54_Q12_RUNNING_Q%d_DESCRIPTIVE" % epoch
    board.update(status=status, latest_A54=entry, latest_report=str(report), mainline_validated=False)
    board["A54_Q%d_milestone" % epoch] = milestone
    save(canonical[0], board)
    qup, qfw = progress["optimizer_updates"], progress["full_model_forwards"]
    counts = {"model_forwards": 185612 + 9 + 10159 + qfw, "formal_detector_training_updates": 155854 + 9708 + qup,
        "capacity_only_optimizer_updates": 55, "detector_optimizer_updates_including_capacity": 155903 + 6 + 9708 + qup,
        "training_updates": 155903 + 6 + 9708 + qup, "formal_training_epochs": 192 + 12 + qup // 809}
    state = load(canonical[1])
    state.update(**counts, status=status, recorded_utc=datetime.now(timezone.utc).isoformat(), latest_report=str(report),
        counts_terminal=False, training_counts_terminal=False, current_native_job_terminal=False, counters_observed_lower_bounds_while_job_live=True)
    state["A54_Q%d_milestone" % epoch] = milestone
    state["A54_execution"].update(status=status, snapshot=obs, snapshot_client_receipt=str(obsfile), observed_Q_state=progress, report=str(report))
    state["previous_goal_turn_classification"] = "progress: verified M12 archive/raw/EMA/optimizer/input streams and actual Q initialization/start, not a mere status restatement."
    save(canonical[1], state)
    tracked = canonical + [verification, report, obsfile, qf / metrics_name, qf / flow_name, P / "terminal_helpers_prepared_v1.json",
                          P / "audit_terminal_inputs_A54.py", P / "publish_terminal_A54.py"]
    save(receipt, {"status": "PASS", "phase": "A54_Q%d_DESCRIPTIVE" % epoch, "goal_status": "active", "mainline_confirmed": False,
        "previous_receipt": str(previous), "previous_receipt_sha256": sha(previous), "previous_signed_current_inputs_resolved": resolved,
        "current_files_sha256": {str(f): sha(f) for f in tracked}, "counts_lower_bound": counts, "counts_terminal": False,
        "publication_added_model_operations": 0})
    current.update(status=status, receipt=str(receipt), receipt_sha256=sha(receipt), latest_report=str(report), latest_observation=str(obsfile),
        last_observed_Q_optimizer_updates=qup, last_observed_Q_completed_epochs=qup // 809, mainline_confirmed=False,
        terminal_helpers_prepared=str(P / "terminal_helpers_prepared_v1.json"))
    save(P / "CURRENT_A54.json", current)
    print(json.dumps({"status": status, "deltas_pp": deltas, "counts_lower_bound": counts, "receipt_sha256": sha(receipt)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, choices=[4, 8], required=True)
    parser.add_argument("--observation", required=True)
    args = parser.parse_args()
    main(args.epoch, args.observation)
