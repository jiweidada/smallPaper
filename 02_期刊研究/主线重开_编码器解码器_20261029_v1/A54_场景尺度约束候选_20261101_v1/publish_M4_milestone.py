"""Publish a descriptive snapshot; never imports a detector or changes training."""
from pathlib import Path
from datetime import datetime, timezone
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
    receipt = P / "A54_M4阶段保留回执.json"
    assert not receipt.exists(), "One publication only; do not overwrite a receipt."
    previous = P / "A54_候选与启动保留回执.json"
    prev = load(previous)
    for name, expected in prev["current_files_sha256"].items():
        assert sha(Path(name)) == expected, name
    observer = P / "observer_04_recovered_session.json"
    raw = load(observer)
    assert raw["exit"] == 0
    obs = json.loads(raw["stdout"])
    assert all(obs["frozen_files_unchanged"].values())
    assert obs["supervisor_process"]["live"] and obs["supervisor_process"]["identity_matches"]
    assert obs["arms"]["M"]["process"]["identity_matches"]
    progress = obs["arms"]["M"]["progress"]
    assert progress["status"] == "TRAINING" and obs["arms"]["Q"]["progress"] is None
    bfolder = BASE / "A49_下一主线有界筛选_20261101_v1/completed_B_v1/arm_B"
    bmetrics = bfolder / "epoch04_EMA_metrics.json"
    mmetrics = P / "M4/epoch04_EMA_metrics.json"
    b, m = load(bmetrics), load(mmetrics)
    assert m == obs["arms"]["M"]["metrics"]["4"]
    for key in ["evaluated_GT", "ignored_valid_GT", "max_detections", "area_convention", "scope"]:
        assert b[key] == m[key], key
    bflow = bfolder / "epoch04_input_flow.json"
    mflow = P / "M4/epoch04_input_flow.json"
    bf, mf = load(bflow), load(mflow)
    flow_keys = ["order_sha256", "first_eight_actual_input_sha256", "first_eight_pre_forward_rng_sha256", "all_batches_targets_including_ignore_tiles_sha256"]
    assert all(bf[k] == mf[k] for k in flow_keys)
    assert bf["optimizer_updates"] == mf["optimizer_updates"] == 3236
    deltas = {k: m[k] - b[k] for k in ["AP", "APs", "AP50", "AP75", "APs75", "ARs"]}
    check = {
        "status": "PASS", "epoch": 4, "reading": "Local downloaded originals; no model or evaluator execution",
        "B": b, "M": m, "M_minus_B_pp": deltas,
        "matching_flow_keys": flow_keys,
        "flow_scope": "Whole-epoch order and target/ignore metadata; actual pixels and pre-forward RNG for first eight batches only.",
        "source_files_sha256": {str(f): sha(f) for f in [bmetrics, mmetrics, bflow, mflow, P / "M4/initialization.json", observer]},
        "decision_scope": "Descriptive epoch4 only. Keep fixed12 endpoint, no early success/failure decision or recipe change."
    }
    save(P / "M4/local_independent_verification.json", check)
    canonical = [BASE / "01_候选判别表.json", BASE / "15_当前状态与结果校验.json"]
    before = P / "before_A54_M4"
    before.mkdir()
    resolved = {}
    for f in canonical:
        copy = before / f.name
        shutil.copyfile(f, copy)
        assert sha(copy) == prev["current_files_sha256"][str(f)]
        resolved[str(f)] = {"preserved_path": str(copy), "sha256": sha(copy)}
    shutil.copyfile(P / "CURRENT_A54.json", before / "CURRENT_A54.json")
    report = P / "10_M4阶段结果与当前进度.txt"
    report.write_text(
        "A54：M第4轮结果，属于预定的描述性阶段观察。\n"
        f"同轮B/M AP：{b['AP']:.6f}/{m['AP']:.6f}；M−B {deltas['AP']:+.6f}个百分点。\n"
        f"同轮B/M APs：{b['APs']:.6f}/{m['APs']:.6f}；M−B {deltas['APs']:+.6f}个百分点。\n"
        f"同轮APs75差：{deltas['APs75']:+.6f}个百分点。当前未见明确小目标收益，不能据此替代12轮终点。\n"
        "本地独立核对epoch4图像顺序、前8批实际像素/RNG、全轮标签/忽略区元数据均一致；不是全轮每批像素均有留存。\n"
        f"服务器观测unix={obs['observed_server_unix']}：M完成{progress['completed_epochs']}轮，正在第{progress['epoch']}轮{progress['microbatch']}/809批；"
        f"累计正式更新{progress['optimizer_updates']}，整网前向{progress['full_model_forwards']}。Q尚未启动。\n"
        "原监督器与训练进程身份一致，10个冻结输入哈希均未变。此为实时下界快照，不是终止计数。\n"
        "固定12轮EMA，M后接Q，不挑第4/8轮最优，不因中途结果修改投影/正则/监督/增益，不追加救援实验。\n"
        "计分为已有忽略区过滤COCO式指标，不是VisDrone官方挑战计分；程序时延包含诊断开销，不能据此宣称部署速度。\n"
        "主线有效性、因果归因、独立泛化与完整新颖性尚未确立；总目标继续active。\n", encoding="utf-8")
    milestone = {"phase": "A54_M4_DESCRIPTIVE", "goal_status": "active", "mainline_confirmed": False,
        "evaluation": check, "observation": str(observer), "snapshot": obs, "report": str(report),
        "counts_are_observed_lower_bounds_while_live": True}
    save(P / "M4/milestone.json", milestone)
    board = load(canonical[0])
    entry = next(x for x in board["candidates"] if x["id"] == "A54_SCENE_SCALE_FIELD")
    entry["latest_milestone"] = {"epoch": 4, "B_M_metrics": {"B": b, "M": m}, "M_minus_B_pp": deltas,
        "scope": "Descriptive only; fixed12 pending", "report": str(report)}
    entry["evidence_scope"] = "GT-only scale statistics, bounded capacity, formal training, descriptive M4 metrics; no Q result or fixed12 decision."
    board["latest_A54"] = entry
    save(canonical[0], board)
    state = load(canonical[1])
    updates, forwards, epochs = [progress[k] for k in ["optimizer_updates", "full_model_forwards", "completed_epochs"]]
    state.update(recorded_utc=datetime.now(timezone.utc).isoformat(), latest_report=str(report),
        model_forwards=185612 + 9 + forwards, formal_detector_training_updates=155854 + updates,
        capacity_only_optimizer_updates=55, detector_optimizer_updates_including_capacity=155903 + 6 + updates,
        training_updates=155903 + 6 + updates, formal_training_epochs=192 + epochs,
        training_update_scope="Observed A54 snapshot: A53 exact155903 including49 capacity + A54 capacity6 + observed A54 formal updates. Live lower bound; reused B not counted twice.",
        full_model_forwards_scope="A53 exact185612 + A54 capacity9 + observed A54 formal whole-model forwards; live lower bound.",
        formal_training_epochs_scope="A53 exact192 formal epochs + fully completed A54 formal epochs at observation; partial epochs excluded.",
        counts_terminal=False, training_counts_terminal=False, current_native_job_terminal=False,
        counters_observed_lower_bounds_while_job_live=True, full_forward_cap_closed=False,
        A54_M4_milestone=milestone)
    state["A54_execution"].update(snapshot=obs, snapshot_client_receipt=str(observer), observed_M_progress=progress,
        observed_Q_state="not started at this snapshot", report=str(report))
    save(canonical[1], state)
    tracked = canonical + [report, observer, P / "M4/local_independent_verification.json", P / "M4/milestone.json", mmetrics, mflow, P / "M4/initialization.json"]
    save(receipt, {"status": "PASS", "phase": "A54_M4_DESCRIPTIVE", "goal_status": "active", "mainline_confirmed": False,
        "previous_receipt": str(previous), "previous_receipt_sha256": sha(previous),
        "previous_signed_current_inputs_resolved": resolved,
        "unchanged_prior_signed_files": {n: h for n, h in prev["current_files_sha256"].items() if n not in resolved},
        "current_files_sha256": {str(f): sha(f) for f in tracked},
        "counts_snapshot": {k: state[k] for k in ["model_forwards", "formal_detector_training_updates", "capacity_only_optimizer_updates", "detector_optimizer_updates_including_capacity", "training_updates", "formal_training_epochs"]},
        "counts_terminal": False, "new_model_ops_for_this_publication": 0})
    current = load(P / "CURRENT_A54.json")
    current.update(receipt=str(receipt), receipt_sha256=sha(receipt), latest_report=str(report),
        latest_observation=str(observer), last_observed_M_completed_epochs=epochs,
        last_observed_M_optimizer_updates=updates, latest_M4_verification=str(P / "M4/local_independent_verification.json"))
    save(P / "CURRENT_A54.json", current)
    print(json.dumps({"status": "PUBLISHED_DESCRIPTIVE_M4", "deltas_pp": deltas, "counts_lower_bound": load(receipt)["counts_snapshot"], "receipt_sha256": sha(receipt)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
