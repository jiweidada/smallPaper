from __future__ import annotations
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

OUT = Path(r"C:\Users\jiwei\Desktop\小论文投稿\01_会议论文\实验数据汇总_20261108")
PAPER = OUT.parent
RECOVERY = PAPER / "远程公平对照_20261103/模块消融/NO_MSCA/recovery30715_20261109"
EVIDENCE = RECOVERY / "completion_evidence_round320_20261111"
SNAP280 = OUT / "verified_snapshots/NO_MSCA_round280_verified_20261110"
METRICS = ["AP","AP50","AP75","APs","APm","APl","AR1","AR10","AR100","ARs","ARm","ARl"]

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def file_sha(path: Path) -> str:
    return sha(path.read_bytes())

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

# Validate the immutable prior snapshot and preserve its exact delivery prefix.
old_manifest_path = SNAP280 / "snapshot_manifest.json"
old_manifest = read_json(old_manifest_path)
assert file_sha(old_manifest_path) == "883acf595c6fe7ebd7cd86a52e879b90d507c6052536f6d90b0aafe60cabe730"
for item in old_manifest["files"]:
    p = SNAP280 / item["file"]
    assert p.is_file() and p.stat().st_size == item["bytes"] and file_sha(p) == item["sha256"], item["file"]
old_csv = (SNAP280 / "逐轮指标.csv").read_bytes()
current_csv_path = OUT / "逐轮指标.csv"
current_csv_bytes = current_csv_path.read_bytes()
assert current_csv_bytes.startswith(old_csv), "Old CSV byte prefix changed"
all_rounds = read_csv(current_csv_path)
assert len(all_rounds) == 1280, len(all_rounds)
counts = {}
for row in all_rounds:
    counts[row["arm"]] = counts.get(row["arm"], 0) + 1
assert counts == {"BASE":320, "FULL":320, "NO_MPS":320, "NO_MSCA":320}, counts

# Compare each newly appended row with the native accepted epoch record and exact raw-line digest.
source_path = EVIDENCE / "epoch0319/epochs.jsonl"
source_raw = source_path.read_bytes().splitlines()
source_rows = [json.loads(line) for line in source_raw]
assert len(source_raw) == len(source_rows) == 320
assert file_sha(source_path) == "f5db3b074c24853808a521835390b27bf4f7fd4a9d69155d454ac0a9ed37e4ff"
assert len(all_rounds[-40:]) == 40
for offset, row in enumerate(all_rounds[-40:]):
    index = 280 + offset
    src = source_rows[index]
    assert row["arm"] == "NO_MSCA" and int(row["round_1based"]) == index + 1
    assert int(row["actual_epoch_index_0based"]) == src["epoch"]
    assert int(row["last_epoch_state_counter"]) == src["last_epoch"]
    for key, source_key in [
        ("optimizer_step_attempts","optimizer_step_attempts"),
        ("ema_updates","ema_updates"),
    ]:
        assert int(row[key]) == int(src[source_key]), (index,key,row[key],src[source_key])
    assert row["seed"] == "0"
    assert abs(float(row["train_loss_mean"]) - float(src["train_loss_mean"])) <= 1e-15
    for metric in METRICS:
        assert float(row[metric]) == float(src["metrics"][metric]), (index,metric)
    assert row["weight_source"] == src["weight_source"]
    assert row["evaluated_ema_tensor_sha256"] == src["ema_tensor_sha256"]
    assert row["prediction_sha256"] == src["prediction_sha256"]
    assert row["evaluation_arrays_sha256"] == src["evaluation_arrays_sha256"]
    assert row["protocol_sha256"] == src["protocol_sha256"]
    assert row["configuration_sha256"] == src["config_sha256"]
    assert row["source_id"] == "NO_MSCA_NATIVE_EPOCH_ROWS_R320"
    assert row["source_sha256"] == file_sha(source_path)
    assert int(row["source_record_index_0based"]) == index
    assert row["source_locator"] == f"line {index+1}"
    assert row["source_raw_record_sha256"] == sha(source_raw[index])
    canonical = json.dumps(src, sort_keys=True, separators=(",",":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    assert row["normalized_row_sha256"] == sha(canonical)
    assert row["checkpoint_sha256_present_in_original_source"] == src.get("checkpoint_sha256", "")
    assert json.loads(row["first_executed_lrs"]) == src["first_executed_lrs"]
    assert json.loads(row["post_final_update_lrs"]) == src["post_final_update_lrs"]

# Bind terminal and best-AP outputs to saved lineage rows; verify data tables.
summary = read_json(OUT / "实验汇总.json")
audit_path = RECOVERY / "lineage_second_recovery_through_round320.json"
audit = read_json(audit_path)
assert file_sha(audit_path) == "66ba7a9b2f61c550222230aa0a661994f585abdc8d0d69cee9430ba727cb1734"
assert audit["status"] == "VERIFIED_ABLATION_LINEAGE_AND_SAVED_ENDPOINTS"
assert audit["completed_rounds"] == 320 and audit["validation_images"] == 548
assert audit["predictions"] == 164400 and audit["model_forwards"] == audit["evaluator_runs"] == 0
assert audit["maximum_latest_metric_error"] == 0
terminal = audit["endpoints"]["terminal"]
best = audit["endpoints"]["best"]
assert terminal["epoch_index"] == 319 and best["epoch_index"] == 301
assert best["epoch_index"] == max(range(320), key=lambda i: source_rows[i]["metrics"]["AP"])
for index, endpoint in [(319,terminal),(301,best)]:
    assert source_rows[index]["metrics"] == endpoint["metrics"]
for role, endpoint in [("primary", terminal), ("secondary", best)]:
    item = next(x for x in summary["completed_endpoints"] if x["arm"] == "NO_MSCA" and x["endpoint_role"] == role)
    assert item["round_1based"] == endpoint["epoch_index"] + 1
    assert item["trainable_parameters"] == 3699662
    for metric in METRICS:
        assert item["metrics_fraction"][metric] == endpoint["metrics"][metric]

info = read_csv(OUT / "终点信息.csv")
assert sum(x["arm"] == "NO_MSCA" for x in info) == 2
metric_rows = read_csv(OUT / "终点指标.csv")
n0_metric_rows = [x for x in metric_rows if x["arm"] == "NO_MSCA"]
assert len(n0_metric_rows) == 24
for row in n0_metric_rows:
    endpoint = terminal if row["endpoint_role"] == "primary" else best
    assert int(row["round_1based"]) == endpoint["epoch_index"] + 1
    assert float(row["value_fraction"]) == endpoint["metrics"][row["metric"]]
delta_rows = read_csv(OUT / "FULL差值.csv")
n0_deltas = [x for x in delta_rows if x["comparison"] == "FULL-NO_MSCA"]
assert len(n0_deltas) == 24
for row in n0_deltas:
    role = row["endpoint_role"]
    endpoint = terminal if role == "primary" else best
    full = next(x for x in summary["completed_endpoints"] if x["arm"] == "FULL" and x["endpoint_role"] == role)
    expected = (full["metrics_fraction"][row["metric"]] - endpoint["metrics"][row["metric"]]) * 100
    assert abs(float(row["difference_percentage_points"]) - expected) <= 1e-12
progress = read_csv(OUT / "NO_MSCA进度.csv")
assert len(progress) == 12 and all(x["status"] == "completed_320_rounds" for x in progress)
for row in progress:
    assert float(row["round320_primary_value"]) == terminal["metrics"][row["metric"]]
    assert float(row["round320_secondary_value"]) == best["metrics"][row["metric"]]
    assert row["native_backup_client_alive"] == "COMPLETED_EXIT0_AND_FINAL_BACKUP_VERIFIED"

# Verify the expanded source index against the summary and preserve its previous byte prefix.
source_csv_path = OUT / "输入来源.csv"
source_csv_rows = read_csv(source_csv_path)
source_summary = read_json(OUT / "实验汇总.json")["input_sources"]
assert len(source_csv_rows) == 43 and len(source_summary) == 43
old_source_csv = (SNAP280 / "输入来源.csv").read_bytes()
assert source_csv_path.read_bytes().startswith(old_source_csv)
for got, expected in zip(source_csv_rows, source_summary):
    assert all(str(got[k]) == str(expected[k]) for k in got.keys()), got["source_id"]
assert all(not Path(x["relative_to_remote_root"]).is_absolute() and "C:" not in x["relative_to_remote_root"] for x in source_summary[-7:])
assert len([x for x in source_summary if x["source_id"].endswith("R320") or "R302" in x["source_id"]]) >= 7
# Verify the workbook import/export check, final workbook hash and all rendered previews.
checks_path = OUT / "_build/round320_workbook_checks.json"
checks = read_json(checks_path)
xlsx_path = OUT / "outputs/实验数据汇总_20261108/实验数据汇总.xlsx"
assert file_sha(xlsx_path) == "e900dfdc0eb60d1a9d4f66d1c4ef8e9fcd1ad7cbc9343d11c921838704824ebf"
assert checks["status"] == "ROUND320_WORKBOOK_FINAL_VISUAL_REFINEMENT_EXPORTED_AND_REIMPORTED"
assert checks["xlsx_sha256"] == file_sha(xlsx_path)
assert checks["total_endpoint_difference_formulas"] == 72
assert checks["NO_MSCA_percent_precision_aligned_with_peer_columns"] is True
assert checks["NO_MSCA_configuration_source_shortened_for_readability"] is True
preview_hashes = {}
for name, expected in checks["previews"].items():
    p = OUT / "_build/previews" / name
    assert p.is_file() and file_sha(p) == expected
    preview_hashes[name] = expected
status_path = EVIDENCE / "monitor_status.json"
receipt_path = EVIDENCE / "completed_run_receipt.json"
supervisor_path = EVIDENCE / "completed_supervisor.json"
for p, expected in [
    (status_path,"b9d16d44695939c8f01285c580b7d8e39b531d79b3567fe2380f15aaf9d2e0db"),
    (receipt_path,"0053cd79f6e2925c02120303701f4491e62c151becaf2a8ad5ad457c09071777"),
    (supervisor_path,"84f86c4be37084977a23bb63a52a4728a8455ae88967b74150fdcdf53ea1890a"),
]:
    assert file_sha(p) == expected
status, receipt, supervisor = read_json(status_path), read_json(receipt_path), read_json(supervisor_path)
assert status["status"] == "COMPLETED_AND_BACKED_UP" and status["epoch"] == 319
assert receipt["status"] == "COMPLETED" and receipt["completed_epochs"] == 320 and receipt["actual_loop_epoch"] == 319
assert supervisor["status"] == "COMPLETED" and supervisor["exit_code"] == 0

result = {
    "status": "ROUND320_PARENT_READBACK_PASSED",
    "checked_utc": datetime.now(timezone.utc).isoformat(),
    "prior_round280_manifest_sha256": file_sha(old_manifest_path),
    "prior_csv_exact_byte_prefix_bytes": len(old_csv),
    "prior_csv_rows_preserved_exactly": 1240,
    "current_csv_sha256": file_sha(current_csv_path),
    "current_csv_rows": len(all_rounds),
    "input_sources_csv_rows": len(source_csv_rows),
    "input_sources_csv_sha256": file_sha(source_csv_path),
    "prior_input_sources_csv_exact_prefix_bytes": len(old_source_csv),
    "rows_by_arm": counts,
    "native_cumulative_records": 320,
    "new_rows_round281_320_checked_against": [
        "metrics","optimizer counters","loss","EMA weight source and SHA","prediction and evaluation-array SHA",
        "learning-rate vectors","protocol/configuration SHA","source line and raw/normalized row digests"
    ],
    "terminal_round_1based": terminal["epoch_index"] + 1,
    "best_AP_round_1based": best["epoch_index"] + 1,
    "terminal_metrics_sha256": terminal["metrics_sha256"],
    "best_metrics_sha256": best["metrics_sha256"],
    "trainable_parameters": 3699662,
    "workbook_sha256": file_sha(xlsx_path),
    "workbook_reimport_formula_count": checks["total_endpoint_difference_formulas"],
    "previews_sha256": preview_hashes,
    "native_worker_status": status["status"],
    "supervisor_observed_worker_exit_code": supervisor["exit_code"],
    "supervisor_exit_code": supervisor["exit_code"],
    "model_forwards": 0,
    "evaluator_runs": 0,
    "new_training_runs": 0
}
out = OUT / "_build/round320_parent_readback.json"
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
