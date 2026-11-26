from pathlib import Path
import csv
import datetime as dt
import hashlib
import io
import json
import zipfile

ROOT = Path(r"C:\Users\jiwei\Desktop\小论文投稿\01_会议论文")
HERE = Path(__file__).resolve().parent
REMOTE = ROOT / "远程公平对照_20261103"
SNAP = ROOT / "实验数据汇总_20261108" / "verified_snapshots" / "NO_MSCA_round320_verified_20261111"
METRICS = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))

def record(p):
    return {"file": p.relative_to(ROOT).as_posix(), "sha256": sha(p)}

def jsonl(p):
    lines = p.read_bytes().splitlines()
    return [(json.loads(line), hashlib.sha256(line).hexdigest()) for line in lines if line.strip()]

def main():
    HERE.mkdir(parents=True, exist_ok=True)
    manifest = SNAP / "snapshot_manifest.json"
    assert sha(manifest) == "958b7bd88b206b5ea7693bffedf5bbdde141e37eb52d17c1cb6340244eb557af"
    frozen = {x["file"]: x["sha256"] for x in read(manifest)["files"]}
    csv_path = SNAP / "逐轮指标.csv"
    assert sha(csv_path) == frozen[csv_path.name]
    table = list(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8-sig"))))
    by_key = {(r["arm"], int(r["round_1based"])): r for r in table}
    sources = {
        "BASE_ORIGINAL": REMOTE / "remote_evidence" / "interrupted_BASE_20261105" / "runs" / "BASE" / "epochs.jsonl",
        "BASE_SAVED_THROUGH_308": REMOTE / "断点续训_20261105" / "backup_epoch0307" / "epochs.jsonl",
        "BASE_RECOVERY_309_320": REMOTE / "断点续训_20261105" / "BASE_30166_r308" / "terminal_evidence" / "run" / "epochs.jsonl",
    }
    native = {key: jsonl(p) for key, p in sources.items()}
    counts = {}
    for round_num in range(1, 321):
        row = by_key[("BASE", round_num)]
        if round_num <= 214:
            source_id, index = "BASE_ORIGINAL", round_num - 1
        elif round_num <= 308:
            source_id, index = "BASE_SAVED_THROUGH_308", round_num - 215
        else:
            source_id, index = "BASE_RECOVERY_309_320", round_num - 309
        source, raw_sha = native[source_id][index]
        assert row["source_id"] == source_id and row["source_sha256"] == sha(sources[source_id])
        assert int(row["source_record_index_0based"]) == index and row["source_raw_record_sha256"] == raw_sha
        assert source["epoch"] == round_num - 1 == int(row["actual_epoch_index_0based"])
        assert int(row["optimizer_step_attempts"]) == source["optimizer_step_attempts"]
        for name in METRICS:
            assert float(row[name]) == source["metrics"][name]
        for out_key, native_key in [("evaluated_ema_tensor_sha256", "ema_tensor_sha256"), ("prediction_sha256", "prediction_sha256"), ("evaluation_arrays_sha256", "evaluation_arrays_sha256")]:
            assert row[out_key] == source[native_key]
        counts[source_id] = counts.get(source_id, 0) + 1
    assert counts == {"BASE_ORIGINAL": 214, "BASE_SAVED_THROUGH_308": 94, "BASE_RECOVERY_309_320": 12}
    receipt_path = REMOTE / "diagnostic_completion_receipt_20261105.json"
    receipt = read(receipt_path)
    result_path = REMOTE / "诊断准备_20261105" / "remote_final" / "diagnostic_result.json"
    result = read(result_path)
    authorization = read(REMOTE / "诊断准备_20261105" / "diagnostic_user_authorization_20261105.json")
    pair_path = REMOTE / "断点续训_20261105" / "BASE_FULL_pair_verified_20261108.json"
    pair = read(pair_path)
    anchor = pair["lineages"]["BASE"]["origin_sha256"]
    assert anchor == authorization["input_checkpoint_sha256"] == "33bb8012aa6b43a505b2f1a554189ed6abeeb4c39dc5cd6f20e83726fb922f65"
    assert receipt["status"] == "ONE_AUTHORIZED_DIAGNOSTIC_COMPLETED_AND_EVIDENCE_VERIFIED"
    assert receipt["optimizer_updates"] == result["optimizer_updates"] == 17787
    assert receipt["full_validation_passes"] == result["full_validation_passes"] == len(receipt["phases"]) == 14
    checked = []
    for evidence in receipt["evidence"]:
        p = REMOTE / evidence["path"]
        assert p.is_file() and sha(p) == evidence["sha256"]
        checked.append(record(p))
    diagnostic = {p["epoch_index"] + 1: p for p in receipt["phases"] if p["phase"].startswith("epoch_")}
    assert sorted(diagnostic) == list(range(215, 226))
    excluded = [x for x, _ in native["BASE_ORIGINAL"][214:225]]
    assert len(excluded) == 11
    assert all(float(by_key[("BASE", r)]["AP"]) != diagnostic[r]["AP"] for r in diagnostic)
    assert all(float(by_key[("BASE", r)]["AP"]) != excluded[r-215]["metrics"]["AP"] for r in diagnostic)

    provenance_path = ROOT / "修订记录_20261103" / "第三十三轮实验主线重组_20261108" / "learning_curve_provenance.json"
    curves_path = provenance_path.with_name("paired_learning_curve_rows.json")
    curves, provenance = read(curves_path), read(provenance_path)
    for arm in ["BASE", "FULL"]:
        assert len(curves[arm]) == 320
        for round_num, row in enumerate(curves[arm], 1):
            for name in ["AP", "APs"]:
                assert row["metrics"][name] == float(by_key[(arm, round_num)][name])
    manuscript = ROOT / "稿件" / "MPSMA-DEIM_conference_EN_revised.docx"
    assert sha(manuscript) == "8f5ba9fd3bcdd9fa8b2f8f297937b6d7de0afc3c77993024fadd7b855fbfa189"
    with zipfile.ZipFile(manuscript) as z:
        media = {name: hashlib.sha256(z.read(name)).hexdigest() for name in z.namelist() if name.startswith("word/media/")}
    expected_figure = provenance["figure_sha256"]["paired_validation_curves.png"]
    matched_media = [k for k, v in media.items() if v == expected_figure]
    assert len(matched_media) == 1

    series = []
    original_anchor_AP = native["BASE_ORIGINAL"][213][0]["metrics"]["AP"]
    diag_anchor = next(p for p in receipt["phases"] if p["phase"] == "initial_ema")
    for branch, label in [("original_excluded", "Original branch (excluded after round 214)"), ("authorized_diagnostic", "Separate 11-round diagnostic"), ("retained_continuation", "Continuation retained in Figure 6")]:
        anchor_AP = diag_anchor["AP"] if branch == "authorized_diagnostic" else original_anchor_AP
        series.append({"branch": branch, "round_1based": 214, "AP_fraction": anchor_AP, "AP_percent": 100*anchor_AP})
        for r in range(215, 226):
            value = excluded[r-215]["metrics"]["AP"] if branch == "original_excluded" else diagnostic[r]["AP"] if branch == "authorized_diagnostic" else float(by_key[("BASE", r)]["AP"])
            series.append({"branch": branch, "round_1based": r, "AP_fraction": value, "AP_percent": 100*value})
    output_csv = HERE / "BASE_three_branches_round214_225.csv"
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(series[0])); writer.writeheader(); writer.writerows(series)
    report = {
        "status": "BASE_320_ROWS_AND_THREE_BRANCH_PROVENANCE_VERIFIED",
        "observed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_manuscript": record(manuscript), "immutable_data_snapshot": record(manifest), "per_round_csv": record(csv_path),
        "native_sources": [dict(source_id=key, **record(path)) for key, path in sources.items()],
        "BASE_all_320_rounds_checked_against_native_rows": {"slice_counts": counts, "metrics_per_row": 12, "maximum_metric_difference": 0, "counters_and_EMA_prediction_array_hashes_match": True},
        "diagnostic_completion": {"receipt": record(receipt_path), "result": record(result_path), "completed_epochs": 11, "optimizer_updates": 17787, "full_validation_passes": 14, "completion_receipt_referenced_files_rehashed": checked},
        "common_saved_anchor_sha256": anchor,
        "diagnostic_rows_in_manuscript_BASE_trajectory": 0,
        "original_post_anchor_rows_excluded_from_manuscript": 11,
        "retained_post_anchor_rows_in_round215_225": 11,
        "figure6_binding": {"curve_rows": record(curves_path), "provenance": record(provenance_path), "BASE_and_FULL_AP_APs_all_320_rounds_match_snapshot": True, "embedded_media_part": matched_media[0], "embedded_png_sha256": expected_figure},
        "branch_table": record(output_csv),
        "interpretation": "Diagnostic and retained continuation are distinct branches; neither diagnostic rows nor the 11 discarded original rows are substituted into the final retained BASE trajectory. The diagnostic did not reproduce the original drop and does not identify its cause. The BASE/FULL comparison remains conditional on different continuation histories.",
        "new_model_forwards": 0, "new_evaluator_runs": 0, "new_training_runs": 0,
    }
    (HERE / "branch_provenance_audit_rev40.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:report[k] for k in ["status", "BASE_all_320_rounds_checked_against_native_rows", "diagnostic_rows_in_manuscript_BASE_trajectory", "original_post_anchor_rows_excluded_from_manuscript", "interpretation"]}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
