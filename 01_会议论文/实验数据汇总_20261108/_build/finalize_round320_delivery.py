from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

OUT = Path(r"C:\Users\jiwei\Desktop\小论文投稿\01_会议论文\实验数据汇总_20261108")
PAPER = OUT.parent
RECOVERY = PAPER / "远程公平对照_20261103/模块消融/NO_MSCA/recovery30715_20261109"
EVIDENCE = RECOVERY / "completion_evidence_round320_20261111"
DEST = OUT / "verified_snapshots/NO_MSCA_round320_verified_20261111"
METRICS = ["AP","AP50","AP75","APs","APm","APl","AR1","AR10","AR100","ARs","ARm","ARl"]

def sha_bytes(data): return hashlib.sha256(data).hexdigest()
def sha_file(path): return sha_bytes(path.read_bytes())
def read_json(path): return json.loads(path.read_text(encoding="utf-8-sig"))
def write_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")

assert not DEST.exists(), f"Refusing to overwrite immutable snapshot: {DEST}"
parent_readback_path = OUT / "_build/round320_parent_readback.json"
parent_readback = read_json(parent_readback_path)
assert parent_readback["status"] == "ROUND320_PARENT_READBACK_PASSED"
assert parent_readback["current_csv_rows"] == 1280
assert parent_readback["input_sources_csv_rows"] == 43

# Retain the historical export audit while making the current workbook state explicit.
export_path = OUT / "导出核查.json"
previous_export = read_json(export_path)
workbook_path = OUT / "outputs/实验数据汇总_20261108/实验数据汇总.xlsx"
workbook_checks_path = OUT / "_build/round320_workbook_checks.json"
workbook_checks = read_json(workbook_checks_path)
export_current = {
    "status": "ROUND320_WORKBOOK_REIMPORTED_VISUALLY_REVIEWED_AND_SOURCE_BOUND",
    "xlsx": str(workbook_path),
    "xlsx_sha256": sha_file(workbook_path),
    "numeric_metric_cells_verified": 96,
    "comparison_formula_cells_verified": 72,
    "NO_MSCA_primary_and_secondary_metric_cells_verified": 24,
    "NO_MSCA_primary_and_secondary_blank_cells_verified": 0,
    "source_index_rows_verified": 43,
    "original_source_records_preserved_byte_prefix": 36,
    "round320_source_records_added_and_verified": 7,
    "endpoint_and_completion_source_hashes_added_to_workbook": 19,
    "rendered_sheets_visually_reviewed": ["终点对比","NO_MSCA进度","来源与权重"],
    "native_excel_opened": False,
    "note": "ArtifactTool export/reimport, all 72 difference formulas, both NO_MSCA endpoints, source hashes and the three rendered sheets were verified. No model forward, evaluator run or training was performed during this data update.",
    "NO_MSCA_accepted_rounds": 320,
    "NO_MSCA_progress_metric_cells_verified": 12,
    "completed_sheet_values_formulas_sources_preserved": True,
    "workbook_error_scan": workbook_checks["workbook_error_scan"],
    "round320_parent_readback_sha256": sha_file(parent_readback_path),
    "round320_workbook_checks_sha256": sha_file(workbook_checks_path),
    "round320_preview_sha256": workbook_checks["previews"],
    "update_local_date": "2026-11-11",
    "previous_round280_export_check": previous_export,
}
write_json(export_path, export_current)

# Finalize the round-320 update report before building the immutable package.
check_path = OUT / "第320轮更新核查.json"
check = read_json(check_path)
check.update({
    "status": "ROUND320_DATA_WORKBOOK_SOURCE_INDEX_AND_SNAPSHOT_VERIFIED",
    "local_date": "2026-11-11",
    "parent_readback_sha256": sha_file(parent_readback_path),
    "workbook_sha256": sha_file(workbook_path),
    "workbook_checks_sha256": sha_file(workbook_checks_path),
    "export_check_sha256": sha_file(export_path),
    "input_sources_csv_rows": 43,
    "input_sources_csv_sha256": sha_file(OUT / "输入来源.csv"),
    "prior_input_sources_csv_exact_byte_prefix_rows": 36,
    "per_epoch_csv_sha256": sha_file(OUT / "逐轮指标.csv"),
    "per_epoch_rows_after_update": 1280,
    "snapshot_destination": str(DEST),
    "snapshot_verified": True,
    "model_forwards": 0,
    "evaluator_runs": 0,
    "new_training_runs": 0,
    "workbook_and_snapshot_pending": False,
})
write_json(check_path, check)

# Refresh the delivery manifest from the existing inventory, keeping history and adding round-320 QA artifacts.
delivery_path = OUT / "交付清单.json"
previous_delivery = read_json(delivery_path)
inventory = {}
for item in previous_delivery["files"]:
    rel = item["file"].replace("\\","/")
    if rel == "交付清单.json":
        continue
    p = OUT / Path(rel)
    assert p.is_file(), f"Missing prior delivery file: {rel}"
    inventory[rel] = {"file": rel, "bytes": p.stat().st_size, "sha256": sha_file(p)}
for rel in [
    "第320轮更新核查.json",
    "_build/round320_workbook_checks.json",
    "_build/round320_parent_readback.json",
    "_build/previews/round320_endpoints_full_after.png",
    "_build/previews/round320_progress_after.png",
    "_build/previews/round320_sources_full_after.png",
]:
    p=OUT/Path(rel)
    assert p.is_file(), rel
    inventory[rel]={"file":rel,"bytes":p.stat().st_size,"sha256":sha_file(p)}

delivery = {
    "status": "ROUND320_COMPLETE_CURRENT_DELIVERY",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "NO_MSCA_accepted_rounds": 320,
    "per_epoch_csv_rows": 1280,
    "input_source_rows": 43,
    "workbook_sha256": sha_file(workbook_path),
    "round320_parent_readback_sha256": sha_file(parent_readback_path),
    "round320_workbook_checks_sha256": sha_file(workbook_checks_path),
    "NO_MSCA_terminal_round_1based": 320,
    "NO_MSCA_best_AP_round_1based": 302,
    "NO_MSCA_terminal_AP": 0.22219763674879622,
    "NO_MSCA_terminal_APs": 0.1472717517007563,
    "NO_MSCA_best_AP": 0.2253384467031726,
    "NO_MSCA_trainable_parameters": 3699662,
    "source_evidence_packet": {
        "binder": str(RECOVERY / "round320_completion_and_endpoints_20261111.json"),
        "binder_sha256": sha_file(RECOVERY / "round320_completion_and_endpoints_20261111.json"),
        "packet_directory": str(EVIDENCE),
        "packet_file_count": len([p for p in EVIDENCE.rglob("*") if p.is_file()]),
        "packet_bytes": sum(p.stat().st_size for p in EVIDENCE.rglob("*") if p.is_file()),
    },
    "previous_verified_snapshot": str(OUT / "verified_snapshots/NO_MSCA_round280_verified_20261110"),
    "verified_snapshot_destination": str(DEST),
    "files": [inventory[k] for k in sorted(inventory)],
    "manifest_self_reference_note": "This manifest excludes itself and the snapshot_manifest.json to avoid recursive hash references. The immutable snapshot manifest binds this manifest by SHA-256.",
    "updated_local_date": "2026-11-11",
}
write_json(delivery_path,delivery)

# Assemble a self-contained immutable snapshot of all current deliverables and their compact source packet.
DEST.mkdir(parents=True,exist_ok=False)
files=[]
for item in delivery["files"]:
    rel=item["file"]
    source=OUT/Path(rel)
    target=DEST/Path(rel)
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    assert target.stat().st_size==item["bytes"] and sha_file(target)==item["sha256"],rel
    files.append({"file":rel,"bytes":target.stat().st_size,"sha256":sha_file(target)})
# Bind the current delivery manifest itself into the snapshot.
manifest_copy=DEST/"交付清单.json"
shutil.copy2(delivery_path,manifest_copy)
files.append({"file":"交付清单.json","bytes":manifest_copy.stat().st_size,"sha256":sha_file(manifest_copy)})

binder_src=RECOVERY/"round320_completion_and_endpoints_20261111.json"
binder_dst=DEST/"source_evidence/round320_completion_and_endpoints_20261111.json"
binder_dst.parent.mkdir(parents=True,exist_ok=True)
shutil.copy2(binder_src,binder_dst)
files.append({"file":"source_evidence/round320_completion_and_endpoints_20261111.json",
              "bytes":binder_dst.stat().st_size,"sha256":sha_file(binder_dst)})
for source in sorted(EVIDENCE.rglob("*")):
    if source.is_file():
        rel=Path("source_evidence/completion_evidence_round320_20261111")/source.relative_to(EVIDENCE)
        target=DEST/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        files.append({"file":rel.as_posix(),"bytes":target.stat().st_size,"sha256":sha_file(target)})

snapshot = {
    "status": "ROUND320_CURRENT_DELIVERY_BYTE_VERIFIED",
    "local_date": "2026-11-11",
    "NO_MSCA_independently_verified_rounds": 320,
    "per_epoch_csv_rows": 1280,
    "input_source_rows": 43,
    "path_resolution_base": "snapshot_manifest.json parent directory",
    "source_delivery_manifest_sha256": sha_file(delivery_path),
    "parent_readback_sha256": sha_file(parent_readback_path),
    "prior_verified_snapshot": str(OUT / "verified_snapshots/NO_MSCA_round280_verified_20261110"),
    "prior_snapshot_manifest_sha256": "883acf595c6fe7ebd7cd86a52e879b90d507c6052536f6d90b0aafe60cabe730",
    "NO_MSCA_terminal_round_1based": 320,
    "NO_MSCA_best_AP_round_1based": 302,
    "source_evidence_packet_copied": True,
    "source_evidence_packet_bytes": delivery["source_evidence_packet"]["packet_bytes"] + binder_src.stat().st_size,
    "files": sorted(files,key=lambda x:x["file"]),
    "snapshot_manifest_files_exclude_self": True,
}
snapshot_path=DEST/"snapshot_manifest.json"
write_json(snapshot_path,snapshot)

# Final byte-level readback over all manifest entries; no later mutability is allowed.
reloaded=read_json(snapshot_path)
assert reloaded["source_delivery_manifest_sha256"]==sha_file(DEST/"交付清单.json")
for item in reloaded["files"]:
    p=DEST/Path(item["file"])
    assert p.is_file() and p.stat().st_size==item["bytes"] and sha_file(p)==item["sha256"],item["file"]
assert len(reloaded["files"]) == len(files)
assert sha_file(OUT/"verified_snapshots/NO_MSCA_round280_verified_20261110/snapshot_manifest.json")=="883acf595c6fe7ebd7cd86a52e879b90d507c6052536f6d90b0aafe60cabe730"
print(json.dumps({
 "status": "ROUND320_DATA_DELIVERY_AND_IMMUTABLE_SNAPSHOT_PASSED",
 "delivery_manifest_sha256": sha_file(delivery_path),
 "snapshot_manifest": str(snapshot_path),
 "snapshot_manifest_sha256": sha_file(snapshot_path),
 "snapshot_file_count": len(files),
 "data_csv_rows": 1280,
 "input_source_rows": 43,
 "workbook_sha256": sha_file(workbook_path),
 "round320_update_check_sha256": sha_file(check_path),
},ensure_ascii=False,indent=2))
