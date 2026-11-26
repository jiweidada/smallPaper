from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import zipfile

from lxml import etree as ET


ROOT = Path(r"C:\Users\jiwei\Desktop\小论文投稿\01_会议论文")
HERE = Path(__file__).resolve().parent
MASTER = ROOT / "稿件" / "MPSMA-DEIM_conference_EN_revised.docx"
CANDIDATE = HERE / "MPSMA-DEIM_conference_EN_revised_rev39.docx"
BACKUP = HERE / "MPSMA-DEIM_conference_EN_revised_rev38_backup.docx"
ORIGINAL = ROOT / "稿件" / "MPSMA-DEIM_conference_EN.docx"
SNAP = ROOT / "实验数据汇总_20261108" / "verified_snapshots" / "NO_MSCA_round320_verified_20261111"
INDEX_DIR = ROOT / "修订记录_20261103" / "主要结论证据索引"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main", "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}
SOURCE_SHA = "f598650e48c02ca2ffa46d5d9cc72cc8e24531641817000fe050f0fbc4068b23"
CANDIDATE_SHA = "8f5ba9fd3bcdd9fa8b2f8f297937b6d7de0afc3c77993024fadd7b855fbfa189"
NOW = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def write_json(p: Path, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"file": rel(p), "sha256": sha(p)}


def text(e):
    return "".join(n.text or "" for n in e.iter() if n.tag.endswith("}t"))


def record(p: Path):
    return {"file": rel(p), "sha256": sha(p)}


def main():
    assert sha(MASTER) == sha(BACKUP) == SOURCE_SHA
    assert sha(CANDIDATE) == CANDIDATE_SHA
    assert sha(ORIGINAL) == "ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323"
    with zipfile.ZipFile(CANDIDATE) as z:
        assert z.testzip() is None
        parts = {n: z.read(n) for n in z.namelist()}
    with zipfile.ZipFile(BACKUP) as z:
        old_parts = {n: z.read(n) for n in z.namelist()}
    doc = ET.fromstring(parts["word/document.xml"])
    old_doc = ET.fromstring(old_parts["word/document.xml"])
    body = doc.find("w:body", NS)
    old_body = old_doc.find("w:body", NS)
    changed_parts = sorted(n for n in parts if parts[n] != old_parts[n])
    assert changed_parts == ["word/comments.xml", "word/document.xml"]
    changed_paras = [i for i, (a, b) in enumerate(zip(body, old_body)) if text(a) != text(b)]
    assert changed_paras == [5, 10, 13, 20, 71, 72, 73, 79]
    assert len(body) == len(old_body)
    tables = [[ [text(tc) for tc in tr.findall("w:tc", NS)] for tr in t.findall("w:tr", NS)] for t in body.findall("w:tbl", NS)]
    manifest_path = SNAP / "snapshot_manifest.json"
    assert sha(manifest_path) == "958b7bd88b206b5ea7693bffedf5bbdde141e37eb52d17c1cb6340244eb557af"
    frozen = {item["file"]: item["sha256"] for item in read_json(manifest_path)["files"]}
    metrics_path, info_path = SNAP / "终点指标.csv", SNAP / "终点信息.csv"
    assert sha(metrics_path) == frozen[metrics_path.name] and sha(info_path) == frozen[info_path.name]
    metric_rows = list(csv.DictReader(io.StringIO(metrics_path.read_text(encoding="utf-8-sig"))))
    info_rows = list(csv.DictReader(io.StringIO(info_path.read_text(encoding="utf-8-sig"))))
    assert len(metric_rows) == 96 and len(info_rows) == 8
    assert all(r["completed_rounds"] == "320" and r["seed"] == "0" and r["validation_images"] == "548" for r in info_rows)
    metric = {(r["arm"], r["endpoint_role"], r["metric"]): float(r["value_fraction"]) for r in metric_rows}
    info = {(r["arm"], r["endpoint_role"]): r for r in info_rows}
    for row in tables[0][1:]:
        role, arm, round_str, *values = row
        key = (arm, role.lower())
        assert round_str == info[key]["round_1based"]
        for m, value in zip(["AP", "AP50", "AP75", "APs", "AR100"], values):
            assert value == f"{metric[(*key, m)]:.4f}"
    for row in tables[1][1:]:
        arm, *values = row
        for m, value in zip(["AP", "APs", "AR100", "ARl"], values[:4]):
            assert value == f"{metric[(arm, 'primary', m)]:.4f}"
        assert values[4] == f"{int(info[(arm, 'primary')]['trainable_parameters']) / 1e6:.4f}"
    comparisons = {}
    for other, metric_names in [("BASE", ["AP", "APs"]), ("NO_MPS", ["AP", "APs", "ARl"]), ("NO_MSCA", ["AP", "APs", "AR100"])]:
        comparisons[other] = {m: 100 * (metric[("FULL", "primary", m)] - metric[(other, "primary", m)]) for m in metric_names}
    assert [f"{comparisons['BASE'][m]:.2f}" for m in ["AP", "APs"]] == ["3.53", "3.23"]
    assert [f"{comparisons['NO_MPS'][m]:.2f}" for m in ["AP", "APs", "ARl"]] == ["1.57", "1.77", "-1.69"]
    assert [f"{comparisons['NO_MSCA'][m]:.2f}" for m in ["AP", "APs", "AR100"]] == ["0.33", "0.66", "0.28"]
    params = {arm: int(info[(arm, "primary")]["trainable_parameters"]) for arm in ["BASE", "FULL", "NO_MPS", "NO_MSCA"]}
    assert params["FULL"] - params["NO_MPS"] == 5325
    assert params["FULL"] - params["NO_MSCA"] == 3841280
    assert f"{params['FULL']/params['BASE']:.2f}" == "2.02"
    assert info[("NO_MSCA", "secondary")]["round_1based"] == "302"
    assert f"{metric[('NO_MSCA', 'secondary', 'AP')]:.4f}" == "0.2253"
    numeric_path = HERE / "candidate_numeric_check_rev39.json"
    numeric_ref = write_json(numeric_path, {
        "status": "PASSED_CANDIDATE_TABLES_AND_REVISED_NUMERIC_SENTENCES",
        "checked_utc": NOW, "candidate_sha256": CANDIDATE_SHA,
        "snapshot_manifest": record(manifest_path), "source_metrics": record(metrics_path), "source_endpoints": record(info_path),
        "table_1_all_cells_match_frozen_csv": True, "table_2_all_cells_match_frozen_csv": True,
        "new_sentence_differences_percentage_points_FULL_minus_other": comparisons,
        "parameter_counts": params,
        "primary_metrics_not_mixed_with_validation_selected_secondary": True,
        "NO_MSCA_secondary_round": 302, "NO_MSCA_secondary_AP": metric[("NO_MSCA", "secondary", "AP")],
        "scope": "Readback from the immutable CSVs and previously audited source chain; no new inference or evaluator execution.",
    })

    acronym_path = HERE / "acronym_check_rev39.json"
    acr = read_json(acronym_path)
    assert acr["status"] == "partial" and not acr["unread_sections"]
    definitions = {
        "MSCA": "Multi-scale Context-aware Attention (MSCA)", "SiLU": "sigmoid-weighted linear unit (SiLU)",
        "NMS": "non-maximum suppression (NMS)", "ESE": "Effective squeeze-excitation (ESE)",
        "ReLU": "rectified linear unit (ReLU)", "AP": "average precision (AP)",
        "COCO": "Common Objects in Context (COCO)", "IoU": "intersection-over-union (IoU)", "TF32": "TensorFloat-32 (TF32)",
    }
    full_text = "\n".join(text(p) for p in body.findall("w:p", NS))
    for key, value in definitions.items():
        assert value.lower() in full_text.lower()
    assert {r["acronym"] for r in acr["coverage_limits"]} <= set(definitions)
    proper_names = {"FINE", "DFINE", "MPSMA", "SimAM", "DMF", "TF", "NVIDIA", "RTX", "NO"}
    metric_symbols = {"HW", "AP50", "AP75", "ARl"}
    assert {x["acronym"] for x in acr["findings"]} <= proper_names | metric_symbols
    manual_acr_ref = write_json(HERE / "acronym_manual_disposition_rev39.json", {
        "status": "MANUAL_REVIEW_COMPLETED_AUTOMATED_REPORT_REMAINS_PARTIAL", "automated_report": record(acronym_path),
        "automated_findings_count": len(acr["findings"]), "unconfirmed_definition_count": len(acr["coverage_limits"]),
        "unconfirmed_definitions_checked_in_text": definitions,
        "named_methods_vendors_or_run_identifier_fragments": sorted(proper_names),
        "metric_or_shape_symbols": {"HW": "Product of height H and width W.", "AP50": "AP at IoU 0.50, defined in Section 4.1.", "AP75": "AP at IoU 0.75, defined in Section 4.1.", "ARl": "Large-object counterpart of AR@100, defined in Section 4.1."},
        "abstract_first_use_definitions_checked": ["UAV", "MPS", "MSCA", "COCO", "EMA", "AP", "APs", "FULL", "BASE", "NO_MPS", "NO_MSCA"],
        "all_advisories_dispositioned": True, "automated_clean_certificate": False,
    })

    page_paths = [HERE / "rendered_word" / f"page-{i}.png" for i in range(1, 10)]
    assert all(p.exists() and p.stat().st_size > 80000 for p in page_paths)
    assert len(list((HERE / "rendered_word").glob("page-*.png"))) == 9
    assert read_json(HERE / "render_metadata_rev39.json")["page_count"] == 9
    visual_ref = write_json(HERE / "visual_qa_rev39.json", {
        "status": "ALL_NINE_PAGES_RENDERED_AND_VISUALLY_INSPECTED", "checked_utc": NOW, "candidate_sha256": CANDIDATE_SHA,
        "render_attempts": record(HERE / "render_attempts_rev39.json"),
        "renderer": "Bundled renderer diagnosed missing soffice.exe; hidden read-only Word export with macros disabled, then bundled Poppler.",
        "pdf": record(HERE / "rendered_word" / "manuscript.pdf"),
        "pages": [dict(page=i, **record(p), bytes=p.stat().st_size) for i, p in enumerate(page_paths, 1)],
        "visual_findings": "All pages inspected. Tables, captions and all six figures are legible; six native equations show all expected symbols; no clipping, overlap or broken tables. Page 8 ends before the separate references page as in the source layout.",
        "font_warnings": "Poppler reported Symbol and ArialUnicode mapping warnings; rendered symbols were visually present.",
        "target_venue_template_applied": False,
    })

    previous_citation = ROOT / "修订记录_20261103" / "第三十八轮第320轮结果核验_20261111" / "citation_carryforward_rev38.json"
    metadata_path = HERE / "reference3_crossref_registered_metadata.json"
    registered = read_json(metadata_path)["message"]
    assert registered["DOI"].lower() == "10.1109/igarss53475.2024.10640832" and registered["page"] == "9134-9138"
    assert registered["publisher"] == "IEEE"
    citation_ref = write_json(HERE / "citation_audit_rev39.json", {
        "revision": 39, "observed_utc": NOW, "manuscript_sha256": CANDIDATE_SHA,
        "prior_citation_audit": record(previous_citation),
        "bibliography_and_in_text_citations_unchanged": True, "reference_count": 13, "new_bibliography_entries": 0,
        "reference_3_pagination_gap_resolved": {"source": "Publisher-registered Crossref DOI metadata", "url": "https://api.crossref.org/works/10.1109/IGARSS53475.2024.10640832", "raw_response": record(metadata_path), "page": registered["page"], "publisher": registered["publisher"], "scope": "Registered metadata, not full-text access or an independent content review."},
        "full_text_unavailable_references": [3, 10], "already_cited_method_summaries": "Narrow descriptions remain supported by the official abstracts reviewed in rev38; no additional technical claims were introduced.",
    })

    preliminary_path = HERE / "targeted_patch_and_preliminary_check_rev39.json"
    preliminary = read_json(preliminary_path)
    method_ref = write_json(HERE / "method_content_check_rev39.json", {
        "checked_utc": NOW, "source_binding": record(Path(preliminary["method_source_binding_file"])),
        "four_method_files_rehashed": preliminary["four_method_sources_rehashed"],
        "direct_source_inspection": [
            {"source": "hgnetv2_MSCA_EnhancedStem.py:347-430", "finding": "EnhancedStemBlock detail uses two stride-one blocks plus average pool 2; main uses stride-two then stride-one; high-pass subtracts padded average pool and is resized to main; concatenation precedes residual spatial weighting and final stride-two output. Even input dimensions align floor-pool and ceil-convolution shapes."},
            {"source": "MSCA.py:89-196", "finding": "Lite averages identity and two pooled/interpolated maps, computes channel-specific spatial energy, and weights the original input. Dilated branches use reduction/group count C/8 and multiplier eight; context and channel gates multiply the original input. Means/variances use denominator HW, matching equations and Figures 4-5."},
            {"source": "simam.py:30-39", "finding": "Stage-two SimAM uses n=HW-1, as stated in Section 3.3."},
        ],
        "existing_figures_and_equations": "All six figures and native equations are byte-identical to rev38; visually compared with the retained method descriptions. No model was constructed or executed.",
        "audit_boundary": "Source inspection and rendering consistency support implementation description, not mechanism efficacy or runtime performance.",
    })

    # Only now replace the maintained file, with a byte-exact previous-version backup.
    assert sha(MASTER) == SOURCE_SHA and sha(BACKUP) == SOURCE_SHA and sha(CANDIDATE) == CANDIDATE_SHA
    temporary = MASTER.with_name("MPSMA-DEIM_conference_EN_revised.rev39.tmp.docx")
    shutil.copyfile(CANDIDATE, temporary)
    assert sha(temporary) == CANDIDATE_SHA
    os.replace(temporary, MASTER)
    assert sha(MASTER) == CANDIDATE_SHA
    preliminary["status"] = "REV39_PROMOTED_AFTER_NUMERIC_CONTENT_AND_VISUAL_REVIEW"
    preliminary["promoted_manuscript_sha256"] = sha(MASTER)
    preliminary["numeric_check"] = numeric_ref
    preliminary["visual_qa"] = visual_ref
    preliminary["acronym_dispositions"] = manual_acr_ref
    preliminary["citation_audit"] = citation_ref
    preliminary["method_content_check"] = method_ref
    write_json(preliminary_path, preliminary)

    previous_index = INDEX_DIR / "主要结论证据索引_第三十八轮.json"
    index_path = INDEX_DIR / "主要结论证据索引_第三十九轮.json"
    index_ref = write_json(index_path, {
        "revision": 39, "observed_utc": NOW, "manuscript": record(MASTER), "previous_evidence_index": record(previous_index),
        "unchanged_scientific_data_snapshot": record(manifest_path),
        "targeted_patch_and_readback": record(preliminary_path), "numeric_check": numeric_ref, "method_content_check": method_ref,
        "visual_qa": visual_ref, "citation_audit": citation_ref, "acronym_manual_dispositions": manual_acr_ref,
        "claim_scope": "Fixed-round primary and explicitly separate validation-selected secondary results; descriptive single-seed configuration comparisons with documented recovery histories. Capacity and surrounding architecture changes prevent component-only attribution.",
        "changes": ["Abstract defines its initialisms and uses only round320 primary numbers.", "Introduction and related work state the implemented integration and its relation to prior designs.", "MPS input X and the even-dimension condition are explicit.", "Conditional results state accuracy, recall and capacity trade-offs without repeated wording.", "Conclusion is shorter and preserves the single-seed, local-validation and continuation limits.", "Stale comment 3 saying MSCA is ongoing is corrected.", "Reference 3 pagination now matches IEEE-registered Crossref metadata."],
        "unchanged": {"tables": 2, "figures": 6, "native_equations": 6, "reference_entries": 13, "comments": 7},
        "new_model_forwards": 0, "new_evaluator_runs": 0, "new_training_runs": 0,
        "venue_selected": False, "submission_ready": False,
    })
    audit_path = HERE / "current_goal_acceptance_audit_rev39.json"
    audit_ref = write_json(audit_path, {
        "revision": 39, "observed_utc": NOW, "manuscript": record(MASTER), "previous_revision_backup": record(BACKUP), "original_user_manuscript": record(ORIGINAL),
        "evidence_index": index_ref,
        "original_requirements": [
            {"requirement": "Trace primary experiments and conditional removals; correct unfair comparisons and over-attribution.", "state": "Evidence supports the reported descriptive comparisons within stated limits.", "evidence": [numeric_ref, record(manifest_path), index_ref], "remaining_limit": "Different continuation histories, one seed and capacity-changing removals do not establish pure module effects or variability."},
            {"requirement": "Align methods, formulas, architecture figures and parameters with actual implementations.", "state": "Current source and preserved figure/formula descriptions checked within the static audit scope.", "evidence": [method_ref, visual_ref], "remaining_limit": "Static consistency does not validate mechanism efficacy or deployment cost."},
            {"requirement": "Clarify contributions and revise abstract, introduction, related work, analysis, conclusion, English and citations.", "state": "Eight targeted paragraphs and one stale comment corrected; full manuscript read and all nine pages inspected.", "evidence": [record(preliminary_path), citation_ref, manual_acr_ref, visual_ref], "remaining_limit": "Full texts [3] and [10] were not obtained; no independent peer review or target-venue fit judgment is claimed."},
            {"requirement": "Keep the original backup, a maintained revision and important change/issue records.", "state": "Original untouched; rev38 exact backup and promoted rev39 retained; current pointer and logs updated.", "evidence": [record(ORIGINAL), record(BACKUP), record(MASTER)]},
        ],
        "content_and_existing_evidence_revision_reviewed": True,
        "important_evidence_limits_still_present": ["No matched-history or capacity-matched evidence isolates the MSCA gate contribution.", "Single-seed local validation results provide no variance estimate or independent test performance.", "Unretained physical work after instance losses remains unquantified.", "Current n-model latency and onboard performance are unmeasured."],
        "submission_status": "Target-venue submission readiness is unproven; the user has deferred venue selection and template adaptation. Existing evidence supports the narrowed descriptive claims, not stronger causal or deployment claims.",
        "venue_template_and_page_limit_checked": False, "submission_ready": False,
        "new_experiment_authorization_requested_or_used": False, "new_training_runs": 0, "model_forwards": 0, "evaluator_runs": 0,
    })
    progress_path = HERE / "active_goal_progress_rev39_20261112.json"
    progress_ref = write_json(progress_path, {
        "goal_turn_classification": "progress", "previous_goal_turn_classification": "progress", "observed_utc": NOW,
        "goal_status": "active", "objective_scope": "Complete MPSMA-DEIM English conference manuscript revision and evidence verification; venue adaptation follows venue selection.",
        "manuscript": record(MASTER), "revision": 39, "evidence_index": index_ref, "acceptance_audit": audit_ref,
        "concrete_progress": ["Corrected abstract first-use definitions and endpoint presentation.", "Refined design contribution and literature connection.", "Explained numerical accuracy/capacity trade-offs and adverse ARl outcome.", "Corrected a stale unfinished-experiment comment.", "Closed reference 3 pagination metadata gap.", "Rendered and inspected all nine pages, then promoted the checked candidate."],
        "reason_goal_not_complete": "Conference/template requirements are unknown; important experimental inference limits remain explicitly recorded.",
        "new_experiments": 0, "continuous_background_monitoring": False,
        "remote_state_note": "No new remote observation in this turn; prior point-in-time status records retain their original timestamps and cannot be treated as live monitoring.",
    })
    state_path = ROOT / "远程公平对照_20261103" / "模块消融" / "current_execution_state.json"
    shutil.copyfile(state_path, HERE / "current_execution_state_before_rev39.json")
    state = read_json(state_path)
    state.update({"observed_local_utc": NOW, "paper_revision": 39, "paper_sha256": sha(MASTER), "current_evidence_index": str(index_path), "current_evidence_index_sha256": index_ref["sha256"], "current_goal_acceptance_audit": str(audit_path), "current_goal_acceptance_audit_sha256": audit_ref["sha256"], "latest_active_goal_progress": str(progress_path), "latest_active_goal_progress_sha256": progress_ref["sha256"], "citation_binding_evidence": str(HERE / "citation_audit_rev39.json"), "citation_binding_evidence_sha256": citation_ref["sha256"], "goal_status": "active", "last_goal_turn_classification": "progress", "paper_submission_ready": False, "final_comparisons_pending": False, "next_action": "Venue-specific template and page-limit adaptation after venue selection; retain current descriptive claim scope and documented evidence limits.", "operational_work_status": "ALL_FOUR_320_ROUND_RUNS_BACKED_UP; REV39_CONTENT_AND_SOURCE_REVIEW_COMPLETED; VENUE_FORMAT_DEFERRED; NO_CONTINUOUS_REMOTE_MONITOR"})
    for key in ["data200_refresh_pending", "data240_refresh_pending", "data280_refresh_pending", "data296_refresh_pending", "data320_refresh_pending"]:
        state[key] = False
    write_json(state_path, state)

    current_entry_path = ROOT / "当前修订入口.txt"
    shutil.copyfile(current_entry_path, HERE / "当前修订入口_before_rev39.txt")
    note = (
        "第三十九轮内容修订与现有证据核查（2026-11-12 UTC）\n\n"
        f"当前维护稿：{rel(MASTER)}\nSHA256：{sha(MASTER)}\n"
        f"第38轮精确备份：{rel(BACKUP)}\n原始稿保留，原始SHA256：{sha(ORIGINAL)}\n\n"
        "本轮修改：摘要补全EMA/AP等首次定义，只报告固定320轮主端点；引言明确MPS/MSCA集成位置和贡献边界；相关工作与本方法的连接更清楚；MPS输入X与偶数尺寸条件更明确；消融结果合并重复说明，保留NO_MPS的ARl不利结果并把MSCA精度差与384万参数增量一并解释；结论精简；旧批注“MSCA仍待完成”已更正。\n\n"
        "核查：8处正文段落、1条批注变更；2张表、6幅图、6条原生公式、13条引用和7条批注保留。两表及修改句数字与固定数据快照一致；9页Word渲染逐页检查通过。缩略语脚本报告partial，名称和度量符号提示已逐项人工判读，没有把脚本称为自动全通过。\n\n"
        "引用：[3]页码9134–9138已与IEEE登记在Crossref的元数据核对；[3]、[10]全文仍未取得，正文只保留已由官方摘要支持的简述。未新增参考文献。\n\n"
        "实验：BASE/FULL/NO_MPS/NO_MSCA均有320轮保留轨迹，数据表共1280行。NO_MSCA固定320轮AP/APs为0.2222/0.1473；AP最佳次要端点为302轮、AP0.2253。末轮与最佳端点不混用。本轮没有训练、模型前向或评价器重跑。\n\n"
        "尚未完成：会议未选择，模板、页数和投稿清单尚未适配。不同恢复历史、单seed、本地验证集选择、容量变化及未知未保留物理计算量仍限制归因；当前n模型延迟和机载性能未测。现有数据支持稿中收窄后的配置比较，不能据此声称纯模块因果效果、显著性或部署性能。未宣布可投稿。\n\n"
        f"证据索引：{rel(index_path)}\n验收记录：{rel(audit_path)}\n"
        f"数据快照：{rel(SNAP)}\n"
        "远程状态：本轮未重新查询；既有点时观测不等于持续后台监控。\n"
        "历史当前入口已完整备份于本轮目录；完整历次修改保留在修订说明与未解决问题.txt。\n"
    )
    current_entry_path.write_text(note, encoding="utf-8")
    (HERE / "revision_notes_rev39.txt").write_text(note, encoding="utf-8")
    header = "【第39轮已更新：2026-11-12 UTC】\n" + note + "\n以下为历史记录。\n\n"
    for p in [ROOT / "目标自主执行入口_20261109.txt", ROOT / "修订记录_20261103" / "修订说明与未解决问题.txt", ROOT / "远程公平对照_20261103" / "结果并入与验收清单.txt"]:
        previous = p.read_text(encoding="utf-8-sig")
        p.write_text(header + previous, encoding="utf-8")
    chain = {
        "status": "REV39_PROMOTION_AND_CURRENT_POINTERS_READ_BACK_PASSED", "checked_utc": NOW,
        "master": record(MASTER), "backup": record(BACKUP), "original": record(ORIGINAL),
        "evidence_index": record(index_path), "acceptance_audit": record(audit_path), "progress": record(progress_path), "current_state": record(state_path),
        "current_entry": record(current_entry_path), "goal_status": state["goal_status"], "submission_ready": False,
    }
    assert sha(MASTER) == CANDIDATE_SHA and read_json(state_path)["current_evidence_index_sha256"] == sha(index_path)
    write_json(HERE / "postpromotion_chain_readback_rev39.json", chain)
    print(json.dumps(chain, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
