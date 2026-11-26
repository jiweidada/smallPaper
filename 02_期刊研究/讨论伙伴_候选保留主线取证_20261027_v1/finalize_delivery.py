"""Finalize visible discussion, current claims, source integrity and artifact inventory."""
import csv
import datetime
import hashlib
import json
from pathlib import Path
RUN = Path(__file__).resolve().parent
def read(p):
    return json.loads(p.read_text("utf-8-sig"))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name, obj):
    with (RUN / name).open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)

pdfs = read(RUN / "sources_and_reading.json")["records"]
pdfs += read(RUN / "sources_extension_after_H1.json")["records"]
pdfs += read(RUN / "sources_extension_latest.json")["local_PDF_records"]
assert len(pdfs) == 16
checks = []
for r in pdfs:
    p = Path(r["pdf"])
    assert sha(p) == r["sha256"] and r["structural_preflight"] == "PASS"
    assert all(1 <= n <= r["pages"] for n in r["actual_text_pages_read"])
    checks.append({"kind": "local_PDF", "path": str(p), "sha256": r["sha256"], "unchanged": True})
bindings = read(RUN / "local_code_sources.json") + read(RUN / "partner_analysis_sources.json")
bindings += read(RUN / "sources_extension_after_H1.json")["parent_prefix_sources"]
bindings += read(RUN / "current_mechanism_status.json")["protocol_and_results"]
bindings += read(RUN / "cache_producer_binding_addendum.json")["files"]
bindings += read(RUN / "sources_extension_latest.json")["partner_bindings"]
bindings += read(RUN / "class_history_status.json")["bindings"]
bindings += read(RUN / "swap_status.json")["bindings"]
unique = {}
for r in bindings:
    path = r["path"]
    if path in unique:
        assert unique[path] == r["sha256"]
    unique[path] = r["sha256"]
for path, expected in unique.items():
    assert sha(Path(path)) == expected
    checks.append({"kind": "code_or_partner_record", "path": path, "sha256": expected, "unchanged": True})
prior = RUN.parent / "raw_EMA逐GT对照_20261026_v1/跨状态逐GT响应4x4_20261026_v1/protocol.json"
expected = "a1ea55445363e9db94d4902b4afd434c07d82c568f8099f86150ded2e766a275"
assert sha(prior) == expected
checks.append({"kind": "original_response_protocol", "path": str(prior), "sha256": expected, "unchanged": True})

with (RUN / "claim_to_source_v2.csv").open(encoding="utf-8-sig", newline="") as stream:
    claims = list(csv.DictReader(stream))
for r in claims:
    if r["claim_id"] == "Hpre_information":
        r.update(evidence_kind="partner_frozen_probe", sources="sources_extension_latest.json;06_Hpre关闭与类别身份假说_新增原文边界.txt",
                 anchors="2192/2192 images saturated;proxy-critical0", operation_scope="this fixed pre-NMS/association/truncated single-deletion proxy",
                 supports="代理退化恒0，按冻结关口停止", does_not_support="不证明pre所有信息无用，空层没有负显著区间", status="STOP_THIS_FROZEN_PROXY")
    if r["claim_id"] == "Hswap_same_budget":
        r.update(evidence_kind="partner_frozen_intervention", sources="swap_status.json;09_本轮最新结论与下一条双向信息命题.txt",
                 anchors="all4conditions;12CPUevaluations", operation_scope="one same-class same-score-band swap with same-deletion score reference",
                 supports="四条件AP/APs负，预定推进关失败，关闭该规则", does_not_support="不撤销有限资格信息，不否定所有语义证据", status="STOP_THIS_FIXED_SWAP")
def add(cid, kind, sources, anchors, scope, support, limits, status):
    claims.append(dict(zip(claims[0], (cid, kind, sources, anchors, scope, support, limits, status))))
add("Hclass_qualification", "partner_frozen_information_test", "class_history_status.json;07_类别身份信息阳性及关键性反证_交换前.txt", "B-minus-A M2 comparable strata",
    "164400pairs per condition with60115/49210/39716/40316 comparable pairs", "两seed/rawEMA有有限资格信息", "不是全域因果、TP/AP或完整近邻胜出", "POSITIVE_BOUNDED_INFORMATION")
add("Hclass_criticality", "partner_predeclared_secondary", "class_history_status.json", "GT-qualified B-union-C M2", "different predefined domain/outcome",
    "关键性风险差四点估计负，不支持保护解释", "不能与主资格效应相减，部分区间含0", "NOT_A_KEY_WITNESS_PROTECTOR")
add("ordinary_class_ensemble", "primary_precedent_and_inference", "source_ensemble_extension.json;08_简单跨层分类集成的反解释与原文范围.txt", "Cascade R-CNN Table2;DETRDistill4.1,6",
    "official indexed excerpts;not complete local PDFs", "多阶段类别平均/逐阶段类别传递已有先例", "未证明本例固定线性平均与binary predictor同效果", "COMPETING_EXPLANATION_NOT_EXCLUDED")
add("Hlineage_dual_information", "discussion_plan_only", "09_本轮最新结论与下一条双向信息命题.txt", "persistent neighbour intersection and identity-permutation countercontrol",
    "unchanged two-stage marginal observations;addition and deletion separate", "给出重复簇高J/孤立关键低J反例及置换和标签修正", "尚未冻结完可执行参数，未运行；不声称新方法成立", "UNEXECUTED_FALSIFIABLE_PLAN")
with (RUN / "claim_to_source_v3.csv").open("x", encoding="utf-8", newline="") as stream:
    w = csv.DictWriter(stream, fieldnames=list(claims[0]))
    w.writeheader(); w.writerows(claims)

discussion = read(RUN / "discussion_record_final.json")
lines = ["两对话本轮续议实际原文（交付快照）", "2026-10-27。按每个对话自己的消息顺序保存，不伪造统一时间轴。只含用户可见原文及实际发送的对话消息，不含内部推理或命令输出。", "直接人类授权：01a17c46-913a-7e12-9d12-d6fddabc1163，已实际读取。", ""]
count = 0
for s in discussion["threads"]:
    lines += ["=" * 48, s["title"], "threadId=" + s["threadId"], ""]
    for t in s["turns"]:
        lines.append("turnId=" + t["turnId"])
        for item in t["items"]:
            count += 1
            kind = {"direct_user_message": "用户原文", "visible_assistant_message": "助手可见消息", "sent_thread_message": "已发往另一对话的原文"}[item["kind"]]
            lines += ["", "[" + kind + "] " + item["id"], item.get("text") or ""]
        lines.append("")
with (RUN / "10_两对话本轮续议实际原文.txt").open("x", encoding="utf-8") as stream:
    stream.write("\n".join(lines) + "\n")
audit = {"status": "PASS", "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
         "local_PDFs": 16, "local_detection_PDF_versions": 15, "local_distinct_detection_papers": 13, "mathematical_background_PDFs": 1,
         "online_method_records_with_explicit_partial_reading": 3, "online_abstract_only_leads": 1,
         "unchanged_source_checks": checks, "unique_code_and_result_bindings": len(unique), "original4x4_protocol_unchanged": True,
         "visible_discussion_items": count, "discussion_scope": "two-chat current-round delivery snapshot;partner may still be writing its own final record",
         "this_chat_operations": {"prediction_or_checkpoint_loads": 0, "model_forwards": 0, "head_calls": 0, "evaluator_calls": 0, "training": 0},
         "partner_operations_attributed": {"cap_probe_CPU_evaluations": 8, "single_swap_CPU_evaluations": 12, "other_graph_and_information_probes_evaluations": 0, "model_forwards": 0, "training": 0},
         "scope_limits": ["Local PDF structural PASS is separate from actual reading pages.", "Online indexed excerpts are not complete PDF reading and have no invented PDF SHA.", "Partner protocols/results verified by source hashes and read;not independently re-executed.", "Hpre and swap failures retained;positive qualification and negative criticality retained separately.", "Hlineage is an unexecuted plan, not a proven innovation."]}
save("final_source_audit_v2.json", audit)
files = []
for p in sorted(RUN.rglob("*")):
    if p.is_file() and "__pycache__" not in p.parts and p.name != "delivery_manifest.json":
        files.append({"relative_path": str(p.relative_to(RUN)), "bytes": p.stat().st_size, "sha256": sha(p)})
save("delivery_manifest.json", {"date": "2026-10-27", "entry_point": "09_本轮最新结论与下一条双向信息命题.txt",
                               "discussion": "10_两对话本轮续议实际原文.txt", "current_claim_matrix": "claim_to_source_v3.csv",
                               "source_audit": "final_source_audit_v2.json", "files": files})
print(f"PASS:{len(checks)} source bindings unchanged;{count} visible discussion items;{len(files)} artifact hashes; no new forward/evaluation/training by this chat")
