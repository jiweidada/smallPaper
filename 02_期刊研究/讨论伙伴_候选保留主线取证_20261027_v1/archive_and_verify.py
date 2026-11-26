"""Human-readable discussion and final source integrity audit; no experimental execution."""
import hashlib
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
def load(path):
    return json.loads(path.read_text("utf-8-sig"))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

records = load(RUN / "sources_and_reading.json")["records"]
extension = load(RUN / "sources_extension_after_H1.json")
records += extension["records"]
assert len(records) == 14
checks = []
for row in records:
    p = Path(row["pdf"])
    valid = sha(p) == row["sha256"]
    assert valid
    assert row["structural_preflight"] == "PASS"
    assert all(1 <= n <= row["pages"] for n in row["actual_text_pages_read"])
    checks.append({"kind": "PDF", "slug": row["slug"], "path": str(p), "sha256": row["sha256"], "unchanged": valid})
bindings = load(RUN / "local_code_sources.json") + load(RUN / "partner_analysis_sources.json") + extension["parent_prefix_sources"]
bindings += load(RUN / "current_mechanism_status.json")["protocol_and_results"]
for row in bindings:
    p = Path(row["path"])
    assert sha(p) == row["sha256"]
    checks.append({"kind": "code_or_partner_record", **row, "unchanged": True})

discussion = load(RUN / "discussion_record_current.json")
lines = ["两对话研究讨论原文节录与当前共识", "2026-10-27；本轮进行中快照。只保留用户可见文本及已发送消息，不含内部推理、命令输出或评分。", "",
         "直接人类授权已核：对方用户消息01a17c46-913a-7e12-9d12-d6fddabc1163，要求继续寻找主线、逐条反证。", "",
         "当前实质进展：cap必要性与类别前缀硬预算假说关闭；同池贪心基数差不能解释C−B主要增量；单节点关键候选的补回和误删均有GT图证书。", "下一假说分Hpre额外信息与Hswap同预算作用两个关口；pre代理尚未在本对话执行。详情见03与claim_to_source_v2.csv。", ""]
for thread in discussion["threads"]:
    lines.extend(["=" * 48, thread["title"], "threadId=" + thread["threadId"], ""])
    for turn in thread["turns"]:
        lines.append("turnId=" + turn["turnId"])
        for item in turn["items"]:
            label = {"direct_user_message": "用户原文", "visible_assistant_message": "助手可见消息", "sent_thread_message": "发往另一对话的原文"}[item["kind"]]
            lines.extend(["", "[" + label + "] " + item["id"], item.get("text") or ""])
        lines.append("")
with (RUN / "04_两对话实际讨论记录.txt").open("x", encoding="utf-8") as stream:
    stream.write("\n".join(lines) + "\n")
audit = {"status": "PASS", "date": "2026-10-27", "source_checks": checks,
         "PDF_count": 14, "distinct_detection_papers": 11, "detection_pdf_versions": 13,
         "mathematical_background_PDFs": 1,
         "discussion": {"threads": len(discussion["threads"]), "visible_or_sent_items": sum(len(t["items"]) for s in discussion["threads"] for t in s["turns"]), "scope": "current ongoing round snapshot"},
         "new_experiments_by_this_chat": {"prediction_or_checkpoint_loads": 0, "model_forwards": 0, "head_calls": 0, "evaluator_calls": 0, "training": 0},
         "partner_work_scope": "Partner's cap probe used8CPU evaluator calls; capacity/prefix/critical graph checks used0. These are explicitly attributed, not this-chat execution.",
         "provenance_limits": ["Structural PDF PASS is not full-paper reading; page scopes are recorded.", "Parent graph checks are read and source-bound, not independently re-executed.", "S2 rate limits and wrong tentative IDs are retained with explicit publisher fallbacks/errata."]}
with (RUN / "final_source_audit.json").open("x", encoding="utf-8") as stream:
    json.dump(audit, stream, ensure_ascii=False, indent=2)
print(f"PASS:14 PDF sources and{len(bindings)} code/result bindings unchanged; visible discussion saved;0 experimental calls")
