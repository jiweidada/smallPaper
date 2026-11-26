"""Archive visible scope discussion and verify sealed document bindings only."""
import datetime
import hashlib
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(n):
    return json.loads((RUN / n).read_text(encoding="utf-8"))

def binding(p):
    return {"path": str(p), "sha256": digest(p), "bytes": p.stat().st_size}

review = read("scope_correction_review.json")
record = read("discussion_record_scope_correction_final.json")
source_bindings = []
for item in review["sources"]:
    p = Path(item["path"])
    assert digest(p) == item["sha256"], str(p)
    source_bindings.append(binding(p))

manifest_names = [
    "delivery_manifest.json",
    "delivery_manifest_v2_addendum.json",
    "delivery_manifest_Hlineage_execution.json",
]
prior_before = {n: digest(RUN / n) for n in manifest_names}
old_addendum = read(manifest_names[1])
base_ref = old_addendum["base_manifest"]
assert digest(Path(base_ref["path"])) == base_ref["sha256"]
old_execution = read(manifest_names[2])
for row in old_execution["base_deliveries_preserved"]:
    assert digest(Path(row["path"])) == row["sha256"], row["path"]
for row in old_execution["new_files"]:
    assert digest(Path(row["path"])) == row["sha256"], row["path"]

allowed = {"direct_user_message", "visible_assistant_message", "sent_thread_message"}
visible_count = 0
sent_count = 0
direct_human_count = 0
lines = [
    "原两问题范围勘误：双方实际可见讨论原文",
    "2026-10-27。此为交付时可见内容快照，按对话及回合归组；不推断跨对话逐秒顺序。",
    "来源为read_thread返回的用户正文、助手可见正文、send_message_to_thread实际提示正文。",
    "不含内部推理或命令输出；当前本对话仍在进行，末尾归档通知及最终答复不在此快照中。",
    "范围勘误正文见15号；旧实验结果与旧封存保持原样。",
    "",
]
counts = []
for thread in record["threads"]:
    local_count = 0
    lines += [f"对话：{thread['title']}", f"threadId：{thread['threadId']}；hostId：{thread['hostId']}", ""]
    for turn in reversed(thread["turns"]):
        lines += [f"回合：{turn['turnId']}；快照状态：{turn['status']}", ""]
        for item in turn["items"]:
            assert item["kind"] in allowed
            assert isinstance(item["text"], str) and item["text"]
            visible_count += 1
            local_count += 1
            if item["kind"] == "sent_thread_message":
                sent_count += 1
                label = f"已发送对话消息；目标={item['targetThreadId']}；工具状态={item.get('status')}"
            elif item["kind"] == "direct_user_message":
                direct_human_count += 1
                label = "人类用户正文"
            else:
                label = "助手可见正文；phase=" + str(item.get("phase"))
            lines += [f"[{visible_count}] {label}", f"消息标识：{item['id']}", item["text"].rstrip(), ""]
    counts.append({"threadId": thread["threadId"], "visible_items": local_count})
assert len(record["threads"]) == 2
assert sent_count == 7
assert direct_human_count == 1
assert visible_count == 16
with (RUN / "16_原两问题范围勘误_双方讨论原文.txt").open("x", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

for n, h in prior_before.items():
    assert digest(RUN / n) == h
for row in source_bindings:
    assert digest(Path(row["path"])) == row["sha256"]

checks = {
    "status": "PASS",
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": "Document-only scope erratum and actual visible discussion archive",
    "stable_source_documents_verified": len(source_bindings),
    "source_bindings_unchanged": source_bindings,
    "old_base_manifest_references_verified": True,
    "old_Hlineage_delivery_files_verified": len(old_execution["new_files"]),
    "old_delivery_manifests_preserved_during_archive": True,
    "visible_discussion_items": visible_count,
    "visible_discussion_per_thread": counts,
    "actual_sent_thread_messages": sent_count,
    "direct_human_scope_question_archived": True,
    "private_reasoning_and_command_outputs_excluded_by_allowlist": True,
    "interpretation": "15 and CURRENT_SCOPE narrow the overbroad no-mainline wording in13 to the tested BASE24 final-candidate branch. Existing A/B/C,4x4 and later branch results remain sealed. Original early-problem causes and effective intervention remain unresolved.",
    "operations_this_scope_review": review["operations"],
}
with (RUN / "scope_correction_delivery_checks.json").open("x", encoding="utf-8") as f:
    json.dump(checks, f, ensure_ascii=False, indent=2)
new_names = [
    "15_范围勘误_原两问题与最终候选支线.txt",
    "CURRENT_SCOPE_范围勘误入口.txt",
    "archive_scope_correction.py",
    "scope_correction_review.json",
    "discussion_record_scope_correction.json",
    "scope_correction_partner_actual_turn.json",
    "discussion_record_scope_correction_final.json",
    "16_原两问题范围勘误_双方讨论原文.txt",
    "finalize_scope_correction_archive.py",
    "scope_correction_delivery_checks.json",
]
manifest = {
    "date": "2026-10-27",
    "status": "SCOPE_CORRECTION_DELIVERY_COMPLETE",
    "latest_entry": "CURRENT_SCOPE_范围勘误入口.txt",
    "scientific_scope_erratum": "15_范围勘误_原两问题与最终候选支线.txt",
    "actual_discussion": "16_原两问题范围勘误_双方讨论原文.txt",
    "prior_deliveries_preserved": [binding(RUN / n) for n in manifest_names],
    "new_files": [binding(RUN / n) for n in new_names],
    "interpretation_change_only": True,
    "old_results_overwritten": False,
}
with (RUN / "delivery_manifest_scope_correction.json").open("x", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print(f"PASS: {len(source_bindings)} stable sources unchanged; {len(old_execution['new_files'])} prior delivery files unchanged; {visible_count} visible items / {sent_count} actual sent messages archived; 0 experiments or literature searches.")

