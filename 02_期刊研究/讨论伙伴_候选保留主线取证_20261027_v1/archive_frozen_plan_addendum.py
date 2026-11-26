"""Append verified plan and denominator completion; preserve the previous delivery unchanged."""
import datetime
import hashlib
import json
from pathlib import Path
RUN = Path(__file__).resolve().parent
P = RUN.parent / "候选保留_机制与创新假说_20261027_v1"
PLAN = P / "Hlineage_下一最小零训练检验_冻结计划.json"
SUPPORT = P / "class_history_information/support_denominators_v1/results.json"
EXPECTED = "62f105a7fd24394a1034798cff261fc5905d0aded4da349b03f660fbc35a6e04"
def read(p):
    return json.loads(p.read_text("utf-8-sig"))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def bind(p):
    return {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}
def save(name, value):
    with (RUN / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)

before = {str(p): sha(p) for p in (PLAN, SUPPORT, RUN / "class_history_status.json", RUN / "delivery_manifest.json")}
assert before[str(PLAN)] == EXPECTED
plan, support = read(PLAN), read(SUPPORT)
assert plan["status"] == "FROZEN_PLAN_NOT_EXECUTED"
assert plan["execution"]["executed"] is False and plan["execution"]["new_code_implemented"] is False
assert support["status"] == "PASS"
base = read(RUN / "delivery_manifest.json")
base_entries = {r["relative_path"]: r["sha256"] for r in base["files"]}
assert base_entries["class_history_status.json"] == sha(RUN / "class_history_status.json")
prior = read(RUN / "class_history_status.json")
for name, row in support["conditions"].items():
    old = prior["partner_snapshot"]["conditions"][name]["domains"]["B_minus_A"]["M2"]
    assert row["full_images"] == row["images_with_at_least_one_supported_candidate"] == 548
    assert row["full_candidate_pairs"] == 164400
    assert row["supported_candidate_pairs"] == old["overlap_candidates"]
    assert row["supported_strata"] == old["overlap_strata"]
    assert row["weighted_RD_exactly_reproduced"] == old["risk_difference"]

save("frozen_plan_addendum.json", {
    "date": "2026-10-27", "bindings": [bind(p) for p in (PLAN, SUPPORT)],
    "plan_snapshot": plan, "support_snapshot": support,
    "status_change": "Hlineage now has a source-bound frozen plan; remains unimplemented/unexecuted. Supersedes only the earlier not-yet-fully-frozen plan status in09 and claim_to_source_v3.",
    "unchanged_conclusions": "Hpre constant-zero failure; Hclass finite qualification association and negative criticality reference; failed one-swap rule; original4x4 allGT/fixed raw-A populations.",
    "operations_by_this_chat": {"model_forwards": 0, "prediction_cache_loads": 0, "evaluator_calls": 0, "training": 0}
})
lines = [
    "冻结计划与分母补全：上轮交付后的追加记录",
    "2026-10-27。09、claim_to_source_v3及原delivery_manifest按历史状态保留；本文件只更新Hlineage计划冻结状态并补全图像分母。",
    "",
    "一、Hclass分母补全已经实际读取PASS",
    "raw seed1/2可比候选60115/49210；EMA seed1/2为39716/40316，各完整域164400。四条件的可比候选都覆盖548/548图，但仍不是全部候选可比。支持格数2192/2350/1573/1482，原M2风险差逐值精确复现，无新分档或新检验。",
    "",
    "二、下一条计划已冻结，未实现、未执行",
    "文件：" + str(PLAN),
    "SHA256：" + EXPECTED,
    "实际JSON状态FROZEN_PLAN_NOT_EXECUTED，executed=false，new_code_implemented=false。本次只读静态计划及旧结果，没有运行它。",
    "",
    "冻结要点：先按原血缘构造两个邻域，再置换支持身份连线；anchor固定，保留原q排除。主强置换限final类别×4×4空间×maxscore(.1/.5)块内，弱全局只诊断；另与类/IoU几何最优对应比较。own-query固定mean/max分类融合控制保持不变。",
    "删点与加点分别以实际A cap后池P为基准，测试类A导出数≤99；两侧主分析均限GT-qualified。加点r(P+f)−r(P)仍是额外节点的图松界，固定300联合交换和AP是独立后续关口。",
    "真实对应须在两个seed及raw/EMA、补入与删除两域均超过主强置换95分位及几何参照；未过关停止这一版。稀疏/常数/不可比的情况必须如实保留。上述是计划判据，没有新结果或新意成立结论。",
    "",
    "三、前述结论保持",
    "Hpre恒0退化和单次交换失败保留；Hclass资格正差与关键性负差不混称。原raw×EMA4×4、C−B主/B−A参照、全GT后固定raw A阶段均不变。",
    "",
    "源文件和对话追加原文：frozen_plan_addendum.json、frozen_plan_message_addendum.json。只追加新文件，原327项交付清单未改写。"
]
with (RUN / "11_冻结计划与分母补全_续议追加记录.txt").open("x", encoding="utf-8") as stream:
    stream.write("\n".join(lines) + "\n")
assert all(sha(Path(path)) == expected for path, expected in before.items())
save("final_source_audit_delta.json", {
    "status": "PASS", "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "checks": [{"path": p, "sha256": h, "unchanged_after_reading": True} for p, h in before.items()],
    "all4_RD_exactly_reproduced_from_prior_record": True,
    "all4_supported_image_denominators548": True,
    "plan_implemented": False, "plan_executed": False,
    "this_chat_model_forwards": 0, "this_chat_evaluator_calls": 0, "this_chat_training": 0
})
names = ["frozen_plan_message_addendum.json", "archive_frozen_plan_addendum.py", "frozen_plan_addendum.json", "11_冻结计划与分母补全_续议追加记录.txt", "final_source_audit_delta.json"]
save("delivery_manifest_v2_addendum.json", {
    "date": "2026-10-27", "base_manifest": bind(RUN / "delivery_manifest.json"),
    "base_manifest_preserved": True, "latest_addendum_entry": "11_冻结计划与分母补全_续议追加记录.txt",
    "new_files": [bind(RUN / n) for n in names]
})
print("PASS: frozen-plan SHA exact;4 supported-image denominators completed; prior results and delivery manifest unchanged;0 experimental calls")
