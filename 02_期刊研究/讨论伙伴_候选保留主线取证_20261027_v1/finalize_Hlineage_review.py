"""Finalize source-bound scientific review; no prediction arrays or statistics executed."""
import datetime
import hashlib
import json
from pathlib import Path
RUN = Path(__file__).resolve().parent
P = RUN.parent / "Hlineage执行_20261027_v1"
def read(p):
    return json.loads(p.read_text("utf-8-sig"))
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def bind(p):
    return {"path": str(p), "sha256": digest(p), "bytes": p.stat().st_size}
def save(name, obj):
    with (RUN / name).open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)

old = read(RUN / "Hlineage_completed_result_review.json")
for b in old["bindings"]:
    assert digest(Path(b["path"])) == b["sha256"]
r = read(P / "attempt01/results.json")
v = read(P / "attempt01/verification.json")
a = read(P / "audit_v1/results.json")
parity = read(P / "argmax_parity_v1/results.json")
assert r["status"] == "COMPLETE" and v["status"] == a["status"] == parity["status"] == "PASS"
assert r["protocol_sha256"] == digest(P / "protocol.json") == "47fe05725470eaa6d69e3fa2df171ca93799b4aadd9162652ba3d42370f66623"
assert len(r["conditions"]) == len(a["conditions"]) == len(parity["conditions"]) == 8
assert r["strict_information_advance_gate_pass"] is False
assert a["all_CMI_exactly_recomputed_within_1e_12"] is True
for c in parity["conditions"]:
    assert c["images"] == 548 and c["queries"] == 164400
    assert c["argmax_mismatches"] == c["logit_top_ties"] == c["sigmoid_top_ties"] == 0
for name, c in r["conditions"].items():
    m = c["primary_qualified"]
    assert m["advance_gate_pass"] is False
    assert len(m["strong_null_CMI_bits_199"]) == len(m["weak_null_CMI_bits_199"]) == 199
    assert m["real_CMI_bits"] > m["weak_null95_bits"]
    assert a["conditions"][name]["real_exceeds_strong_null_min"] is False
    assert a["conditions"][name]["geometry_reaches_exact_entropy_ceiling_integer_certificate"] is False
    assert a["conditions"][name]["real_geo_joint_tables_identical"] is False
paths = [P / n for n in ("protocol.json", "run.py", "attempt01/results.json", "attempt01/verification.json", "audit_saved_counts.py", "audit_v1/results.json", "audit_v1/integer_and_entropy_audit.csv", "check_argmax_parity.py", "argmax_parity_v1/results.json", "argmax_parity_v1/per_condition.csv")]
before = {str(p): digest(p) for p in paths}
review = {
    "date": "2026-10-27", "scientific_conclusion": "No innovation mainline validated by the present round's mechanism, actual-action and nearest-method evidence.",
    "Hlineage_status": "DOES_NOT_ADVANCE_0_OF_8_FROZEN_GATES; not an absence-of-all-lineage-information theorem",
    "supersedes": "Only pending execution/audit status in09,11,claim_to_source_v3 and first Hlineage review; older records remain historical.",
    "bindings": [bind(p) for p in paths],
    "final_integer_audit": {k: val for k, val in a.items() if k != "input_sha256"},
    "final_argmax_audit": {k: val for k, val in parity.items() if k != "input_sha256"},
    "scientific_limits": ["Deletion-side support is feasible in all4conditions, so failure is not all attributable to sample shortage.",
                          "Raw1/raw2/EMA2 insertion domains fail the frozen feasibility threshold; EMA1 insertion is feasible but fails comparisons.",
                          "All8 real values exceed weak95 and lie below all199 strong values; weak-null positives cannot replace the frozen primary reference.",
                          "No global geometry-entropy ceiling or identical contingency-table certificate was found; this is actual-comparison failure, not universal mathematical impossibility.",
                          "Finite plugin-CMI occupancy bias, coarse controls and geometric continuity remain limitations. Strong-reference percentile is not a calibrated P value.",
                          "Near equality of real/geometry J bins is a measured feature observation, not exact equivalence of query correspondences or complete methods.",
                          "Original candidate-retention effects and finite Hclass qualification information remain; failed readout and failed information definitions are separate."],
    "operations_by_this_chat": {"prediction_cache_loads": 0, "statistical_recomputations": 0, "model_forwards": 0, "evaluator_calls": 0, "training": 0}
}
save("Hlineage_final_scientific_review.json", review)

lines = [
    "本轮实测后最新结论：没有找到通过验证的创新主线",
    "2026-10-27。上轮停在计划处过早，并无授权/资源阻碍；本轮已实际完成执行、双方结果讨论及最终审计。09/11及旧状态矩阵保留为历史，当前以本文件和Hlineage_final_scientific_review.json为准。",
    "",
    "一、Hlineage已经执行完，不再是未执行计划",
    "执行协议SHA47fe05725470eaa6d69e3fa2df171ca93799b4aadd9162652ba3d42370f66623；父计划SHA62f105a7fd24394a1034798cff261fc5905d0aded4da349b03f660fbc35a6e04不变。",
    "对方results COMPLETE、verification PASS；实际196.25秒。2192缓存绑定，2192原A cap后图容量逐图精确重现，2206输入未变；0新前向/评价/训练。此PASS指运行和完整性核验，研究推进门槛为0/8。",
    "本对话实际读全部8域及静态代码，不加载预测数组或重算统计。",
    "",
    "二、八域的完整分母与比较（CMI单位bits）",
    "条件 / 方向 | qualified候选/图像 | informative候选/图像 | real / geo / strong95 | 可行性关 / 推进关"
]
for name, c in r["conditions"].items():
    m = c["primary_qualified"]
    lines.append(f"{name} | {m['qualified_pairs']}/{m['images_with_qualified_pairs']} | {m['informative_pairs']}/{m['informative_images']} | {m['real_CMI_bits']:.8f} / {m['geometry_CMI_bits']:.8f} / {m['strong_null95_bits']:.8f} | {m['information_feasible']} / {m['advance_gate_pass']}")
lines += [
    "",
    "四组删除侧均有数千个informative候选、490/516/511/520张支持图，不能把失败全归于样本不足。补入侧raw1/raw2/EMA2分别48/20/81个informative候选，未过预定100候选可行性关；raw1/2正标签仅9/7，raw2支持图仅19。EMA1补入117候选/90图可行，仍未过信息比较。",
    "8域真实CMI均超过弱全局置换95分位，却都低于全部199个强置换样本；不能用弱对照正号替代预定强门槛。raw1/2补入及EMA2删除还未胜几何参照。",
    "",
    "三、两项最终审计已结清",
    "argmax_parity_v1 PASS：2192缓存×pre/final，共1315200次query-stage比较，logit赢家与FP32 sigmoid赢家差异0，最高分并列0。静态检查发现的潜在数值语义偏差在本缓存中不存在。",
    "audit_v1 PASS：8域real/geo CMI及条件标签熵上界复算差≤1e−12。没有任何域达到几何整数熵上界，也没有全等real/geo联合列联表，因此不能把本次失败解释为整个检验数学上不可能。",
    "real与geometry的J分档98.876%—99.488%逐行相同；差异逐域为287/26005、27/3462、355/31583、42/4027、237/32152、26/4115、258/31525、21/4098。这是当前特征的观测，不是query对应或方法完全等价证明。",
    "",
    "四、最终科学限定与结论",
    "关闭/不推进的是这个冻结的双向信息定义。强置换CMI较高不能解释为随机对应具有更好的因果或检测性能；插件CMI的细格占用偏差、粗控制和精细几何延续仍限制推论。强参照95分位不是校准P值；也不能由此证明所有血缘信息为零。",
    "原候选保留问题及原raw×EMA4×4证据保持；Hclass在有限可比格中的资格信息保持；pre容量代理恒0、固定单交换作用失败及Hlineage未过关分别记录，不互相冒充。两个seed的raw/EMA仍是相关保存状态，不是四个seed。",
    "当前证据没有把候选机制、实际作用与文献差异连成可保留的创新主线。双方均认同本轮如实报告未找到主线；没有现有证据支持下一条具体命题，故不另造名称、不加实验或检索。本轮结束的理由是实际证据链未接通，不能再以冻结计划、授权或资源为理由。",
    "",
    "实际讨论见14_Hlineage实测与最终审计_双方讨论原文.txt；完整源快照及SHA见Hlineage_final_scientific_review.json和本轮交付清单。未覆盖既有结果、计划或旧交付。"
]
with (RUN / "13_Hlineage实测与最终科学结论.txt").open("x", encoding="utf-8") as stream:
    stream.write("\n".join(lines) + "\n")
disc = read(RUN / "discussion_record_Hlineage_execution.json")
transcript = ["Hlineage实际执行与最终科学限定：双方可见讨论原文", "2026-10-27，交付快照；按各对话内部顺序保存，不伪造统一时间轴，不含内部推理或命令输出。", ""]
n = 0
for s in disc["threads"]:
    transcript += ["=" * 48, s["title"], "threadId=" + s["threadId"], ""]
    for t in s["turns"]:
        transcript.append("turnId=" + t["turnId"])
        for item in t["items"]:
            n += 1
            transcript += ["", "[" + item["kind"] + "] " + item["id"], item.get("text") or ""]
with (RUN / "14_Hlineage实测与最终审计_双方讨论原文.txt").open("x", encoding="utf-8") as stream:
    stream.write("\n".join(transcript) + "\n")
with (RUN / "LATEST_当前入口.txt").open("x", encoding="utf-8") as stream:
    stream.write("当前结果：13_Hlineage实测与最终科学结论.txt\n实际讨论：14_Hlineage实测与最终审计_双方讨论原文.txt\n源绑定：Hlineage_final_scientific_review.json\n旧09/11及README均为历史阶段入口；Hlineage目前已执行且未过关，不再是待运行计划。\n原raw×EMA4×4及固定raw A人口不变，本轮没有找到通过验证的创新主线。\n")
assert all(digest(Path(p)) == h for p, h in before.items())
save("Hlineage_review_delivery_checks.json", {"status": "PASS", "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "source_bindings_unchanged": [bind(p) for p in paths], "visible_discussion_items": n,
    "scientific_gate_passes": 0, "scientific_gate_total": 8,
    "argmax_protocol_parity_cleared": True, "integer_entropy_audit_cleared": True,
    "this_chat_model_forwards": 0, "this_chat_evaluator_calls": 0, "this_chat_training": 0})
names = ["12_Hlineage执行前可辨识性核查.txt", "read_Hlineage_result_review.py", "Hlineage_completed_result_review.json", "discussion_record_Hlineage_execution.json", "finalize_Hlineage_review.py", "Hlineage_final_scientific_review.json", "13_Hlineage实测与最终科学结论.txt", "14_Hlineage实测与最终审计_双方讨论原文.txt", "LATEST_当前入口.txt", "Hlineage_review_delivery_checks.json"]
save("delivery_manifest_Hlineage_execution.json", {"date": "2026-10-27", "base_deliveries_preserved": [bind(RUN / x) for x in ("delivery_manifest.json", "delivery_manifest_v2_addendum.json")],
    "latest_entry": "13_Hlineage实测与最终科学结论.txt", "new_files": [bind(RUN / x) for x in names]})
print(f"IntegrityPASS;scientific gate0/8;{n} visible discussion records;10 source bindings unchanged; no new experimental calls by this chat")
