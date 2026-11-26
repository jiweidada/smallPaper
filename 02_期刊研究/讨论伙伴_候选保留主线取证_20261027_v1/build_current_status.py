"""Record partner's frozen results and arithmetic contrasts; never load predictions or evaluate."""
import csv
import hashlib
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
PARTNER = RUN.parent / "候选保留_机制与创新假说_20261027_v1"
SUB = PARTNER / "matching_and_witness"

def read(path):
    return json.loads(path.read_text("utf-8-sig"))

def receipt(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

result = read(SUB / "attempt01/results.json")
verification = read(SUB / "attempt01/verification.json")
assert result["status"] == "COMPLETE" and verification["status"] == "PASS"
assert receipt(SUB / "protocol.json")["sha256"] == result["protocol_sha256"]
rows = []
for state in ("raw", "EMA"):
    for seed in (1, 2):
        for th in (50, 75):
            for hi, lo in (("C", "B"), ("B", "A")):
                a = result["conditions"][f"{state}_s{seed}_{th}_{hi}"]
                b = result["conditions"][f"{state}_s{seed}_{th}_{lo}"]
                assert a["GT"] == b["GT"] == 38759 and a["images"] == b["images"] == 548
                assert a["full_pool_bound"] == b["full_pool_bound"]
                actual = a["actual_TP"] - b["actual_TP"]
                capacity = a["same_cap_pool_max"] - b["same_cap_pool_max"]
                gap_change = a["greedy_cardinality_gap"] - b["greedy_cardinality_gap"]
                assert actual == capacity - gap_change
                rows.append({"state": state, "seed": seed, "IoU": th / 100,
                             "contrast": hi + "-" + lo, "GT": 38759, "images": 548,
                             "actual_TP_delta": actual, "same_cap_pool_max_delta": capacity,
                             "greedy_gap_change": gap_change,
                             "critical_missing_cap_delta": a["critical_missing_cap"] - b["critical_missing_cap"],
                             "interpretation": "Exact arithmetic identity of partner aggregates; not causal shares or AP."})

out = {"date": "2026-10-27", "role": "Read partner results and compute descriptive arithmetic only; no prediction data loaded.",
       "protocol_and_results": [receipt(SUB / name) for name in
                                ("protocol.json", "run.py", "attempt01/results.json", "attempt01/verification.json")],
       "partner_result_snapshot": result, "partner_verification_snapshot": verification,
       "contrasts": rows,
       "code_lines_read": "matching_and_witness/run.py:67-89 for critical-node definition and witness reference; not a claim of full independent reimplementation.",
       "limits": ["Full-graph criticality is rank loss after deleting a candidate, not a unique GT identity or a deployable label.",
                  "Raw matched-pair witness records exist; no EMA matched-pair records are invented.",
                  "Single-node and collective bottlenecks are different; residual is not allocated to a new cause.",
                  "Original all-GT and fixed-raw-A response populations remain unchanged; these are additional descriptive conditions, not replacement cohorts."],
       "operations_by_this_chat": {"prediction_cache_loads_for_this_update": 0, "evaluator_calls": 0,
                                   "model_forwards": 0, "head_calls": 0, "training": 0}}
with (RUN / "current_mechanism_status.json").open("x", encoding="utf-8") as stream:
    json.dump(out, stream, ensure_ascii=False, indent=2)
with (RUN / "mechanism_contrasts.tsv").open("x", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

with (RUN / "claim_to_source.csv").open(encoding="utf-8-sig", newline="") as stream:
    claims = list(csv.DictReader(stream))
for claim in claims:
    if claim["claim_id"] == "prefix_budget_gap":
        claim.update(evidence_kind="partner_constructive_certificate", sources="sources_extension_after_H1.json",
                     anchors="class_prefix_bound all4384 cases", operation_scope="fixed eligible universe / optimistic prefix-cap matching relaxation",
                     supports="全部案例构造达到Mfree且完整前缀闭包≤300，关闭此硬不可达原因",
                     does_not_support="不证明实际分数贪心已最优，也不约束改变原分数/资格的整套方法", status="REFUTED_SCOPED_HYPOTHESIS")
    if claim["claim_id"] == "scarce_witness":
        claim.update(evidence_kind="superseded_hypothesis", sources="02_H1证伪后_匹配见证层级与新增近邻.txt;current_mechanism_status.json",
                     anchors="H2a / H2b1 / H2b2 split before new results", operation_scope="separate pool matching, output selection, individual and collective criticality",
                     supports="旧宽H2已拆分，必须分别解释和反证", does_not_support="一层阴性不能连同另一层关闭，也不能事后换名", status="SUPERSEDED_BY_SPLIT")
def add(cid, kind, sources, anchors, scope, supports, limits, status):
    claims.append(dict(zip(claims[0], (cid, kind, sources, anchors, scope, supports, limits, status))))
add("H2a_CminusB_matching", "partner_executed_bound", "current_mechanism_status.json", "all24 conditions /16 contrasts", "same exact native cap-retained pair pool",
    "同池贪心基数差存在但C-B变化很小；不支持它作为C-B主要来源", "基数界不是AP排序界，也不否认组替代问题", "POSITIVE_SMALL_GAP_LIMITED_EXPLANATION")
add("H2b1_individual_critical", "partner_executed_deletion_checks", "current_mechanism_status.json", "320812 checks;raw actual pair witnesses", "full fixed GT graph with original eligibility and cap bound",
    "两seed新增与丢失均有删除会降低整图容量的实际匹配pair", "不推出GT身份唯一性、独立病因份额或无GT可辨识", "POSITIVE_MECHANISM_CERTIFICATES")
add("H2b2_collective", "retained_alternative", "02_H1证伪后_匹配见证层级与新增近邻.txt", "single versus collective counterexample", "jointly removed node groups",
    "个体关键以外仍有容量残差", "尚未分配组原因，不从残差推断独立因果", "NOT_CLOSED_NOT_ISOLATED")
add("LDDP_direct_neighbour", "primary_method", "lddp", "PDF3-5 Algorithm2", "multiclass output-count-limited global DPP selection",
    "多类质量/空间/语义多样性与限制输出数的宽泛集合选择已有先例", "没有复现该方法或证明与拟议见证关系等效", "VERIFIED")
add("QUBO_coverage_neighbour", "primary_method", "qubo_pedestrian", "PDF2-4", "confidence/overlap global subset and foreground coverage",
    "高分框妨碍多个目标与全局覆盖不是首次提出", "前景覆盖不等同GT匹配图，但差异不自动构成方法空白", "VERIFIED")
add("matching_rank_classical", "mathematical_background", "matroid_notes", "PDF2-4,6-7", "transversal matroid and rank",
    "匹配秩/次模等数学是经典背景", "不证明已有检测规则达到对应实际AP", "CLASSICAL_NOT_NEW_THEORY")
add("Hpre_information", "unexecuted_falsifiable_hypothesis", "03_当前主线与pre见证反证意见.txt", "Hpre", "saved pre units / final-to-pre relations",
    "拟检验预测关系对真实容量损伤是否具有跨seed/状态额外辨识", "代理字段/单位/参数尚未冻结，未运行；跨层本身已有近邻", "PROPOSED_INFORMATION_GATE")
add("Hswap_same_budget", "unexecuted_falsifiable_hypothesis", "03_当前主线与pre见证反证意见.txt", "Hswap", "same actual budget and unchanged evaluation semantics",
    "拟检验信息能否减少补回一个关键候选同时删掉另一个的失败交换", "Hpre阳性不保证此关阳性或AP收益", "PROPOSED_SEPARATE_EFFECT_GATE")
with (RUN / "claim_to_source_v2.csv").open("x", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(claims[0]))
    writer.writeheader()
    writer.writerows(claims)
print("PASS: partner receipts,16 arithmetic contrasts and current claim states saved;0 prediction/evaluation calls")
