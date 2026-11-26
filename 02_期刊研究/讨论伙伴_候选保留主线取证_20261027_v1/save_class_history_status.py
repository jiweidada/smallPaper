"""Read frozen class-history results; preserve finite overlap and outcome distinctions."""
import hashlib
import json
from pathlib import Path
RUN = Path(__file__).resolve().parent
P = RUN.parent / "候选保留_机制与创新假说_20261027_v1/class_history_information"
def read(p):
    return json.loads(p.read_text("utf-8-sig"))
def bind(p):
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
r = read(P / "attempt01/results.json")
v = read(P / "attempt01/verification.json")
assert r["status"] == "COMPLETE" and v["status"] == "PASS"
assert bind(P / "protocol.json")["sha256"] == r["protocol_sha256"]
rows = []
for name, c in r["conditions"].items():
    d = c["domains"]["B_minus_A"]
    m = d["M2"]
    assert d["counts"]["candidate_pairs"] == 164400
    q = c["domains"]["GT_qualified"]["M2"]
    rows.append({"condition": name, "primary_domain": "B_export minus A_export", "primary_outcome": "same-class GT edge50 qualification",
                 "total_pairs": 164400, "overlap_pairs": m["overlap_candidates"], "overlap_predictor1": m["overlap_proxy1"],
                 "overlap_predictor0": m["overlap_proxy0"], "overlap_strata": m["overlap_strata"],
                 "risk_difference": m["risk_difference"], "image_bootstrap95": m["image_bootstrap95"],
                 "secondary_different_domain": "GT-qualified pairs in B-union-C", "secondary_outcome": "critical50 among qualified",
                 "secondary_M2_risk_difference": q["risk_difference"], "secondary_M2_image_bootstrap95": q["image_bootstrap95"]})
out = {"date": "2026-10-27", "bindings": [bind(P / name) for name in ("protocol.json", "attempt01/results.json", "attempt01/verification.json")],
       "partner_snapshot": r, "partner_verification": v, "main_and_secondary_rows": rows,
       "scope": "Positive qualification information in finite comparable coarse strata; no all-pair causal effect, no TP/AP or unique-witness protection claim. The secondary different-domain criticality result is retained, never subtracted from primary effect.",
       "operations_by_this_chat": {"prediction_cache_loads": 0, "model_forwards": 0, "evaluator_calls": 0, "training": 0}}
with (RUN / "class_history_status.json").open("x", encoding="utf-8") as stream:
    json.dump(out, stream, ensure_ascii=False, indent=2)
print("PASS: frozen class-history source hashes and both outcome levels saved;0 experimental calls")
