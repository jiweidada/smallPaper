"""Read and bind partner's completed result; no prediction arrays or statistics are run."""
import hashlib
import json
from pathlib import Path
RUN = Path(__file__).resolve().parent
P = RUN.parent / "Hlineage执行_20261027_v1"
def read(p):
    return json.loads(p.read_text("utf-8-sig"))
def bind(p):
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
r = read(P / "attempt01/results.json")
v = read(P / "attempt01/verification.json")
assert r["status"] == "COMPLETE" and v["status"] == "PASS"
assert bind(P / "protocol.json")["sha256"] == r["protocol_sha256"] == "47fe05725470eaa6d69e3fa2df171ca93799b4aadd9162652ba3d42370f66623"
assert r["parent_plan_sha256"] == "62f105a7fd24394a1034798cff261fc5905d0aded4da349b03f660fbc35a6e04"
assert len(r["conditions"]) == 8
rows = []
for name, c in r["conditions"].items():
    p = c["primary_qualified"]
    rows.append({"condition": name, "full_counts": c["full_counts"],
                 "primary": {k: value for k, value in p.items() if k not in ("strong_null_CMI_bits_199", "weak_null_CMI_bits_199")},
                 "saved_strong_replicates": len(p["strong_null_CMI_bits_199"]), "saved_weak_replicates": len(p["weak_null_CMI_bits_199"])})
out = {"date": "2026-10-27", "kind": "Partner completed-result reading and static-code review; no independent data execution",
       "bindings": [bind(P / n) for n in ("protocol.json", "run.py", "attempt01/results.json", "attempt01/verification.json")],
       "partner_result_snapshot": r, "partner_verification_snapshot": v, "all8_domains_read": rows,
       "static_code_anchors_read": "run.py24-58,65-137; full source read through focused excerpts",
       "static_observations": ["final-to-pre IoU orientation andpre-to-final geometric assignment directions consistent", "sets frozen before relabeling; anchor fix excludes originalq", "all observed controls share identical CMI computation and1e-12 comparison tolerance", "actual nativeA cap ranks bound to old per-image results"],
       "pending_at_first_review": ["logit-vs-sigmoid argmax equality audit", "integer(C,Jgeo,Y) purity/CMI-upper-bound audit"],
       "interpretation": "All8 frozen advance gates failed. Deletion is feasible in all4 conditions; insertion feasibility fails in raw1/raw2/EMA2. Strong-versus-weak reference distinction and finite plugin-CMI occupancy bias remain explicit. Random correspondence is not established as a better causal or detection policy.",
       "operations_by_this_chat": {"prediction_cache_loads": 0, "model_forwards": 0, "evaluator_calls": 0, "statistical_recomputations": 0, "training": 0}}
with (RUN / "Hlineage_completed_result_review.json").open("x", encoding="utf-8") as stream:
    json.dump(out, stream, ensure_ascii=False, indent=2)
print("PASS: completed result and all8 domains source-bound;199+199 saved reference values per domain retained; consistency audits pending,0 experimental calls")
