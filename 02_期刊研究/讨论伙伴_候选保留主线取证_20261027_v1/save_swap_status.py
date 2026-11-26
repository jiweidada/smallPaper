"""Read partner's completed one-swap diagnostic; preserve all contrasts and IDs."""
import hashlib
import json
from pathlib import Path
RUN = Path(__file__).resolve().parent
P = RUN.parent / "候选保留_机制与创新假说_20261027_v1/single_swap_readout"
def read(p):
    return json.loads(p.read_text("utf-8-sig"))
def bind(p):
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
r = read(P / "attempt01/results.json")
v = read(P / "attempt01/verification.json")
assert r["status"] == "COMPLETE" and v["status"] == "PASS"
assert bind(P / "protocol.json")["sha256"] == r["protocol_sha256"]
rows = []
for condition, c in r["conditions"].items():
    assert c["GT"] == 38759 and c["images"] == 548
    assert len(set(x["exported"] for x in c["arms"].values())) == 1
    assert len(set(x["evaluated"] for x in c["arms"].values())) == 1
    for name, d in c["contrasts"].items():
        hi, lo = name.split("_minus_")
        for th, idx in ((50, 0), (75, 5)):
            gains, losses = d[f"gained{th}_IDs"], d[f"lost{th}_IDs"]
            assert len(set(gains)) == len(gains) and len(set(losses)) == len(losses)
            assert not set(gains) & set(losses)
            assert len(gains) - len(losses) == c["arms"][hi]["TP_all10"][idx] - c["arms"][lo]["TP_all10"][idx]
        rows.append({"condition": condition, "contrast": name, "images": 548, "GT": 38759,
                     "images_swapped": c["images_swapped"], "H_S_different_images": c["different_H_S_images"],
                     "AP_delta_pp": d["delta_pp"]["AP"], "APs_delta_pp": d["delta_pp"]["APs"],
                     "gain50": len(d["gained50_IDs"]), "loss50": len(d["lost50_IDs"]),
                     "gain75": len(d["gained75_IDs"]), "loss75": len(d["lost75_IDs"])})
assert r["strict_readout_advance_gate_pass"] is False
out = {"date": "2026-10-27", "bindings": [bind(P / name) for name in ("protocol.json", "attempt01/results.json", "attempt01/verification.json")],
       "partner_snapshot": r, "partner_verification": v, "compact_contrasts": rows,
       "decision": "Stop this fixed one-swap rule. Qualification information, criticality, safe deletion and AP are distinct gates.",
       "ID_convention_note": "All current contrasts share the same positive-GT-ID clone. Small A metric changes against historical original-ID metrics are recorded by partner; exact10TP and50/75 hit-set parity passed. Do not splice unlike ID conventions.",
       "operations_by_this_chat": {"prediction_cache_loads": 0, "model_forwards": 0, "evaluator_calls": 0, "training": 0}}
with (RUN / "swap_status.json").open("x", encoding="utf-8") as stream:
    json.dump(out, stream, ensure_ascii=False, indent=2)
print("PASS: all12 source contrasts and exact gain/loss arithmetic saved; frozen swap gate failed;0 experimental calls")
