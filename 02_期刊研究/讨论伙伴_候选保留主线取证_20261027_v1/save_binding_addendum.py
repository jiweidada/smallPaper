"""Bind static cache-producer source semantics without executing the producer."""
import hashlib
import json
from pathlib import Path
RUN = Path(__file__).resolve().parent
ROOT = RUN.parent
REPLAY = ROOT / "raw_EMA逐GT对照_20261026_v1/同后端CPU重放_20261026_v1"
NATIVE = ROOT / "code/dn_pre_frozen_diagnostic_20261020_v1/native/engine/deim/dfine_decoder.py"
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
receipt = json.loads((REPLAY / "complete.json").read_text("utf-8-sig"))
declared = receipt["source_sha256"]
for p in (NATIVE, REPLAY / "replay.py"):
    assert declared[str(p)] == digest(p)
out = {"date": "2026-10-27", "kind": "static source and historical receipt reading only",
       "files": [{"path": str(p), "sha256": digest(p)} for p in (NATIVE, REPLAY / "replay.py", REPLAY / "complete.json")],
       "anchors": {"replay.py": "55-58: decoder return output[4]/[5] saved as pre_boxes/pre_logits",
                   "native_dfine_decoder.py": "484-490,497-499,513-514: first decoder point, pre score before LQE;356-358 query-shared additive LQE"},
       "conclusion": "Saved pre is the first decoder observation, with direct score_head[0] logits before LQE; shares encoder-selected query ancestry with final. It is not CLSC's intermediate encoder prediction set or statistically independent evidence.",
       "additional_primary_formula_reading": {"clsc": "Re-read PDF2-3 exact local-neighbourhood/TopK quality-weighted mean support formula", "lddp": "Re-read PDF3 Algorithm2 andPDF5 section2.2 final candidate spatial/semantic DPP inference"},
       "operations": {"source_file_execution": 0, "prediction_cache_loads": 0, "model_forwards": 0, "evaluator_calls": 0, "training": 0}}
with (RUN / "cache_producer_binding_addendum.json").open("x", encoding="utf-8") as stream:
    json.dump(out, stream, ensure_ascii=False, indent=2)
print("PASS: actual native implementation matches historical CPU cache receipt; producer not executed")
