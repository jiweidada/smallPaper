"""353: user-ruled revision of 352. Reuses 352_precheck.py functions unmodified.
Primary = frozen historical GPU predictions; sensitivity = sealed 352 CPU predictions.
No model forward, no training. Writes only into this directory."""
import sys, json, importlib.util, shutil, hashlib
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "c_axis_precheck_20261025_v1"
spec = importlib.util.spec_from_file_location("p352", V1 / "352_precheck.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)

def sha(p): return P.sha(p)

# Frozen inputs from 352 copied verbatim (byte-identical, SHA recorded), so analyze() reads them from OUT.
for name in ["352_fixed_cells.json", "352_cohorts.json", "352_gt_geometry.json", "352_protocol_frozen.json"]:
    if not (HERE / name).exists(): shutil.copyfile(V1 / name, HERE / name)
    assert sha(HERE / name) == sha(V1 / name)
P.OUT = HERE  # all writes go here; 'x' mode, never overwrites

data = P.read(P.VAL / "annotations/val.json")
gt = P.read(HERE / "352_gt_geometry.json")
reports = {}
for tag, src in [("historical", lambda l: P.GOLD / f"{l}_640_predictions.json"),
                 ("cpu", lambda l: V1 / f"352_{l}_predictions.json")]:
    stats, matches = {}, {}
    for label in ["baseline", "c"]:
        p = src(label)
        if tag == "historical":
            ev = P.read(P.GOLD / f"{label}_640_evaluation.json")
            assert sha(p) == ev["predictions_sha256"]
        rows = P.read(p)
        P.validate_predictions(rows, [i["id"] for i in data["images"]])
        stats[label], matches[label] = P.evaluate(f"353_{tag}_{label}", rows, data)
        if tag == "historical":
            ref = ev["metrics"]
            d = max(abs(stats[label][k] - ref[k]) for k in P.NAMES)
            assert d < 1e-12, ("historical evaluator mismatch", label, d)
        del rows
    reports[tag] = P.analyze(f"353_{tag}", stats, matches, gt)

h, c = reports["historical"], reports["cpu"]
consistent = bool(h["primary_pass"] == c["primary_pass"])
decision = "pass" if (h["primary_pass"] and consistent) else (
    "insufficient" if not h["gates"]["eligible_support"] else "mixed")
result = {"revision": "353 user ruling A (2026-10-25)", "decision": decision,
          "primary_source": "historical GPU predictions", "sensitivity_source": "352 CPU predictions",
          "historical_gates": h["gates"], "cpu_gates": c["gates"],
          "historical_primary_pass": h["primary_pass"], "cpu_primary_pass": c["primary_pass"],
          "consistent": consistent, "training_started": False,
          "training_status": "未启动：按修订，结果交用户确认后再决定",
          "model_forwards": 0, "optimizer_steps": 0,
          "script_sha256": sha(Path(__file__)), "reused_352_script_sha256": sha(V1 / "352_precheck.py")}
P.put("353_result.json", result)
print(json.dumps(result, ensure_ascii=False, indent=1))
