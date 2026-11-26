"""Read fixed cache metadata only; no torch import or tensor-storage access.

Only the A/B development00..07 whitelist is permitted. This does not execute
producer scripts, native math, matching, gradients, forward passes or training.
"""
from pathlib import Path
from collections import OrderedDict
import ast
import hashlib
import io
import json
import pickle
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "持续寻找研究主线_20261020_v2"
BASE = ROOT / "data/08_匹配聚合覆盖探针_20261019_v1"
CACHE = BASE / "isolated_v3"
OUT = Path(__file__).with_name("BASE24_schema_preflight.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class TensorMeta:
    def __init__(self, storage, offset, shape, stride, requires_grad=False, hooks=None, metadata=None):
        self.info = {"shape": list(shape), "stride": list(stride),
                     "requires_grad_saved": bool(requires_grad),
                     "storage": storage, "offset": offset}


class MetadataOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return TensorMeta
        if module == "torch" and name in ("FloatStorage", "LongStorage"):
            return "torch." + name
        raise ValueError("Unsupported global; no fallback execution: " + module + "." + name)

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage"):
            raise ValueError("Unsupported persistent object")
        _, dtype, key, device, size = pid
        return {"dtype": dtype, "key": key, "saved_device": device, "elements": size}


def skeleton(value):
    if isinstance(value, TensorMeta):
        return {"tensor_metadata": value.info}
    if isinstance(value, dict):
        return {str(k): skeleton(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [skeleton(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("Unexpected metadata object: " + type(value).__name__)


def literal_assign(path, name):
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(n, ast.Name) and n.id == name for n in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def main():
    assert not OUT.exists()
    reader = OLD / "223_BASE24_cached_GO_repair_probe.py"
    expected = literal_assign(reader, "INPUT_HASHES")
    rows, sources = [], {}
    for letter in ("A", "B"):
        seal_path = CACHE / letter / "frozen_predictions/development/seal.json"
        complete_path = CACHE / letter / "development/complete.json"
        seal, complete = read(seal_path), read(complete_path)
        for path in (seal_path, complete_path):
            rel = path.relative_to(CACHE).as_posix()
            sources[str(path)] = sha(path)
            assert sources[str(path)] == expected[rel]
        predrows = {r["batch"]: r for r in seal["files"]}
        priorrows = {r["batch"]: r for r in complete["files"]}
        assert sorted(predrows) == sorted(priorrows) == list(range(8))
        for batch in range(8):
            pred = CACHE / letter / "frozen_predictions/development" / predrows[batch]["relative"]
            prior = CACHE / letter / "development" / priorrows[batch]["relative"]
            for path in (pred, prior):
                rel = path.relative_to(CACHE).as_posix()
                sources[str(path)] = sha(path)
                assert sources[str(path)] == expected[rel]
            with zipfile.ZipFile(pred) as z:
                members = [n for n in z.namelist() if n.endswith("/data.pkl")]
                assert len(members) == 1
                value = MetadataOnly(io.BytesIO(z.read(members[0]))).load()
            a = read(prior)
            output = value["outputs"]
            assert output["pred_logits"].info["shape"] == [4, 300, 10]
            assert output["pred_boxes"].info["shape"] == [4, 300, 4]
            assert len(value["targets"]) == len(a["branch_matches"]) == 4
            assert all("final" in r for r in a["branch_matches"])
            rows.append({"letter": letter, "batch": batch,
                         "prediction_sha256": sources[str(pred)],
                         "analysis_sha256": sources[str(prior)],
                         "prediction_schema": skeleton(value),
                         "analysis_keys": list(a), "branch_names": a["branch_names"],
                         "branch_match_keys_per_image": [list(r) for r in a["branch_matches"]],
                         "analysis_input_keys_per_image": [list(r) for r in a["inputs"]],
                         "saved_scalar_total_keys": list(a.get("batch_losses", {}))})
    for f in (reader, OLD / "209_C0_monotone_GO_repair.py", OLD / "197_C0_cached_DN_descriptive.py",
              BASE / "baseline_recipes.json", BASE / "package_manifest.json", Path(__file__)):
        sources[str(f)] = sha(f)
    result = {"scope": "A/B development00..07 schema/provenance only",
              "source_hashes": sources, "rows": rows,
              "tensor_storage_members_opened": 0, "torch_imported": False,
              "native_scripts_imported_or_executed": 0, "gradient_statistics_computed": 0,
              "model_constructions": 0, "model_forwards": 0, "backward_or_VJP": 0,
              "updates": 0, "heldout_reads": 0}
    assert all(sha(p) == h for p, h in sources.items())
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"saved": str(OUT), "fixed_batches": len(rows),
                      "tensor_storage_members_opened": 0, "gradient_statistics_computed": 0,
                      "first_prediction_keys": list(rows[0]["prediction_schema"]),
                      "first_analysis_keys": rows[0]["analysis_keys"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
