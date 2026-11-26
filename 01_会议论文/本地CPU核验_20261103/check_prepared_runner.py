"""Static/script-guard checks only; never import torch, models, or an evaluator."""

import ast
import email
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
RUNNER = ROOT / "evaluate_cpu.py"


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=str(RUNNER))
    forbidden_imports = {"engine.solver", "engine.optim", "train"}
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    if any(name == bad or name.startswith(bad + ".") for name in imports for bad in forbidden_imports):
        raise AssertionError("Unexpected training/solver import")
    forbidden_calls = {"deploy", "backward", "fit", "calculate_flops"}
    if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
           and node.func.attr in forbidden_calls for node in ast.walk(tree)):
        raise AssertionError("Unexpected training/deployment/profiling call")
    load_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr == "load_state_dict"]
    if len(load_calls) != 2 or not all(any(item.arg == "strict" and isinstance(item.value, ast.Constant)
                                         and item.value.value is True for item in node.keywords)
                                      for node in load_calls):
        raise AssertionError("Both model and postprocessor must load strictly")
    run_directories_before = sorted(str(path) for path in (ROOT / "results").glob("*"))
    guarded = subprocess.run([sys.executable, "-I", str(RUNNER)], text=True, capture_output=True, timeout=15)
    run_directories_after = sorted(str(path) for path in (ROOT / "results").glob("*"))
    if guarded.returncode != 2 or "No evaluation requested" not in guarded.stderr:
        raise AssertionError("Default invocation did not reject execution as expected")
    if run_directories_before != run_directories_after:
        raise AssertionError("The rejected invocation created a result directory")

    historical = (WORKSPACE / "02_期刊研究" / "前期调研_码率受限航拍小目标_20261028_v1"
                  / "本机CPU短测_20261029_v1")
    wheel_records = []
    for subfolder in ("wheelhouse_cpu", "wheelhouse_support", "wheels"):
        for path in sorted((historical / subfolder).glob("*")):
            if not path.is_file() or ".whl" not in path.name:
                continue
            record = {"path": str(path), "bytes": path.stat().st_size,
                      "zip_container_readable": zipfile.is_zipfile(path)}
            if record["zip_container_readable"]:
                with zipfile.ZipFile(path) as archive:
                    metadata_files = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
                    if len(metadata_files) == 1:
                        metadata = email.message_from_bytes(archive.read(metadata_files[0]))
                        record["package"] = metadata["Name"]
                        record["version"] = metadata["Version"]
                        record["requires_dist"] = metadata.get_all("Requires-Dist", [])
                        record["sha256"] = sha256(path)
            wheel_records.append(record)
    (ROOT / "local_dependency_inventory.json").write_text(json.dumps({
        "scope": "Read-only metadata and hashes for known historical wheel directories; no package imported or installed",
        "wheels": wheel_records,
        "limits": "Readable ZIP and package metadata do not establish full CRC integrity, installation success, or model compatibility. Files with partial suffixes must not be installed before validation.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = {
        "status": "RUNNER_DRAFT_STATIC_CHECKS_ONLY_NOT_AUTHORIZED_NOT_RUN",
        "runner_sha256": sha256(RUNNER), "syntax_valid": True,
        "strict_loading_calls": len(load_calls), "default_guard_returncode": guarded.returncode,
        "default_guard_message": guarded.stderr.strip(),
        "new_result_directories": 0,
        "imports_runtime_or_model_during_checks": False,
        "checkpoint_loaded": False, "model_constructed": False,
        "inference_run": False, "evaluation_run": False,
        "runtime_compatibility_verified": False,
        "selected_module_registration": "Uses unchanged frozen Python modules via package shells; skips broad package initializers and original solver entry",
        "unverified": ["Dependency import compatibility", "Strict EMA checkpoint/model compatibility",
                       "Forward and postprocessor execution on CPU", "Full evaluator execution and metrics"],
        "local_dependency_inventory_sha256": sha256(ROOT / "local_dependency_inventory.json"),
    }
    (ROOT / "runner_preparation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "guard_returncode": guarded.returncode,
                      "wheel_files_inspected": len(wheel_records), "model_or_eval_run": False}))


if __name__ == "__main__":
    main()
