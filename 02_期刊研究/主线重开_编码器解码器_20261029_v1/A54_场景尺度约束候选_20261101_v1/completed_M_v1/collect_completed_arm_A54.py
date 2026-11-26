"""Archive an exited arm; validates hashes without any detector execution."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PROTOCOL_SHA = "20d73d282fa24357257fcd0a136d90346b414c44ffe9cb7973a4c4160e6a2166"
MODULE_SHA = "91a952243abbe8446b5efaed054dc1ec2a076d593b900e981a2d80b6dbc710a8"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(4 << 20), b""):
            h.update(data)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, obj):
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def process(identity):
    try:
        parts = Path("/proc", str(identity["pid"]), "stat").read_text().split(") ", 1)[1].split()
    except FileNotFoundError:
        return {"pid": identity["pid"], "owned_live": False, "reason": "missing"}
    match = int(parts[19]) == identity["start_ticks"]
    return {"pid": identity["pid"], "owned_live": match and parts[0] not in ("Z", "X"),
            "identity_matches": match, "state": parts[0]}


def exact_module_initial_state():
    """Post hoc CPU reconstruction; its tensor digest must match actual startup."""
    target = ROOT / "new_module_untrained_initial_state.pth"
    receipt = ROOT / "new_module_untrained_initial_state_receipt.json"
    if target.exists():
        assert sha(target) == load(receipt)["file_sha256"]
        assert load(receipt)["tensor_sha256"] == MODULE_SHA
        return [target, receipt]
    import torch
    from scene_scale_prior import SceneScalePrior, NEW_INIT_SEED
    torch.set_num_threads(2)
    stats = load(ROOT / "scale_statistics_FROZEN_v1.json")
    torch.manual_seed(NEW_INIT_SEED)
    state = SceneScalePrior(192, "M", stats["class_offsets"], stats["initial_log_scale"]).state_dict()
    h = hashlib.sha256()
    for name, tensor in state.items():
        h.update(name.encode()); h.update(tensor.contiguous().numpy().tobytes())
    assert h.hexdigest() == MODULE_SHA
    torch.save({"state_dict": state, "module_seed": NEW_INIT_SEED, "reconstructed_after_training_started": True}, target)
    save(receipt, {"status": "EXACT_STARTUP_DIGEST_MATCH", "file_sha256": sha(target), "tensor_sha256": h.hexdigest(),
        "scope": "CPU reconstruction after training started; same tensor digest as formal startup and capacity, not a previously saved startup file.",
        "new_whole_model_forwards": 0, "new_optimizer_updates": 0})
    return [target, receipt]


def main(arm):
    folder = ROOT / ("arm_" + arm)
    result, end = load(folder / "result.json"), load(folder / "exit.json")
    assert result["status"] == "COMPLETED" and result["epochs"] == 12
    assert result["optimizer_updates"] == 9708 and result["full_model_forwards"] == 10159
    assert result["model_parameters"] == 9809322 and result["extra_parameters"] == 6210
    assert end["exit_code"] == 0
    proc = process(load(folder / "child_identity.json"))
    assert not proc["owned_live"]
    protocol = ROOT / "protocol_FROZEN_v1.json"
    assert sha(protocol) == result["protocol_sha256"] == PROTOCOL_SHA
    plan = load(protocol)
    local = {name: sha(ROOT / name) == h for name, h in plan["files_sha256"].items()}
    external = {name: sha(Path(name)) == h for name, h in plan["external_sha256"].items()}
    assert all(local.values()) and all(external.values())
    initial = load(folder / "initialization.json")
    assert initial["new_module_initialization_sha256"] == MODULE_SHA and initial["scratch_no_capacity_weights"]
    auditfile = ROOT / (arm + "_completed_source_audit.json")
    save(auditfile, {"status": "PASS", "arm": arm, "observed_server_unix": time.time(), "owned_process": proc,
        "local_frozen_files": local, "external_frozen_inputs": external,
        "new_detector_forwards": 0, "new_optimizer_updates": 0})
    stable_names = ["protocol_FROZEN_v1.json", "supervise_A54.py", "collect_completed_arm_A54.py",
                    "capacity_v1_result.json", "capacity_v1.log", "capacity_v2/capacity_protocol_FROZEN_v2.json",
                    "capacity_v2/capacity_result.json", "capacity_v2/capacity.log"]
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    files += [ROOT / n for n in plan["files_sha256"]]
    files += [ROOT / n for n in stable_names if (ROOT / n).is_file()]
    files += [auditfile] + exact_module_initial_state()
    files = sorted(set(files))
    assert all(not f.is_symlink() and f.resolve().is_relative_to(ROOT) for f in files)
    manifest = {"arm": arm, "status": "COMPLETED_EXIT0", "files": {
        f.relative_to(ROOT).as_posix(): {"bytes": f.stat().st_size, "sha256": sha(f)} for f in files},
        "source_hashes_unchanged": True, "new_detector_forwards": 0, "new_optimizer_updates": 0}
    target = ROOT / (arm + "_completed_bundle_v1.zip")
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        z.writestr("bundle_manifest.json", json.dumps(manifest, indent=2))
        for f in files:
            z.write(f, f.relative_to(ROOT).as_posix())
    assert all(sha(ROOT / n) == row["sha256"] for n, row in manifest["files"].items())
    ready = {"status": "BUNDLE_READY", "arm": arm, "path": str(target), "bytes": target.stat().st_size,
        "sha256": sha(target), "payload_files": len(files), "endpoint": result["endpoint_metrics"],
        "source_hashes_unchanged": True, "new_detector_forwards": 0, "new_optimizer_updates": 0}
    save(ROOT / (arm + "_bundle_ready.json"), ready)
    print(json.dumps(ready), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["M", "Q"], required=True)
    main(parser.parse_args().arm)
