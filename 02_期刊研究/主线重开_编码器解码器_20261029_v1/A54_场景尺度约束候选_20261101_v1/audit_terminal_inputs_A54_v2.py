"""Terminal-only identity, source and recorded-stream audit. No model imports."""
from pathlib import Path
import hashlib
import json
import time

ROOT = Path(__file__).resolve().parent
PROTOCOL_SHA = "20d73d282fa24357257fcd0a136d90346b414c44ffe9cb7973a4c4160e6a2166"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def process(identity):
    try:
        parts = Path("/proc", str(identity["pid"]), "stat").read_text().split(") ", 1)[1].split()
    except FileNotFoundError:
        return {"pid": identity["pid"], "owned_live": False, "reason": "missing"}
    match = int(parts[19]) == identity["start_ticks"]
    return {"pid": identity["pid"], "identity_matches": match, "state": parts[0],
            "owned_live": match and parts[0] not in ("Z", "X")}


def main():
    target = ROOT / "terminal_frozen_input_audit.json"
    assert not target.exists()
    final = load(ROOT / "three_arm_result.json")
    assert final["status"] == "COMPLETED"
    assert load(ROOT / "supervisor_progress.json")["status"] == "COMPLETED"
    processes = {"supervisor": process(load(ROOT / "supervisor_identity.json"))}
    rows = {}
    for arm in ["M", "Q"]:
        folder = ROOT / ("arm_" + arm)
        result = load(folder / "result.json")
        assert result["status"] == "COMPLETED" and result["epochs"] == 12
        assert result["optimizer_updates"] == 9708 and result["full_model_forwards"] == 10159
        assert result["model_parameters"] == 9809322 and result["extra_parameters"] == 6210
        assert load(folder / "exit.json")["exit_code"] == 0
        processes[arm] = process(load(folder / "child_identity.json"))
        rows[arm] = result
    assert not any(p["owned_live"] for p in processes.values())
    protocol = ROOT / "protocol_FROZEN_v1.json"
    assert sha(protocol) == PROTOCOL_SHA
    plan = load(protocol)
    local = {name: sha(ROOT / name) == h for name, h in plan["files_sha256"].items()}
    external = {name: sha(Path(name)) == h for name, h in plan["external_sha256"].items()}
    assert all(local.values()) and all(external.values())
    initial = {arm: load(ROOT / ("arm_" + arm) / "initialization.json") for arm in ["M", "Q"]}
    assert initial["M"] == initial["Q"]
    assert initial["M"]["new_module_initialization_sha256"] == "91a952243abbe8446b5efaed054dc1ec2a076d593b900e981a2d80b6dbc710a8"
    assert initial["M"]["all_native_initial_tensors_identical"] and initial["M"]["scratch_no_capacity_weights"]
    flow_keys = ["order_sha256", "first_eight_actual_input_sha256", "first_eight_pre_forward_rng_sha256", "all_batches_targets_including_ignore_tiles_sha256"]
    flow_checks = {}
    for e in range(1, 13):
        flows = {a: load(ROOT / ("arm_" + a) / ("epoch%02d_input_flow.json" % e)) for a in ["M", "Q"]}
        flows["B"] = load(Path("/root/uav_a49_20261101_v1/arm_B") / ("epoch%02d_input_flow.json" % e))
        flow_checks[str(e)] = {k: flows["B"][k] == flows["M"][k] == flows["Q"][k] for k in flow_keys}
        assert all(flow_checks[str(e)].values())
        assert all(f["optimizer_updates"] == 809 * e for f in flows.values())
    capacity = load(ROOT / "capacity_v2/capacity_result.json")
    assert capacity["status"] == "PASS" and capacity["cumulative_A54_capacity_forwards"] == 9
    assert sha(ROOT / "capacity_result.json") == "2c23b227ed03f60b7944a97c229c7d9e9196e00df996da88e9d57deaa905417b"
    failed_capacity = load(ROOT / "capacity_result.json")
    assert failed_capacity["whole_model_forwards"] == 1 and failed_capacity["capacity_only_optimizer_updates"] == 0
    assert capacity["whole_model_forwards"] == 8 and capacity["capacity_only_optimizer_updates"] == 6
    assert final["new_formal_model_forwards"] == 20318 and final["new_formal_optimizer_updates"] == 19416
    result = {"status": "PASS", "phase": "A54_TERMINAL_ONLY", "observed_server_unix": time.time(),
        "protocol_sha256": sha(protocol), "owned_processes": processes, "owned_jobs_live": False,
        "auditor_version": 2, "auditor_sha256": sha(Path(__file__)),
        "failed_capacity_original_remote_file": str(ROOT / "capacity_result.json"),
        "failed_capacity_original_sha256": sha(ROOT / "capacity_result.json"),
        "local_frozen_files": local, "external_frozen_inputs": external, "recorded_stream_checks": flow_checks,
        "capacity_forwards_including_v1_failed_checker": 9, "formal_new_model_forwards": 20318,
        "new_detector_forwards_for_this_audit": 0, "new_optimizer_updates_for_this_audit": 0,
        "scope": "Terminal identity, frozen inputs and recorded stream keys. Pixels/RNG cover first8 batches each epoch; labels/order cover all809 batches. No model/evaluator rerun."}
    with target.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({"status": "PASS", "local_files": len(local), "external_inputs": len(external),
        "owned_jobs_live": False, "new_model_ops": 0}), flush=True)


if __name__ == "__main__":
    main()
