"""Integrity and recorded-flow verification on CPU; no detector/evaluator call."""
from pathlib import Path, PurePosixPath
import argparse
import collections
import hashlib
import json
import math
import sys
import zipfile

P = Path(__file__).resolve().parent
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


def main(arm):
    output = P / (arm + "_completed_local_verification.json")
    assert not output.exists()
    ready = load(P / (arm + "_bundle_ready.json"))
    bundle = P / (arm + "_completed_bundle_v1.zip")
    assert bundle.stat().st_size == ready["bytes"] and sha(bundle) == ready["sha256"]
    dest = P / ("completed_" + arm + "_v1")
    already_extracted = dest.exists()
    if not already_extracted:
        dest.mkdir(exist_ok=False)
    with zipfile.ZipFile(bundle) as z:
        manifest = json.loads(z.read("bundle_manifest.json"))
        assert len(z.namelist()) == len(set(z.namelist())) == len(manifest["files"]) + 1
        assert set(z.namelist()) == set(manifest["files"]) | {"bundle_manifest.json"}
        assert len(manifest["files"]) == ready["payload_files"]
        if already_extracted:
            assert load(dest / "bundle_manifest.json") == manifest
        for name, expected in manifest["files"].items():
            rel = PurePosixPath(name)
            assert not rel.is_absolute() and ".." not in rel.parts
            target = (dest / Path(*rel.parts)).resolve()
            assert target.is_relative_to(dest.resolve())
            if already_extracted:
                assert target.is_file() and target.stat().st_size == expected["bytes"] and sha(target) == expected["sha256"], name
                continue
            target.parent.mkdir(exist_ok=True, parents=True)
            h, count = hashlib.sha256(), 0
            with z.open(name) as src, target.open("xb") as out:
                for data in iter(lambda: src.read(1 << 20), b""):
                    out.write(data); h.update(data); count += len(data)
            assert h.hexdigest() == expected["sha256"] and count == expected["bytes"], name
    if not already_extracted:
        (dest / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    folder = dest / ("arm_" + arm)
    result, init = load(folder / "result.json"), load(folder / "initialization.json")
    assert result["status"] == "COMPLETED" and load(folder / "exit.json")["exit_code"] == 0
    assert result["epochs"] == 12 and result["optimizer_updates"] == 9708 and result["full_model_forwards"] == 10159
    assert result["model_parameters"] == 9809322 and result["extra_parameters"] == 6210
    assert sha(dest / "protocol_FROZEN_v1.json") == result["protocol_sha256"] == init["protocol_sha256"] == PROTOCOL_SHA
    metrics = load(folder / "epoch12_EMA_metrics.json")
    assert metrics == result["endpoint_metrics"] == ready["endpoint"]
    assert load(dest / (arm + "_completed_source_audit.json"))["status"] == "PASS"
    bf = P.parent / "A49_下一主线有界筛选_20261101_v1/completed_B_v1/arm_B"
    flow_keys = ["order_sha256", "first_eight_actual_input_sha256", "first_eight_pre_forward_rng_sha256", "all_batches_targets_including_ignore_tiles_sha256"]
    checks = {}
    for e in range(1, 13):
        actual, baseline = load(folder / ("epoch%02d_input_flow.json" % e)), load(bf / ("epoch%02d_input_flow.json" % e))
        assert actual["optimizer_updates"] == baseline["optimizer_updates"] == 809 * e
        checks[str(e)] = {k: actual[k] == baseline[k] for k in flow_keys}
        assert all(checks[str(e)].values())
    sys.path.insert(0, str(P.parents[1] / "code/dn_pre_frozen_diagnostic_20261020_v1/runtime"))
    import torch
    import numpy as np
    torch.set_num_threads(4)
    initial_path = P.parent / "A43_ECDet三组蒸馏最小验证_20261031_v1/common_init_10class.pth"
    assert sha(initial_path) == init["common_init_sha256"] == result["initialization_sha256"]
    native = torch.load(initial_path, map_location="cpu", weights_only=True)["model"]
    reconstruction = dest / "new_module_untrained_initial_state.pth"
    reconstruction_receipt = load(dest / "new_module_untrained_initial_state_receipt.json")
    assert sha(reconstruction) == reconstruction_receipt["file_sha256"]
    extra = torch.load(reconstruction, map_location="cpu", weights_only=True)["state_dict"]
    h = hashlib.sha256()
    for name, value in extra.items():
        h.update(name.encode()); h.update(value.contiguous().numpy().tobytes())
    assert h.hexdigest() == reconstruction_receipt["tensor_sha256"] == init["new_module_initialization_sha256"] == MODULE_SHA
    prefix = "decoder.scene_scale_prior."
    expected = dict(native)
    expected.update({prefix + k: v for k, v in extra.items()})
    parameter_names = ["gain", "map.0.weight", "map.0.bias", "map.2.weight", "map.2.bias"]
    assert sum(extra[k].numel() for k in parameter_names) == 6210
    ckpt = torch.load(folder / "epoch12.pth", map_location="cpu", weights_only=True)
    assert ckpt["epoch"] == 12 and ckpt["arm"] == arm and ckpt["optimizer_updates"] == ckpt["ema"]["updates"] == 9708
    assert ckpt["protocol_sha256"] == PROTOCOL_SHA
    # Native StateEMA averages all floating buffers. Model this arithmetic,
    # rather than requiring those EMA buffers to be mathematically constant.
    constant = extra["class_offsets"].numpy().copy()
    ema_reference = constant.copy()
    for u in range(1, 9709):
        decay = .9999 * (1 - math.exp(-u / 1000))
        product = np.float32(ema_reference * np.float32(decay))
        ema_reference = np.float32(product.astype(np.float64) + constant.astype(np.float64) * np.float64(np.float32(1 - decay)))
    ema_reference = torch.from_numpy(ema_reference)
    state_checks = {}
    for label, state in [("raw", ckpt["model"]), ("EMA", ckpt["ema"]["module"])]:
        assert set(state) == set(expected)
        changed, structural = 0, []
        for name, tensor in state.items():
            assert tensor.shape == expected[name].shape and tensor.dtype == expected[name].dtype, name
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                assert name.endswith(".anchors") and torch.equal(tensor, expected[name]), name
                structural.append(name)
            changed += int(not torch.equal(tensor, expected[name]))
        bn = ["encoder.fpn_blocks.0.cv4.norm", "encoder.fpn_blocks.1.cv4.norm", "encoder.pan_blocks.0.cv4.norm", "encoder.pan_blocks.1.cv4.norm"]
        assert all(int(state[k + ".num_batches_tracked"]) == 9708 for k in bn)
        changes = {k: float((state[prefix + k] - extra[k]).abs().max()) for k in parameter_names}
        assert changed > 0 and all(v > 0 for v in changes.values())
        offset_reference = extra["class_offsets"] if label == "raw" else ema_reference
        assert torch.equal(state[prefix + "class_offsets"], offset_reference), (label, "constant-buffer EMA arithmetic mismatch")
        state_checks[label] = {"changed_state_tensors": changed, "new_parameter_changes": changes,
            "class_offsets_equal_declared_reference": True,
            "class_offsets_bitwise_unchanged_from_initial": torch.equal(state[prefix + "class_offsets"], extra["class_offsets"]),
            "class_offsets_reference_scope": "Unchanged initial buffer" if label == "raw" else "FP32 mul plus fused-scaled-add recurrence for9708 native EMA steps; bit-exact match, no relaxed tolerance",
            "four_BN_counters": 9708, "unchanged_nonfinite_anchor_buffers": structural}
    steps = collections.Counter(int(v["step"]) for v in ckpt["optimizer"]["state"].values())
    baseline_steps = load(P.parent / "A49_下一主线有界筛选_20261101_v1/B_completed_local_verification.json")["optimizer_step_histogram"]
    assert baseline_steps == {"9708": 482} and dict(steps) == {9708: 482 + 5}, dict(steps)
    for row in ckpt["optimizer"]["state"].values():
        assert all(torch.isfinite(row[k]).all() for k in ["exp_avg", "exp_avg_sq"])
    verified = {"status": "PASS", "arm": arm, "epochs": 12, "optimizer_updates": 9708, "full_model_forwards": 10159,
        "model_parameters": 9809322, "extra_parameters": 6210, "endpoint_metrics": metrics,
        "bundle_sha256": ready["sha256"], "payload_files_verified": len(manifest["files"]),
        "checkpoint_state_checks": state_checks, "optimizer_step_histogram": dict(steps),
        "recorded_flow_checks_against_B": checks, "new_detector_forwards": 0, "new_optimizer_updates": 0,
        "verifier_version": 2, "previous_checker_failure_preserved": str(P / "verification_v1_failure"),
        "constant_buffer_EMA_roundoff_max": float((ema_reference - extra["class_offsets"]).abs().max()),
        "scope": "CPU archive, raw/EMA/optimizer and recorded input-flow checks. No prediction rerun, significance or independent-test claim. EMA fixed-buffer arithmetic checked bit-exactly."}
    output.write_text(json.dumps(verified, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in verified.items() if k not in ["checkpoint_state_checks", "recorded_flow_checks_against_B"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["M", "Q"], required=True)
    main(parser.parse_args().arm)
