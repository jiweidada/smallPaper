"""Authorized compatibility check: hashes, imports and strict loading; no forward pass."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

from evaluate_cpu import (BUNDLE, block_network, digest, register_selected_source, verify_inputs)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-id", choices=("N0", "N4", "S1"), required=True)
parser.add_argument("--receipt-suffix", choices=("", "utf8"), default="")
args = parser.parse_args()
destination = BUNDLE / "compatibility"
destination.mkdir(exist_ok=True)
receipt_stem = args.run_id + ("_" + args.receipt_suffix if args.receipt_suffix else "")
receipt_path = destination / (receipt_stem + ".json")
if receipt_path.exists():
    raise FileExistsError("Preflight receipt already exists; inspect it before repeating")
report = {"run_id": args.run_id, "started_utc": datetime.now(timezone.utc).isoformat(),
          "receipt_suffix": args.receipt_suffix,
          "status": "CHECKING", "inference_run": False, "evaluation_run": False,
          "runner_sha256": digest(BUNDLE / "evaluate_cpu.py"),
          "authorization_sha256": digest(BUNDLE / "authorization.json")}
try:
    sys.addaudithook(block_network)
    import torch
    torch.set_num_threads(8)
    manifest, run, cfg, expected_ids = verify_inputs(args.run_id)
    workspace, yaml_utils, evaluator_module = register_selected_source(cfg["DEIM"]["backbone"])
    merged = lambda: yaml_utils.merge_config(deepcopy(cfg), inplace=False, overwrite=False)
    model = workspace.create(cfg["model"], merged())
    postprocessor = workspace.create(cfg["postprocessor"], merged())
    checkpoint = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
    ema = checkpoint.get("ema")
    if not isinstance(ema, dict) or not isinstance(ema.get("module"), dict):
        raise ValueError("Expected EMA module missing")
    state = ema["module"]
    expected = model.state_dict()
    report.update({"checkpoint_sha256": run["checkpoint_sha256"],
                   "missing_keys": sorted(set(expected) - set(state)),
                   "unexpected_keys": sorted(set(state) - set(expected)),
                   "shape_mismatches": [key for key in sorted(set(expected) & set(state))
                       if not torch.is_tensor(state[key]) or list(state[key].shape) != list(expected[key].shape)],
                   "model_parameters": sum(value.numel() for value in model.parameters()),
                   "model_trainable_parameters": sum(value.numel() for value in model.parameters() if value.requires_grad),
                   "checkpoint_last_epoch": checkpoint.get("last_epoch")})
    if any(report[key] for key in ("missing_keys", "unexpected_keys", "shape_mismatches")):
        raise ValueError("EMA/model mismatch; no inference permitted")
    model.load_state_dict(state, strict=True)
    postprocessor.load_state_dict(checkpoint["postprocessor"], strict=True)
    loader_cfg = merged()
    batch = loader_cfg["val_dataloader"].pop("total_batch_size")
    loader = workspace.create("val_dataloader", loader_cfg, batch_size=batch)
    if len(loader.dataset) != 548 or sorted(loader.dataset.ids) != expected_ids:
        raise ValueError("Validation dataset differs from the frozen image set")
    evaluator = evaluator_module.CocoEvaluator(loader.dataset.coco, ["bbox"])
    p = evaluator.coco_eval["bbox"].params
    report.update({"status": "STRICT_LOAD_AND_CONSTRUCTION_OK_NO_FORWARD",
                   "strict": True, "dataset_count": len(loader.dataset), "validation_batch_size": batch,
                   "max_dets": [int(value) for value in p.maxDets],
                   "categories": [int(value) for value in p.catIds]})
except BaseException as error:
    report.update({"status": "FAILED_BEFORE_INFERENCE", "error": str(error), "error_type": type(error).__name__})
    (destination / (receipt_stem + ".traceback.txt")).write_text(traceback.format_exc(), encoding="utf-8")
    raise
finally:
    report["ended_utc"] = datetime.now(timezone.utc).isoformat()
    receipt_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
