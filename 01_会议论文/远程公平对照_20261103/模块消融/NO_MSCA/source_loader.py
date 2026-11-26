"""Proposed one-pass CPU accuracy check. NOT RUN or authorized by this file.

Default invocation exits before importing any model/runtime. An explicit CLI
confirmation prevents accidental use; it does not replace the user's consent.
This runner has no training, deployment conversion, timing, or profiling path.
"""

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import traceback
import types


BUNDLE = Path(__file__).resolve().parent
SNAPSHOT = BUNDLE / "source_snapshot"
METRICS = ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl")
PACKAGES = ("torch", "torchvision", "numpy", "Pillow", "PyYAML", "opencv-python",
            "tqdm", "psutil", "faster-coco-eval")
NAMESPACE_PACKAGES = ("engine", "engine.core", "engine.data", "engine.data.dataset",
                      "engine.data.transforms", "engine.misc", "engine.backbone",
                      "engine.deim", "engine.extre_module", "engine.extre_module.custom_nn",
                      "engine.extre_module.custom_nn.attention")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    if not path.resolve().is_relative_to((BUNDLE / "results").resolve()):
        raise ValueError("Output must remain in this bundle's results directory")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_hash(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"Input hash mismatch: {path.name}")
    return actual


def verify_inputs(run_id):
    manifest_path = BUNDLE / "preparation_manifest.json"
    manifest = read_json(manifest_path)
    run = next(item for item in manifest["prepared_runs"] if item["run_id"] == run_id)
    config_path = BUNDLE / run["prepared_configuration"]
    require_hash(config_path, run["prepared_configuration_sha256"])
    require_hash(BUNDLE / "source_manifest.json", manifest["source_manifest_sha256"])
    require_hash(BUNDLE / "validation_image_manifest.json", manifest["image_manifest_sha256"])
    for item in read_json(BUNDLE / "source_manifest.json"):
        path = (BUNDLE / item["snapshot"]).resolve()
        if not path.is_relative_to(SNAPSHOT.resolve()):
            raise ValueError("Source snapshot path escapes bundle")
        require_hash(path, item["sha256"])
    require_hash(Path(run["checkpoint"]), run["checkpoint_sha256"])
    annotation = Path(manifest["validation_annotation"])
    require_hash(annotation, manifest["validation_annotation_sha256"])
    image_root = annotation.parent.parent / "images"
    image_manifest = read_json(BUNDLE / "validation_image_manifest.json")
    for item in image_manifest:
        image_path = (image_root / item["file_name"]).resolve()
        if not image_path.is_relative_to(image_root.resolve()):
            raise ValueError("Image path escapes validation directory")
        require_hash(image_path, item["sha256"])

    cfg = read_json(config_path)
    if cfg["device"] != "cpu" or cfg["use_amp"] or not cfg["use_ema"]:
        raise ValueError("Unexpected device, AMP, or EMA setting")
    if cfg.get("ram_cache") or cfg.get("remap_mscoco_category"):
        raise ValueError("Unexpected image cache or category remapping")
    loader_cfg = cfg["val_dataloader"]
    if loader_cfg["shuffle"] or loader_cfg["drop_last"] or loader_cfg["num_workers"] != 0:
        raise ValueError("Unexpected validation loader setting")
    if loader_cfg["total_batch_size"] != run["validation_batch_size"]:
        raise ValueError("Validation batch size changed")
    if Path(loader_cfg["dataset"]["ann_file"]) != annotation:
        raise ValueError("Validation annotation path changed")
    if Path(loader_cfg["dataset"]["img_folder"]) != image_root:
        raise ValueError("Validation image path changed")
    if cfg[cfg["DEIM"]["backbone"]].get("pretrained"):
        raise ValueError("Pretrained downloads are forbidden")
    if cfg["seed"] != 0 or cfg["num_classes"] != 10:
        raise ValueError("Unexpected seed or category count")
    if cfg["evaluator"] != {"type": "CocoEvaluator", "iou_types": ["bbox"]}:
        raise ValueError("Unexpected evaluator recipe")
    ids = sorted(item["image_id"] for item in image_manifest)
    if len(ids) != 548 or len(set(ids)) != 548:
        raise ValueError("Expected exactly 548 unique validation images")
    return manifest, run, cfg, ids


def disable_unused_profiling_import():
    # These source modules only call calculate_flops in their standalone demos.
    # Keep their bytes intact; fail if profiling is ever reached in this run.
    attention = SNAPSHOT / "engine/extre_module/custom_nn/attention"
    for filename in ("ema.py", "simam.py", "lsk.py", "DeformableLKA.py"):
        tree = ast.parse((attention / filename).read_text(encoding="utf-8"))
        for statement in tree.body:
            is_demo = (isinstance(statement, ast.If) and isinstance(statement.test, ast.Compare)
                       and isinstance(statement.test.left, ast.Name) and statement.test.left.id == "__name__"
                       and len(statement.test.comparators) == 1
                       and isinstance(statement.test.comparators[0], ast.Constant)
                       and statement.test.comparators[0].value == "__main__")
            if not is_demo and any(isinstance(node, ast.Name) and node.id == "calculate_flops"
                                   for node in ast.walk(statement)):
                raise ValueError("Profiling symbol unexpectedly used outside a standalone demo")
    if "calflops" in sys.modules:
        raise RuntimeError("Unexpected preloaded profiling dependency")
    disabled = types.ModuleType("calflops")
    disabled.__spec__ = importlib.machinery.ModuleSpec("calflops", loader=None)

    def forbidden(*args, **kwargs):
        raise RuntimeError("Profiling is excluded from this authorized accuracy check")

    disabled.calculate_flops = forbidden
    sys.modules["calflops"] = disabled


def register_selected_source(backbone_name, include_evaluation=True):
    # Empty package shells skip broad __init__.py imports of unrelated models.
    # Actual selected modules execute byte-identical archived source files.
    if any(name == "engine" or name.startswith("engine.") for name in sys.modules):
        raise RuntimeError("Run in a fresh Python process; engine was already imported")
    for name in NAMESPACE_PACKAGES:
        package = types.ModuleType(name)
        package.__path__ = [str(SNAPSHOT.joinpath(*name.split(".")))]
        package.__package__ = name
        package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = package
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, package)
    workspace = importlib.import_module("engine.core.workspace")
    for name in ("register", "create", "GLOBAL_CONFIG"):
        setattr(sys.modules["engine.core"], name, getattr(workspace, name))
    data_loader = importlib.import_module("engine.data.dataloader")
    sys.modules["engine.data"].DataLoader = data_loader.DataLoader
    importlib.import_module("engine.misc.dist_utils")
    disable_unused_profiling_import()
    evaluator_module = None
    if include_evaluation:
        importlib.import_module("engine.data.transforms._transforms")
        importlib.import_module("engine.data.transforms.container")
        importlib.import_module("engine.data.dataset.coco_dataset")
        evaluator_module = importlib.import_module("engine.data.dataset.coco_eval")
    backbone_modules = {"HGNetv2": "hgnetv2", "HGNetv2_MSCA_EnhancedStem": "hgnetv2_MSCA_EnhancedStem"}
    importlib.import_module("engine.backbone." + backbone_modules[backbone_name])
    for module_name in ("deim", "hybrid_encoder", "dfine_decoder", "postprocessor"):
        importlib.import_module("engine.deim." + module_name)
    yaml_utils = importlib.import_module("engine.core.yaml_utils")
    return workspace, yaml_utils, evaluator_module


def block_network(event, args):
    if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.bind", "socket.sendto"}:
        raise RuntimeError("Network access is disabled for this local evaluation")


def evaluate_once(run_id, threads):
    manifest, run, cfg, expected_ids = verify_inputs(run_id)
    output = BUNDLE / "results" / run_id
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)  # Never overwrite a prior/partial run or auto-retry.
    receipt = {
        "status": "INITIALIZING", "run_id": run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(), "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "threads": threads, "device": "cpu",
        "configuration_sha256": run["prepared_configuration_sha256"],
        "checkpoint_sha256": run["checkpoint_sha256"], "weight_source": "ema.module",
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "annotation_sha256": manifest["validation_annotation_sha256"],
        "image_manifest_sha256": manifest["image_manifest_sha256"],
        "runner_sha256": digest(Path(__file__)), "completed_images": 0,
        "inference_passes": 1, "deployment_conversion": False,
        "training": False, "timing_benchmark": False, "profiling": False,
        "profiling_import_adapter": "Fail-closed placeholder for unused standalone-demo calculate_flops imports; selected neural-network source bytes unchanged",
        "authorization_sha256": digest(BUNDLE / "authorization.json"),
    }
    save_json(output / "receipt.json", receipt)
    try:
        sys.dont_write_bytecode = True
        sys.addaudithook(block_network)
        import numpy as np
        import torch

        receipt["package_versions"] = {name: importlib.metadata.version(name) for name in PACKAGES}
        receipt["torch_build"] = torch.__config__.show()
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        torch.set_default_dtype(torch.float32)
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        workspace, yaml_utils, evaluator_module = register_selected_source(cfg["DEIM"]["backbone"])

        def merged():
            return yaml_utils.merge_config(deepcopy(cfg), inplace=False, overwrite=False)

        model = workspace.create(cfg["model"], merged())
        postprocessor = workspace.create(cfg["postprocessor"], merged())
        # Restricted deserialization only. Failure is reported, never retried unsafely.
        checkpoint = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
        ema = checkpoint.get("ema")
        if not isinstance(ema, dict) or not isinstance(ema.get("module"), dict):
            raise ValueError("Checkpoint has no expected EMA module; ordinary-model fallback is forbidden")
        state = ema["module"]
        expected_state = model.state_dict()
        loading = {
            "missing_keys": sorted(set(expected_state) - set(state)),
            "unexpected_keys": sorted(set(state) - set(expected_state)),
            "shape_mismatches": [key for key in sorted(set(state) & set(expected_state))
                                 if not torch.is_tensor(state[key]) or list(state[key].shape) != list(expected_state[key].shape)],
            "checkpoint_last_epoch": checkpoint.get("last_epoch"),
            "ema_updates": ema.get("updates"), "strict": True,
        }
        save_json(output / "loading_diagnostics.json", loading)
        if any(loading[key] for key in ("missing_keys", "unexpected_keys", "shape_mismatches")):
            raise ValueError("EMA state does not strictly match the archived current model")
        model.load_state_dict(state, strict=True)
        if "postprocessor" not in checkpoint:
            raise ValueError("Checkpoint postprocessor state missing")
        postprocessor.load_state_dict(checkpoint["postprocessor"], strict=True)
        receipt["model_parameters"] = sum(parameter.numel() for parameter in model.parameters())
        receipt["model_trainable_parameters"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        del checkpoint, state, expected_state, ema
        model.eval().to("cpu")
        postprocessor.eval().to("cpu")

        loader_config = merged()
        batch_size = loader_config["val_dataloader"].pop("total_batch_size")
        loader = workspace.create("val_dataloader", loader_config, batch_size=batch_size)
        if len(loader.dataset) != 548 or sorted(loader.dataset.ids) != expected_ids:
            raise ValueError("Constructed validation loader differs from the frozen image set")
        evaluator = evaluator_module.CocoEvaluator(loader.dataset.coco, ["bbox"])
        evaluator.cleanup()
        params = evaluator.coco_eval["bbox"].params
        # Check the historically observed local COCO-style protocol, without overriding defaults.
        if list(params.maxDets) != [1, 10, 100] or sorted(params.catIds) != list(range(10)):
            raise ValueError("Evaluator categories or maxDets differ from the audited local protocol")
        if not np.allclose(params.iouThrs, np.linspace(0.5, 0.95, 10), rtol=0, atol=1e-12):
            raise ValueError("Unexpected evaluator IoU thresholds")
        if not np.allclose(params.recThrs, np.linspace(0, 1, 101), rtol=0, atol=1e-12):
            raise ValueError("Unexpected evaluator recall thresholds")
        expected_areas = [[0, 1e10], [0, 1024], [1024, 9216], [9216, 1e10]]
        if np.asarray(params.areaRng).tolist() != expected_areas or params.useCats != 1:
            raise ValueError("Unexpected evaluator area ranges or category handling")

        receipt["status"] = "EVALUATING"
        receipt["validation_batch_size"] = batch_size
        save_json(output / "receipt.json", receipt)
        predictions, seen_ids = [], []
        # This reproduces det_engine.evaluate's prediction/evaluator sequence,
        # omitting its unused criterion.eval(), timers, logging helper and TIDE.
        with torch.no_grad():
            for samples, targets in loader:
                samples = samples.to("cpu", non_blocking=True)
                targets = [{key: value.to("cpu", non_blocking=True) for key, value in target.items()}
                           for target in targets]
                outputs = model(samples)
                sizes = torch.stack([target["orig_size"] for target in targets], dim=0)
                results = postprocessor(outputs, sizes)
                if len(results) != len(targets):
                    raise ValueError("Postprocessor batch length mismatch")
                batch_ids = [target["image_id"].item() for target in targets]
                if any(image_id in seen_ids for image_id in batch_ids) or len(set(batch_ids)) != len(batch_ids):
                    raise ValueError("Repeated validation image")
                mapped = dict(zip(batch_ids, results))
                for prediction in results:
                    for key in ("boxes", "scores"):
                        if not torch.isfinite(prediction[key]).all():
                            raise ValueError("Nonfinite prediction")
                predictions.extend(evaluator.prepare_for_coco_detection(mapped))
                evaluator.update(mapped)
                seen_ids.extend(batch_ids)
                receipt["completed_images"] = len(seen_ids)
                save_json(output / "receipt.json", receipt)
                print(f"{run_id}: {len(seen_ids)}/548 images", flush=True)
        if sorted(seen_ids) != expected_ids:
            raise ValueError("Incomplete or differing validation image set")
        save_json(output / "predictions.json", predictions)
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()
        coco_eval = evaluator.coco_eval["bbox"]
        values = np.asarray(coco_eval.stats, dtype=float)
        if values.shape != (12,) or not np.isfinite(values).all():
            raise ValueError("Unexpected metric vector")
        protocol = {
            "iou_type": "bbox", "image_ids": [int(value) for value in coco_eval.params.imgIds],
            "category_ids": [int(value) for value in coco_eval.params.catIds],
            "max_dets": [int(value) for value in coco_eval.params.maxDets],
            "iou_thresholds": np.asarray(coco_eval.params.iouThrs).tolist(),
            "recall_thresholds": np.asarray(coco_eval.params.recThrs).tolist(),
            "area_ranges": np.asarray(coco_eval.params.areaRng).tolist(),
            "use_categories": int(coco_eval.params.useCats),
        }
        result = {
            "run_id": run_id, "metrics": dict(zip(METRICS, values.tolist())),
            "checkpoint_sha256": run["checkpoint_sha256"], "weight_source": "ema.module",
            "reference_best_log_epoch": run["reference_best_log_epoch"],
            "reference_best_AP": run["reference_best_AP"],
            "AP_difference_percentage_points": 100 * (float(values[0]) - run["reference_best_AP"]),
            "protocol": protocol, "predictions_sha256": digest(output / "predictions.json"),
            "interpretation": "A current-code, current-runtime check of a retained checkpoint; not proof of historical source identity or a fair training ablation.",
        }
        save_json(output / "metrics.json", result)
        np.savez_compressed(output / "evaluation_arrays.npz", precision=coco_eval.eval["precision"],
                            recall=coco_eval.eval["recall"], scores=coco_eval.eval["scores"])
        receipt["status"] = "COMPLETED"
        receipt["metrics_sha256"] = digest(output / "metrics.json")
        receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(output / "receipt.json", receipt)
        return 0
    except BaseException as error:
        receipt["status"] = "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED"
        receipt["error_type"] = type(error).__name__
        receipt["error"] = str(error)
        receipt["ended_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(output / "receipt.json", receipt)
        (output / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", choices=("N0", "N4", "S1"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--confirm-local-cpu-evaluation", action="store_true",
                        help="Use only after explicit user authorization; this flag itself is not consent")
    args = parser.parse_args()
    if not args.confirm_local_cpu_evaluation:
        parser.error("No evaluation requested. Explicit user authorization and the confirmation flag are required.")
    if args.run_id is None or not 1 <= args.threads <= 16:
        parser.error("Choose one frozen run ID and 1 to 16 CPU threads")
    return evaluate_once(args.run_id, args.threads)


if __name__ == "__main__":
    raise SystemExit(main())
