"""Cache the bound seed1 epoch132 EMA detector without training or interventions.

The native validation loader is used for both splits. Only its image/annotation
paths change. Native postprocessor outputs are retained exactly; a passive
pre-hook captures final classification-head inputs in FP32. This script does
not evaluate AP: predictions.json is handed to the separately bound evaluator.

--max-images is an engineering subset, never a complete scientific cache.
Existing split directories are refused. No old code, images or checkpoints are
modified, and no network connection is made by this script.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
CHECKPOINT_SHA = "30febfa18e6e21dc67b16169bd26875b82340eb3435534719a963904dadf68b5"
BINDING_FILES_SHA = "195d652530d5ba59beffb259e89f041ceec80ba87f5d620d7dca6ea8bb81f829"
ANNOTATIONS = {
    "train": ("32c1cae852e084430ea74503eb660f0707ec6b5e65d0998ce3ec8b448265c883", 6471),
    "val": ("0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd", 548),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_sources(native_root, binding_path):
    """Validate both the immutable 468-file manifest and its actual source set."""
    root = Path(native_root).resolve()
    files = read_json(binding_path)["files"]
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    require(len(files) == 468 and hashlib.sha256(canonical).hexdigest() == BINDING_FILES_SHA,
            "Not the reviewed native 468-file binding")
    actual = {p.relative_to(root).as_posix() for p in (root / "engine").rglob("*.py")}
    require(actual == set(files), "Native engine file set differs from binding")
    for rel, expected in files.items():
        path = (root / rel).resolve()
        require(root in path.parents, "Binding path escapes native root")
        value = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        require(value == expected, f"Native source changed: {rel}")
    return files


def load_recipe(path):
    """Native YAML resolves includes; saved args/plan JSON is already merged."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = read_json(path)
        data = data.get("yaml_cfg", data.get("native_baseline_config", data))
    else:
        from engine.core.yaml_utils import load_config
        data = load_config(str(path))
    cfg = copy.deepcopy(data)
    cfg.pop("__include__", None)
    cfg.pop("config", None)
    require(cfg.get("num_classes") == 10 and cfg.get("use_focal_loss") is True,
            "Only the bound 10-class focal baseline is supported")
    require(cfg.get("remap_mscoco_category") is False, "Category remapping is forbidden")
    require(cfg.get("model") == "DEIM" and cfg["DFINETransformer"]["num_queries"] == 300,
            "Unexpected baseline architecture")
    require(cfg["DFINETransformer"]["hidden_dim"] == 256, "Expected h256")
    require(cfg["HGNetv2"].get("pretrained") is False, "Do not download backbone weights")
    require(cfg.get("ram_cache", False) is False, "Native RAM-cache resizing is not allowed")
    val = cfg["val_dataloader"]
    require(not val.get("shuffle", False) and not val.get("drop_last", False),
            "Validation order must be deterministic and complete")
    require(val.get("total_batch_size", val.get("batch_size")) == 4,
            "Preserve the original batch4 inference shape")
    require(val["collate_fn"] == {"type": "BatchImageCollateFunction"},
            "Unexpected validation collate recipe")
    require(val["dataset"]["transforms"] == {
        "type": "Compose", "ops": [
            {"type": "Resize", "size": [640, 640]},
            {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
        ]}, "Validation transformations differ from the reviewed native recipe")
    return cfg


def state_sha(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FinalFeatures:
    """Passive clone of the tensor entering dec_score_head[eval_idx], before LQE."""
    def __init__(self, model):
        self.values = []
        decoder = model.decoder
        self.eval_idx = decoder.eval_idx
        self.handle = decoder.dec_score_head[self.eval_idx].register_forward_pre_hook(self.capture)

    def capture(self, module, inputs):
        self.values.append(inputs[0].detach().clone())

    def reset(self):
        self.values.clear()

    def finish(self, outputs, torch):
        require(len(self.values) == 1, "Final score-head hook must fire exactly once")
        value = self.values[0]
        require(value.dtype == torch.float32 and value.shape[-1] == 256,
                "Features must be native FP32 h256, not a cast of AMP features")
        require(value.shape[:2] == outputs["pred_logits"].shape[:2], "Query feature shape mismatch")
        require(bool(torch.isfinite(value).all()), "Nonfinite query features")
        return value

    def close(self):
        self.handle.remove()


def run_split(args, split, recipe, model, postprocessor, torch, np):
    from engine.core import YAMLConfig
    from engine.deim.postprocessor import scale_boxes_to_original_image
    from torchvision.ops import box_convert

    initial_model_sha = state_sha(model)
    ann = Path(getattr(args, split + "_ann")).resolve()
    images = Path(getattr(args, split + "_images")).resolve()
    expected_sha, expected_count = ANNOTATIONS[split]
    require(sha(ann) == expected_sha, f"{split} annotation SHA differs from binding")
    infos = {int(x["id"]): x for x in read_json(ann)["images"]}
    require(len(infos) == expected_count, f"Unexpected {split} image count")
    directory = Path(args.out).resolve() / split
    directory.mkdir(parents=True, exist_ok=False)
    arrays = directory / "arrays"
    arrays.mkdir()
    cfg_dict = copy.deepcopy(recipe)
    # Never instantiate train_dataloader, optimizer, criterion, scaler or EMA.
    cfg_dict["val_dataloader"]["dataset"]["ann_file"] = str(ann)
    cfg_dict["val_dataloader"]["dataset"]["img_folder"] = str(images)
    # Native load_config checks the suffix; JSON syntax is also valid YAML.
    resolved = directory / "resolved_eval_config.yaml"
    write_json(resolved, cfg_dict)
    loader_cfg = YAMLConfig(str(resolved))
    loader = loader_cfg.val_dataloader
    expected_ids = sorted(infos)
    require(list(loader.dataset.ids) == expected_ids, "Native dataset image order changed")
    limit = min(args.max_images or expected_count, expected_count)
    meta = {
        "schema": "sod_roi12_frozen_native_cache_v1", "split": split,
        "status": "RUNNING", "engineering_subset": args.max_images is not None,
        "requested_max_images": args.max_images, "dataset_images": expected_count,
        "image_root": str(images), "annotation": str(ann), "annotation_sha256": expected_sha,
        "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": CHECKPOINT_SHA,
        "weights": "seed1_epoch132_ema.module", "training_source_root": str(Path(args.native_root).resolve()),
        "source_binding": str(Path(args.source_binding).resolve()),
        "source_binding_sha256": sha(args.source_binding), "native_bound_files": 468,
        "input_config": str(Path(args.config).resolve()), "input_config_sha256": sha(args.config),
        "resolved_config_sha256": sha(resolved), "script_sha256": sha(__file__),
        "validation_transforms": cfg_dict["val_dataloader"]["dataset"]["transforms"],
        "batch_size": 4, "dtype": "float32", "feature_dtype": "float32", "autocast": False,
        "device": str(args.device), "torch": torch.__version__,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "num_top_queries": postprocessor.num_top_queries,
        "score_threshold": postprocessor.score_threshold,
        "feature_source": "pre-forward input to model.decoder.dec_score_head[eval_idx]; before LQE",
        "box_format": "xyxy original-image coordinates; exact native PostProcessor result",
        "evaluation": "Not performed here; predictions.json awaits the bound standard evaluator",
        "images_copied": False, "training": False,
    }
    write_json(directory / "run_metadata.json", meta)
    records, predictions, seen = [], [], []
    capture = None
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for batch_index, (samples, targets) in enumerate(loader):
                require(isinstance(samples, torch.Tensor) and samples.dtype == torch.float32,
                        "Expected native FP32 RGB image batch")
                samples = samples.to(args.device, non_blocking=True)
                sizes = torch.stack([t["orig_size"] for t in targets]).to(args.device)
                # The first real batch additionally proves the passive hook has no effect.
                if capture is None:
                    plain = model(samples)
                    capture = FinalFeatures(model)
                    outputs = model(samples)
                    for key in ("pred_boxes", "pred_logits"):
                        require(torch.equal(plain[key], outputs[key]), f"Hook changes {key}")
                    del plain
                else:
                    capture.reset()
                    outputs = model(samples)
                features = capture.finish(outputs, torch)
                for key in ("pred_boxes", "pred_logits"):
                    require(outputs[key].dtype == torch.float32 and bool(torch.isfinite(outputs[key]).all()),
                            f"Expected finite native FP32 {key}")
                native = postprocessor(outputs, sizes, for_eval=True)
                # IMPORTANT: outputs.pred_logits includes native LQE; do not recompute scores from h.
                scores, flat_index = outputs["pred_logits"].sigmoid().flatten(1).topk(300, dim=-1)
                query_ids, labels = flat_index // 10, flat_index % 10
                all_boxes = scale_boxes_to_original_image(outputs["pred_boxes"], sizes, None)
                chosen_boxes = all_boxes.gather(1, query_ids.unsqueeze(-1).repeat(1, 1, 4))
                for j, target in enumerate(targets):
                    if len(seen) == limit:
                        break
                    image_id = int(target["image_id"].item())
                    require(image_id == expected_ids[len(seen)], "Batch order differs from native dataset")
                    info = infos[image_id]
                    require(sizes[j].tolist() == [info["width"], info["height"]], "Original size changed")
                    keep = scores[j] > postprocessor.score_threshold
                    q = query_ids[j][keep]
                    pred = native[j]
                    for key, reconstructed in (("labels", labels[j][keep]),
                                               ("scores", scores[j][keep]),
                                               ("boxes", chosen_boxes[j][keep])):
                        require(torch.equal(reconstructed, pred[key]), f"Native postprocessor {key} mismatch")
                    image_path = (images / info["file_name"]).resolve()
                    require(images in image_path.parents, "Image path escapes bound image directory")
                    path = arrays / f"{image_id:06d}.npz"
                    np.savez_compressed(path,
                        boxes=pred["boxes"].detach().cpu().numpy(),
                        labels=pred["labels"].detach().cpu().numpy(),
                        scores=pred["scores"].detach().cpu().numpy(),
                        query_ids=q.cpu().numpy(), features=features[j, q].cpu().numpy(),
                        output_index=np.arange(len(q), dtype=np.int64),
                        image_id=np.array(image_id, dtype=np.int64),
                        width=np.array(info["width"], dtype=np.int64),
                        height=np.array(info["height"], dtype=np.int64),
                        file_name=np.array(info["file_name"]))
                    xywh = box_convert(pred["boxes"], "xyxy", "xywh").cpu().tolist()
                    for label, score, box in zip(pred["labels"].tolist(), pred["scores"].tolist(), xywh):
                        predictions.append(dict(image_id=image_id, category_id=label, score=score, bbox=box))
                    records.append({"image_id": image_id, "file_name": info["file_name"],
                        "width": info["width"], "height": info["height"], "image_path": str(image_path),
                        "image_sha256": sha(image_path), "cache_file": path.relative_to(directory).as_posix(),
                        "sha256": sha(path), "bytes": path.stat().st_size, "outputs": len(q)})
                    seen.append(image_id)
                print(json.dumps({"split": split, "images_done": len(seen), "of": limit,
                                  "elapsed_s": round(time.perf_counter() - started, 2)}), flush=True)
                if len(seen) == limit:
                    break
        require(seen == expected_ids[:limit], "Cache coverage incomplete")
        require(sha(ann) == expected_sha and sha(args.checkpoint) == CHECKPOINT_SHA,
                "Bound input changed during caching")
        verify_sources(args.native_root, args.source_binding)
        require(state_sha(model) == initial_model_sha, "Frozen detector state changed")
        write_json(directory / "predictions.json", predictions)
        write_json(directory / "manifest.json", {
            **meta, "status": "ENGINEERING_SUBSET_COMPLETE" if args.max_images is not None else "COMPLETE",
            "images_cached": len(seen), "outputs_cached": len(predictions), "files": records,
            "images": [{key: record[key] for key in
                        ("image_id", "file_name", "width", "height", "cache_file")}
                       for record in records],
            "image_id_scope": "Split-local; cross-split leakage checks must use filenames/image hashes, not IDs",
            "artifact_sha256": {name: sha(directory / name) for name in
                                ("predictions.json", "resolved_eval_config.yaml", "run_metadata.json")},
            "native_postprocessor_exact_all_items": True, "first_batch_hook_outputs_exact": True,
            "model_unchanged": True, "elapsed_s": time.perf_counter() - started})
    except Exception:
        write_json(directory / "failure.json", {"status": "FAILED_PARTIAL_DO_NOT_USE",
                    "images_saved": len(seen), "traceback": traceback.format_exc()})
        raise
    finally:
        if capture is not None:
            capture.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-root", "source-binding", "config", "checkpoint", "out"):
        parser.add_argument("--" + name, required=True)
    for split in ("train", "val"):
        parser.add_argument("--" + split + "-ann")
        parser.add_argument("--" + split + "-images")
    parser.add_argument("--split", choices=("val", "train", "both"), default="val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-images", type=int, help="Engineering subset only; never a full cache")
    args = parser.parse_args(argv)
    require(args.max_images is None or args.max_images > 0, "max-images must be positive")
    splits = ("val", "train") if args.split == "both" else (args.split,)
    for split in splits:
        require(getattr(args, split + "_ann") and getattr(args, split + "_images"),
                f"--{split}-ann and --{split}-images are required")
    require("engine" not in sys.modules, "Start a fresh interpreter for native source binding")
    verify_sources(args.native_root, args.source_binding)
    require(sha(args.checkpoint) == CHECKPOINT_SHA, "Expected bound seed1 epoch132 last.pth")
    sys.path.insert(0, str(Path(args.native_root).resolve()))
    import numpy as np
    import torch
    from engine.core import YAMLConfig
    recipe = load_recipe(args.config)
    # A separate config object loads the model only; loader objects remain split-specific.
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model_recipe_path = out / "model_recipe.yaml"
    if model_recipe_path.exists():
        require(read_json(model_recipe_path) == recipe, "Existing output root belongs to another recipe")
    else:
        write_json(model_recipe_path, recipe)
    config = YAMLConfig(str(model_recipe_path))
    torch.manual_seed(1)
    model = config.model.float().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    require(checkpoint["last_epoch"] == 131, "Expected complete epoch132 checkpoint")
    state = checkpoint["ema"]["module"]
    model.load_state_dict(state, strict=True)
    require(all(torch.equal(v, state[k]) for k, v in model.state_dict().items()), "EMA load differs")
    del checkpoint, state
    model.requires_grad_(False).to(args.device)
    postprocessor = config.postprocessor.to(args.device).eval()
    require(postprocessor.num_top_queries == 300 and postprocessor.num_classes == 10
            and postprocessor.use_focal_loss and not postprocessor.remap_mscoco_category
            and postprocessor.score_threshold == .001, "Native output protocol differs")
    before = state_sha(model)
    for split in splits:
        run_split(args, split, recipe, model, postprocessor, torch, np)
        require(state_sha(model) == before, "Frozen detector state changed")
    print(json.dumps({"status": "CACHE_COMPLETE", "splits": list(splits),
                      "engineering_subset": args.max_images is not None, "model_state_sha256": before}), flush=True)


if __name__ == "__main__":
    main()
