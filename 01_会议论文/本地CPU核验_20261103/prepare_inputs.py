"""Freeze local inputs and proposed CPU settings; never import or run a model.

Standard library only. This preparation is not experimental authorization.
Existing files may be reused only when their bytes match.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path


BUNDLE = Path(__file__).resolve().parent
PAPER = BUNDLE.parent
AUDIT = PAPER / "修订记录_20261103"
SOURCE = Path(r"C:\Users\jiwei\Desktop\DEIM")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_bytes(path, content):
    if not path.resolve().is_relative_to(BUNDLE):
        raise ValueError("Destination must remain inside this preparation bundle")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Refusing to replace a differing file: {path.name}")
    else:
        path.write_bytes(content)


def save_json(path, value):
    save_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def main():
    proposal_path = AUDIT / "proposed_reevaluation.json"
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    if proposal["status"] != "PROPOSED_NOT_AUTHORIZED_NOT_RUN":
        raise ValueError("This preparation script is for the unapproved proposal only")

    # Reuse already archived source evidence, without silently accepting changes.
    prior_manifest = json.loads((AUDIT / "source_manifest.json").read_text(encoding="utf-8"))
    for entry in prior_manifest:
        if sha256(Path(entry["source"])) != entry["sha256"]:
            raise ValueError("An original audited source file has changed")
        if sha256(Path(entry["snapshot"])) != entry["sha256"]:
            raise ValueError("An audit snapshot has changed")

    sources = sorted((SOURCE / "engine").rglob("*.py"))
    sources += [SOURCE / name for name in ("train.py", "requirements.txt", "AGENTS.md")]
    source_manifest = []
    for path in sources:
        resolved = path.resolve()
        if path.is_symlink() or not resolved.is_relative_to(SOURCE.resolve()):
            raise ValueError("Unexpected source link or path outside the original repository")
        relative = path.relative_to(SOURCE)
        destination = BUNDLE / "source_snapshot" / relative
        save_bytes(destination, path.read_bytes())
        source_manifest.append({
            "original": str(path),
            "snapshot": destination.relative_to(BUNDLE).as_posix(),
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
        })
    save_json(BUNDLE / "source_manifest.json", source_manifest)

    annotation = Path(proposal["validation_annotation"])
    if sha256(annotation) != proposal["validation_annotation_sha256"]:
        raise ValueError("Validation annotation hash differs from the audited proposal")
    annotation_data = json.loads(annotation.read_text(encoding="utf-8"))
    image_root = annotation.parent.parent / "images"
    images = annotation_data["images"]
    if len(images) != proposal["image_count"] or len({x["id"] for x in images}) != len(images):
        raise ValueError("Unexpected validation image count or duplicate image IDs")
    image_manifest = []
    for item in sorted(images, key=lambda row: row["id"]):
        path = image_root / item["file_name"]
        if not path.resolve().is_relative_to(image_root.resolve()) or not path.is_file():
            raise ValueError("Missing image or image path outside the validation directory")
        image_manifest.append({
            "image_id": item["id"], "file_name": item["file_name"],
            "width": item["width"], "height": item["height"],
            "bytes": path.stat().st_size, "sha256": sha256(path),
        })
    save_json(BUNDLE / "validation_image_manifest.json", image_manifest)

    prepared = []
    for item in proposal["minimum_scope"]:
        run_id = item["run_id"]
        original_args = Path(item["configuration"])
        args = json.loads(original_args.read_text(encoding="utf-8"))
        original = args["yaml_cfg"]
        cfg = deepcopy(original)
        checkpoint = Path(item["checkpoint"])
        if sha256(checkpoint) != item["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed: {run_id}")
        if not cfg["use_ema"]:
            raise ValueError("Expected the archived EMA validation recipe")
        backbone = cfg["DEIM"]["backbone"]
        if cfg[backbone].get("pretrained", False):
            raise ValueError("Unexpected pretrained download setting")

        changes = []

        def replace(keys, value):
            target = cfg
            for key in keys[:-1]:
                target = target[key]
            changes.append({"path": ".".join(keys), "before": target.get(keys[-1]), "after": value})
            target[keys[-1]] = value

        if "__include__" in cfg:
            changes.append({"path": "__include__", "before": cfg.pop("__include__"),
                            "after": "REMOVED: args.json already contains resolved settings"})
        replace(["device"], "cpu")
        replace(["use_amp"], False)
        replace(["resume"], str(checkpoint))
        replace(["tuning"], None)
        replace(["output_dir"], str(BUNDLE / "results" / run_id))
        replace(["val_dataloader", "num_workers"], 0)
        replace(["val_dataloader", "pin_memory"], False)
        replace(["val_dataloader", "dataset", "img_folder"], str(image_root))
        replace(["val_dataloader", "dataset", "ann_file"], str(annotation))

        # Preserve architecture, postprocessing, image transforms, and batch size.
        for key in ("DEIM", backbone, "HybridEncoder", "DFINETransformer", "PostProcessor"):
            if cfg.get(key) != original.get(key):
                raise ValueError(f"Unexpected change in {key}")
        for key in ("total_batch_size", "shuffle", "drop_last", "collate_fn"):
            if cfg["val_dataloader"][key] != original["val_dataloader"][key]:
                raise ValueError(f"Unexpected validation change in {key}")
        if cfg["val_dataloader"]["dataset"]["transforms"] != original["val_dataloader"]["dataset"]["transforms"]:
            raise ValueError("Validation image transforms changed")

        config_path = BUNDLE / "configs" / f"{run_id}.cpu_evaluation.json"
        save_json(config_path, cfg)
        save_bytes(BUNDLE / "configs" / f"{run_id}.retained_args.json", original_args.read_bytes())
        prepared.append({
            **item,
            "retained_args_sha256": sha256(original_args),
            "prepared_configuration": config_path.relative_to(BUNDLE).as_posix(),
            "prepared_configuration_sha256": sha256(config_path),
            "validation_batch_size": cfg["val_dataloader"]["total_batch_size"],
            "overrides": changes,
        })

    manifest = {
        "status": "STATIC_PREPARATION_ONLY_NOT_AUTHORIZED_NOT_RUN",
        "proposal_sha256": sha256(proposal_path),
        "prepared_runs": prepared,
        "source_file_count": len(source_manifest),
        "source_manifest_sha256": sha256(BUNDLE / "source_manifest.json"),
        "image_manifest_sha256": sha256(BUNDLE / "validation_image_manifest.json"),
        "validation_annotation": str(annotation),
        "validation_annotation_sha256": sha256(annotation),
        "validation_image_count": len(images),
        "verification_performed": "File existence, SHA-256, JSON parsing, and configuration comparisons only",
        "environment_created": False,
        "dependencies_installed": False,
        "checkpoint_deserialized_by_torch": False,
        "model_constructed": False,
        "inference_run": False,
        "evaluation_run": False,
        "runner_ready": False,
        "remaining_before_execution": [
            "Explicit user authorization for local CPU evaluation and isolated environment setup",
            "Resolve available local dependency packages; package downloads need separate clarification if no remote access is required",
            "Prepare an evaluation-only entry point without deployment fusion, profiling, timing, TIDE, or training",
            "Register only the selected model/data modules to avoid unrelated experimental CUDA imports",
            "Require strict loading of checkpoint['ema']['module']; do not fall back to unaveraged model weights",
            "Record runtime versions and verify all input hashes again before the authorized inference pass",
        ],
        "scope_limits": proposal["not_authorized"],
        "interpretation_limit": proposal["limits"],
    }
    save_json(BUNDLE / "preparation_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "source_files": len(source_manifest),
                      "validation_images": len(images), "run_ids": [x["run_id"] for x in prepared]}))


if __name__ == "__main__":
    main()
