"""Prepare verified saved target-policy tensors for an arithmetic-only replay.

This script never imports torch, constructs a model, evaluates a detector, or
opens a network connection. It uses exclusive creation for every output.
"""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import traceback
import zipfile

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = ROOT / "outputs/decoder_sampling_gpu_replay_20261011_v1/inputs"
PROTOCOL = ROOT / "英文小论文/code/decoder_sampling_geometry_20261011_v4/protocol.json"
PROTOCOL_SHA = "b94d3ca50170d901c50739188e5a69c2001f04d1eb5b6739e9cf56e43c5552c7"
BOX_OPS_SHA = "398fc107fc003b0fabaf36154594d50a8e0aa54195056b343031960c0aff0975"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf8")


def write_json(path, value):
    with path.open("xb") as stream:
        stream.write(json_bytes(value))


def checked(data, expected, description):
    if isinstance(expected, str):
        assert digest(data) == expected, description
    else:
        assert len(data) == expected["bytes"], description
        assert digest(data) == expected["sha256"], description
    return data


def main():
    assert not OUTPUT.exists(), "Output already exists; no overwrite or automatic retry."
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir()
    counts = dict(seeds=0, batches=0, original_records=0,
                  tensor_descriptors=0, unique_tensor_blobs=0,
                  stage_files=0, GT_boxes=0, native_source_files=0)
    try:
        protocol_raw = checked(PROTOCOL.read_bytes(), PROTOCOL_SHA, "v4 protocol")
        p = json.loads(protocol_raw)
        assert p["schema"] == "saved_native_decoder_sampling_geometry_v4"
        assert p["source_models_are_previous_target_policy_not_new_FDR"] is True
        assert p["arm"] == "target_protect" and p["epoch_zero_based"] == 11
        assert p["points_per_level"] == [3, 6, 3]
        archive = OUTPUT / "inputs.zip"
        entries = {}
        verified_sources = {}
        native_hashes = dict(p["native_source_sha256"])
        native_hashes["engine/deim/box_ops.py"] = BOX_OPS_SHA
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=3, allowZip64=True) as out:
            def put(name, data):
                assert not name.startswith("/") and ".." not in Path(name).parts
                metadata = dict(sha256=digest(data), bytes=len(data))
                if name in entries:
                    assert entries[name] == metadata, name
                    return False
                out.writestr(name, data)
                entries[name] = metadata
                return True

            put("source_protocol_v4.json", protocol_raw)
            for seed in p["seeds"]:
                binding = p["sources"][str(seed)]
                old = ROOT / f"outputs/scale_protection12_20261010_v1/terminal_collection/seed{seed}_target_protect_v1"
                download = ROOT / f"outputs/candidate_all_module_capture_20261010_v1/seed{seed}_download_v1"
                report_path = old / "verification_report_v1.json"
                checked(report_path.read_bytes(), binding["target_report_sha256"], "target report")
                report = read(report_path)
                assert report["status"] == "PASS_FIXED12_DOWNLOADED_EVIDENCE_AND_SAVED_WEIGHT_RELOAD"
                raw_report_path = download / "raw_verification_report_v1.json"
                checked(raw_report_path.read_bytes(), binding["raw_report_sha256"], "raw report")
                raw_report = read(raw_report_path)
                assert raw_report["status"] == "PASS_DOWNLOADED_ALL_CALLED_MODULE_RAW_OUTPUTS"
                assert raw_report["seed"] == seed and raw_report["original_capture_exit_code"] == 0
                raw_metadata_path = download / f"module_candidate_panel_seed{seed}_raw_v1_metadata.json"
                checked(raw_metadata_path.read_bytes(), binding["metadata_sha256"], "raw metadata")
                raw_metadata = read(raw_metadata_path)
                raw_archive = download / f"module_candidate_panel_seed{seed}_raw_v1.zip"
                assert raw_archive.stat().st_size == binding["archive_bytes"] == raw_metadata["bytes"]
                assert sha(raw_archive) == binding["archive_sha256"] == raw_metadata["sha256"] == raw_report["archive_sha256"]

                # Bind each stage/native manifest to the already verified terminal
                # archive, rather than trusting loose local manifest files.
                terminal_archive = old / f"terminal_target_protect_seed{seed}_v1.zip"
                terminal_metadata = read(old / f"terminal_target_protect_seed{seed}_v1_metadata.json")
                assert terminal_archive.stat().st_size == terminal_metadata["bytes"]
                assert sha(terminal_archive) == report["archive_sha256"] == terminal_metadata["sha256"]
                with zipfile.ZipFile(terminal_archive) as terminal:
                    terminal_manifest_raw = terminal.read("manifest.json")
                assert terminal_manifest_raw == (old / "extracted/manifest.json").read_bytes()
                terminal_manifest = json.loads(terminal_manifest_raw)
                assert terminal_manifest["seed"] == seed and terminal_manifest["arm"] == "target_protect"
                tm = terminal_manifest["files"]
                stage_manifest_path = old / "extracted/run/stages_epoch12/manifest.json"
                stage_manifest_raw = checked(stage_manifest_path.read_bytes(), tm["run/stages_epoch12/manifest.json"], "stage manifest")
                stage_manifest = json.loads(stage_manifest_raw)
                put(f"seed{seed}/stages/manifest.json", stage_manifest_raw)
                plan_raw = checked((old / "extracted/prepared/plan.json").read_bytes(), tm["prepared/plan.json"], "recipe")
                recipe = json.loads(plan_raw)["model_recipe"]["DFINETransformer"]
                assert recipe["num_points"] == p["points_per_level"] and recipe["num_layers"] == 3
                for relative, expected_sha in native_hashes.items():
                    name = "native/" + relative
                    data = checked((old / "extracted" / name).read_bytes(), tm[name], name)
                    checked(data, expected_sha, name)
                    if put(name, data):
                        counts["native_source_files"] += 1
                verified_sources[str(seed)] = dict(
                    terminal_archive_sha256=report["archive_sha256"],
                    terminal_archive_bytes=terminal_archive.stat().st_size,
                    terminal_manifest_sha256=digest(terminal_manifest_raw),
                    stage_manifest_sha256=digest(stage_manifest_raw),
                    prepared_plan_sha256=digest(plan_raw),
                    raw_archive_sha256=binding["archive_sha256"],
                    raw_archive_bytes=binding["archive_bytes"],
                    selected_batches=[])
                seen_ids = []
                with zipfile.ZipFile(raw_archive) as raw_zip:
                    capture_manifest_raw = checked(raw_zip.read("capture/manifest.json"), binding["capture_manifest_sha256"], "capture manifest")
                    capture_manifest = json.loads(capture_manifest_raw)
                    put(f"seed{seed}/capture/manifest.json", capture_manifest_raw)
                    for batch in p["native_batches"]:
                        sub = f"batch_{batch:03d}/"
                        prefix = "capture/" + sub
                        record_raw = checked(raw_zip.read(prefix + "record.json"), capture_manifest["files"][sub + "record.json"], "record")
                        record = json.loads(record_raw)
                        metadata = record["metadata"]
                        assert metadata["seed"] == seed and metadata["arm"] == "target_protect"
                        assert metadata["checkpoint_sha256"] == binding["checkpoint_sha256"]
                        ids = metadata["image_ids"]
                        assert len(ids) == 4 and ids == list(range(batch * 4, batch * 4 + 4))
                        seen_ids.extend(ids)
                        put(f"seed{seed}/" + prefix + "record.json", record_raw)
                        counts["original_records"] += 1
                        descriptors = []

                        def event(name, invocation=0):
                            found = [e for e in record["invocation_events"] if e["module"] == name and e["invocation"] == invocation]
                            assert len(found) == 1, (seed, batch, name, invocation)
                            descriptors.append(found[0]["output"])

                        for level in range(3):
                            event(f"decoder.input_proj.{level}")
                            event("decoder.integral", level)
                        for layer in range(3):
                            name = f"decoder.decoder.layers.{layer}.cross_attn"
                            for suffix in (".sampling_offsets", ".attention_weights", ""):
                                event(name + suffix)
                        assert len(descriptors) == 15
                        for offset, iid in enumerate(ids):
                            stage_name = f"image_{iid:06d}.npz"
                            stage_path = old / "extracted/run/stages_epoch12" / stage_name
                            stage_raw = checked(stage_path.read_bytes(), stage_manifest["files"][stage_name], "stage")
                            checked(stage_raw, tm["run/stages_epoch12/" + stage_name], "terminal stage")
                            with np.load(io.BytesIO(stage_raw), allow_pickle=False) as stage:
                                for key in ("encoder_initial_boxes", "pre_after_first_layer_boxes", "final_boxes"):
                                    assert stage[key].shape == (300, 4) and np.isfinite(stage[key]).all()
                            put(f"seed{seed}/stages/" + stage_name, stage_raw)
                            counts["stage_files"] += 1
                            targets = {x["key"]["value"]: x["value"] for x in record["native_targets"]["items"][offset]["entries"]}
                            descriptors.extend([targets["boxes"], targets["labels"]])
                            assert targets["boxes"]["shape"][1:] == [4]
                            assert targets["labels"]["shape"] == targets["boxes"]["shape"][:1]
                            counts["GT_boxes"] += targets["boxes"]["shape"][0]
                        assert len(descriptors) == 23
                        for desc in descriptors:
                            assert desc["kind"] == "tensor"
                            blob_name = prefix + desc["blob_file"]
                            blob = checked(raw_zip.read(blob_name), desc["blob_sha256"], blob_name)
                            checked(blob, capture_manifest["files"][sub + desc["blob_file"]], blob_name)
                            value = np.load(io.BytesIO(blob), allow_pickle=False)
                            assert list(value.shape) == desc["shape"] and str(value.dtype) == desc["storage_numpy_dtype"]
                            assert np.isfinite(value).all()
                            if put(f"seed{seed}/" + blob_name, blob):
                                counts["unique_tensor_blobs"] += 1
                            counts["tensor_descriptors"] += 1
                        counts["batches"] += 1
                        verified_sources[str(seed)]["selected_batches"].append(dict(batch=batch, image_ids=ids, tensor_descriptors=len(descriptors), record_sha256=digest(record_raw)))
                assert seen_ids == p["image_ids"]
                counts["seeds"] += 1
                print(json.dumps(dict(status="SEED_INPUTS_VERIFIED", seed=seed,
                                      selected_images=len(seen_ids)), ensure_ascii=False), flush=True)
            assert counts["seeds"] == 2 and counts["batches"] == 4
            assert counts["stage_files"] == 16 and counts["GT_boxes"] == 1920
            manifest = dict(schema="saved_decoder_sampling_gpu_replay_inputs_v1",
                            status="VERIFIED_SAVED_TARGET_POLICY_TENSOR_INPUTS",
                            files=entries, source_sources=p["sources"],
                            source_protocol_sha256=PROTOCOL_SHA,
                            source_native_hashes=native_hashes,
                            verified_sources=verified_sources,
                            read_counts=counts, prepare_script_sha256=sha(__file__),
                            source_arm="target_protect", source_epoch_zero_based=11,
                            new_model_forwards=0, new_evaluator_calls=0,
                            optimizer_updates=0, epochs_added=0,
                            retained_records_are_original_full_records=True,
                            subset_note="Only the specified replay tensors and native boxes/labels are included; other descriptors in the original records are intentionally not materialized.")
            manifest_raw = json_bytes(manifest)
            out.writestr("manifest.json", manifest_raw)
        # Read back every output entry before offering this package for upload.
        with zipfile.ZipFile(archive) as written:
            assert set(written.namelist()) == set(entries) | {"manifest.json"}
            assert len(written.namelist()) == len(entries) + 1
            assert written.read("manifest.json") == manifest_raw
            for name, item in entries.items():
                checked(written.read(name), item, "output readback: " + name)
        assert "torch" not in sys.modules
        assert not any(name.startswith("pycocoeval") for name in sys.modules)
        metadata = dict(status="PASS_VERIFIED_CPU_INPUT_PACKAGE", archive=str(archive),
                        archive_bytes=archive.stat().st_size, archive_sha256=sha(archive),
                        manifest_sha256=digest(manifest_raw),
                        prepare_script_sha256=sha(__file__),
                        created_local_utc=datetime.now(timezone.utc).isoformat(),
                        files=len(entries), read_counts=counts,
                        new_model_forwards=0, new_evaluator_calls=0,
                        optimizer_updates=0, epochs_added=0,
                        every_output_entry_readback_verified=True)
        write_json(OUTPUT / "inputs_metadata.json", metadata)
        print(json.dumps(metadata, ensure_ascii=False), flush=True)
    except BaseException:
        write_json(OUTPUT / "failure.json", dict(status="INPUT_PREPARATION_FAILED",
                   traceback=traceback.format_exc(), read_counts=counts,
                   new_model_forwards=0, new_evaluator_calls=0,
                   optimizer_updates=0, epochs_added=0))
        raise


if __name__ == "__main__":
    main()
