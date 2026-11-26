"""CPU-only, exclusive preparation of saved tensors and deterministic GT owners.

Does not import torch, construct a model, evaluate, or contact a server.
All original records are retained; only explicitly needed tensor blobs exist.
"""
from collections import defaultdict
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
OUTPUT = ROOT / "outputs/sampling_location_intervention_20261011_v1/inputs"
OLD = ROOT / "outputs/decoder_sampling_gpu_replay_20261011_v1/inputs"
OLD_META_SHA = "28e7322b1e969a8f6b09ea0c2bcedb1759b11af6398aa5ce0d6edd8db7ba5cf2"
OLD_ZIP_SHA = "948d1357863f3a1c17e53704d1935973acfff59bf0baf562b3f74b805e61b71a"
PROTOCOL = ROOT / "英文小论文/code/decoder_sampling_geometry_20261011_v5/protocol.json"
PROTOCOL_SHA = "c54261e25dfdc5ec9f030478a5312e8e7e7494703c97fc0672a8f98d118b16a1"
PANEL = ROOT / "英文小论文/code/candidate_all_module_capture_20261010_v1/panel_protocol.json"
PANEL_SHA = "f251e2a4c8ebd4c4737d46a76aa71aede595d346d2e19b011bf60437de9a3fc9"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def jb(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf8")


def checked(data, expected, description):
    if isinstance(expected, dict):
        assert len(data) == expected["bytes"], description
        expected = expected["sha256"]
    assert digest(data) == expected, description
    return data


def write(path, value):
    with path.open("xb") as stream:
        stream.write(jb(value))


def xyxy(box):
    return np.concatenate((box[..., :2] - box[..., 2:] / 2,
                           box[..., :2] + box[..., 2:] / 2), -1)


def iou(gt, pred):
    wh = np.maximum(np.minimum(gt[:, None, 2:], pred[None, :, 2:]) -
                    np.maximum(gt[:, None, :2], pred[None, :, :2]), 0)
    intersection = wh.prod(-1)
    ga = np.maximum(gt[:, 2:] - gt[:, :2], 0).prod(-1)
    pa = np.maximum(pred[:, 2:] - pred[:, :2], 0).prod(-1)
    return intersection / np.maximum(ga[:, None] + pa[None, :] - intersection, 1e-30)


def map_annotations(annotations, image, boxes, labels):
    """Replay source dataset filtering/FP32 box conversion then verify EXACTLY.

    COCO createIndex appends annotations in JSON order, getAnnIds(image) retains
    that order, loadAnns retains IDs order. Dataset then drops iscrowd and invalid
    clipped XYXY boxes. Validation Resize scales x/y independently to 640.
    No cache, masks, category remapping, or additional validation sanitization.
    """
    source = [ann for ann in annotations if ann["image_id"] == image["id"]]
    noncrowd = [ann for ann in source if ann.get("iscrowd", 0) == 0]
    reconstructed = np.asarray([ann["bbox"] for ann in noncrowd], dtype=np.float32).reshape(-1, 4)
    reconstructed[:, 2:] += reconstructed[:, :2]
    reconstructed[:, 0::2] = np.clip(reconstructed[:, 0::2], 0, image["width"])
    reconstructed[:, 1::2] = np.clip(reconstructed[:, 1::2], 0, image["height"])
    keep = (reconstructed[:, 2] > reconstructed[:, 0]) & (reconstructed[:, 3] > reconstructed[:, 1])
    selected = [ann for ann, retained in zip(noncrowd, keep) if retained]
    reconstructed = reconstructed[keep]
    reconstructed *= np.asarray([640 / image["width"], 640 / image["height"],
                                  640 / image["width"], 640 / image["height"]], dtype=np.float32)
    reconstructed_labels = np.asarray([ann["category_id"] for ann in selected], dtype=np.int64)
    assert boxes.dtype == np.float32 and labels.dtype == np.int64
    assert np.array_equal(boxes, reconstructed), ("Native annotation boxes not exact", image["id"])
    assert np.array_equal(labels, reconstructed_labels), ("Native annotation labels not exact", image["id"])
    return selected, dict(
        original_annotation_count=len(source), native_GT_count=len(selected),
        crowd_filtered_annotation_ids=[ann["id"] for ann in source if ann.get("iscrowd", 0) != 0],
        invalid_clipped_box_annotation_ids=[ann["id"] for ann, retained in zip(noncrowd, keep) if not retained],
        native_order_annotation_ids=[ann["id"] for ann in selected],
        native_boxes_and_labels_exactly_reproduced=True,
        source_image=image, source_annotations=source)


def main():
    assert not OUTPUT.exists(), "Output exists; no overwrite or automatic retry."
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir()
    counts = dict(seeds=0, batches=0, images=0, native_GT=0, original_records=0,
                  new_norm1_blobs=0, reused_verified_entries=0)
    try:
        old_meta_raw = checked((OLD / "inputs_metadata.json").read_bytes(), OLD_META_SHA, "Old package metadata")
        old_meta = json.loads(old_meta_raw)
        assert old_meta["status"] == "PASS_VERIFIED_CPU_INPUT_PACKAGE"
        assert (OLD / "inputs.zip").stat().st_size == old_meta["archive_bytes"] == 157647925
        assert sha(OLD / "inputs.zip") == old_meta["archive_sha256"] == OLD_ZIP_SHA
        protocol_raw = checked(PROTOCOL.read_bytes(), PROTOCOL_SHA, "Source v5 protocol")
        protocol = json.loads(protocol_raw)
        assert protocol["schema"] == "saved_native_decoder_sampling_geometry_v5"
        panel_raw = checked(PANEL.read_bytes(), PANEL_SHA, "Source panel protocol")
        panel = json.loads(panel_raw)
        assert panel["annotation_sha256"] == "0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd"
        owners = dict(schema="sampling_location_single_query_owners_v1", seeds={},
            selection="All native input short-side <4px GT with pre best IoU>0; lowest query ID on exact IoU ties; lowest annotation ID owns a shared query within an image.",
            selected_using_final_outputs=False, pre_IoU_arithmetic="Original FP32 pre cxcywh converted to XYXY, then mixed with FP64 native GT/640, identical to source v5 reader.",
            delta_arithmetic="FP64 native GT center/640 minus original FP32 encoder reference center promoted to FP64.")
        bindings = {}
        entries = {}
        archive = OUTPUT / "inputs.zip"
        with zipfile.ZipFile(OLD / "inputs.zip") as old_zip, zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as out:
            old_manifest_raw = checked(old_zip.read("manifest.json"), old_meta["manifest_sha256"], "Old root manifest")
            old_manifest = json.loads(old_manifest_raw)
            assert len(old_zip.namelist()) == len(set(old_zip.namelist()))
            assert set(old_zip.namelist()) == set(old_manifest["files"]) | {"manifest.json"}
            assert old_manifest["source_sources"] == protocol["sources"]

            def put(name, data):
                assert not name.startswith("/") and ".." not in Path(name).parts
                item = dict(sha256=digest(data), bytes=len(data))
                if name in entries:
                    assert entries[name] == item, name
                    return False
                out.writestr(name, data)
                entries[name] = item
                return True

            # Every reused entry is checked again, not just the zip checksum.
            for name, item in old_manifest["files"].items():
                put(name, checked(old_zip.read(name), item, "Reused input: " + name))
                counts["reused_verified_entries"] += 1
            put("source_protocol_v5.json", protocol_raw)
            put("source_panel_protocol.json", panel_raw)
            put("provenance/previous_inputs_metadata.json", old_meta_raw)
            put("provenance/previous_inputs_manifest.json", old_manifest_raw)
            for seed in protocol["seeds"]:
                binding = protocol["sources"][str(seed)]
                terminal = ROOT / f"outputs/scale_protection12_20261010_v1/terminal_collection/seed{seed}_target_protect_v1/extracted"
                download = ROOT / f"outputs/candidate_all_module_capture_20261010_v1/seed{seed}_download_v1"
                terminal_manifest_raw = checked((terminal / "manifest.json").read_bytes(), old_manifest["verified_sources"][str(seed)]["terminal_manifest_sha256"], "Terminal manifest")
                tm = json.loads(terminal_manifest_raw)["files"]
                plan_raw = checked((terminal / "prepared/plan.json").read_bytes(), tm["prepared/plan.json"], "Original recipe")
                config = json.loads(plan_raw)["historical_resolved_config"]
                val = config["val_dataloader"]
                assert config["remap_mscoco_category"] is False and config["ram_cache"] is False
                assert val["dataset"]["return_masks"] is False and val["shuffle"] is False
                assert val["dataset"]["transforms"]["ops"] == [dict(type="Resize", size=[640, 640]), dict(type="ConvertPILImage", dtype="float32", scale=True)]
                put(f"seed{seed}/mapping_evidence/prepared_plan.json", plan_raw)
                for name in ("native/engine/data/dataset/coco_dataset.py", "native/engine/data/transforms/_transforms.py", "evaluator/pycocoeval/coco.py"):
                    source_raw = checked((terminal / name).read_bytes(), tm[name], name)
                    put("mapping_evidence/" + name, source_raw)
                annotations_raw = checked((terminal / "reference/val_annotations.json").read_bytes(), panel["annotation_sha256"], "Annotations")
                annotations = json.loads(annotations_raw)
                assert len({a["id"] for a in annotations["annotations"]}) == len(annotations["annotations"])
                images = {im["id"]: im for im in annotations["images"]}
                checkpoint = terminal / "run/last.pth"
                assert sha(checkpoint) == binding["checkpoint_sha256"], "Checkpoint entire-file SHA"
                raw_report = checked((download / "raw_verification_report_v1.json").read_bytes(), binding["raw_report_sha256"], "Raw verification")
                assert json.loads(raw_report)["status"] == "PASS_DOWNLOADED_ALL_CALLED_MODULE_RAW_OUTPUTS"
                raw_metadata = checked((download / f"module_candidate_panel_seed{seed}_raw_v1_metadata.json").read_bytes(), binding["metadata_sha256"], "Raw metadata")
                raw_zip_path = download / f"module_candidate_panel_seed{seed}_raw_v1.zip"
                assert raw_zip_path.stat().st_size == binding["archive_bytes"] == json.loads(raw_metadata)["bytes"]
                assert json.loads(raw_metadata)["sha256"] == binding["archive_sha256"]
                put(f"seed{seed}/provenance/raw_verification.json", raw_report)
                put(f"seed{seed}/provenance/raw_metadata.json", raw_metadata)
                seed_owners, zero, conflicts, tie_rows, image_audits = [], [], [], [], []
                with zipfile.ZipFile(raw_zip_path) as raw_zip:
                    cap_raw = checked(raw_zip.read("capture/manifest.json"), binding["capture_manifest_sha256"], "Raw capture manifest")
                    assert cap_raw == old_zip.read(f"seed{seed}/capture/manifest.json")
                    cap = json.loads(cap_raw)
                    for batch in protocol["native_batches"]:
                        relative = f"batch_{batch:03d}/"
                        prefix = f"seed{seed}/capture/" + relative
                        original_prefix = "capture/" + relative
                        rec_raw = checked(raw_zip.read(original_prefix + "record.json"), cap["files"][relative + "record.json"], "Original full record")
                        assert rec_raw == old_zip.read(prefix + "record.json")
                        record = json.loads(rec_raw)
                        assert record["metadata"]["checkpoint_sha256"] == binding["checkpoint_sha256"]
                        assert record["metadata"]["seed"] == seed
                        norm = [e for e in record["invocation_events"] if e["module"] == "decoder.decoder.layers.0.norm1" and e["invocation"] == 0]
                        assert len(norm) == 1
                        desc = norm[0]["output"]
                        assert desc["shape"] == [4, 300, 256] and desc["storage_numpy_dtype"] == "float32"
                        norm_raw = checked(raw_zip.read(original_prefix + desc["blob_file"]), desc["blob_sha256"], "norm1 descriptor")
                        checked(norm_raw, cap["files"][relative + desc["blob_file"]], "norm1 manifest")
                        norm_array = np.load(io.BytesIO(norm_raw), allow_pickle=False)
                        assert norm_array.shape == (4, 300, 256) and norm_array.dtype == np.float32 and np.isfinite(norm_array).all()
                        assert put(prefix + desc["blob_file"], norm_raw)
                        counts["new_norm1_blobs"] += 1
                        for off, iid in enumerate(record["metadata"]["image_ids"]):
                            ds = {e["key"]["value"]: e["value"] for e in record["native_targets"]["items"][off]["entries"]}
                            def tensor(d):
                                data = checked(old_zip.read(prefix + d["blob_file"]), d["blob_sha256"], "GT descriptor")
                                array = np.load(io.BytesIO(data), allow_pickle=False)
                                assert list(array.shape) == d["shape"] and str(array.dtype) == d["storage_numpy_dtype"]
                                return array
                            boxes, labels = tensor(ds["boxes"]), tensor(ds["labels"])
                            selected, mapping = map_annotations(annotations["annotations"], images[iid], boxes, labels)
                            mapping.update(seed=seed, image_id=iid, batch=batch, image_offset=off,
                                           native_boxes_blob_sha256=ds["boxes"]["blob_sha256"], native_labels_blob_sha256=ds["labels"]["blob_sha256"])
                            with np.load(io.BytesIO(old_zip.read(f"seed{seed}/stages/image_{iid:06d}.npz")), allow_pickle=False) as stage:
                                pre = stage["pre_after_first_layer_boxes"].astype(np.float32)
                                ref = stage["encoder_initial_boxes"].astype(np.float32)
                            gt = boxes.astype(np.float64)
                            overlaps = iou(gt / 640, xyxy(pre))
                            queries, maxima = overlaps.argmax(-1), overlaps.max(-1)
                            tiny = np.min(gt[:, 2:] - gt[:, :2], axis=-1) < 4
                            candidates = []
                            for g in np.flatnonzero(tiny):
                                identity = dict(seed=seed, batch=batch, image_offset=off, image_id=iid,
                                                GT_native_index=int(g), annotation_id=int(selected[g]["id"]))
                                if maxima[g] == 0:
                                    zero.append(dict(identity, pre_IoU=0.0))
                                    continue
                                q = int(queries[g])
                                tied = np.flatnonzero(overlaps[g] == maxima[g])
                                assert q == int(tied.min())
                                if len(tied) > 1:
                                    tie_rows.append(dict(identity, query=q, tied_query_ids=tied.tolist()))
                                center = (gt[g, :2] + gt[g, 2:]) / 2 / 640
                                encoder_center = ref[q, :2].astype(np.float64)
                                candidates.append(dict(identity, query=q, pre_IoU=float(maxima[g]),
                                    GT_center_normalized=center.tolist(), encoder_reference_center=encoder_center.tolist(),
                                    delta=(center - encoder_center).tolist()))
                            grouped = defaultdict(list)
                            for row in candidates:
                                grouped[row["query"]].append(row)
                            for q in sorted(grouped):
                                group = sorted(grouped[q], key=lambda row: row["annotation_id"])
                                seed_owners.append(group[0])
                                if len(group) > 1:
                                    conflicts.append(dict(seed=seed, image_id=iid, query=q,
                                        owner_annotation_id=group[0]["annotation_id"], owner_GT_native_index=group[0]["GT_native_index"],
                                        excluded_annotation_ids=[r["annotation_id"] for r in group[1:]],
                                        excluded_GT_native_indices=[r["GT_native_index"] for r in group[1:]]))
                            mapping.update(under4_native_GT=int(tiny.sum()), positive_pre_IoU_under4_GT=len(candidates),
                                           zero_pre_IoU_under4_GT=int(tiny.sum()) - len(candidates), unique_owners=len(grouped))
                            image_audits.append(mapping)
                            counts["images"] += 1
                            counts["native_GT"] += len(gt)
                        counts["batches"] += 1
                        counts["original_records"] += 1
                excluded_conflict_count = sum(len(row["excluded_annotation_ids"]) for row in conflicts)
                assert len({(r["image_id"], r["query"]) for r in seed_owners}) == len(seed_owners)
                assert sum(r["under4_native_GT"] for r in image_audits) == 134
                assert len(seed_owners) + excluded_conflict_count + len(zero) == 134
                owners["seeds"][str(seed)] = dict(owners=seed_owners, unique_owner_count=len(seed_owners),
                    native_GT_count=sum(r["native_GT_count"] for r in image_audits), under4_native_GT_count=134,
                    zero_pre_IoU_excluded_count=len(zero), zero_pre_IoU_excluded=zero,
                    conflict_excluded_count=excluded_conflict_count, shared_query_conflicts=conflicts,
                    exact_pre_IoU_ties=tie_rows, image_mapping_audits=image_audits)
                bindings[str(seed)] = dict(checkpoint_sha256=binding["checkpoint_sha256"],
                    checkpoint_full_file_hash_checked=True, checkpoint_bytes=checkpoint.stat().st_size,
                    raw_archive_sha256=binding["archive_sha256"], raw_archive_bytes=binding["archive_bytes"],
                    raw_archive_whole_hash_reused_from_verified_input_package=True,
                    raw_selected_records_manifest_and_new_blobs_hash_checked=True,
                    terminal_manifest_sha256=digest(terminal_manifest_raw), annotation_sha256=panel["annotation_sha256"])
                counts["seeds"] += 1
                print(json.dumps(dict(seed=seed, unique_owners=len(seed_owners), zero_IoU_excluded=len(zero), conflict_excluded=excluded_conflict_count, native_GT=960)), flush=True)
            assert [owners["seeds"][str(s)]["unique_owner_count"] for s in [1, 2]] == [88, 85]
            owners_raw = jb(owners)
            put("owners.json", owners_raw)
            assert counts["images"] == 16 and counts["native_GT"] == 1920 and counts["new_norm1_blobs"] == 4
            manifest = dict(schema="sampling_location_intervention_inputs_v1", status="VERIFIED_CPU_INPUTS_WITH_DETERMINISTIC_NATIVE_GT_OWNERS",
                files=entries, source_protocol_sha256=PROTOCOL_SHA, source_panel_protocol_sha256=PANEL_SHA,
                source_sources=protocol["sources"], source_native_hashes=old_manifest["source_native_hashes"],
                verified_sources=bindings, read_counts=counts, prepare_script_sha256=sha(__file__),
                reused_package_sha256=OLD_ZIP_SHA, reused_manifest_sha256=old_meta["manifest_sha256"],
                source_arm="target_protect", source_epoch_zero_based=11, new_model_forwards=0,
                new_evaluator_calls=0, optimizer_updates=0, epochs_added=0,
                subset_note="Original full records retained. Only required tensor outputs are materialized. Original GT boxes/labels cover every GT, including nonowners and other scales.")
            manifest_raw = jb(manifest)
            out.writestr("manifest.json", manifest_raw)
        with zipfile.ZipFile(archive) as written:
            assert set(written.namelist()) == set(entries) | {"manifest.json"}
            assert len(written.namelist()) == len(entries) + 1
            assert written.read("manifest.json") == manifest_raw
            for name, item in entries.items():
                checked(written.read(name), item, "Readback: " + name)
        assert "torch" not in sys.modules and not any(s.startswith("pycocoeval") for s in sys.modules)
        write(OUTPUT / "owners.json", owners)
        metadata = dict(status="PASS_VERIFIED_CPU_INPUT_PACKAGE_WITH_NATIVE_GT_OWNERS", archive=str(archive),
            archive_bytes=archive.stat().st_size, archive_sha256=sha(archive), manifest_sha256=digest(manifest_raw),
            owners_sha256=digest(owners_raw), prepare_script_sha256=sha(__file__), files=len(entries),
            created_local_utc=datetime.now(timezone.utc).isoformat(), read_counts=counts,
            owner_counts={s: r["unique_owner_count"] for s, r in owners["seeds"].items()},
            every_output_entry_readback_verified=True, new_model_forwards=0, new_evaluator_calls=0,
            optimizer_updates=0, epochs_added=0)
        write(OUTPUT / "inputs_metadata.json", metadata)
        print(json.dumps(metadata, ensure_ascii=False), flush=True)
    except BaseException:
        write(OUTPUT / "failure.json", dict(status="CPU_INPUT_PREPARATION_FAILED", traceback=traceback.format_exc(),
            read_counts=counts, new_model_forwards=0, new_evaluator_calls=0, optimizer_updates=0, epochs_added=0))
        raise


if __name__ == "__main__":
    main()
