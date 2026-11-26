"""Verify and describe one downloaded, frozen sampling-location intervention.

Pure CPU NumPy arithmetic. No model/evaluator/torch imports or network access.
The --destination extraction directory and --output analysis directory must both
be new. The local protocol.json must already bind this script before execution.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import traceback
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INPUTS = ROOT / "outputs/sampling_location_intervention_20261011_v1/inputs/inputs.zip"
ARMS = ("T", "Cplus", "Cminus")
GROUPS = ("under4", "4to8", "8to16", "ge16", "all")
THRESHOLDS = (0.5, 0.75)


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


def write(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def checked(data, expected, description):
    if isinstance(expected, dict):
        assert len(data) == expected["bytes"], description
        expected = expected["sha256"]
    assert digest(data) == expected, description
    return data


def zip_names(z):
    names = z.namelist()
    assert len(names) == len(set(names)), "Duplicate zip entries"
    assert len(names) == len({name.casefold() for name in names}), "Case-insensitive zip path collision"
    reserved = {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)}
    for item in z.infolist():
        name = item.filename
        parts = PurePosixPath(name).parts
        assert name and not name.startswith(("/", "\\")) and "\\" not in name
        assert ":" not in name and all(x not in ("..", ".", "") for x in parts)
        assert all(not x.endswith((".", " ")) and x.split(".", 1)[0].casefold() not in reserved for x in parts)
        assert PurePosixPath(name).as_posix() == name, name
        assert not item.is_dir(), "Expected only explicit regular files"
        mode = item.external_attr >> 16
        assert not stat.S_ISLNK(mode), "Zip symlink forbidden"
        assert not item.flag_bits & 1, "Encrypted zip member forbidden"
    return set(names)


def extract_exclusive(z, destination):
    destination.mkdir(parents=True, exist_ok=False)
    for name in z.namelist():
        target = destination.joinpath(*PurePosixPath(name).parts)
        assert target.resolve().is_relative_to(destination.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(z.read(name))


def xyxy(boxes):
    return np.concatenate((boxes[..., :2] - boxes[..., 2:] / 2,
                           boxes[..., :2] + boxes[..., 2:] / 2), axis=-1)


def iou(gt, boxes):
    wh = np.maximum(np.minimum(gt[:, None, 2:], boxes[None, :, 2:]) -
                    np.maximum(gt[:, None, :2], boxes[None, :, :2]), 0)
    inter = wh.prod(-1)
    ga = np.maximum(gt[:, 2:] - gt[:, :2], 0).prod(-1)
    ba = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(-1)
    return inter / np.maximum(ga[:, None] + ba[None, :] - inter, 1e-30)


def arrays(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as saved:
        assert set(saved.files) == {"final_boxes", "final_logits", "pre_boxes", "pre_logits"}
        result = {key: saved[key].copy() for key in saved.files}
    for key, array in result.items():
        expected = (4, 300, 4 if key.endswith("boxes") else 10)
        assert array.shape == expected and array.dtype == np.float32 and np.isfinite(array).all(), key
    return result


def summary(values):
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        return dict(n=0, mean=None, median=None, min=None, max=None,
                    positive_fraction=None, negative_fraction=None, zero_fraction=None)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                min=float(x.min()), max=float(x.max()), positive_fraction=float((x > 0).mean()),
                negative_fraction=float((x < 0).mean()), zero_fraction=float((x == 0).mean()))


def masks(gt_pixels):
    # Cohorts use original native FP32 coordinates promoted to FP64, before any
    # normalization round trip, matching the owner-selection/source-v5 rule.
    short = np.min(gt_pixels[:, 2:] - gt_pixels[:, :2], axis=-1)
    return dict(under4=short < 4, **{"4to8": (short >= 4) & (short < 8),
                "8to16": (short >= 8) & (short < 16)}, ge16=short >= 16,
                all=np.ones(len(gt_pixels), dtype=bool))


def boundary_match(t, c):
    # The runner records equal-point mean bilinear valid support mass. These
    # predeclared strata do not remove any case from the main analysis.
    return bool(abs(t["bilinear_valid_mass_after"] - c["bilinear_valid_mass_after"]) <= 1e-6
                and t["normalized_inside_after"] == c["normalized_inside_after"])


def csv_write(path, rows):
    assert rows
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_cases(owner_metrics, group_metrics, pair_rows, selected_case_ids):
    selected = set(selected_case_ids)
    result = dict(unique_owner_cases=len(selected), arms={}, paired_T_minus_control={}, geometry={})
    for arm in ARMS:
        rows = [r for r in owner_metrics if r["case_index"] in selected and r["arm"] == arm]
        result["arms"][arm] = {key: summary([r[key] for r in rows]) for key in
            ("owner_final_IoU_delta", "owner_true_class_logit_delta", "selected_cross_L2_change",
             "other_query_box_L2_mean", "other_query_logits_L2_mean")}
    for arm in ("Cplus", "Cminus"):
        rows = [r for r in pair_rows if r["case_index"] in selected and r["control_arm"] == arm]
        result["paired_T_minus_control"][arm] = {
            key: summary([r[key] for r in rows]) for key in
            ("owner_IoU_T_minus_control", "owner_true_class_logit_T_minus_control")}
    for group in GROUPS:
        for threshold in THRESHOLDS:
            tag = f"{group}_IoU{threshold:g}"
            grouped = {}
            for arm in ARMS:
                rows = [r for r in group_metrics if r["case_index"] in selected and r["arm"] == arm
                        and r["group"] == group and r["threshold"] == threshold]
                # Within one case, the four images form one intervention batch.
                # Across cases we retain averages/medians, never a synthetic union
                # or a summed whole-dataset rescue count.
                by_case = {}
                for case in selected:
                    current = [r for r in rows if r["case_index"] == case]
                    if current:
                        by_case[case] = {k: sum(r[k] for r in current) for k in ("gained", "lost", "net")}
                grouped[arm] = by_case
            result["geometry"][tag] = dict(arms={arm: {
                key: summary([r[key] for r in grouped[arm].values()]) for key in ("gained", "lost", "net")}
                for arm in ARMS}, paired_net_T_minus_control={arm: summary([
                    grouped["T"][case]["net"] - grouped[arm][case]["net"] for case in sorted(selected)
                    if case in grouped["T"] and case in grouped[arm]]) for arm in ("Cplus", "Cminus")})
    means = [result["paired_T_minus_control"][arm]["owner_IoU_T_minus_control"]["mean"]
             for arm in ("Cplus", "Cminus")]
    result["descriptive_mean_owner_IoU_T_exceeds_both_directions"] = (
        bool(all(x > 0 for x in means)) if all(x is not None for x in means) else None)
    result["comparison_definition"] = "Strictly positive paired mean T minus each direction for owner final IoU; descriptive only, no training gate or causal claim."
    return result


def main(args):
    destination, output = Path(args.destination).resolve(), Path(args.output).resolve()
    assert not destination.exists() and not output.exists(), "Exclusive destinations required"
    assert not destination.is_relative_to(output) and not output.is_relative_to(destination)
    output.mkdir(parents=True, exist_ok=False)
    try:
        p_raw = (HERE / "protocol.json").read_bytes()
        p, p_sha = json.loads(p_raw), digest(p_raw)
        assert p["schema"] == "sampling_location_suffix_intervention_v1"
        assert p["analysis_sha256"] == sha(__file__)
        for name, expected in p["code_sha256"].items():
            checked((HERE / name).read_bytes(), expected, "Frozen local code: " + name)
        spec = p["seeds"][str(args.seed)]
        metadata_raw = Path(args.metadata).read_bytes()
        metadata = json.loads(metadata_raw)
        assert metadata["seed"] == args.seed and metadata["protocol_sha256"] == p_sha
        assert metadata["child_exit_code"] == 0 and metadata["timed_out"] is False
        assert Path(args.archive).stat().st_size == metadata["archive_bytes"]
        assert sha(args.archive) == metadata["archive_sha256"]
        assert INPUTS.stat().st_size == p["inputs_bytes"] and sha(INPUTS) == p["inputs_sha256"]
        with zipfile.ZipFile(args.archive) as z, zipfile.ZipFile(INPUTS) as inputs:
            zn, inn = zip_names(z), zip_names(inputs)
            im_raw = checked(inputs.read("manifest.json"), p["inputs_manifest_sha256"], "Inputs manifest")
            im = json.loads(im_raw)
            assert inn == set(im["files"]) | {"manifest.json"}
            for name, entry in im["files"].items():
                checked(inputs.read(name), entry, "Input entry: " + name)
            assert z.read("code/protocol.json") == p_raw
            for name in ("runner.py", "controller.py"):
                data = checked(z.read("code/" + name), p["code_sha256"][name], "Remote frozen code")
                assert data == (HERE / name).read_bytes()
            completion = json.loads(z.read("run/completion.json"))
            start = json.loads(z.read("run/start.json"))
            assert completion["child_exit_code"] == 0 and completion["timed_out"] is False
            assert completion["child_live"] is False and completion["no_retry"] is True
            assert completion["child_pid"] == start["child_pid"]
            assert start["protocol_sha256"] == p_sha and start["gpu_uuid"] == spec["gpu_uuid"]
            assert start["timeout_seconds"] == p["timeout_seconds"]
            om_raw = z.read("run/outputs/manifest.json")
            om = json.loads(om_raw)
            expected_zn = {"code/runner.py", "code/controller.py", "code/protocol.json", "run/start.json",
                           "run/completion.json", "run/console.log", "run/outputs/manifest.json"}
            expected_zn |= {"run/outputs/" + name for name in om["files"]}
            assert zn == expected_zn, ("Unexpected terminal archive entries", zn ^ expected_zn)
            for name, entry in om["files"].items():
                assert PurePosixPath(name).name == name and name != "manifest.json"
                checked(z.read("run/outputs/" + name), entry, "Output entry: " + name)
            report = json.loads(z.read("run/outputs/report.json"))
            assert report["status"] == "PASS_SAMPLING_LOCATION_SUFFIX_INTERVENTION"
            assert report["seed"] == args.seed and report["protocol_sha256"] == p_sha
            assert report["checkpoint_sha256"] == spec["checkpoint_sha256"]
            assert report["inputs_sha256"] == p["inputs_sha256"]
            assert report["script_sha256"] == p["code_sha256"]["runner.py"]
            assert report["gpu_uuid"] == spec["gpu_uuid"]
            assert report["AP_evaluation"] is False and report["trained_method"] is False
            owners_doc = json.loads(inputs.read("owners.json"))
            owners = owners_doc["seeds"][str(args.seed)]["owners"]
            audits = {a["image_id"]: a for a in owners_doc["seeds"][str(args.seed)]["image_mapping_audits"]}
            assert len(owners) == spec["owner_count"] == report["owner_count"]
            assert report["cases"] == 3 * len(owners)
            assert len({(o["image_id"], o["query"]) for o in owners}) == len(owners)
            counts = report["counts"]
            assert counts["decoder_suffix_calls"] == 2 + 3 * len(owners) <= spec["max_suffix_calls"]
            assert counts["attempted_suffix_calls"] == counts["decoder_suffix_calls"]
            assert report["immutable_saved_inputs_version_checks_passed"] is True
            assert report["nonselected_query_sampling_locations_unchanged"] is True
            assert counts["control_calls"] == 2 and counts["intervention_calls"] == 3 * len(owners)
            for key in ("full_detector_forwards", "backbone_forwards", "encoder_forwards", "first_self_attention_forwards",
                        "BASE_forwards", "evaluator_calls", "optimizer_updates", "epochs_added"):
                assert counts[key] == 0, key
            expected_output_files = {"start.json", "resolved.yml", "report.json"}
            for batch in (25, 105):
                expected_output_files.update((f"control_{batch:03d}.json", f"N_batch_{batch:03d}.npz"))
            for case_index in range(len(owners)):
                expected_output_files.add(f"progress_{case_index:03d}.json")
                for arm in ARMS:
                    expected_output_files.update((f"case_{case_index:03d}_{arm}.json", f"case_{case_index:03d}_{arm}.npz"))
            assert set(om["files"]) == expected_output_files
            runner_start = json.loads(z.read("run/outputs/start.json"))
            assert runner_start["seed"] == args.seed and runner_start["protocol_sha256"] == p_sha
            assert runner_start["pid"] == start["child_pid"]
            final_progress = json.loads(z.read(f"run/outputs/progress_{len(owners)-1:03d}.json"))
            assert final_progress["completed_owners"] == len(owners) and final_progress["counts"] == counts
            batches, control_checks = {}, []
            for batch in (25, 105):
                prefix = f"seed{args.seed}/capture/batch_{batch:03d}/"
                record = json.loads(inputs.read(prefix + "record.json"))
                ids = record["metadata"]["image_ids"]
                assert record["metadata"]["checkpoint_sha256"] == spec["checkpoint_sha256"]
                gt, gt_pixels, labels, stage_boxes, stage_logits = [], [], [], [], []
                for offset, iid in enumerate(ids):
                    descriptors = {e["key"]["value"]: e["value"] for e in record["native_targets"]["items"][offset]["entries"]}
                    def load(desc):
                        raw = checked(inputs.read(prefix + desc["blob_file"]), desc["blob_sha256"], "GT tensor")
                        array = np.load(io.BytesIO(raw), allow_pickle=False)
                        assert list(array.shape) == desc["shape"] and str(array.dtype) == desc["storage_numpy_dtype"]
                        return array
                    gt_pixels.append(load(descriptors["boxes"]).astype(np.float64))
                    gt.append(gt_pixels[-1] / 640)
                    labels.append(load(descriptors["labels"]).astype(np.int64))
                    assert len(gt[-1]) == audits[iid]["native_GT_count"]
                    with np.load(io.BytesIO(inputs.read(f"seed{args.seed}/stages/image_{iid:06d}.npz")), allow_pickle=False) as stage:
                        stage_boxes.append(stage["final_boxes"])
                        stage_logits.append(stage["final_logits"])
                n = arrays(z.read(f"run/outputs/N_batch_{batch:03d}.npz"))
                controls = json.loads(z.read(f"run/outputs/control_{batch:03d}.json"))
                assert controls == next(c for c in report["controls"] if c["batch"] == batch)
                assert controls["cross_maxabs"] <= p["controls"]["cross_maxabs"]
                assert controls["cross_relative_L2"] <= p["controls"]["cross_relative_L2"]
                boxes_error = float(np.abs(n["final_boxes"] - np.stack(stage_boxes)).max())
                logits_diff = n["final_logits"].astype(np.float64) - np.stack(stage_logits).astype(np.float64)
                logits_error = float(np.abs(logits_diff).max())
                logits_relative = float(np.linalg.norm(logits_diff) / max(np.linalg.norm(np.stack(stage_logits).astype(np.float64)), 1e-30))
                assert boxes_error <= p["controls"]["boxes_maxabs"]
                assert logits_error <= p["controls"]["logits_maxabs"] and logits_relative <= p["controls"]["logits_relative_L2"]
                native_iou = [iou(gt[b], xyxy(n["final_boxes"][b].astype(np.float64))) for b in range(4)]
                support_disagreement = {}
                for threshold in THRESHOLDS:
                    support_disagreement[str(threshold)] = sum(int(np.sum(
                        (native_iou[b].max(-1) >= threshold) !=
                        (iou(gt[b], xyxy(stage_boxes[b].astype(np.float64))).max(-1) >= threshold))) for b in range(4))
                assert all(v == 0 for v in support_disagreement.values())
                assert controls["GT_support_differences"] == support_disagreement
                control_checks.append(dict(batch=batch, recomputed_boxes_maxabs=boxes_error,
                    recomputed_logits_maxabs=logits_error, recomputed_logits_relative_L2=logits_relative,
                    recomputed_support_disagreements=support_disagreement,
                    cross_attention_control_source="Remote measured control record; cross tensor itself was not saved."))
                batches[batch] = dict(ids=ids, gt=gt, gt_pixels=gt_pixels, labels=labels, n=n, native_iou=native_iou)
            # Verify everything before extracting the terminal package exclusively.
            extract_exclusive(z, destination)
            owner_rows, group_rows, pair_rows, boundary_rows = [], [], [], []
            gt_csv = output / "case_GT_geometry.csv"
            gt_fields = ["seed", "case_index", "arm", "owner_image_id", "owner_annotation_id", "affected_image_id",
                         "GT_native_index", "annotation_id", "input_short_px", "N_any_query_IoU", "arm_any_query_IoU",
                         "IoU0.5_N_supported", "IoU0.5_arm_supported", "IoU0.75_N_supported", "IoU0.75_arm_supported"]
            with gt_csv.open("x", encoding="utf-8-sig", newline="") as gt_stream:
                writer = csv.DictWriter(gt_stream, fieldnames=gt_fields)
                writer.writeheader()
                for case_index, owner in enumerate(owners):
                    batch = batches[owner["batch"]]
                    b, q, g = owner["image_offset"], owner["query"], owner["GT_native_index"]
                    assert batch["ids"][b] == owner["image_id"]
                    assert audits[owner["image_id"]]["native_order_annotation_ids"][g] == owner["annotation_id"]
                    label = int(batch["labels"][b][g])
                    arm_info, arm_metrics = {}, {}
                    for arm in ARMS:
                        prefix = f"run/outputs/case_{case_index:03d}_{arm}"
                        info = json.loads(z.read(prefix + ".json"))
                        assert info["case_index"] == case_index and info["arm"] == arm and info["owner"] == owner
                        assert info["output"] == f"case_{case_index:03d}_{arm}.npz"
                        assert np.isfinite(info["bilinear_valid_mass_after"])
                        assert -1e-12 <= info["bilinear_valid_mass_after"] <= 1 + 1e-12
                        assert 0 <= info["normalized_inside_after"] <= info["points_total"] == 96
                        current = arrays(z.read(prefix + ".npz"))
                        all_iou = [iou(batch["gt"][idx], xyxy(current["final_boxes"][idx].astype(np.float64))) for idx in range(4)]
                        other = np.ones(300, dtype=bool); other[q] = False
                        row = dict(seed=args.seed, case_index=case_index, arm=arm, batch=owner["batch"], image_id=owner["image_id"],
                            annotation_id=owner["annotation_id"], GT_native_index=g, query=q, true_class=label,
                            N_owner_final_IoU=float(batch["native_iou"][b][g, q]), arm_owner_final_IoU=float(all_iou[b][g, q]),
                            owner_final_IoU_delta=float(all_iou[b][g, q] - batch["native_iou"][b][g, q]),
                            N_owner_true_class_logit=float(batch["n"]["final_logits"][b, q, label]),
                            arm_owner_true_class_logit=float(current["final_logits"][b, q, label]),
                            owner_true_class_logit_delta=float(np.float64(current["final_logits"][b, q, label]) - batch["n"]["final_logits"][b, q, label]),
                            selected_cross_L2_change=float(info["selected_cross_L2_change"]),
                            other_query_box_L2_mean=float(np.linalg.norm(current["final_boxes"][b, other].astype(np.float64) - batch["n"]["final_boxes"][b, other], axis=-1).mean()),
                            other_query_logits_L2_mean=float(np.linalg.norm(current["final_logits"][b, other].astype(np.float64) - batch["n"]["final_logits"][b, other], axis=-1).mean()),
                            bilinear_valid_mass_after=float(info["bilinear_valid_mass_after"]),
                            normalized_inside_after=int(info["normalized_inside_after"]))
                        owner_rows.append(row); arm_metrics[arm] = row; arm_info[arm] = info
                        boundary_rows.append(dict(seed=args.seed, case_index=case_index, arm=arm,
                                                  complete_remote_case_record_json=json.dumps(info, ensure_ascii=False, sort_keys=True)))
                        for idx, iid in enumerate(batch["ids"]):
                            gt = batch["gt"][idx]
                            base_max, current_max = batch["native_iou"][idx].max(-1), all_iou[idx].max(-1)
                            gt_pixels = batch["gt_pixels"][idx]
                            group_masks = masks(gt_pixels)
                            for gg in range(len(gt)):
                                writer.writerow(dict(seed=args.seed, case_index=case_index, arm=arm,
                                    owner_image_id=owner["image_id"], owner_annotation_id=owner["annotation_id"],
                                    affected_image_id=iid, GT_native_index=gg, annotation_id=audits[iid]["native_order_annotation_ids"][gg],
                                    input_short_px=float(np.min(gt_pixels[gg, 2:] - gt_pixels[gg, :2])),
                                    N_any_query_IoU=float(base_max[gg]), arm_any_query_IoU=float(current_max[gg]),
                                    **{f"IoU{threshold:g}_{tag}_supported": bool(values[gg] >= threshold)
                                       for threshold in THRESHOLDS for tag, values in (("N", base_max), ("arm", current_max))}))
                            for group, mask in group_masks.items():
                                for threshold in THRESHOLDS:
                                    before, after = base_max[mask] >= threshold, current_max[mask] >= threshold
                                    gained, lost = int(np.sum(~before & after)), int(np.sum(before & ~after))
                                    group_rows.append(dict(seed=args.seed, case_index=case_index, arm=arm,
                                        owner_image_id=owner["image_id"], affected_image_id=iid, affected_is_owner_image=iid == owner["image_id"],
                                        group=group, threshold=threshold, GT_count=int(mask.sum()), N_supported=int(before.sum()),
                                        arm_supported=int(after.sum()), gained=gained, lost=lost, net=gained - lost))
                    for arm in ("Cplus", "Cminus"):
                        pair_rows.append(dict(seed=args.seed, case_index=case_index, owner_image_id=owner["image_id"],
                            annotation_id=owner["annotation_id"], control_arm=arm,
                            owner_IoU_T_minus_control=arm_metrics["T"]["owner_final_IoU_delta"] - arm_metrics[arm]["owner_final_IoU_delta"],
                            owner_true_class_logit_T_minus_control=arm_metrics["T"]["owner_true_class_logit_delta"] - arm_metrics[arm]["owner_true_class_logit_delta"],
                            boundary_matched=boundary_match(arm_info["T"], arm_info[arm]),
                            bilinear_valid_mass_difference=arm_info["T"]["bilinear_valid_mass_after"] - arm_info[arm]["bilinear_valid_mass_after"],
                            normalized_inside_count_difference=arm_info["T"]["normalized_inside_after"] - arm_info[arm]["normalized_inside_after"]))
            csv_write(output / "case_owner_metrics.csv", owner_rows)
            csv_write(output / "case_group_geometry.csv", group_rows)
            csv_write(output / "case_pairwise_comparisons.csv", pair_rows)
            csv_write(output / "case_boundary_records.csv", boundary_rows)
            geometry_lookup = {(r["case_index"], r["arm"], r["affected_image_id"], r["group"], r["threshold"]): r
                               for r in group_rows}
            geometry_pairs = []
            for row in group_rows:
                if row["arm"] != "T":
                    continue
                for arm in ("Cplus", "Cminus"):
                    control = geometry_lookup[(row["case_index"], arm, row["affected_image_id"], row["group"], row["threshold"])]
                    geometry_pairs.append(dict(seed=args.seed, case_index=row["case_index"],
                        owner_image_id=row["owner_image_id"], affected_image_id=row["affected_image_id"],
                        affected_is_owner_image=row["affected_is_owner_image"], group=row["group"],
                        threshold=row["threshold"], control_arm=arm, GT_count=row["GT_count"],
                        gained_T_minus_control=row["gained"] - control["gained"],
                        lost_T_minus_control=row["lost"] - control["lost"],
                        net_T_minus_control=row["net"] - control["net"]))
            csv_write(output / "case_geometry_T_minus_controls.csv", geometry_pairs)
            ids = list(range(len(owners)))
            strata = {}
            for comparison in ("Cplus", "Cminus", "both"):
                for matched in (True, False):
                    eligible = []
                    for case in ids:
                        rows = [r for r in pair_rows if r["case_index"] == case and
                                (comparison == "both" or r["control_arm"] == comparison)]
                        if all(r["boundary_matched"] for r in rows) == matched:
                            eligible.append(case)
                    strata[f"{comparison}_{'matched' if matched else 'unmatched'}"] = summarize_cases(owner_rows, group_rows, pair_rows, eligible)
            by_image = {str(iid): summarize_cases(owner_rows, group_rows, pair_rows,
                [idx for idx, owner in enumerate(owners) if owner["image_id"] == iid]) for iid in sorted(audits)}
            by_affected_image = {}
            for iid in sorted(audits):
                selected_rows = [row for row in group_rows if row["affected_image_id"] == iid]
                selected_cases = sorted({row["case_index"] for row in selected_rows})
                by_affected_image[str(iid)] = dict(
                    native_GT_count=audits[iid]["native_GT_count"], paired_case_count=len(selected_cases),
                    cases_whose_owner_is_this_image=sum(owners[idx]["image_id"] == iid for idx in selected_cases),
                    cases_whose_owner_is_another_image=sum(owners[idx]["image_id"] != iid for idx in selected_cases),
                    geometry=summarize_cases(owner_rows, selected_rows, pair_rows, selected_cases)["geometry"])
            result = dict(status="PASS_VERIFIED_SAMPLING_LOCATION_INTERVENTION_DESCRIPTION", seed=args.seed,
                verified_archive_sha256=metadata["archive_sha256"], metadata_sha256=digest(metadata_raw),
                protocol_sha256=p_sha, input_archive_sha256=p["inputs_sha256"], input_manifest_sha256=p["inputs_manifest_sha256"],
                output_manifest_sha256=digest(om_raw), analyzer_sha256=sha(__file__),
                process_completion=completion, control_recomputations=control_checks,
                remote_counts=counts, owner_count=len(owners), overall=summarize_cases(owner_rows, group_rows, pair_rows, ids),
                by_owner_image=by_image, by_affected_image=by_affected_image, boundary_strata=strata,
                notes=["Every intervention is compared against the actual N output of its original batch, not against original stage arrays.",
                       "Counts are gained/lost within each single-query case over its complete four-image batch. Across cases only paired distributions/means/medians are reported; no cumulative whole-image or dataset gain is claimed.",
                       "Owners are unique image-query pairs, but different cases share images and other GT. Cases are not independent statistical samples.",
                       "Boundary strata retain all cases. Matched means equal normalized inside point count and absolute mean bilinear valid mass difference <=1e-6.",
                       "selected_cross_L2_change and boundary metadata are remote measurements retained from case records; final IoU/logit/geometry statistics are recomputed from actual saved arrays.",
                       "GT-directed shifts are oracle diagnostics on a reused eight-image panel. Directional differences do not establish missing query information, a unique root cause, trainability, AP benefit, or a training gate."],
                AP_evaluated=False, p_values_computed=False, causal_root_established=False, automatic_training_authorized_by_result=False,
                new_model_forwards=0, evaluator_calls=0, optimizer_updates=0, epochs_added=0,
                completed_local_utc=datetime.now(timezone.utc).isoformat())
            write(output / "summary.json", result)
        assert "torch" not in sys.modules and not any(name.startswith("pycocoeval") for name in sys.modules)
        write(output / "analysis_manifest.json", dict(schema="sampling_location_cpu_analysis_manifest_v1",
            files={f.name: dict(sha256=sha(f), bytes=f.stat().st_size) for f in output.iterdir() if f.is_file()},
            analyzer_sha256=sha(__file__), actual_execution_completed=True))
        print(json.dumps(dict(status=result["status"], seed=args.seed, owner_count=len(owners),
            summary=str(output / "summary.json"), descriptive_T_exceeds_both_directions=result["overall"]["descriptive_mean_owner_IoU_T_exceeds_both_directions"])), flush=True)
    except BaseException:
        write(output / "failure.json", dict(status="CPU_ANALYSIS_OR_VERIFICATION_FAILED", traceback=traceback.format_exc(),
            new_model_forwards=0, evaluator_calls=0, optimizer_updates=0, epochs_added=0))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(1, 2), required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--destination", required=True, help="New directory for verified terminal extraction")
    parser.add_argument("--output", required=True, help="New directory for CPU analysis artifacts")
    main(parser.parse_args())
