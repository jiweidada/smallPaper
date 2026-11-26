"""Fulfil frozen secondary-output categories using cached predictions only."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import math
import sys

P = Path(__file__).resolve().parent
A49 = P.parent / "A49_下一主线有界筛选_20261101_v1"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    protocol = load(P / "secondary_analysis_scope_v1.json")
    assert sha(Path(__file__)) == protocol["script_sha256"]
    out = P / "secondary_cached_analysis_v1"
    out.mkdir(exist_ok=False)
    sys.path.insert(0, str(A49 / "postprocess_ignore_strata/deps"))
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    folders = {"B": A49 / "completed_B_v1/arm_B", "M": P / "completed_M_v1/arm_M", "Q": P / "completed_Q_v1/arm_Q"}
    gt = load(folders["B"] / "epoch12_EMA_GT.json")
    assert len(gt["images"]) == 548 and len(gt["annotations"]) == 38602
    images, annotations = {x["id"]: x for x in gt["images"]}, {x["id"]: x for x in gt["annotations"]}
    assert all(not a["iscrowd"] and a["area"] > 0 for a in annotations.values())
    subsets = {"all": set(annotations), "small": set(), "small_lt2": set(), "small_2to4": set(), "small_4to8": set(), "small_ge8": set()}
    rows = {}
    for aid, a in annotations.items():
        im = images[a["image_id"]]
        side = min(a["bbox"][2] * 640 / im["width"], a["bbox"][3] * 640 / im["height"])
        binname = "lt2" if side < 2 else "2to4" if side < 4 else "4to8" if side < 8 else "ge8"
        if a["area"] <= 1024:
            subsets["small"].add(aid); subsets["small_" + binname].add(aid)
        rows[aid] = {"GT_id": aid, "image_id": a["image_id"], "category_id": a["category_id"],
            "original_area": a["area"], "input_short_side": side, "input_short_side_bin": binname, "arms": {}}
    assert len(subsets["small"]) == 26444 and len(subsets["small_4to8"]) == 10200
    all_sources, arm_results = {}, {}
    for arm, folder in folders.items():
        files = [folder / ("epoch12_EMA_" + suffix) for suffix in ["GT.json", "predictions.json", "metrics.json"]]
        all_sources.update({str(f): sha(f) for f in files})
        assert load(files[0]) == gt
        pred, reference = load(files[1]), load(files[2])
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            api = COCO(); api.dataset = gt; api.createIndex()
            dt = api.loadRes(pred)
            ev = COCOeval(api, dt, "bbox"); ev.params.maxDets = [1, 10, 300]
            ev.evaluate(); ev.accumulate()
        (out / (arm + "_COCO_stdout.txt")).write_text(capture.getvalue(), encoding="utf-8")
        precision, recall = ev.eval["precision"], ev.eval["recall"]
        def ap(area=0, iou=None):
            values = precision[:, :, :, area, -1]
            if iou is not None:
                values = values[np.isclose(ev.params.iouThrs, iou)]
            values = values[values >= 0]
            return float(values.mean() * 100)
        def ar(area):
            values = recall[:, :, area, -1]; values = values[values >= 0]
            return float(values.mean() * 100)
        metrics = {"AP": ap(), "AP50": ap(iou=.5), "AP75": ap(iou=.75), "APs": ap(1), "APs75": ap(1, .75), "APm": ap(2), "APl": ap(3), "ARs": ar(1)}
        differences = {k: metrics[k] - reference[k] for k in metrics}
        assert max(abs(v) for v in differences.values()) <= 1e-8, differences
        matched = {50: {}, 75: {}}
        for item in ev.evalImgs:
            if item is None or item["aRng"] != ev.params.areaRng[0] or item["maxDet"] != 300:
                continue
            for threshold in [50, 75]:
                t = int(np.argmin(abs(ev.params.iouThrs - threshold / 100)))
                for aid, did, ignored in zip(item["gtIds"], item["gtMatches"][t], item["gtIgnore"]):
                    if did > 0 and not ignored:
                        matched[threshold][int(aid)] = int(did)
        for aid, row in rows.items():
            did = matched[50].get(aid)
            entry = {"matched50": did is not None, "matched75": aid in matched[75], "matched50_detection_id": did,
                     "signed_log_geomean_size_error_if_matched50": None}
            if did is not None:
                gb, pb = annotations[aid]["bbox"], dt.anns[did]["bbox"]
                assert min(gb[2], gb[3], pb[2], pb[3]) > 0
                entry["signed_log_geomean_size_error_if_matched50"] = .5 * (math.log(pb[2] / gb[2]) + math.log(pb[3] / gb[3]))
            row["arms"][arm] = entry
        summaries = {}
        for name, ids in subsets.items():
            size_errors = [rows[i]["arms"][arm]["signed_log_geomean_size_error_if_matched50"] for i in ids if i in matched[50]]
            summaries[name] = {"GT": len(ids), "matched50": len(ids & matched[50].keys()), "matched75": len(ids & matched[75].keys()),
                "recall50_percent": len(ids & matched[50].keys()) * 100 / len(ids),
                "recall75_percent": len(ids & matched[75].keys()) * 100 / len(ids),
                "size_error_TP50_count": len(size_errors), "mean_absolute_log_geomean_size_error_on_own_TP50": float(np.mean(np.abs(size_errors))) if size_errors else None,
                "mean_signed_log_geomean_size_error_on_own_TP50": float(np.mean(size_errors)) if size_errors else None}
        arm_results[arm] = {"metric_reproduction": metrics, "metric_minus_original_pp": differences, "groups": summaries}
        del ev, dt, api, pred
    old = load(P.parent / "A53_原版漏检与机制候选筛选_20261101_v1/error_audit_result_v4.json")
    assert all(arm_results["B"]["groups"]["all"][k] == old["all_GT"][k] for k in ["GT", "matched50", "matched75"])
    common = {}
    for name, ids in subsets.items():
        cohort = [i for i in ids if all(rows[i]["arms"][a]["matched50"] for a in folders)]
        common[name] = {"common_TP50_GT": len(cohort), "arms": {a: {
            "mean_absolute_log_geomean_size_error": float(np.mean([abs(rows[i]["arms"][a]["signed_log_geomean_size_error_if_matched50"]) for i in cohort])) if cohort else None
        } for a in folders}}
    rows_path = out / "GT_rows.jsonl"
    with rows_path.open("x", encoding="utf-8") as f:
        for aid in sorted(rows):
            f.write(json.dumps(rows[aid], allow_nan=False) + "\n")
    assert all(sha(Path(f)) == h for f, h in all_sources.items())
    result = {"status": "DESCRIPTIVE_SECONDARY_COMPLETE", "primary_gates_changed": False,
        "scope_protocol_sha256": sha(P / "secondary_analysis_scope_v1.json"), "arms": arm_results,
        "common_matched50_cohort": common, "source_files_sha256": all_sources, "GT_rows_sha256": sha(rows_path),
        "COCO_evaluate_calls": 3, "new_detector_forwards": 0, "new_optimizer_updates": 0,
        "limits": protocol["limits"]}
    save(out / "result.json", result)
    print(json.dumps({"status": result["status"], "small_4to8": {a: v["groups"]["small_4to8"] for a, v in arm_results.items()},
        "COCO_evaluate_calls": 3, "new_detector_forwards": 0}), flush=True)


if __name__ == "__main__":
    main()
