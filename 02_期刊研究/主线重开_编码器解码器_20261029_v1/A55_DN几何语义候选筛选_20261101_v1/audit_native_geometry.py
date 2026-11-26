"""CPU-only native DN construction and fixed training-annotation sample audit."""
from pathlib import Path
from collections import defaultdict
from unittest.mock import patch
import ast
import hashlib
import json
import sys
import time

P = Path(__file__).resolve().parent
BASE = P.parent
ENG = P.parents[1]
SRC = BASE / "A43_ECDet三组蒸馏最小验证_20261031_v1/source/EdgeCrafter-d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/ecdetseg/engine/edgecrafter"
sys.path.insert(0, str(ENG / "code/dn_pre_frozen_diagnostic_20261020_v1/runtime"))
import torch


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_function(path, name, env):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name]
    assert len(nodes) == 1
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
    return env[name]


def overlap(a, b):
    # Float64 geometry of the actual float32 references; no prediction changes.
    a, b = a.double(), b.double()
    wh = (torch.minimum(a[:, None, 2:], b[None, :, 2:]) - torch.maximum(a[:, None, :2], b[None, :, :2])).clamp_min(0)
    intersection = wh.prod(-1)
    union = (a[:, 2:] - a[:, :2]).prod(-1)[:, None] + (b[:, 2:] - b[:, :2]).prod(-1)[None, :] - intersection
    return intersection / union


def main():
    started = time.monotonic()
    scope = load(P / "02_核查范围_FROZEN_v1.json")
    assert sha(Path(__file__)) == scope["script_sha256"]
    assert all(sha(Path(f)) == h for f, h in scope["source_files_sha256"].items())
    output = P / "03_原生DN几何核查_v1.json"
    assert not output.exists()
    torch.set_num_threads(4)
    env = {"torch": torch, "Tensor": torch.Tensor}
    to_xy = selected_function(SRC / "box_ops.py", "box_cxcywh_to_xyxy", env)
    to_cw = selected_function(SRC / "box_ops.py", "box_xyxy_to_cxcywh", env)
    selected_function(SRC / "utils.py", "inverse_sigmoid", env)
    captured = {}
    def record_conversion(boxes):
        result = to_cw(boxes)
        captured["inverted"] = (result[..., 2:] < 0).any(-1).clone()
        captured["pre_abs_cxcywh"] = result.detach().clone()
        return result
    env["box_xyxy_to_cxcywh"] = record_conversion
    native = selected_function(SRC / "denoising.py", "get_contrastive_denoising_training_group", env)
    echo_class = lambda labels: labels.to(torch.float32).unsqueeze(-1)
    gt = torch.tensor([[.5, .5, 4 / 640, 4 / 640]], dtype=torch.float32)
    target = {"boxes": gt, "labels": torch.tensor([0])}
    def inward_sign(values, *args, **kwargs):
        return torch.tensor([1, 1, 0, 0], dtype=values.dtype).expand_as(values).clone()
    with patch.object(torch, "rand_like", side_effect=lambda x, *a, **k: torch.full_like(x, .99)), patch.object(torch, "randint_like", side_effect=inward_sign):
        _, reference, _, meta = native([target], 10, 300, echo_class, num_denoising=1, label_noise_ratio=0., box_noise_scale=1.)
    synthetic_boxes = reference.sigmoid()
    synthetic_iou = float(overlap(to_xy(synthetic_boxes[0, 1:2]), to_xy(gt))[0, 0])
    assert meta["dn_positive_idx"][0].tolist() == [0] and captured["inverted"][0, 1]
    assert synthetic_iou > .97
    synthetic = {"negative_slot": 1, "source_GT_cxcywh": gt.tolist(), "post_clip_pre_abs_cxcywh": captured["pre_abs_cxcywh"][0, 1].tolist(),
        "native_negative_reference_cxcywh": synthetic_boxes[0, 1].tolist(), "IoU_to_source_GT": synthetic_iou,
        "scope": "Controlled native-generator example, label noise disabled only here; not a detector prediction or frequency estimate."}
    manifest = load(Path(scope["train_manifest"]))
    byfile = {r["file"]: r for r in manifest["rows"]}
    selected, annotation_hashes = [], {}
    for file in scope["selected_images"]:
        m = byfile[file]
        annotation = Path(scope["raw_train_annotations"]) / (Path(file).stem + ".txt")
        assert sha(annotation) == m["annotation_sha256"]
        annotation_hashes[str(annotation)] = sha(annotation)
        boxes, labels, areas, sides = [], [], [], []
        for line in annotation.read_text().splitlines():
            x, y, w, h, score, category, trunc, occ = map(float, line.split(",")[:8])
            x0, x1 = max(0., min(x, m["width"])), max(0., min(x + w, m["width"]))
            y0, y1 = max(0., min(y, m["height"])), max(0., min(y + h, m["height"]))
            if score <= 0 or not 1 <= int(category) <= 10 or x1 <= x0 or y1 <= y0:
                continue
            boxes.append([(x0 + x1) / (2 * m["width"]), (y0 + y1) / (2 * m["height"]), (x1 - x0) / m["width"], (y1 - y0) / m["height"]])
            labels.append(int(category) - 1); areas.append((x1 - x0) * (y1 - y0))
            sides.append(min((x1 - x0) * 640 / m["width"], (y1 - y0) * 640 / m["height"]))
        assert len(boxes) == m["valid_gt"]
        selected.append({"file": file, "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long), "areas": torch.tensor(areas, dtype=torch.float64), "sides": torch.tensor(sides, dtype=torch.float64)})
    totals = defaultdict(lambda: defaultdict(int))
    affected = defaultdict(set)
    records, examples = [], []
    generator_calls = 1
    for draw in range(scope["draws"]):
        for batch_index in range(0, len(selected), 8):
            batch = selected[batch_index:batch_index + 8]
            torch.manual_seed(scope["seed"] + 1009 * draw + batch_index)
            classes, refs, _, meta = native(batch, 10, 300, echo_class, num_denoising=100, label_noise_ratio=.5, box_noise_scale=1.)
            generator_calls += 1
            refs = refs.sigmoid()
            maximum = max(len(t["labels"]) for t in batch)
            groups = meta["dn_num_group"]
            for image_index, t in enumerate(batch):
                count = len(t["labels"])
                if not count:
                    continue
                pos = torch.cat([torch.arange(count) + 2 * maximum * k for k in range(groups)])
                assert torch.equal(pos, meta["dn_positive_idx"][image_index])
                neg = pos + maximum
                owner = torch.arange(count).repeat(groups)
                box = to_xy(refs[image_index, neg]); truth = to_xy(t["boxes"])
                ov = overlap(box, truth)
                source_iou = ov[torch.arange(len(owner)), owner]
                other = ov.clone(); other[torch.arange(len(owner)), owner] = -1
                matching_input_class = classes[image_index, neg, 0].long()[:, None] == t["labels"][None, :]
                cross_same = torch.where(matching_input_class, other, torch.full_like(other, -1)).max(-1).values
                cross = other.max(-1).values
                masks = {"all": torch.ones(len(owner), dtype=torch.bool), "small": t["areas"][owner] <= 1024,
                    "primary_small_shortside_lt8": (t["areas"][owner] <= 1024) & (t["sides"][owner] < 8)}
                for binname in ["lt2", "2to4", "4to8", "ge8"]:
                    side = t["sides"][owner]
                    band = side < 2 if binname == "lt2" else ((side >= 2) & (side < 4)) if binname == "2to4" else ((side >= 4) & (side < 8)) if binname == "4to8" else side >= 8
                    masks["small_" + binname] = (t["areas"][owner] <= 1024) & band
                tests = {"inverted_before_abs": captured["inverted"][image_index, neg],
                    "source_IoU_ge0_5": source_iou >= .5, "other_IoU_ge0_5": cross >= .5,
                    "same_input_class_other_IoU_ge0_5": cross_same >= .5, "same_input_class_other_IoU_ge0_7": cross_same >= .7}
                perimage = {"file": t["file"], "draw": draw, "GT": count, "groups": groups, "valid_negative_slots": len(neg)}
                for cohort, mask in masks.items():
                    row = {"negative_slots": int(mask.sum()), **{key: int((test & mask).sum()) for key, test in tests.items()}}
                    for k, value in row.items():
                        totals[cohort][k] += value
                    if row["same_input_class_other_IoU_ge0_5"]:
                        affected[cohort].add(t["file"])
                    perimage[cohort] = row
                records.append(perimage)
                if len(examples) < 20:
                    for index in torch.where(tests["same_input_class_other_IoU_ge0_5"])[0].tolist():
                        if len(examples) >= 20:
                            break
                        examples.append({"file": t["file"], "draw": draw, "source_GT_index": int(owner[index]),
                            "negative_reference_cxcywh": refs[image_index, neg[index]].tolist(), "input_class": int(classes[image_index, neg[index], 0]),
                            "best_other_same_input_class_IoU": float(cross_same[index]), "source_IoU": float(source_iou[index])})
    summaries = {}
    for cohort, counts in totals.items():
        summaries[cohort] = {**dict(counts), "affected_images": len(affected[cohort]),
            "same_input_class_other_IoU_ge0_5_rate": counts["same_input_class_other_IoU_ge0_5"] / counts["negative_slots"] if counts["negative_slots"] else None}
    primary = summaries["primary_small_shortside_lt8"]
    gates = {"primary_cross_same_class_rate_ge0_03": primary["same_input_class_other_IoU_ge0_5_rate"] is not None and primary["same_input_class_other_IoU_ge0_5_rate"] >= .03,
             "affected_images_ge20": primary["affected_images"] >= 20}
    assert generator_calls == 65
    assert all(sha(Path(f)) == h for f, h in scope["source_files_sha256"].items())
    record_path = P / "03_逐图逐次计数_v1.jsonl"
    with record_path.open("x", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    result = {"status": "CPU_NATIVE_GEOMETRY_AUDIT_COMPLETE", "scope_sha256": sha(P / "02_核查范围_FROZEN_v1.json"),
        "synthetic_native_example": synthetic, "selected_train_images": len(selected), "selected_valid_GT": sum(len(t["labels"]) for t in selected),
        "selected_annotation_hashes": annotation_hashes, "draws": scope["draws"], "native_generator_calls": generator_calls,
        "summaries": summaries, "entry_gates": gates, "prevalence_gate_pass": all(gates.values()),
        "first20_examples_not_representative_sample": examples, "per_image_draw_counts_sha256": sha(record_path),
        "new_detector_forwards": 0, "new_optimizer_updates": 0, "validation_images_used": 0,
        "candidate_selected_for_training": False, "limits": scope["limits"], "seconds": time.monotonic() - started}
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ["status", "synthetic_native_example", "selected_train_images", "selected_valid_GT", "summaries", "entry_gates", "prevalence_gate_pass", "seconds"]}), flush=True)


if __name__ == "__main__":
    main()
