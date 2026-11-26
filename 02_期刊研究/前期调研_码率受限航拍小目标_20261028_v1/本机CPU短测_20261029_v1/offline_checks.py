"""Read-only train annotation audit and deterministic software fixtures (no ML)."""
from pathlib import Path
from collections import Counter
from itertools import combinations
import hashlib
import json
import math
import time

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[2]
ANN = WORK / "英文小论文/相位机制论文_20261018/修订_20261018_v2/data/annotations/train/converted.json"


def main():
    started = time.perf_counter()
    raw = ANN.read_bytes()
    data = json.loads(raw)
    images = data["images"]
    annotations = data["annotations"]
    lookup = {im["id"]: im for im in images}
    cat_ids = {c["id"] for c in data["categories"]}
    invalid = []
    outside = 0
    bad_ref = 0
    nonfinite = 0
    valid = 0
    for ann in annotations:
        if ann["image_id"] not in lookup or ann["category_id"] not in cat_ids:
            bad_ref += 1
            continue
        im = lookup[ann["image_id"]]
        box = ann.get("bbox", [])
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            nonfinite += 1
            continue
        x, y, w, h = box
        if w <= 0 or h <= 0:
            invalid.append({"annotation_id": ann["id"], "file_name": im["file_name"], "bbox": box})
            continue
        valid += 1
        if x < 0 or y < 0 or x + w > im["width"] or y + h > im["height"]:
            outside += 1
    names = [im["file_name"] for im in images]
    ordered = sorted(names, key=lambda s: (hashlib.sha256(("compress-dev-v1:" + s).encode()).hexdigest(), s))
    dev = set(ordered[:300])
    fit = set(names) - dev
    samples = sorted(fit)[:3]
    audit = {
        "status": "PASS_WITH_DATA_CAVEAT",
        "scope": "train annotation metadata only; no data edits; no val/test reads",
        "source": str(ANN), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "images": len(images), "annotations": len(annotations),
        "categories": data["categories"], "valid_positive_size_annotations": valid,
        "nonpositive_size_boxes": invalid, "outside_image_positive_size_boxes": outside,
        "nonfinite_or_malformed_boxes": nonfinite, "bad_image_or_category_references": bad_ref,
        "duplicate_image_ids": len(images) - len(lookup),
        "duplicate_annotation_ids": len(annotations) - len({a["id"] for a in annotations}),
        "duplicate_filenames": len(names) - len(set(names)),
        "nonpositive_image_dimensions": sum(im["width"] <= 0 or im["height"] <= 0 for im in images),
        "class_annotation_counts": dict(Counter(a["category_id"] for a in annotations)),
        "split_dry_run": {"dev": len(dev), "fit": len(fit), "overlap": len(dev & fit),
                          "manifest_created": False, "scene_disjoint_claimed": False},
        "candidate_fit_images_for_smoke": [im for name in samples for im in images if im["file_name"] == name],
        "first_image_record": images[0], "first_annotation_record": annotations[0],
        "seconds": round(time.perf_counter() - started, 4),
    }
    assert len(lookup) == len(images) and len(set(names)) == len(images)
    assert bad_ref == 0 and nonfinite == 0
    assert len(fit) == 6171 and len(dev) == 300 and not dev & fit
    (OUT / "annotation_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    # Small scalar fixtures exercise both odd dimensions and padding, independently of NumPy/Torch.
    width, height = 79, 45
    left, top, right, bottom = 7, 5, 72, 40
    cases = []
    for overlap in (0.0, 6.0):
        max_sum_error = 0.0
        min_weight, max_weight = 1.0, 0.0
        endpoints = True
        padding_ok = True
        convex_ok = True
        zero_residual_ok = True
        for y in range(height):
            for x in range(width):
                inside = left <= x < right and top <= y < bottom
                if inside:
                    mid_x, mid_y = (left + right) / 2, (top + bottom) / 2
                    if overlap:
                        rx = min(1.0, max(0.0, (x + .5 - mid_x) / overlap + .5))
                        ry = min(1.0, max(0.0, (y + .5 - mid_y) / overlap + .5))
                    else:
                        rx, ry = float(x + .5 >= mid_x), float(y + .5 >= mid_y)
                    weights = [(1-rx)*(1-ry), rx*(1-ry), (1-rx)*ry, rx*ry]
                    max_sum_error = max(max_sum_error, abs(sum(weights) - 1))
                    min_weight, max_weight = min(min_weight, *weights), max(max_weight, *weights)
                else:
                    weights = [0.0] * 4
                degraded = ((x * 13 + y * 7) % 256) / 255 if inside else 114/255
                candidate = ((x * 3 + y * 17 + 19) % 256) / 255 if inside else degraded
                for bits in range(16):
                    gate = sum(weights[i] for i in range(4) if bits & (1 << i))
                    mixed = degraded if bits == 0 else candidate if bits == 15 else degraded + gate * (candidate - degraded)
                    endpoints &= (bits != 0 or mixed == degraded) and (bits != 15 or mixed == candidate)
                    convex_ok &= min(degraded, candidate) - 1e-12 <= mixed <= max(degraded, candidate) + 1e-12
                    padding_ok &= inside or mixed == degraded
                    zero_residual_ok &= degraded + gate * (degraded - degraded) == degraded
        assert max_sum_error < 1e-12 and min_weight >= 0 and max_weight <= 1
        assert endpoints and padding_ok and convex_ok and zero_residual_ok
        cases.append({"overlap_pixels": overlap, "partition_sum_max_error": max_sum_error,
                      "weight_range": [min_weight, max_weight], "endpoints_exact": endpoints,
                      "padding_unchanged": padding_ok, "within_endpoint_bounds": convex_ok,
                      "zero_residual_identity": zero_residual_ok})

    # Known artificial polynomial: verifies coefficient extraction/held-out indexing, not a detector result.
    pairs = list(combinations(range(4), 2))
    a = [.1, -.2, .3, .4]
    b = dict(zip(pairs, [.07, -.11, .13, .02, -.04, .09]))
    def u(bits):
        return sum(a[i] for i in range(4) if bits & (1 << i)) + sum(v for (i, j), v in b.items() if bits & (1 << i) and bits & (1 << j))
    single = [u(1 << i) for i in range(4)]
    recovered = {(i, j): u((1 << i) | (1 << j)) - single[i] - single[j] for i, j in pairs}
    fit_subsets = [bits for bits in range(16) if bits.bit_count() <= 2]
    held_out = [bits for bits in range(16) if bits.bit_count() >= 3]
    max_error = max(abs(u(bits) - sum(single[i] for i in range(4) if bits & (1 << i)) - sum(v for (i, j), v in recovered.items() if bits & (1 << i) and bits & (1 << j))) for bits in held_out)
    assert len(fit_subsets) == 11 and len(held_out) == 5 and set(fit_subsets).isdisjoint(held_out)
    assert max_error < 1e-12
    fixture = {"status": "PASS", "data_kind": "deterministic synthetic software fixtures; no scientific effect estimate",
               "partition_cases": cases, "subset_count": 16, "construction_subsets": fit_subsets,
               "combination_holdout_subsets": held_out, "known_quadratic_recovery_max_error": max_error,
               "model_calls": 0, "training_steps": 0}
    (OUT / "offline_fixture_checks.json").write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"annotation_counts": [len(images), len(annotations)], "invalid_size_boxes": invalid,
                      "outside_boxes": outside, "split_dry_run": audit["split_dry_run"],
                      "software_fixture_status": "PASS", "candidate_fit_images": samples,
                      "elapsed_seconds": round(time.perf_counter()-started, 4)}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
