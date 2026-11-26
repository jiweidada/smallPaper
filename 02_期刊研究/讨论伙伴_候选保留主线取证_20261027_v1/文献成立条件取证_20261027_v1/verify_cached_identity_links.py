"""Read-only foreign-key/order/hash verification of already saved targets.
No tensor/model import, transform, image read, or scientific outcome recomputation.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
DATA = ROOT / "持续寻找研究主线_20261020_v2/49_augmentation_replay_audit_v4"
ANN = ROOT.parent / "outputs/scale_protection12_20261010_v1/terminal_collection/seed1_target_protect_v1/extracted/reference/train_annotations.json"
NATIVE = ROOT / "code/dn_pre_frozen_diagnostic_20261020_v1/native/engine/data/dataloader.py"

def read(p):
    return json.loads(p.read_text(encoding="utf-8"))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

paths = [DATA / n for n in ["01_transform_traces.json", "02_plain_batch_identity.json", "03_results.json", "05_tagged_source_identity.json"]] + [ANN, NATIVE, Path(__file__).resolve()]
before = {str(p): sha(p) for p in paths}
result49 = read(DATA / "03_results.json")
assert before[str(ANN)] == result49["annotation_sha256"]
traces = read(DATA / "01_transform_traces.json")
tagged = read(DATA / "05_tagged_source_identity.json")
plain = read(DATA / "02_plain_batch_identity.json")
source = read(ANN)
ann_by_id = {a["id"]: (a["image_id"], a["category_id"]) for a in source["annotations"]}
key = lambda v: (v["seed"], v["batch"], v["position"])
t_by_key = {key(v): v for v in tagged}
assert len(t_by_key) == len(tagged)
groups = {}
pre_foreign_keys = 0
for event in traces:
    k = key(event)
    link = t_by_key[k]
    assert link["source_ids"] == event["source_ids"]
    assert link["sources"] == event["sources"]
    assert link["boxes"] == event["final_GT"]
    assert len(link["instances"]) == len(event["final_classes"]) == len(link["boxes"])
    for instance, cls in zip(link["instances"], event["final_classes"]):
        assert ann_by_id[instance["annotation_id"]] == (instance["source_image_id"], cls)
        assert link["sources"][instance["source_slot"]]["image_id"] == instance["source_image_id"]
        pre_foreign_keys += 1
    groups.setdefault((event["seed"], event["batch"]), []).append(event)

index_rows = []
for (seed, batch), events in sorted(groups.items()):
    events.sort(key=lambda v: v["position"])
    row = plain[str(seed)][batch]
    assert len(events) == len(row["samples"]) == len(row["classes"]) == len(row["mixup"]) == 4
    flags = [w is not None for w in row["mixup"]]
    assert all(flags) or not any(flags)
    mixed = all(flags)
    for i, event in enumerate(events):
        sample = row["samples"][i]
        assert sample["image_sha256"] == event["final640_sha256"]
        assert sample["classes"] == event["final_classes"]
        assert sample["source_ids"] == event["source_ids"]
        pre_array = np.asarray(event["final_GT"], dtype=np.float32).reshape(-1, 4)
        assert hashlib.sha256(pre_array.tobytes()).hexdigest() == sample["boxes_sha256"]

        origins = [i, (i - 1) % 4] if mixed else [i]
        classes = []
        boxes = []
        segments = []
        start = 0
        for origin in origins:
            e = events[origin]
            n = len(e["final_classes"])
            classes += e["final_classes"]
            boxes += e["final_GT"]
            segments.append({
                "precollate_event": {"seed": seed, "batch": batch, "position": e["position"]},
                "output_row_start": start,
                "output_row_stop": start + n,
                "lookup": "05_tagged_source_identity.instances[output_row-output_row_start]",
            })
            start += n
        assert classes == row["classes"][i]
        target_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        target_sha = hashlib.sha256(target_array.tobytes()).hexdigest()
        assert target_sha == row["boxes_sha256"][i]
        if mixed:
            assert len(row["mixup"][i]) == len(classes)
        index_rows.append({
            "seed": seed, "batch": batch, "output_slot": i, "mixed": mixed,
            "segments": segments, "target_count": len(classes),
            "classes_exact": True, "boxes_sha256_exact": True,
            "boxes_sha256": target_sha,
        })

assert len(groups) == 64 and len(index_rows) == 256
assert all(sha(Path(p)) == h for p, h in before.items())
index = {
    "status": "CACHED_TARGET_IDENTITY_INDEX_COMPLETE",
    "scope": "Final surviving GT occurrence identities only; no deleted-GT or pixel-visibility claim",
    "unit": "occurrence, not unique originalGT",
    "sources": before,
    "rows": index_rows,
}
with (RUN / "postcollate_identity_index.json").open("x", encoding="utf-8") as f:
    json.dump(index, f, ensure_ascii=False, indent=2)
review = {
    "status": "PASS_METADATA_ONLY",
    "date_client": "2026-10-27",
    "source_hashes_unchanged": before,
    "cached_batches_checked": len(groups),
    "cached_output_target_lists_checked": len(index_rows),
    "precollate_survivor_foreign_key_checks": pre_foreign_keys,
    "precollate_target_boxes_and_classes_match": True,
    "postcollate_target_boxes_sha256_and_classes_match": True,
    "postcollate_surviving_GT_identity_recoverable": True,
    "missing_fields_retained": [
        "complete per-operation deleted-GT timeline/reasons",
        "per-object semantic pixel mask",
        "complete postcollate RGB arrays in49",
    ],
    "operations": {
        "new_data_transforms": 0, "new_pixel_support_masks": 0,
        "new_deletion_or_visibility_outcomes": 0, "model_loads": 0,
        "model_forwards": 0, "evaluations": 0, "training": 0,
    },
    "authorization_scope_correction": "The direct human no-new-forward wording was in this chat's earlier rawBC/4x4 task. It must not be automatically generalized to the peer root chat's newly authorized research stage. This branch stays read-only.",
}
with (RUN / "04_现成缓存身份闭合_只读核验.json").open("x", encoding="utf-8") as f:
    json.dump(review, f, ensure_ascii=False, indent=2)
print(f"PASS: {len(groups)} cached batches / {len(index_rows)} target lists preserve exact classes and FP32 box-byte hashes; {pre_foreign_keys} saved surviving-GT foreign keys valid. No transforms,pixel-support computation,models,or scientific outcome rerun.")

