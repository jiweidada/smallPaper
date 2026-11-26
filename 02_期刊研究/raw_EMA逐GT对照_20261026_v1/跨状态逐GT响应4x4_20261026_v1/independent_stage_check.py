"""Independently verify fixed raw-A stage cells via original source identity sets."""
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
RUN = Path(__file__).resolve().parent
OUT = RUN / "attempt01"
started = time.monotonic()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

protocol = json.loads((RUN / "protocol.json").read_text("utf-8"))
assert sha(RUN / "protocol.json") == (RUN / "protocol.sha256").read_text("utf-8").strip()
base = Path(protocol["input_root"])
names = {"raw": "raw_GT", "EMA": "EMA_GT", "stages": "original_A_stages"}
paths = {label: base / protocol["inputs"][name]["path"] for label, name in names.items()}
before = {label: sha(path) for label, path in paths.items()}
for label, name in names.items():
    assert before[label] == protocol["inputs"][name]["sha256"]
target = OUT / "fixed_raw_A_stage_4x4.csv"
target_hash = sha(target)
states = ("00", "01", "10", "11")
stages = ("G", "S", "K", "C", "M", "T")
sets, group_ids, identities = defaultdict(set), defaultdict(set), defaultdict(set)
for branch in ("raw", "EMA"):
    with paths[branch].open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            seed = row["seed"]
            ident = (row["image_id"], row["annotation_id"], row["category_id"])
            assert ident not in identities[branch, seed]
            identities[branch, seed].add(ident)
            for th in ("50", "75"):
                for contrast, first, second in (("C_minus_B", "B", "C"), ("B_minus_A", "A", "B")):
                    state = row[first + th] + row[second + th]
                    assert state in states
                    sets[branch, seed, th, contrast, state].add(ident)
with paths["stages"].open(encoding="utf-8-sig", newline="") as stream:
    for row in csv.DictReader(stream):
        seed = row["seed"]
        ident = (row["image_id"], row["annotation_id"], row["category_id"])
        assert ident not in identities["stages", seed]
        identities["stages", seed].add(ident)
        for th in ("50", "75"):
            stage = row["raw_state" + th]
            assert stage in stages
            group_ids[seed, th, stage].add(ident)
for seed in ("1", "2"):
    assert identities["raw", seed] == identities["EMA", seed] == identities["stages", seed]
    assert len(identities["raw", seed]) == 38759
    for th in ("50", "75"):
        combined = set().union(*(group_ids[seed, th, stage] for stage in stages))
        assert combined == identities["raw", seed]
        assert sum(len(group_ids[seed, th, stage]) for stage in stages) == 38759
saved = {}
with target.open(encoding="utf-8-sig", newline="") as stream:
    for row in csv.DictReader(stream):
        assert row["group_type"] == "raw_A_stage" and row["group"] == row["fixed_raw_A_stage"]
        th = "50" if row["iou"] == "0.5" else "75" if row["iou"] == "0.75" else None
        assert th is not None
        key = (row["seed"], th, row["contrast"], row["group"], row["raw_response"], row["EMA_response"])
        assert key not in saved
        saved[key] = (int(row["count"]), int(row["group_GT"]))
count = 0
for seed in ("1", "2"):
    for th in ("50", "75"):
        for contrast in ("C_minus_B", "B_minus_A"):
            for stage in stages:
                group = group_ids[seed, th, stage]
                for raw_state in states:
                    raw_in_group = group & sets["raw", seed, th, contrast, raw_state]
                    for ema_state in states:
                        key = (seed, th, contrast, stage, raw_state, ema_state)
                        expected = len(raw_in_group & sets["EMA", seed, th, contrast, ema_state])
                        assert saved[key] == (expected, len(group)), (key, saved[key], expected, len(group))
                        count += 1
assert count == len(saved) == 768
after = {label: sha(path) for label, path in paths.items()}
assert before == after and sha(target) == target_hash
record = {"status": "PASS", "method": "Intersect raw-A stage identities from original stage table with original raw and EMA hit-status identity sets",
          "fixed_raw_A_stage_cells_checked": count, "source_GT_per_seed": 38759,
          "source_sha256_before": before, "source_sha256_after": after, "stage_matrix_sha256": target_hash,
          "new_evaluator_calls": 0, "new_model_forwards": 0, "training": 0,
          "elapsed_seconds": time.monotonic() - started}
with (RUN / "independent_stage_check.json").open("x", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, ensure_ascii=False)
print(f"PASS: {count} fixed raw-A-stage cells independently reproduced; 0 evaluations or model forwards")
