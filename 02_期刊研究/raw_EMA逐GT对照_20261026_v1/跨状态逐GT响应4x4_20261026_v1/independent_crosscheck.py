"""Independently verify every full-table cell by intersecting original source GT sets."""
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

def read(path):
    return json.loads(Path(path).read_text("utf-8"))

protocol = read(RUN / "protocol.json")
assert sha(RUN / "protocol.json") == (RUN / "protocol.sha256").read_text("utf-8").strip()
base = Path(protocol["input_root"])
paths = {branch: base / protocol["inputs"][name]["path"] for branch, name in
         (("raw", "raw_GT"), ("EMA", "EMA_GT"))}
before = {b: sha(path) for b, path in paths.items()}
assert before["raw"] == protocol["inputs"]["raw_GT"]["sha256"]
assert before["EMA"] == protocol["inputs"]["EMA_GT"]["sha256"]
matrix_path = OUT / "all_GT_results.json"
matrix_hash = sha(matrix_path)
saved = read(matrix_path)
states = ("00", "01", "10", "11")
assert tuple(saved["state_order"]) == states
sets = defaultdict(set)
identities = defaultdict(set)
for branch, path in paths.items():
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            seed = row["seed"]
            identity = (row["image_id"], row["annotation_id"], row["category_id"])
            assert identity not in identities[branch, seed]
            identities[branch, seed].add(identity)
            for threshold in ("50", "75"):
                for contrast, first, second in (("C_minus_B", "B", "C"), ("B_minus_A", "A", "B")):
                    value = row[first + threshold] + row[second + threshold]
                    assert value in states
                    sets[branch, seed, threshold, contrast, value].add(identity)
count = 0
for seed in ("1", "2"):
    assert identities["raw", seed] == identities["EMA", seed]
    assert len(identities["raw", seed]) == 38759
    for threshold in ("50", "75"):
        for contrast in ("C_minus_B", "B_minus_A"):
            matrix = saved["summaries"][seed][threshold][contrast]["matrix"]
            for i, raw_state in enumerate(states):
                for j, ema_state in enumerate(states):
                    value = len(sets["raw", seed, threshold, contrast, raw_state] &
                                sets["EMA", seed, threshold, contrast, ema_state])
                    assert value == matrix[i][j], (seed, threshold, contrast, raw_state, ema_state, value, matrix[i][j])
                    count += 1
after = {branch: sha(path) for branch, path in paths.items()}
assert before == after and sha(matrix_path) == matrix_hash
output = {"status": "PASS", "method": "Independent set intersections of original raw and EMA source CSVs using image, annotation and class identity",
          "cells_checked": count, "source_GT_per_seed": 38759, "source_sha256_before": before,
          "source_sha256_after": after, "matrix_sha256": matrix_hash,
          "new_evaluator_calls": 0, "new_model_forwards": 0, "training": 0,
          "elapsed_seconds": time.monotonic() - started}
with (RUN / "independent_crosscheck.json").open("x", encoding="utf-8") as stream:
    json.dump(output, stream, indent=2, ensure_ascii=False)
print(f"PASS: {count} full-table cells independently reproduced; 0 evaluations or model forwards")
