"""Frozen descriptive join of saved per-GT records; standard library only."""
import csv
import hashlib
import json
import math
import os
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
RUN = Path(__file__).resolve().parent
OUT = RUN / "attempt01"
OUT.mkdir(exist_ok=False)
START = time.monotonic()
STOP = threading.Event()
OPS = {"new_model_forwards": 0, "head_calls": 0, "evaluator_calls": 0,
       "checkpoint_loads": 0, "model_builds": 0, "training": 0,
       "GPU": False, "remote_compute": False, "test_access": False}
STATES = ("00", "01", "10", "11")
STAGES = ("G", "S", "K", "C", "M", "T")
AREAS = ("small", "medium", "large")
SIZES = ("[0,4)", "[4,8)", "[8,16)", "[16,32)", "[32,inf)")
CONTRASTS = {"C_minus_B": ("B", "C"), "B_minus_A": ("A", "B")}

def dump(name, obj):
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def guard():
    while not STOP.wait(2):
        why = None
        if time.monotonic() - START > 600:
            why = "600 second hard budget"
        elif sum(p.stat().st_size for p in RUN.rglob("*") if p.is_file()) > 100 * 1024**2:
            why = "100 MiB output budget"
        if why:
            try:
                dump("hard_budget_stop.json", {"reason": why, "operations": OPS})
            finally:
                os._exit(124)

threading.Thread(target=guard, daemon=True).start()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read_json(path):
    return json.loads(Path(path).read_text("utf-8"))

def gt_key(row):
    return tuple(int(row[k]) for k in ("seed", "image_id", "annotation_id", "category_id"))

def read_table(path, columns):
    rows = {}
    annotation_keys = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(set(columns).issubset(reader.fieldnames or []), f"Missing columns in {path}")
        for number, row in enumerate(reader, 2):
            key = gt_key(row)
            require(key[0] in (1, 2) and 0 <= key[3] <= 9, f"Invalid identity {path}:{number}: {key}")
            require(key not in rows, f"Duplicate full key {path}:{number}: {key}")
            require((key[0], key[2]) not in annotation_keys, f"Duplicate annotation within seed {path}:{number}: {key}")
            annotation_keys.add((key[0], key[2]))
            rows[key] = {name: row[name] for name in columns}
    require(len(rows) == 77518, f"Unexpected row count in {path}: {len(rows)}")
    return rows

def size_for(short):
    return SIZES[0] if short < 4 else SIZES[1] if short < 8 else SIZES[2] if short < 16 else SIZES[3] if short < 32 else SIZES[4]

def area_for(area):
    return "small" if area < 1024 else "medium" if area < 9216 else "large"

def to_flag(value, context):
    require(value in ("0", "1"), f"Non-binary flag {context}: {value!r}")
    return int(value)

def write_csv(name, fields, iterable):
    with (OUT / name).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(iterable)

def ratio(n, d):
    return n / d if d else None

def matrix_summary(cells):
    m = [[cells[4 * r + c] for c in range(4)] for r in range(4)]
    raw = [sum(row) for row in m]
    ema = [sum(m[r][c] for r in range(4)) for c in range(4)]
    return {"matrix": m, "raw_margins": raw, "EMA_margins": ema,
            "GT": sum(raw), "shared_gain": m[1][1], "raw_gain": raw[1], "EMA_gain": ema[1],
            "shared_gain_fraction_of_raw_gain": ratio(m[1][1], raw[1]),
            "shared_gain_fraction_of_EMA_gain": ratio(m[1][1], ema[1]),
            "shared_loss": m[2][2], "raw_loss": raw[2], "EMA_loss": ema[2],
            "shared_loss_fraction_of_raw_loss": ratio(m[2][2], raw[2]),
            "shared_loss_fraction_of_EMA_loss": ratio(m[2][2], ema[2]),
            "both_states_both_miss": m[0][0], "both_states_both_hit": m[3][3]}

def matrix_rows(seed, th, contrast, group_type, label, stage, cells):
    total = sum(cells)
    for r, raw in enumerate(STATES):
        for c, ema in enumerate(STATES):
            n = cells[4 * r + c]
            yield {"seed": seed, "iou": th / 100, "contrast": contrast, "group_type": group_type,
                   "group": label, "fixed_raw_A_stage": stage, "raw_response": raw, "EMA_response": ema,
                   "count": n, "group_GT": total, "fraction_of_group": ratio(n, total)}

FIELDS = ("seed", "iou", "contrast", "group_type", "group", "fixed_raw_A_stage",
          "raw_response", "EMA_response", "count", "group_GT", "fraction_of_group")

def main():
    protocol_path = RUN / "protocol.json"
    protocol_hash = sha(protocol_path)
    require(protocol_hash == (RUN / "protocol.sha256").read_text("utf-8").strip(), "Frozen protocol SHA mismatch")
    p = read_json(protocol_path)
    require(p["id"] == "raw_EMA_GT_response_4x4_fixed_raw_A_stage_v1", "Unexpected protocol ID")
    root = Path(p["input_root"])
    paths = {name: root / x["path"] for name, x in p["inputs"].items()}
    before = {name: sha(path) for name, path in paths.items()}
    for name, value in before.items():
        require(value == p["inputs"][name]["sha256"], f"Input SHA mismatch before analysis: {name}")
    script_hash = sha(Path(__file__))
    protocol_sha_file_hash = sha(RUN / "protocol.sha256")
    dump("execution_receipt.json", {"protocol_sha256": protocol_hash, "script_sha256": script_hash,
         "runtime": sys.executable, "python_version": sys.version, "bound_input_sha256": before,
         "operations": OPS, "started_unix": time.time()})
    for name in ("raw_B_verification", "prior_stage_verification", "EMA_verification", "raw_operation_verification"):
        require(read_json(paths[name])["status"] == "PASS", f"Prior verification not PASS: {name}")
    raw_ops = read_json(paths["raw_operation_results"])
    for seed in (1, 2):
        boundary = raw_ops["seeds"][str(seed)]["precision_boundaries"]
        for th in (50, 75):
            require(boundary[f"raw_edge_vs_native_bbox_pair_crossings_{th}"] == 0,
                    f"Historical raw geometry threshold crossing at seed {seed}, {th}")

    keycols = ("seed", "image_id", "annotation_id", "category_id")
    flags = tuple(a + str(t) for t in (50, 75) for a in ("A", "B", "C"))
    raw = read_table(paths["raw_GT"], keycols + ("area_original", "short_side_640") + flags)
    ema = read_table(paths["EMA_GT"], keycols + ("area_original", "short_side_640", "size_group") + flags)
    hist = read_table(paths["original_A_stages"], keycols + ("short_side_640", "size_group", "raw_state50", "raw_state75", "raw_hit50", "raw_hit75", "EMA_hit50", "EMA_hit75"))
    rk, ek, hk = set(raw), set(ema), set(hist)
    require(rk == ek == hk, f"Key sets differ: raw-only {len(rk-ek)}, EMA-only {len(ek-rk)}, raw-minus-stage {len(rk-hk)}, stage-minus-raw {len(hk-rk)}")
    keys_by_seed = {s: {k[1:] for k in rk if k[0] == s} for s in (1, 2)}
    require(keys_by_seed[1] == keys_by_seed[2], "Validation population differs between seeds")
    for s in (1, 2):
        group = keys_by_seed[s]
        require(len(group) == 38759 and len({k[0] for k in group}) == 548 and
                len({k[1] for k in group}) == 38759 and 0 in {k[1] for k in group},
                f"Population/GT0 mismatch seed {s}")
    prior = {"raw": read_json(paths["raw_B_results"]), "EMA": read_json(paths["EMA_results"])}
    joined = []
    observed = {(b, s, t, field): set() for b in ("raw", "EMA") for s in (1, 2) for t in (50, 75)
                for field in ("B_new_vs_A", "B_lost_vs_A", "C_new_vs_B", "C_lost_vs_B")}
    for key in sorted(rk):
        s, iid, aid, cat = key
        a, b, h = raw[key], ema[key], hist[key]
        ar, ae = float(a["area_original"]), float(b["area_original"])
        sr, se, sh = float(a["short_side_640"]), float(b["short_side_640"]), float(h["short_side_640"])
        require(all(math.isfinite(v) and v >= 0 for v in (ar, ae, sr, se, sh)), f"Invalid metadata for {key}")
        require(abs(ar-ae) <= 1e-8 and abs(sr-se) <= 1e-8 and abs(sr-sh) <= 1e-8, f"Metadata mismatch for {key}")
        require(b["size_group"] == h["size_group"] == size_for(sr), f"Size group mismatch for {key}")
        row = {"seed": s, "image_id": iid, "annotation_id": aid, "category_id": cat,
               "area_original": ar, "area_group": area_for(ar), "short_side_640": sr, "size_group": size_for(sr)}
        for th in (50, 75):
            stage = h["raw_state" + str(th)]
            require(stage in STAGES, f"Unknown raw A stage for {key}, {th}: {stage}")
            row["raw_A_stage" + str(th)] = stage
            for branch, source in (("raw", a), ("EMA", b)):
                for arm in ("A", "B", "C"):
                    row[branch + "_" + arm + str(th)] = to_flag(source[arm + str(th)], f"{branch} {key} {arm}{th}")
                hist_hit = to_flag(h[branch + "_hit" + str(th)], f"historical {branch} {key} {th}")
                require(row[branch + "_A" + str(th)] == hist_hit, f"A hit mismatch {branch} {key} {th}")
                aa, bb, cc = (row[branch + "_" + arm + str(th)] for arm in ("A", "B", "C"))
                if bb and not aa: observed[branch, s, th, "B_new_vs_A"].add(aid)
                if aa and not bb: observed[branch, s, th, "B_lost_vs_A"].add(aid)
                if cc and not bb: observed[branch, s, th, "C_new_vs_B"].add(aid)
                if bb and not cc: observed[branch, s, th, "C_lost_vs_B"].add(aid)
            require((stage == "T") == bool(row["raw_A" + str(th)]), f"raw A stage/hit mismatch {key} {th}")
        joined.append(row)
    checks = 0
    for (branch, seed, th, name), got in observed.items():
        saved = prior[branch]["seeds"][str(seed)]["sets"][str(th)][name]
        require(len(saved) == len(set(saved)), f"Duplicate IDs in prior result {branch} {seed} {th} {name}")
        expected = set(map(int, saved))
        require(got == expected, f"Prior GT set mismatch {branch} seed {seed} th {th} {name}: missing={len(expected-got)} extra={len(got-expected)}")
        checks += 1
    print(f"Join preflight PASS: {len(joined)} rows; {checks} prior GT-ID sets replayed", flush=True)

    global_counts = Counter()
    for row in joined:
        seed = row["seed"]
        for th in (50, 75):
            for contrast, (old, new) in CONTRASTS.items():
                rs = str(row[f"raw_{old}{th}"]) + str(row[f"raw_{new}{th}"])
                es = str(row[f"EMA_{old}{th}"]) + str(row[f"EMA_{new}{th}"])
                global_counts[seed, th, contrast, rs, es] += 1
    all_matrices = {}
    all_rows = []
    for s in (1, 2):
        all_matrices[str(s)] = {}
        for th in (50, 75):
            all_matrices[str(s)][str(th)] = {}
            for contrast in CONTRASTS:
                cells = [global_counts[s, th, contrast, r, e] for r in STATES for e in STATES]
                require(sum(cells) == 38759, f"Full table population mismatch {s} {th} {contrast}")
                all_matrices[str(s)][str(th)][contrast] = matrix_summary(cells)
                all_rows.extend(matrix_rows(s, th, contrast, "all", "all", "all", cells))
    write_csv("all_GT_4x4.csv", FIELDS, all_rows)
    dump("all_GT_results.json", {"state_order": STATES, "row_axis": "raw", "column_axis": "EMA",
         "population_per_seed": 38759, "summaries": all_matrices})
    print("Full-GT matrices saved before fixed-stage stratification", flush=True)

    counts = Counter()
    for row in joined:
        seed, cat, area, size = row["seed"], str(row["category_id"]), row["area_group"], row["size_group"]
        for th in (50, 75):
            stage = row["raw_A_stage" + str(th)]
            groups = (("raw_A_stage", stage, stage), ("class", cat, "all"), ("area", area, "all"),
                      ("size", size, "all"), ("stage_by_class", cat, stage),
                      ("stage_by_area", area, stage), ("stage_by_size", size, stage))
            for contrast, (old, new) in CONTRASTS.items():
                rs = str(row[f"raw_{old}{th}"]) + str(row[f"raw_{new}{th}"])
                es = str(row[f"EMA_{old}{th}"]) + str(row[f"EMA_{new}{th}"])
                for typ, label, fixed_stage in groups:
                    counts[seed, th, contrast, typ, label, fixed_stage, rs, es] += 1
    def cells_for(s, t, c, typ, label, stage):
        return [counts[s, t, c, typ, label, stage, r, e] for r in STATES for e in STATES]
    def add_vectors(vectors):
        return [sum(v[i] for v in vectors) for i in range(16)]
    group_definitions = (("class", tuple(map(str, range(10)))), ("area", AREAS), ("size", SIZES))
    stages_summary, stage_rows, group_rows = {}, [], []
    partitions_checked = 0
    for s in (1, 2):
        stages_summary[str(s)] = {}
        for th in (50, 75):
            stages_summary[str(s)][str(th)] = {}
            for contrast in CONTRASTS:
                global_cells = [global_counts[s, th, contrast, r, e] for r in STATES for e in STATES]
                stage_mats = {stage: cells_for(s, th, contrast, "raw_A_stage", stage, stage) for stage in STAGES}
                require(add_vectors(stage_mats.values()) == global_cells, f"Stage partition mismatch {s} {th} {contrast}")
                partitions_checked += 1
                stages_summary[str(s)][str(th)][contrast] = {}
                for stage, cells in stage_mats.items():
                    stages_summary[str(s)][str(th)][contrast][stage] = matrix_summary(cells)
                    stage_rows.extend(matrix_rows(s, th, contrast, "raw_A_stage", stage, stage, cells))
                for typ, labels in group_definitions:
                    mats = [cells_for(s, th, contrast, typ, label, "all") for label in labels]
                    require(add_vectors(mats) == global_cells, f"{typ} partition mismatch {s} {th} {contrast}")
                    partitions_checked += 1
                    for label, cells in zip(labels, mats):
                        group_rows.extend(matrix_rows(s, th, contrast, typ, label, "all", cells))
                    for stage in STAGES:
                        stage_type = "stage_by_" + typ
                        parts = [cells_for(s, th, contrast, stage_type, label, stage) for label in labels]
                        require(add_vectors(parts) == stage_mats[stage], f"{stage_type} partition mismatch {s} {th} {contrast} {stage}")
                        partitions_checked += 1
                        for label, cells in zip(labels, parts):
                            group_rows.extend(matrix_rows(s, th, contrast, stage_type, label, stage, cells))
    write_csv("fixed_raw_A_stage_4x4.csv", FIELDS, stage_rows)
    dump("fixed_raw_A_stage_results.json", {"state_order": STATES, "stage_order": STAGES, "summaries": stages_summary})
    write_csv("fixed_class_area_size_4x4.csv", FIELDS, group_rows)
    write_csv("joined_GT.csv", tuple(joined[0]), joined)
    after = {name: sha(path) for name, path in paths.items()}
    changed = [name for name in before if before[name] != after[name]]
    require(not changed, f"Inputs changed during run: {changed}")
    require(sha(protocol_path) == protocol_hash and sha(RUN / "protocol.sha256") == protocol_sha_file_hash,
            "Frozen protocol or hash file changed during run")
    require(sha(Path(__file__)) == script_hash, "Analysis script changed during run")
    dump("input_manifest.json", {"protocol_sha256": protocol_hash, "script_sha256": script_hash,
         "inputs": {name: {"path": str(path), "sha256_before": before[name], "sha256_after": after[name]}
                    for name, path in paths.items()}})
    dump("verification.json", {"status": "PASS", "GT_rows": len(joined), "GT_per_seed": 38759,
         "images_per_seed": 548, "GT0_present_both_seeds": True, "prior_exact_ID_sets_replayed": checks,
         "partitions_reconstruct_cellwise": partitions_checked, "input_files_rehashed": len(after),
         "changed_inputs": changed, "protocol_unchanged": True, "script_unchanged": True,
         "geometry_threshold_crossings_in_prior_raw_map": {"50": 0, "75": 0},
         "operations": OPS, "elapsed_seconds": time.monotonic() - START})
    print(f"PASS: {partitions_checked} cellwise partition checks; {len(after)} input hashes unchanged", flush=True)

try:
    main()
except BaseException as exc:
    try:
        dump("failure.json", {"status": "FAIL", "error": repr(exc), "traceback": traceback.format_exc(),
                              "elapsed_seconds": time.monotonic() - START, "operations": OPS})
    finally:
        raise
finally:
    STOP.set()
