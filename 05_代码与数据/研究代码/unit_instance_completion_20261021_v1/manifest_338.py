"""338 manifest + 337-consistency checker."""
import os, sys, json, hashlib, time
import torch
sys.dont_write_bytecode = True
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OUT)
import build_338 as B
import stats_338 as ST

def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()

def build_manifest(seed_times):
    imgs, by, order = B.load_meta()
    man = {
        "item": "338 manifest",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "coordinate_contract": "stage1_cache_v2.py (337): pred_boxes normalized cxcywh *640 -> xyxy; GT pixel xywh * (640/orig_w,640/orig_h); window = 64x64 px, 10x10 grid, row-major ties",
        "definitions_source": {
            "stage1_cache_v2.py": sha_file(os.path.join(OUT, "stage1_cache_v2.py")),
            "uic_m0.py": sha_file(os.path.join(OUT, "uic_m0.py")),
        },
        "sources": {},
        "entries_per_seed": {},
        "elapsed_secs": seed_times,
    }
    for name, p in [("checkpoint_seed1", B.CKPT[1]), ("checkpoint_seed2", B.CKPT[2]),
                    ("annotation_train", B.ANN), ("sample_plan", B.SP)]:
        man["sources"][name] = {"path": p, "sha256": sha_file(p)}
    for seed in (1, 2):
        d = os.path.join(OUT, f"338_cache_seed{seed}")
        files = sorted(f for f in os.listdir(d) if f.endswith(".pt"))
        man["entries_per_seed"][f"seed{seed}"] = {"count": len(files),
                                                  "dir": d,
                                                  "total_bytes": sum(os.path.getsize(os.path.join(d, f)) for f in files)}
    man["total_entries"] = sum(v["count"] for v in man["entries_per_seed"].values())
    man["disk_bytes"] = sum(v["total_bytes"] for v in man["entries_per_seed"].values())
    json.dump(man, open(os.path.join(OUT, "338_manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return man

def check_337_consistency():
    """Same image_id in full cache must match 337 subset G_W/U_W exactly."""
    p337 = os.path.join(OUT, "337_result_stage1_subset.json")
    d337 = json.load(open(p337, encoding="utf-8"))
    rows = {r["image_id"]: r for r in d337["tests"]["per_image"]}
    out = {"checked": 0, "mismatch": [], "missing": []}
    for iid, r337 in rows.items():
        p = os.path.join(OUT, "338_cache_seed1", f"{iid:06d}.pt")
        if not os.path.exists(p):
            out["missing"].append(iid); continue
        e = torch.load(p, map_location="cpu", weights_only=True)
        ok = int(e["n_G_W"]) == int(r337["n_GW"]) and int(e["n_U_W"]) == int(r337["n_UW"])
        if not ok:
            out["mismatch"].append({"image_id": iid, "337": [r337["n_GW"], r337["n_UW"]],
                                    "338": [int(e["n_G_W"]), int(e["n_U_W"])]})
        out["checked"] += 1
    out["all_consistent"] = (not out["mismatch"] and not out["missing"])
    return out

if __name__ == "__main__":
    man = build_manifest({"seed1": None, "seed2": None})
    cons = check_337_consistency()
    json.dump(cons, open(os.path.join(OUT, "338_337_consistency.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps({"manifest_entries": man["entries_per_seed"], "consistency": cons}, ensure_ascii=False, indent=2)[:2500])
