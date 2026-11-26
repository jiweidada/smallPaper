"""338 independent statistics: reads ONLY from on-disk cache entries, re-verifies SHA."""
import os, sys, json, math, hashlib
import numpy as np
import torch
sys.dont_write_bytecode = True
OUT = os.path.dirname(os.path.abspath(__file__))
P327 = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2\327_训练实施输入索引.json"
ORDER = json.load(open(P327, encoding="utf-8"))["sample_plan"]["development_ids"]

def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k/n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    r = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return [round((c-r)/d, 6), round((c+r)/d, 6)]

def sha_payload(e):
    h = hashlib.sha256()
    for k in ["fullp_boxes","fullp_scores","fullp_logits_row","G_W","U_W","K_g","gt_boxes_640"]:
        t = e[k]
        h.update(k.encode()); h.update(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes())
    return h.hexdigest()

def load_seed(seed, verify=True):
    d = os.path.join(OUT, f"338_cache_seed{seed}")
    rows = []; bad = []
    for iid in ORDER:
        p = os.path.join(d, f"{iid:06d}.pt")
        if not os.path.exists(p):
            bad.append({"image_id": iid, "reason": "missing"}); continue
        e = torch.load(p, map_location="cpu", weights_only=True)
        if verify:
            got = sha_payload(e)
            if got != e.get("sha256"):
                bad.append({"image_id": iid, "reason": "sha_mismatch", "stored": e.get("sha256"), "recomputed": got}); continue
        rows.append({"image_id": iid, "n_gt": e["n_gt"], "n_G_W": e["n_G_W"], "n_U_W": e["n_U_W"],
                     "window_cell": list(e["window_cell"]),
                     "gt_boxes_640": e["gt_boxes_640"], "G_W": e["G_W"]})
    return rows, bad

def dist(values, bins=(0,1,2,3,5)):
    out = {}
    for b in bins:
        out[str(b)] = sum(1 for v in values if v == b)
    out["6+"] = sum(1 for v in values if v >= 6)
    return out

def gt_density_window_counterfactual(row):
    """Diagnostic only: count GT fully inside the 64px window with the most GT centers."""
    gt = row["gt_boxes_640"]
    if gt.numel() == 0: return 0
    g = gt.numpy()
    cx = (g[:,0]+g[:,2])/2; cy = (g[:,1]+g[:,3])/2
    cnt = np.zeros((10,10), dtype=np.int64)
    for x,y in zip(cx,cy):
        if 0 <= x <= 640 and 0 <= y <= 640:
            cnt[min(int(y//64),9), min(int(x//64),9)] += 1
    r, c = np.unravel_index(int(np.argmax(cnt)), (10,10))
    W = [c*64, r*64, min(c*64+64,640), min(r*64+64,640)]
    return int(sum(1 for i in range(g.shape[0])
                   if g[i,0] >= W[0]-1e-6 and g[i,1] >= W[1]-1e-6
                   and g[i,2] <= W[2]+1e-6 and g[i,3] <= W[3]+1e-6))

def main():
    res = {"item": "338 全量缓存统计", "order": "327 sorted development_ids", "seeds": {}}
    allrows = {}
    for seed in (1, 2):
        rows, bad = load_seed(seed)
        allrows[seed] = rows
        n = len(rows)
        gw_pos = sum(1 for r in rows if r["n_G_W"] > 0)
        uw_pos = sum(1 for r in rows if r["n_U_W"] > 0)
        gw_total = sum(r["n_G_W"] for r in rows)
        uw_total = sum(r["n_U_W"] for r in rows)
        uw_hist = [r["n_U_W"] for r in rows]
        overflow_imgs = sum(1 for r in rows if r["n_U_W"] > 4)
        overflow_gt = sum(max(0, r["n_U_W"]-4) for r in rows)
        gt_win_pos = sum(1 for r in rows if gt_density_window_counterfactual(r) > 0)
        res["seeds"][f"seed{seed}"] = {
            "entries": n, "bad_entries": bad[:20], "bad_count": len(bad),
            "G_W_positive": {"count": gw_pos, "rate": round(gw_pos/n,6) if n else None,
                             "wilson95": wilson(gw_pos, n)},
            "U_W_positive": {"count": uw_pos, "rate": round(uw_pos/n,6) if n else None,
                             "wilson95": wilson(uw_pos, n)},
            "G_W_total": gw_total, "U_W_total": uw_total,
            "U_W_over_G_W": round(uw_total/gw_total,6) if gw_total else None,
            "G_W_per_image_dist": dist([r["n_G_W"] for r in rows]),
            "U_W_per_image_dist": dist(uw_hist),
            "U_W_gt4": {"images": overflow_imgs, "overflow_gt_total": overflow_gt},
            "gt_density_window_diagnostic": {"G_W_positive_imgs": gt_win_pos,
                                             "rate": round(gt_win_pos/n,6) if n else None},
        }
    # cross-seed window agreement
    if allrows[1] and allrows[2]:
        m1 = {r["image_id"]: r for r in allrows[1]}
        m2 = {r["image_id"]: r for r in allrows[2]}
        common = sorted(set(m1) & set(m2))
        same = sum(1 for i in common if m1[i]["window_cell"] == m2[i]["window_cell"])
        gw_both = sum(1 for i in common if m1[i]["n_G_W"] > 0 and m2[i]["n_G_W"] > 0)
        gw_same_gt = 0
        for i in common:
            if m1[i]["G_W"].shape == m2[i]["G_W"].shape and m1[i]["G_W"].numel():
                if torch.equal(m1[i]["G_W"], m2[i]["G_W"]): gw_same_gt += 1
        res["cross_seed"] = {"common_images": len(common), "same_window": same,
                             "same_window_rate": round(same/len(common),6) if common else None,
                             "same_window_wilson95": wilson(same, len(common)),
                             "both_G_W_positive": gw_both,
                             "identical_G_W_mask": gw_same_gt}
    # combined (two seeds pooled as independent seed-runs, NOT independent images)
    pooled_n = sum(len(allrows[s]) for s in (1,2))
    pooled_gw = sum(1 for s in (1,2) for r in allrows[s] if r["n_G_W"] > 0)
    pooled_uw = sum(1 for s in (1,2) for r in allrows[s] if r["n_U_W"] > 0)
    res["pooled"] = {"note": "两 seed 同一批图，不是独立样本；仅按 seed-run 合并计数",
                     "n": pooled_n, "G_W_positive": pooled_gw, "U_W_positive": pooled_uw,
                     "G_W_rate": round(pooled_gw/pooled_n,6) if pooled_n else None,
                     "U_W_rate": round(pooled_uw/pooled_n,6) if pooled_n else None}
    json.dump(res, open(os.path.join(OUT, "338_result_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps({k: res[k] for k in ("seeds","cross_seed","pooled")}, ensure_ascii=False, indent=2)[:4000])

if __name__ == "__main__":
    main()
