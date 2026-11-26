"""340 full cache builder: v3 window/M0 + v4 window-local K_g + crop128. development 3235 x 2 seeds. CPU only.
Definitions are frozen to stage1_cache_v2.py (337). No rule changes.
Resumable: completed image_id entries are skipped after SHA verification.
"""
import os, sys, json, time, hashlib
import numpy as np
import torch
sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache_v3 as S
import stage1_cache_v4 as S4
from PIL import Image

OUT = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
SP  = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\08_匹配聚合覆盖探针_20261019_v1\sample_plan.json"
ANN = r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\annotations\train.json"
IMGDIR = r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\images"
CKPT = {
  1: r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1\seed1\run\last.pth",
  2: r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\data\00_实验计划与证据索引\本次24轮双基线模型下载_20261009_v1\seed2\run\last.pth",
}
P327 = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2\327_训练实施输入索引.json"

def load_meta():
    ann = json.load(open(ANN, encoding="utf-8"))
    imgs = {im["id"]: im for im in ann["images"]}
    by = {}
    for a in ann["annotations"]: by.setdefault(a["image_id"], []).append(a)
    order = json.load(open(P327, encoding="utf-8"))["sample_plan"]["development_ids"]
    return imgs, by, order

def entry_path(seed, iid):
    d = os.path.join(OUT, f"340_cache_seed{seed}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{iid:06d}.pt")

def _kg_window(tg, gt, W):
    """340: K_g on the WINDOW-LOCAL 16x16 grid for G_W targets only; 0 elsewhere."""
    K = torch.zeros(gt.shape[0], 256)
    for i in range(gt.shape[0]):
        if bool(tg["G_W"][i]):
            x0, y0, x1, y1 = [float(v) for v in gt[i]]
            K[i] = torch.from_numpy(S4.bilinear_scatter_window([(x0+x1)/2, (y0+y1)/2, 1, 1], W)).float()
    return K

def build_entry(model, im, items):
    pid = im["id"]; ow, oh = im["width"], im["height"]
    pil = Image.open(os.path.join(IMGDIR, im["file_name"])).convert("RGB")
    pb, logits = S.global_forward_640(model, pil)
    p300 = S.norm_cxcywh_to_640_xyxy(pb)
    fullp = S.make_fullp(pb, logits)
    rc, cnt = S.select_window(p300)
    W = S.window_xyxy(rc)
    if items:
        xywh = torch.tensor([a["bbox"] for a in items], dtype=torch.float32)
        labs = torch.tensor([a["category_id"] for a in items], dtype=torch.long)
        gt = S.orig_xywh_to_640_xyxy(xywh, ow, oh)
    else:
        gt = torch.zeros(0, 4); labs = torch.zeros(0, dtype=torch.long)
    tg = S.build_targets_v2(fullp, gt, labs, W)
    return {
        "image_id": pid, "orig_wh": [ow, oh], "file_name": im["file_name"],
        "final300_boxes": pb, "final300_logits": logits,
        "fullp_boxes": fullp["boxes"], "fullp_scores": fullp["scores"],
        "fullp_labels": fullp["labels"], "fullp_query_index": fullp["query_index"],
        "fullp_logits_row": fullp["logits_row"],
        "window_cell": list(rc), "window_xyxy_640": W,
        "G_W": tg["G_W"], "U_W": tg["U_W"],
        "m0_gt_explained": tg["m0"]["gt_explained"], "m0_cand_owner": tg["m0"]["cand_owner"],
        "m0_coverage": int(tg["m0"]["coverage"]),
        "K_g": _kg_window(tg, gt, W), "gt_boxes_640": gt, "gt_labels": labs,
        "crop128": S4.crop128_from_window(pil, W, (ow, oh)),
        "n_gt": int(gt.shape[0]), "n_G_W": int(tg["G_W"].sum()), "n_U_W": int(tg["U_W"].sum()),
    }

def sha_entry(e):
    """SHA over the numeric payload fields."""
    h = hashlib.sha256()
    for k in ["fullp_boxes","fullp_scores","fullp_logits_row","G_W","U_W","K_g","gt_boxes_640","crop128"]:
        t = e[k]
        h.update(k.encode()); h.update(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes())
    return h.hexdigest()
def run_build(seed, limit=None, verbose=True):
    torch.set_num_threads(2)
    imgs, by, order = load_meta()
    if limit: order = order[:limit]
    model = S.load_real_model(CKPT[seed])
    done = 0; skipped = 0; t0 = time.time()
    logp = os.path.join(OUT, f"340_build_seed{seed}.log")
    with open(logp, "a", encoding="utf-8") as lf:
        for n, iid in enumerate(order, 1):
            p = entry_path(seed, iid)
            if os.path.exists(p):
                try:
                    e = torch.load(p, map_location="cpu", weights_only=True)
                    if e.get("image_id") == iid and sha_entry(e) == e.get("sha256"):
                        skipped += 1
                        continue
                except Exception:
                    pass  # corrupt -> rebuild
            e = build_entry(model, imgs[iid], by.get(iid, []))
            e["sha256"] = sha_entry(e)
            tmp = p + ".tmp"
            torch.save(e, tmp)
            os.replace(tmp, p)
            done += 1
            if verbose and (n % 100 == 0 or n == len(order)):
                msg = f"[seed{seed}] {n}/{len(order)} built={done} skipped={skipped} elapsed={time.time()-t0:.1f}s"
                print(msg, flush=True); lf.write(msg+"\n"); lf.flush()
    return {"seed": seed, "total": len(order), "built": done, "skipped": skipped,
            "elapsed_secs": round(time.time()-t0, 1)}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    print(json.dumps(run_build(a.seed, a.limit), ensure_ascii=False))