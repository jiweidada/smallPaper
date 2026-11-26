"""341 training link = train340 + 317v4 (G class_null lr x10). CPU only; NO GPU.
317v4: only G_geom class_null uses base 1e-3 / lo 1e-4 (v3 curve x10); all other params unchanged.
Training entry is marked: 未授权，不得在GPU上运行.
"""
import os, sys, json, math, hashlib, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.dont_write_bytecode = True

OUT = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
BASE_V2 = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2"
HEAD_V2 = BASE_V2 + r"\311_unit_instance_core_v2"
GCAL_V2 = BASE_V2 + r"\320_geometry_calibrated_core_v2"
P327 = BASE_V2 + r"\327_训练实施输入索引.json"
sys.path.insert(0, HEAD_V2); sys.path.insert(0, GCAL_V2)
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\dn_pre_frozen_diagnostic_20261020_v1\native")
import core as core311
from core import CLASSES, DIM, SIDE
from gcal import GeometryCalibratedHead, copy_R_common

def load_order():
    return json.load(open(P327, encoding="utf-8"))["sample_plan"]["development_ids"]

def load_entry(seed, iid):
    p = os.path.join(OUT, f"340_cache_seed{seed}", f"{iid:06d}.pt")
    return torch.load(p, map_location="cpu", weights_only=True)

def collate(entries):
    """Build a batch: rgb crops, P candidates (window-local), GT, M0/U_W, K_g."""
    B = len(entries)
    rgb = torch.stack([e["crop128"] for e in entries])
    # P_W = fullP whose box intersects W with positive area (296 L12)
    P_boxes, P_cls, P_scores, P_valid = [], [], [], []
    maxn = 0
    per = []
    for e in entries:
        W = e["window_xyxy_640"]
        b = e["fullp_boxes"]
        ix = torch.clamp(torch.minimum(b[:,2], W[2]) - torch.maximum(b[:,0], W[0]), min=0)
        iy = torch.clamp(torch.minimum(b[:,3], W[3]) - torch.maximum(b[:,1], W[1]), min=0)
        inter = ix * iy
        keep = inter > 0
        bb = b[keep]
        # window-local xyxy coordinates (unclipped per 317)
        loc = torch.stack([bb[:,0]-W[0], bb[:,1]-W[1], bb[:,2]-W[0], bb[:,3]-W[1]], -1)
        per.append((loc, e["fullp_labels"][keep], e["fullp_scores"][keep]))
        maxn = max(maxn, loc.shape[0])
    for loc, lab, sc in per:
        n = loc.shape[0]
        pad = maxn - n
        P_boxes.append(F.pad(loc, (0,0,0,pad)))
        P_cls.append(F.pad(lab, (0,pad)))
        P_scores.append(F.pad(sc, (0,pad)))
        v = torch.zeros(maxn, dtype=torch.bool); v[:n] = True
        P_valid.append(v)
    return {
        "rgb": rgb,
        "P_boxes": torch.stack(P_boxes),        # window-local xyxy
        "P_cls": torch.stack(P_cls),
        "P_scores": torch.stack(P_scores),
        "P_valid": torch.stack(P_valid),
        "entries": entries,
    }
def build_arm(arm, seed):
    """N/S/R from 311_v2 CompletionHead; G from 320_v2 GeometryCalibratedHead (which subclasses R)."""
    torch.manual_seed(2026102100 + seed)
    if arm == "G":
        return GeometryCalibratedHead()
    return core311.CompletionHead(arm)

def make_common_init(arm, seed):
    """All arms start from the same R/common initialisation (317 §8: G/R initial equality)."""
    torch.manual_seed(2026102100 + seed)
    ref = core311.CompletionHead("R")
    model = build_arm(arm, seed)
    if arm == "G":
        copy_R_common(ref, model)
    elif arm == "S":
        core311.copy_common(ref, model)
    return model

def compute_losses_340(arm, model, batch, denom_mode="batch_all_uw"):
    """Returns dict of losses. L_det from uic_losses.compute_losses (335 parity).
    denom_mode: 'batch_all_uw' (主控解读) or 'per_image_mean' (report only)."""
    sys.path.insert(0, OUT)
    from uic_losses import compute_losses
    rgb = batch["rgb"]; Pb = batch["P_boxes"]; Pc = batch["P_cls"]; Ps = batch["P_scores"]; Pv = batch["P_valid"]
    out = model.forward_features(model.trunk(rgb), Pb, Pc, Ps, Pv) if arm != "G" else model.forward_features(model.trunk(rgb), Pb, Pc, Ps, Pv)
    # new head outputs: 4 queries normalized cxcywh in window-local coords
    nb = out["pred_boxes"]; nl = out["pred_logits"]
    losses = {"loss_det": 0.0, "loss_mass": 0.0, "loss_assoc": 0.0}
    # --- L_det: per image compute_losses with |U_W| denominator per spec ---
    det_terms = []
    per_image_denoms = []
    for i, e in enumerate(batch["entries"]):
        UW = e["U_W"]; gt = e["gt_boxes_640"]; W = e["window_xyxy_640"]
        # U_W GT in window-local normalized cxcywh (window 64px)
        sel = gt[UW]
        if sel.numel():
            x0,y0,x1,y1 = sel[:,0],sel[:,1],sel[:,2],sel[:,3]
            ww = float(W[2]-W[0]); wh = float(W[3]-W[1])
            ubox = torch.stack([(x0+x1)/2-W[0].item(), (y0+y1)/2-W[1].item(),
                                x1-x0, y1-y0], -1) / torch.tensor([ww,wh,ww,wh])
            ulab = e["gt_labels"][UW]
        else:
            ubox = torch.zeros(0,4); ulab = torch.zeros(0,dtype=torch.long)
        per_image_denoms.append(max(1, int(UW.sum())))
        r = compute_losses({"pred_logits": nl[i], "pred_boxes": nb[i]}, ubox, ulab,
                           num_classes=CLASSES, denominator=float(max(1,int(UW.sum()))), k=4)
        det_terms.append(r)
    if denom_mode == "batch_all_uw":
        denom = float(max(1, sum(per_image_denoms)))
        # recompute with the batch denominator
        det_terms = []
        for i, e in enumerate(batch["entries"]):
            UW = e["U_W"]; gt = e["gt_boxes_640"]; W = e["window_xyxy_640"]
            sel = gt[UW]
            if sel.numel():
                x0,y0,x1,y1 = sel[:,0],sel[:,1],sel[:,2],sel[:,3]
                ww=float(W[2]-W[0]); wh=float(W[3]-W[1])
                ubox = torch.stack([(x0+x1)/2-W[0].item(),(y0+y1)/2-W[1].item(), x1-x0, y1-y0],-1)/torch.tensor([ww,wh,ww,wh])
                ulab = e["gt_labels"][UW]
            else:
                ubox=torch.zeros(0,4); ulab=torch.zeros(0,dtype=torch.long)
            det_terms.append(compute_losses({"pred_logits":nl[i],"pred_boxes":nb[i]},ubox,ulab,
                              num_classes=CLASSES, denominator=denom, k=4))
    ld = sum(t["loss_mal"] + 5.0*t["loss_bbox"] + 2.0*t["loss_giou"] for t in det_terms) / len(batch["entries"])
    losses["loss_det"] = ld
    # --- L_mass (296 L38) ---
    mass = out["mass"]  # [B,10,256]
    lm = []
    for i, e in enumerate(batch["entries"]):
        GW = e["G_W"]; K = e["K_g"]
        if GW.sum() == 0:
            lm.append(torch.tensor(0.0)); continue
        mstar = torch.zeros(CLASSES, 256)
        for j in range(e["gt_boxes_640"].shape[0]):
            if bool(GW[j]):
                c = int(e["gt_labels"][j]); mstar[c] += K[j]
        m = mass[i]
        eps = 1e-8
        term = m - mstar + mstar * torch.log((mstar+eps)/(m+eps))
        term = torch.where(mstar > 0, term, torch.zeros_like(term))
        lm.append(term.sum() / max(1, int(GW.sum())))
    losses["loss_mass"] = torch.stack(lm).mean()
    # --- L_assoc (296 L39) ---
    # plan["T"]: [B,J,256] transport to candidates; plan["z"]: [B,J] null prob.
    # For each valid candidate j: if its M0 owner g is in G_W -> -sum_i K_g[i]*log(T[j,i]+1e-8)
    # else -> -log(z[j]+1e-8). Mean within explained group and null group, then mean over non-empty groups.
    T_all = out["plan"]["T"]; z_all = out["plan"]["z"]
    la = []
    for i, e in enumerate(batch["entries"]):
        # need the same P_W subset used in collate
        W = e["window_xyxy_640"]; b = e["fullp_boxes"]
        ix = torch.clamp(torch.minimum(b[:,2], W[2]) - torch.maximum(b[:,0], W[0]), min=0)
        iy = torch.clamp(torch.minimum(b[:,3], W[3]) - torch.maximum(b[:,1], W[1]), min=0)
        keep = (ix*iy) > 0
        n = int(keep.sum())
        owner = e["m0_cand_owner"][keep]         # per kept candidate: GT index or -1
        GW = e["G_W"]; K = e["K_g"]
        T = T_all[i][:, :n]; z = z_all[i][:n]    # T[i]: [256, n] (cells x candidates); z[i]: [n]
        pos_terms = []; neg_terms = []
        for j in range(n):
            o = int(owner[j])
            if o >= 0 and bool(GW[o]):
                pos_terms.append(-(K[o] * torch.log(T[:, j] + 1e-8)).sum())
            else:
                neg_terms.append(-torch.log(z[j] + 1e-8))
        if not pos_terms and not neg_terms:
            la.append(torch.tensor(0.0)); continue
        groups = []
        if pos_terms: groups.append(torch.stack(pos_terms).mean())
        if neg_terms: groups.append(torch.stack(neg_terms).mean())
        la.append(torch.stack(groups).mean())
    losses["loss_assoc"] = torch.stack(la).mean()
    losses["loss_total"] = losses["loss_det"] + losses["loss_mass"] + losses["loss_assoc"]
    return losses, out
KAPPA_LR_MULT = 10.0   # 317v4

def build_optimizer(model, kappa_mult=KAPPA_LR_MULT):
    """317 §5 + 317v4: AdamW lr1e-4; bias/norm no decay; class_null no decay AND lr x10."""
    decay, no_decay, kappa = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if n.startswith("class_null"):
            kappa.append(p)
        elif p.ndim == 1 or n.endswith("bias") or ("norm" in n) or ("bn" in n):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": 1e-4, "lr": 1e-4},
              {"params": no_decay, "weight_decay": 0.0, "lr": 1e-4}]
    if kappa:
        groups.append({"params": kappa, "weight_decay": 0.0, "lr": 1e-4 * kappa_mult})
    return torch.optim.AdamW(groups, lr=1e-4, betas=(0.9, 0.999), eps=1e-8)

def lr_at_kappa(step, total, warmup=100, base=1e-3, lo=1e-4):
    """317v4: class_null curve = v3 curve x10 (base 1e-3, lo 1e-4)."""
    return lr_at(step, total, warmup=warmup, base=base, lo=lo)

def lr_at(step, total, warmup=100, base=1e-4, lo=1e-5):
    if step < warmup:
        return lo + (base - lo) * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return lo + 0.5 * (base - lo) * (1 + math.cos(math.pi * min(1.0, t)))

def capture_rng():
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate()}

def restore_rng(st):
    torch.set_rng_state(st["torch"]); np.random.set_state(st["numpy"]); random.setstate(st["python"])

def save_checkpoint(path, model, opt, step, epoch, shuffle_seed, rng):
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "step": step, "epoch": epoch, "shuffle_seed": shuffle_seed,
                "rng": rng, "lr": opt.param_groups[0]["lr"]}, path)

def load_checkpoint(path, model, opt):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
    for g in opt.param_groups: g["lr"] = ck["lr"]
    restore_rng(ck["rng"])
    return ck["step"], ck["epoch"], ck["shuffle_seed"]

def run_steps(model, opt, batches, start_step, steps, total_steps):
    """Run optimizer steps; returns loss log."""
    log = []
    for k in range(steps):
        b = batches[(start_step + k) % len(batches)]
        lr = lr_at(start_step + k, total_steps)
        kappa_lr = lr_at_kappa(start_step + k, total_steps)
        for gi, g in enumerate(opt.param_groups):
            # 317v4: last group (class_null) uses the x10 curve
            g["lr"] = kappa_lr if gi == len(opt.param_groups) - 1 and len(opt.param_groups) == 3 else lr
        opt.zero_grad(set_to_none=True)
        arm = getattr(model, "arm", "R")
        losses, out = compute_losses_340(arm, model, b)
        losses["loss_total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        log.append({k: float(v.detach()) for k, v in losses.items() if torch.is_tensor(v)})
    return log