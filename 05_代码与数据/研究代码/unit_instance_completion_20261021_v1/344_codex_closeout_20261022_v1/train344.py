"""344 CPU pretraining review link; no real-training entry point. Based on train342; sealed new targets only.
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
from importlib import import_module as _import_module
_ctx344 = _import_module('344_context')
geom = _ctx344.load_geometry()
from importlib import import_module
head342 = import_module('342_head')
import core as core311
from core import CLASSES, DIM, SIDE
from gcal import GeometryCalibratedHead, copy_R_common

def load_order():
    return json.load(open(P327, encoding="utf-8"))["sample_plan"]["development_ids"]

MANIFEST_PATH = _ctx344.HERE / '344_development6470_逐条SHA清单.json'
MANIFEST_SHA256 = 'dfd1b7167f362e9a71646a4973a51520dcc9730f5b4fbf3a101e3c675e74ea12'
_bound_manifest = None
INPUT_COUNTS = {'manifest_SHA_checks': 0, 'entry_SHA_checks': 0, 'payload_SHA_checks': 0, 'entry_loads': 0, 'nonpositive_raw_masks_verified': 0}

def _manifest():
    global _bound_manifest
    assert _ctx344.sha(MANIFEST_PATH) == MANIFEST_SHA256, 'sealed manifest SHA mismatch'
    INPUT_COUNTS['manifest_SHA_checks'] += 1
    if _bound_manifest is None:
        document = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
        records = document['entries']
        assert len(records) == 6470
        _bound_manifest = {(r['seed'],r['image_id']):r for r in records}
        ids = set(load_order())
        assert set(_bound_manifest) == {(s,i) for s in (1,2) for i in ids}
        assert sum(len(r['fullp_nonpositive_indices']) for r in records) == 1
    return _bound_manifest

def _verify_entry_record(seed, iid, record):
    from pathlib import Path
    expected = (_ctx344.HERE/'344_cache_development_seed2/344_001466.pt' if (seed,iid)==(2,1466)
                else _ctx344.S343/f'343_cache_development_seed{seed}'/f'343_{iid:06d}.pt')
    path = Path(record['path'])
    assert path.resolve() == expected.resolve(), 'unapproved cache source'
    assert _ctx344.sha(path) == record['file_sha256'], 'entry file SHA mismatch'
    INPUT_COUNTS['entry_SHA_checks'] += 1
    e = torch.load(path, map_location='cpu', weights_only=True)
    INPUT_COUNTS['entry_loads'] += 1
    assert e['seed']==seed and e['image_id']==iid
    # Reuse the already-approved byte serialization, with no builder import.
    def payload_sha(value):
        h=hashlib.sha256()
        def add(v):
            if torch.is_tensor(v):
                h.update(b'tensor');h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
                h.update(v.contiguous().numpy().tobytes())
            elif isinstance(v,dict):
                for k in sorted(v):
                    if k=='stage3_payload_sha256':continue
                    h.update(k.encode());add(v[k])
            elif isinstance(v,(list,tuple)):
                h.update(str(len(v)).encode())
                for x in v:add(x)
            else:h.update(json.dumps(v,sort_keys=True).encode())
        add(value);return h.hexdigest()
    assert payload_sha(e)==e['stage3_payload_sha256']==record['payload_sha256'], 'payload SHA mismatch'
    INPUT_COUNTS['payload_SHA_checks'] += 1
    assert e['raw_cache_file_sha256']==record['source_340_file_sha256']
    assert e['checkpoint_sha256']==record['checkpoint_sha256']
    bad=geom.fullp_nonpositive_mask(e['final300_boxes'],e['fullp_query_index'])
    assert bad.nonzero().flatten().tolist()==record['fullp_nonpositive_indices']
    if 'fullp_nonpositive_mask' in e:assert torch.equal(bad,e['fullp_nonpositive_mask'])
    assert int(bad.sum())==int((seed,iid)==(2,1466))
    assert not e['m0_legal_mask'][bad].any() and (e['m0_cand_owner'][bad]==-1).all()
    # Old 343 files are immutable; attach their verified all-false field in RAM.
    e['fullp_nonpositive_mask']=bad
    geom.intersecting_fullp_candidates(e['fullp_boxes'],e['window_xyxy_640'],bad)
    INPUT_COUNTS['nonpositive_raw_masks_verified'] += 1
    return e

def load_entry(seed, iid):
    assert seed in (1,2) and iid in set(load_order()), 'development-only input'
    record=_manifest()[seed,iid]
    return _verify_entry_record(seed,iid,record)

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
        keep = geom.intersecting_fullp_candidates(b, W, e["fullp_nonpositive_mask"])
        bb = b[keep]
        # window-normalized xyxy coordinates (unclipped per 317)
        loc = geom.win_norm_xyxy(bb, W)
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
        "P_boxes": torch.stack(P_boxes),        # window-normalized xyxy
        "P_cls": torch.stack(P_cls),
        "P_scores": torch.stack(P_scores),
        "P_valid": torch.stack(P_valid),
        "entries": entries,
    }
def build_arm(arm, seed):
    """N/S/R from 311_v2 CompletionHead; G from 320_v2 GeometryCalibratedHead (which subclasses R)."""
    torch.manual_seed(2026102100 + seed)
    if arm == "G":
        return head342.GeometryCalibratedHead342()
    return head342.CompletionHead342(arm)

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
            ubox = geom.win_norm_cxcywh(sel, W)
            ulab = e["gt_labels"][UW]
        else:
            ubox = torch.zeros(0,4); ulab = torch.zeros(0,dtype=torch.long)
        per_image_denoms.append(int(UW.sum()))
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
                ubox = geom.win_norm_cxcywh(sel, W)
                ulab = e["gt_labels"][UW]
            else:
                ubox=torch.zeros(0,4); ulab=torch.zeros(0,dtype=torch.long)
            det_terms.append(compute_losses({"pred_logits":nl[i],"pred_boxes":nb[i]},ubox,ulab,
                              num_classes=CLASSES, denominator=denom, k=4))
    ld = sum(t["loss_mal"] + 5.0*t["loss_bbox"] + 2.0*t["loss_giou"] for t in det_terms)
    if denom_mode == "per_image_mean":
        ld = ld / len(batch["entries"])
    losses["loss_det"] = ld
    # --- L_mass (296 L38) ---
    mass = out["mass"]  # [B,10,256]
    lm = []
    for i, e in enumerate(batch["entries"]):
        GW = e["G_W"]; K = e["K_g"]
        mstar = torch.zeros(CLASSES, 256)
        for j in range(e["gt_boxes_640"].shape[0]):
            if bool(GW[j]):
                c = int(e["gt_labels"][j]); mstar[c] += K[j]
        m = mass[i]
        eps = 1e-8
        term = m - mstar + mstar * torch.log((mstar+eps)/(m+eps))
        # At m*=0 only 0*log(0) vanishes; the +m penalty remains (296 §5).
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
        keep = geom.intersecting_fullp_candidates(b, W, e["fullp_nonpositive_mask"])
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
    with open(path, "xb") as checkpoint_file:
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "step": step, "epoch": epoch, "shuffle_seed": shuffle_seed,
                    "rng": rng, "lr": opt.param_groups[0]["lr"]}, checkpoint_file)

def load_checkpoint(path, model, opt):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
    # load_state_dict restores EACH group's LR, including G's x10 group.
    restore_rng(ck["rng"])
    return ck["step"], ck["epoch"], ck["shuffle_seed"]

def run_steps(model, opt, batches, start_step, steps, total_steps):
    """Run optimizer steps; returns loss log."""
    assert 0 <= steps <= 20 and total_steps == 1218, "CPU smoke only"
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert sum(len(b["entries"]) for b in batches) <= 32, "development smoke limit"
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

if __name__ == '__main__':
    raise SystemExit('CPU bounded smoke module only; use the sealed 344 engineering validator. No real-training entry.')
