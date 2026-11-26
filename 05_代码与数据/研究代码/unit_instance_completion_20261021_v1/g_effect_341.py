"""341 G-effectiveness metric (317v4 §3). Synthetic-testable; real models only after training."""
import os, sys, json
import numpy as np, torch
sys.dont_write_bytecode=True
BASE = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2"
sys.path.insert(0, BASE + chr(92) + '311_unit_instance_core_v2')
sys.path.insert(0, BASE + chr(92) + '320_geometry_calibrated_core_v2')
import torch.nn.functional as F
import core as core311
from core import CLASSES

def _gen_with_kappa(model, mass, safe_boxes, safe_classes, valid, kappa):
    xy = model.association.xy.to(mass.dtype)[None, :, None, :]
    outside = (safe_boxes[:, None, :, :2] - xy).clamp_min(0.)
    outside = outside + (xy - safe_boxes[:, None, :, 2:]).clamp_min(0.)
    wh = safe_boxes[..., 2:] - safe_boxes[..., :2]
    geometry = (outside / wh[:, None]).sum(dim=-1)
    null_cost = kappa[safe_classes]
    return core311.independent_cap(mass, geometry, null_cost, safe_classes, valid)

def residual_over_mass(model, mass, boxes_norm, classes, valid, kappa):
    plan = _gen_with_kappa(model, mass, boxes_norm, classes, valid, kappa)
    return float(plan['residual'].sum() / mass.sum())

def delta_g_effect(model, entries, kappa_override=None):
    diffs = []
    zero = torch.zeros(CLASSES)
    for e in entries:
        W = e['window_xyxy_640']; b = e['fullp_boxes']
        ix = torch.clamp(torch.minimum(b[:,2], W[2]) - torch.maximum(b[:,0], W[0]), min=0)
        iy = torch.clamp(torch.minimum(b[:,3], W[3]) - torch.maximum(b[:,1], W[1]), min=0)
        keep = (ix*iy) > 0
        n = int(keep.sum())
        if n == 0: continue
        bb = b[keep]; cls = e['fullp_labels'][keep]
        ww = float(W[2]-W[0]); whh = float(W[3]-W[1])
        loc = torch.stack([(bb[:,0]+bb[:,2])/2-W[0], (bb[:,1]+bb[:,3])/2-W[1],
                           bb[:,2]-bb[:,0], bb[:,3]-bb[:,1]], -1) / torch.tensor([ww,whh,ww,whh])
        loc = loc.unsqueeze(0); cls2 = cls.unsqueeze(0)
        valid = torch.ones(1, n, dtype=torch.bool)
        mass = torch.full((1, CLASSES, 256), 1.0/256)
        k = model.class_null if kappa_override is None else kappa_override
        r_k = residual_over_mass(model, mass, loc, cls2, valid, k)
        r_0 = residual_over_mass(model, mass, loc, cls2, valid, zero)
        diffs.append(abs(r_k - r_0))
    return (float(np.mean(diffs)) if diffs else 0.0), len(diffs)

def classify(delta):
    if delta < 0.01: return 'G 未有效校准'
    if delta < 0.03: return '校准能力受限'
    return 'G 可作正式对照'

def measure_trained_G(model, entries):
    """Full report per 317v4 §3: Delta + per-class kappa values/displacement."""
    d, n = delta_g_effect(model, entries)
    kap = model.class_null.detach().tolist()
    return {'delta': d, 'windows': n, 'verdict': classify(d),
            'kappa_final': kap, 'kappa_initial': [0.0]*len(kap),
            'kappa_displacement': [abs(x) for x in kap]}