"""337 stage-1 cache pipeline v2: corrected coordinate handling.

Coordinate contract (from 296 L2/L11/L45 and 317 §2):
  * Global input: natural RGB -> native PIL Resize (640,640)  [stretch, NOT aspect-preserving] -> float32/255
  * Model pred_boxes: normalized cxcywh in that 640x640 plane, values ~[0,1].
    -> multiply cx,cy,w,h by 640, then convert to xyxy -> "640 plane xyxy".
  * GT: annotation pixel xywh in ORIGINAL image (orig_w,orig_h)
    -> xyxy in original pixels -> scale x by 640/orig_w, y by 640/orig_h -> same "640 plane xyxy".
  * Window W: one 64x64-cell window on the 640 plane; W_xyxy in 640-plane pixels.
  * Crop: W in 640 plane -> original image pixels by *orig_w/640, *orig_h/640 (296 L45 formula).
  All M0 / G_W / U_W / K_g operate in the 640 plane.
"""
import os, sys, json, hashlib, time
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("MKL_NUM_THREADS","2")
os.environ.setdefault("MKL_THREADING_LAYER","SEQUENTIAL")
sys.dont_write_bytecode = True

DEIM_NATIVE = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\dn_pre_frozen_diagnostic_20261020_v1\native"
EXP_DIR     = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\fine_residual_screen24_20261009_v2"
BASE_V2     = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2"
HEAD_V2     = BASE_V2 + r"\311_unit_instance_core_v2"
GCAL_V2     = BASE_V2 + r"\320_geometry_calibrated_core_v2"

def sha_file(p, chunk=1<<20):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda: f.read(chunk), b''): h.update(c)
    return h.hexdigest()

def sha_tensor(t):
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()

# ---------- coordinate conversion, two independent functions + unit tests ----------
def norm_cxcywh_to_640_xyxy(boxes_norm, size=640):
    """pred_boxes (normalized cxcywh) -> 640-plane xyxy.  boxes_norm: [N,4]"""
    b = boxes_norm.float() * float(size)
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)

def orig_xywh_to_640_xyxy(gt_xywh, orig_w, orig_h, size=640):
    """annotation pixel xywh (original image) -> 640-plane xyxy (stretch-resize)."""
    x, y, w, h = gt_xywh.float().unbind(-1)
    x1 = x; y1 = y; x2 = x + w; y2 = y + h
    sx = float(size) / float(orig_w); sy = float(size) / float(orig_h)
    return torch.stack([x1*sx, y1*sy, x2*sx, y2*sy], dim=-1)

def window_640_to_orig_xyxy(W, orig_w, orig_h, size=640):
    """Window in 640-plane pixels -> original-image pixel xyxy (296 L45 mapping)."""
    sx = float(orig_w)/float(size); sy = float(orig_h)/float(size)
    return torch.tensor([W[0]*sx, W[1]*sy, W[2]*sx, W[3]*sy], dtype=torch.float32)
# ---------- model / forward ----------
def load_real_model(ckpt_path):
    sys.path.insert(0, DEIM_NATIVE); sys.path.insert(0, EXP_DIR)
    from geometry import ResidualGeometryDEIM
    bundle = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), "bundle", "resolved.yml")
    cfg = json.load(open(bundle, encoding="utf-8"))
    m = ResidualGeometryDEIM(cfg["ResidualGeometryDEIM"]["recipe"], arm="baseline")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    m.load_state_dict(ck["model"], strict=True)
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m

@torch.no_grad()
def global_forward_640(model, pil_img):
    """native eval640: PIL Resize to (640,640) stretch, float32/255. Returns raw pred_boxes (norm cxcywh) and logits."""
    import torchvision.transforms.functional as TF
    img = TF.resize(pil_img, [640, 640])
    x = TF.to_tensor(img).unsqueeze(0)
    out = model(x)
    return out["pred_boxes"][0].detach(), out["pred_logits"][0].detach()

def make_fullp(pred_boxes_norm, logits, num_classes=10, topq=300, score_thr=0.001):
    """317/296 L10: sigmoid -> flatten(query,class) -> top300 -> score strictly > .001.
    Keep query_index so M0 can retrieve the full 300x10 logits row."""
    s = torch.sigmoid(logits)
    topv, topi = torch.topk(s.flatten(0), topq, dim=-1)
    labels = topi % num_classes
    qidx = topi // num_classes
    boxes640 = norm_cxcywh_to_640_xyxy(pred_boxes_norm)[qidx]
    keep = topv > score_thr
    return {"boxes": boxes640[keep], "scores": topv[keep], "labels": labels[keep],
            "query_index": qidx[keep], "logits_row": logits[qidx[keep]]}

def select_window(final300_boxes640, win_px=64, size=640):
    """296 L11 + 262.make_grid/cell_ids: a WINDOW is 64x64 PIXELS; grid stride is 64 px,
    so the 640 plane holds 10x10 = 100 candidate windows (NOT 64x64 cells of 10 px).
    Density = count of ORIGINAL final-300 prediction centers falling in each window;
    highest count; row-major earliest on ties (262: range over y, then x)."""
    import math
    nx = (size + win_px - 1)//win_px      # 10
    ny = nx
    cx = ((final300_boxes640[:,0]+final300_boxes640[:,2])/2).numpy()
    cy = ((final300_boxes640[:,1]+final300_boxes640[:,3])/2).numpy()
    cnt = np.zeros((ny,nx), dtype=np.int64)
    for x,y in zip(cx,cy):
        if not (0 <= x <= size and 0 <= y <= size): continue
        i = min(int(x//win_px), nx-1); j = min(int(y//win_px), ny-1)
        cnt[i,j]+=1
    flat = cnt.reshape(-1)
    idx = int(np.argmax(flat)) if flat.max()>0 else 0
    return (idx//nx, idx%nx), cnt

def window_xyxy(rc, win_px=64, size=640):
    r,c = rc
    x = c*win_px; y = r*win_px
    return torch.tensor([x, y, min(x+win_px,size), min(y+win_px,size)], dtype=torch.float32)

def bilinear_scatter_unit(box_cxcywh_640, side=16, size=640):
    """296/292: unit mass K_g on 16x16 grid at box center, bilinear 4-point, clamp to border.
    Grid coords are within the 640 plane, normalized by size before mapping to cells."""
    cx = float(box_cxcywh_640[0])/size
    cy = float(box_cxcywh_640[1])/size
    gx = cx*side - 0.5; gy = cy*side - 0.5
    x0 = int(np.floor(gx)); y0 = int(np.floor(gy))
    fx = gx - x0; fy = gy - y0
    K = np.zeros(side*side, dtype=np.float64)
    for dy in (0,1):
        for dx in (0,1):
            w = (fx if dx else 1-fx) * (fy if dy else 1-fy)
            xi = min(max(x0+dx,0),side-1); yi = min(max(y0+dy,0),side-1)
            K[yi*side+xi] += w
    s = K.sum()
    if s <= 0:
        xi=min(max(int(round(gx)),0),side-1); yi=min(max(int(round(gy)),0),side-1)
        K[:]=0; K[yi*side+xi]=1.0; s=1.0
    return K/s

def gt_complete_in_window(gt_xyxy_640, W):
    x0,y0,x1,y1 = [float(v) for v in gt_xyxy_640]
    return bool(x0 >= float(W[0])-1e-6 and y0 >= float(W[1])-1e-6
                and x1 <= float(W[2])+1e-6 and y1 <= float(W[3])+1e-6)
def m0_native(fullp, gt_boxes_640, gt_labels, num_classes=10):
    """M0 on fullP x all-image native GT (640 plane). Legal edge = same class & FP32 IoU>=0.5.
    Cost uses the candidate's REAL 300x10 logits row (296 L13: native matcher cost on full-image
    normalized boxes; class term uses the candidate query's full logits)."""
    sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
    from uic_m0 import m0_lex
    N = fullp["boxes"].shape[0]; M = gt_boxes_640.shape[0]
    if N == 0 or M == 0:
        return {"gt_explained": torch.zeros(M, dtype=torch.bool),
                "cand_owner": torch.full((N,), -1, dtype=torch.long),
                "matched_pairs": [], "coverage": 0, "C": torch.zeros(0, M)}
    # normalized cxcywh for cost (296 L13: full-image normalized boxes)
    def to_cxcywh_norm(xyxy):
        x0,y0,x1,y1 = xyxy.unbind(-1)
        return torch.stack([(x0+x1)/2/640, (y0+y1)/2/640, (x1-x0)/640, (y1-y0)/640], -1)
    cb = to_cxcywh_norm(fullp["boxes"])
    gb = to_cxcywh_norm(gt_boxes_640)
    logits = fullp["logits_row"]          # [N,10] real rows
    return m0_lex(cb, logits, gb, gt_labels)

def m0_independent_numpy(fullp, gt_boxes_640, gt_labels, iou_thr=0.5):
    """Independent pure-numpy legal-edge + greedy coverage check (no uic_m0).
    Returns per-GT covered flag from a maximal (not lex-optimal) matching, used only to
    cross-check the *legal edge set* and that M0's coverage is attainable."""
    def iou(a, b):
        lt = np.maximum(a[None,:,:2], b[:,None,:2]); rb = np.minimum(a[None,:,2:], b[:,None,2:])
        wh = np.clip(rb-lt, 0, None); inter = wh[...,0]*wh[...,1]
        aa = np.clip(a[:,2]-a[:,0],0,None)*np.clip(a[:,3]-a[:,1],0,None)
        ab = np.clip(b[:,2]-b[:,0],0,None)*np.clip(b[:,3]-b[:,1],0,None)
        return inter/np.maximum(aa[None,:]+ab[:,None]-inter, 1e-12)
    if fullp["boxes"].shape[0]==0 or gt_boxes_640.shape[0]==0:
        return {"legal_edges": 0, "greedy_coverage": 0}
    I = iou(fullp["boxes"].numpy(), gt_boxes_640.numpy())     # [N,M]
    same = fullp["labels"].numpy()[:,None] == gt_labels.numpy()[None,:]
    legal = same & (I >= iou_thr)
    # simple greedy maximal matching (only for cross-check of edge set / attainability)
    used_c = set(); cov = 0
    for j in range(legal.shape[1]):
        for i in np.nonzero(legal[:, j])[0]:
            if int(i) not in used_c:
                used_c.add(int(i)); cov += 1; break
    return {"legal_edges": int(legal.sum()), "greedy_coverage": int(cov)}

def build_targets_v2(fullp, gt_boxes_640, gt_labels, W):
    """G_W complete-in-W; U_W = G_W minus M0-explained; K_g unit scatter for complete GT."""
    m0 = m0_native(fullp, gt_boxes_640, gt_labels)
    M = gt_boxes_640.shape[0]
    G_W = torch.tensor([gt_complete_in_window(gt_boxes_640[i].tolist(), W) for i in range(M)], dtype=torch.bool) if M else torch.zeros(0,dtype=torch.bool)
    U_W = G_W & (~m0["gt_explained"])
    K = torch.zeros(M, 256)
    for i in range(M):
        if bool(G_W[i]):
            x0,y0,x1,y1 = [float(v) for v in gt_boxes_640[i]]
            K[i] = torch.from_numpy(bilinear_scatter_unit([(x0+x1)/2, (y0+y1)/2, x1-x0, y1-y0])).float()
    return {"m0": m0, "G_W": G_W, "U_W": U_W, "K_g": K}