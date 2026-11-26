"""336 stage-1 cache builder. CPU only, development/val only, never heldout.
Real ResidualGeometryDEIM (native engine snapshot) + 311_v2 M0. No proxy model."""
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

def load_real_model(ckpt_path):
    sys.path.insert(0, DEIM_NATIVE); sys.path.insert(0, EXP_DIR)
    from geometry import ResidualGeometryDEIM
    cfg = json.load(open(os.path.join(os.path.dirname(os.path.dirname(ckpt_path)),"bundle","resolved.yml"), encoding="utf-8"))
    m = ResidualGeometryDEIM(cfg["ResidualGeometryDEIM"]["recipe"], arm="baseline")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    m.load_state_dict(ck["model"], strict=True)
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m

@torch.no_grad()
def global_forward_640(model, pil_img):
    """native eval640: resize to 640x640, float32, /255, CHW."""
    import torchvision.transforms.functional as TF
    img = TF.resize(pil_img, [640,640])
    x = TF.to_tensor(img)                     # [3,640,640] float /255
    x = x.unsqueeze(0)
    out = model(x)
    return out["pred_boxes"][0], out["pred_logits"][0]   # [300,4],[300,10]

def make_fullp(boxes, logits, num_classes=10, topq=300, score_thr=0.001):
    """317: sigmoid->flatten(1)->top300->score strictly > .001; same query different classes kept."""
    s = torch.sigmoid(logits)
    flat = s.flatten(0)                       # [300*10]
    topv, topi = torch.topk(flat, topq, dim=-1)
    labels = topi % num_classes
    boxidx = topi // num_classes
    sel = boxes[boxidx]
    keep = topv > score_thr
    return {"boxes": sel[keep], "scores": topv[keep], "labels": labels[keep],
            "query_index": boxidx[keep]}

def select_window(cx, cy, grid=64, win_px=640):
    """317原文：原final300预测中心数最多格，row-major解并列。返回 (r,c)。"""
    cs = win_px/grid
    cnt = np.zeros((grid,grid), dtype=np.int64)
    for x,y in zip(cx,cy):
        i = min(max(int(x//cs),0),grid-1); j = min(max(int(y//cs),0),grid-1)
        cnt[i,j]+=1
    flat = cnt.reshape(-1)
    if flat.max()==0: return (0,0)
    idx = int(np.argmax(flat))   # first max in row-major
    return (idx//grid, idx%grid)

def window_xyxy(rc, grid=64, win_px=640):
    cs=win_px/grid; r,c=rc
    return torch.tensor([c*cs, r*cs, (c+1)*cs, (r+1)*cs], dtype=torch.float32)


def bilinear_scatter_unit(box_cxcywh, side=16):
    """296/292: unit mass K_g on 16x16 grid by bilinear 4-point at box center, clamped to border.
    Returns [side*side] summing exactly 1."""
    cx, cy = float(box_cxcywh[0]), float(box_cxcywh[1])
    # continuous grid coords (cell centers at (i+0.5)/side)
    gx = cx*side - 0.5
    gy = cy*side - 0.5
    x0 = int(np.floor(gx)); y0 = int(np.floor(gy))
    fx = gx - x0; fy = gy - y0
    K = np.zeros(side*side, dtype=np.float64)
    for dy in (0,1):
        for dx in (0,1):
            w = (fx if dx else 1-fx) * (fy if dy else 1-fy)
            xi = min(max(x0+dx,0),side-1); yi = min(max(y0+dy,0),side-1)
            K[yi*side+xi] += w
    s = K.sum()
    if s <= 0:  # degenerate: put all on nearest cell
        xi=min(max(int(round(gx)),0),side-1); yi=min(max(int(round(gy)),0),side-1)
        K[:]=0; K[yi*side+xi]=1.0; s=1.0
    return K/s, s

def gt_in_window(gt_xyxy, W):
    """complete = fully inside W (317: 完整位于W)."""
    x0,y0,x1,y1 = gt_xyxy
    return bool(x0 >= W[0] and y0 >= W[1] and x1 <= W[2] and y1 <= W[3])

def build_targets(fullp, gt_boxes_xyxy, gt_labels, W, side=16):
    """M0 on fullP x all-image native valid GT; G_W complete-in-W; U_W=G_W minus M0-explained.
    Returns dict with per-GT flags, K_g for G_W, per-candidate owner/null target."""
    sys.path.insert(0, HEAD_V2)
    import core as core311
    N = fullp["boxes"].shape[0]
    M = gt_boxes_xyxy.shape[0]
    if M == 0 or N == 0:
        return {"m0":{"gt_explained":torch.zeros(M,dtype=torch.bool),
                      "cand_owner":torch.full((N,),-1,dtype=torch.long),
                      "matched_pairs":[],"coverage":0},
                "G_W":torch.zeros(M,dtype=torch.bool),
                "U_W":torch.zeros(M,dtype=torch.bool),
                "K_g":torch.zeros(M,side*side)}
    # M0: use the 335-fixed m0 implementation (task1, no cost bias)
    sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
    from uic_m0 import m0_lex
    # convert fullP boxes (xyxy in 640) to normalized cxcywh
    bw = fullp["boxes"].clone()
    cx = (bw[:,0]+bw[:,2])/2/640; cy=(bw[:,1]+bw[:,3])/2/640
    w = (bw[:,2]-bw[:,0])/640; h=(bw[:,3]-bw[:,1])/640
    cb = torch.stack([cx,cy,w,h],-1)
    gb = gt_boxes_xyxy.clone()
    gcx=(gb[:,0]+gb[:,2])/2/640; gcy=(gb[:,1]+gb[:,3])/2/640
    gw=(gb[:,2]-gb[:,0])/640; gh=(gb[:,3]-gb[:,1])/640
    gbn = torch.stack([gcx,gcy,gw,gh],-1)
    # class logits for M0: need 10-dim logits per candidate -> reconstruct from fullP labels/scores
    # build logits where candidate's label has logit=logit(score), others very low
    logits = torch.full((N,10), -20.0)
    for i in range(N):
        logits[i, int(fullp["labels"][i])] = torch.logit(fullp["scores"][i].clamp(1e-6,1-1e-6))
    r = m0_lex(cb, logits, gbn, gt_labels)
    # G_W complete in W
    G_W = torch.tensor([gt_in_window(tuple(gt_boxes_xyxy[i].tolist()), W) for i in range(M)], dtype=torch.bool)
    E_W = r["gt_explained"] & G_W
    U_W = G_W & (~r["gt_explained"])
    K = torch.zeros(M, side*side)
    for i in range(M):
        if bool(G_W[i]):
            kk,_ = bilinear_scatter_unit(gbn[i].tolist(), side)
            K[i] = torch.from_numpy(kk).float()
    return {"m0":r, "G_W":G_W, "U_W":U_W, "K_g":K}
