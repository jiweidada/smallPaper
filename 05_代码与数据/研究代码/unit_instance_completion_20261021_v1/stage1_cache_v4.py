"""340 cache v4: v3 + window-local K_g + RGB128 crop.

DIFF vs v3 (stage1_cache_v3.py):
  1. bilinear_scatter_unit_window(): K_g scattered on the WINDOW-LOCAL 16x16 grid
     (4 px per cell for a 64 px window), using gx=(cx-Wx0)/(Wx1-Wx0)*16-0.5, gy likewise.
     v3 scattered on the whole 640 plane (40 px per cell), unrelated to W. WRONG for L_mass.
  2. crop128: deterministic RGB 128x128 crop from the original image, per 296 L45 formula.
  All other definitions (coords, window selection, M0, G_W/U_W, thresholds) unchanged.
"""
import os, sys, json, hashlib
import numpy as np
import torch
import torch.nn.functional as F
sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1")
import stage1_cache_v3 as S3
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

def bilinear_scatter_window(box_cxcywh_640, W, side=16):
    """296 L21: unit mass on the WINDOW-LOCAL 16x16 grid, bilinear 4-point at the box center.
    gx=(cx-Wx0)/(Wx1-Wx0)*side-0.5 ; gy likewise. Clamp neighbours to [0,side-1]. Sum == 1."""
    x0, y0, x1, y1 = [float(v) for v in W]
    ww = max(x1 - x0, 1e-9); wh = max(y1 - y0, 1e-9)
    cx = float(box_cxcywh_640[0]); cy = float(box_cxcywh_640[1])
    gx = (cx - x0) / ww * side - 0.5
    gy = (cy - y0) / wh * side - 0.5
    ix = int(np.floor(gx)); iy = int(np.floor(gy))
    fx = gx - ix; fy = gy - iy
    K = np.zeros(side * side, dtype=np.float64)
    for dy in (0, 1):
        for dx in (0, 1):
            w = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
            xi = min(max(ix + dx, 0), side - 1)
            yi = min(max(iy + dy, 0), side - 1)
            K[yi * side + xi] += w
    s = K.sum()
    if s <= 0:
        xi = min(max(int(round(gx)), 0), side - 1); yi = min(max(int(round(gy)), 0), side - 1)
        K[:] = 0; K[yi * side + xi] = 1.0; s = 1.0
    return K / s

def crop128_from_window(pil_img, W, orig_wh, size=128, grid=640):
    """296 L45: edge coords [(x0+(u+.5)*W_width/128)*orig_w/640, same for y], minus .5 -> pixel
    centre index; float32 bilinear, border-replicate; RGB/255."""
    ow, oh = orig_wh
    img = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0   # [H,W,3]
    H, Wd = img.shape[:2]
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)             # [1,3,H,W]
    x0, y0, x1, y1 = [float(v) for v in W]
    ww = x1 - x0; wh = y1 - y0
    u = torch.arange(size, dtype=torch.float32) + 0.5                   # (u+.5)
    gx = (x0 + u * ww / size) * (ow / grid) - 0.5                       # pixel-centre index
    gy = (y0 + u * wh / size) * (oh / grid) - 0.5
    # normalised grid_sample coords: align_corners=False -> x_norm = (x+0.5)/W*2-1
    # grid_sample expects grid[..., 0] = x varying along the LAST dim (width) and
    # grid[..., 1] = y varying along dim H; broadcast accordingly.
    xn = ((gx + 0.5) / Wd * 2 - 1).view(1, 1, size).expand(1, size, size)
    yn = ((gy + 0.5) / H * 2 - 1).view(1, size, 1).expand(1, size, size)
    grid = torch.stack([xn, yn], dim=-1)                                # [1,128,128,2]
    out = F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out[0].contiguous()                                          # [3,128,128] float32