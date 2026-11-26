"""348e 只读：(1) tiny Geo0 的中心最近候选是否已与别的 GT IoU>=.5（被邻居占用）；
(2) 仅用标注算术：不同输入形状下 GT 宽/高<4 的数量（无模型）。输出 348e_neighbor_and_resize.json"""
import json, collections
from pathlib import Path
import torch
torch.set_num_threads(2)
ROOT = Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文')
CACHE = ROOT/'code/unit_instance_completion_20261021_v1/342_codex_stage3_20261022_v1'
ANN = Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val\annotations\val.json')
OUT = ROOT/'持续寻找研究主线_20261020_v2/348e_neighbor_and_resize.json'
def iou_mat(a, b):
    lt = torch.max(a[:, None, :2], b[None, :, :2]); rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb-lt).clamp(min=0); inter = wh[..., 0]*wh[..., 1]
    aa = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]); ab = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    return inter/(aa[:, None]+ab[None]-inter).clamp(min=1e-12)
def iou_pair(a, b):
    ix = (torch.min(a[..., 2], b[..., 2])-torch.max(a[..., 0], b[..., 0])).clamp(min=0)
    iy = (torch.min(a[..., 3], b[..., 3])-torch.max(a[..., 1], b[..., 1])).clamp(min=0)
    inter = ix*iy; aa = (a[..., 2]-a[..., 0])*(a[..., 3]-a[..., 1]); ab = (b[..., 2]-b[..., 0])*(b[..., 3]-b[..., 1])
    return inter/(aa+ab-inter).clamp(min=1e-12)
res = {}
for seed in (1, 2):
    acc = collections.defaultdict(int)
    for f in sorted((CACHE/f'342_cache_val_v2_seed{seed}').glob('342_*.pt')):
        d = torch.load(f, map_location='cpu', weights_only=False)
        b = d['final300_boxes'].double()*640
        P = torch.cat([b[:, :2]-b[:, 2:]/2, b[:, :2]+b[:, 2:]/2], -1)
        G = d['gt_boxes_640'].double()
        if len(G) == 0: continue
        I = iou_mat(G, P); mx = I.max(1).values
        gc = (G[:, :2]+G[:, 2:])/2
        nn = torch.cdist(gc, b[:, :2]).argmin(1)
        ss = (G[:, 2:]-G[:, :2]).min(1).values
        g0 = (ss < 4) & (mx < .5)
        Ion = I[:, nn].clone()                    # [G, G]: 第 j 列 = 所有 GT 对 nn[j] 的 IoU
        Ion.fill_diagonal_(0)
        occ = Ion.max(0).values >= .5             # nn[j] 已与别的 GT IoU>=.5
        Qn = P[nn]
        Ox = iou_pair(G, torch.stack([G[:, 0], Qn[:, 1], G[:, 2], Qn[:, 3]], -1))
        acc['tiny_geo0'] += int(g0.sum())
        acc['tiny_geo0_nn_occupied_by_other_gt_iou.5'] += int((g0 & occ).sum())
        acc['tiny_geo0_nn_free_and_Ox_ge.5'] += int((g0 & ~occ & (Ox >= .5)).sum())
        acc['tiny_geo0_nn_occupied_and_Ox_ge.5'] += int((g0 & occ & (Ox >= .5)).sum())
    res[f'seed{seed}'] = dict(acc)
ann = json.load(open(ANN, encoding='utf-8')); imgs = {im['id']: im for im in ann['images']}
shapes = {'640x640_stretch': (640, 640), '853x480_equal_pixels_16:9': (853, 480), '800x512': (800, 512),
          '1088x612_same_short_vs_long': (1088, 612), '960x540': (960, 540), '1280x720': (1280, 720)}
R = {}
for name, (W, H) in shapes.items():
    c = collections.Counter()
    for a in ann['annotations']:
        im = imgs[a['image_id']]; x, y, w, h = a['bbox']
        ww = w*W/im['width']; hh = h*H/im['height']
        c['w_lt4'] += ww < 4; c['h_lt4'] += hh < 4; c['short_lt4'] += min(ww, hh) < 4
        c['short_lt8'] += min(ww, hh) < 8; c['area_lt_16sq'] += ww*hh < 256
    R[name] = dict(c) | {'pixels': W*H}
res['annotation_only_resize_counts'] = R
res['_note'] = 'occupied=中心最近候选与另一GT IoU>=.5；第二部分仅标注算术，无模型；IoU 对轴向独立缩放不变'
json.dump(res, open(OUT, 'x', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
