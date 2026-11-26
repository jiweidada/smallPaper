"""367：集合 IoU 的数学核查（纯数值，无数据、无模型）。
集合 B = { (x1,y1,x2,y2) : |x1-a1|<=d, |y1-b1|<=d, |x2-a2|<=d, |y2-b2|<=d }（四边独立区间，"edge-box" 模型）
以及 B' = { (x,y,w,h) : |x-x0|<=d, |y-y0|<=d, |w-w0|<=d, |h-h0|<=d }（xywh 独立区间，右边界与左边界相关）。
检验 1：对 IoU，"每条边投影到区间"（clip proj）是否等于 max_{b in B} IoU(p,b)。用致密网格暴力求 max 对比。
检验 2：对 GIoU 同样比较，找反例。
检验 3：在 xywh 模型 B' 下，edge-box 投影是否给出可行点/最大值。
"""
import itertools, json, math, random
import numpy as np
random.seed(0); np.random.seed(0)
def iou(p, g):
    iw = max(0, min(p[2], g[2])-max(p[0], g[0])); ih = max(0, min(p[3], g[3])-max(p[1], g[1]))
    i = iw*ih; u = (p[2]-p[0])*(p[3]-p[1])+(g[2]-g[0])*(g[3]-g[1])-i
    return i/u if u > 0 else 0.0
def giou(p, g):
    v = iou(p, g)
    cw = max(p[2], g[2])-min(p[0], g[0]); ch = max(p[3], g[3])-min(p[1], g[1]); c = cw*ch
    iw = max(0, min(p[2], g[2])-max(p[0], g[0])); ih = max(0, min(p[3], g[3])-max(p[1], g[1]))
    u = (p[2]-p[0])*(p[3]-p[1])+(g[2]-g[0])*(g[3]-g[1])-iw*ih
    return v-(c-u)/c
def proj(p, g, d):
    return [min(max(p[k], g[k]-d), g[k]+d) for k in range(4)]
def grid_max(f, p, g, d, n=21, xywh=False):
    t = np.linspace(-d, d, n); best = -9; arg = None
    for e in itertools.product(t, repeat=4):
        if xywh:
            x, y, w, h = g[0]+e[0], g[1]+e[1], (g[2]-g[0])+e[2], (g[3]-g[1])+e[3]
            if w <= 0 or h <= 0: continue
            b = [x, y, x+w, y+h]
        else:
            b = [g[k]+e[k] for k in range(4)]
            if b[2] <= b[0] or b[3] <= b[1]: continue
        v = f(p, b)
        if v > best: best, arg = v, b
    return best, arg
out = {'iou_edgebox': [], 'giou_edgebox': [], 'iou_xywh_vs_edgeproj': []}
cases = []
for _ in range(400):
    w = random.uniform(1.5, 8); h = random.uniform(4, 20); x = random.uniform(0, 10); y = random.uniform(0, 10)
    g = [x, y, x+w, y+h]
    p = [x+random.uniform(-3, 3), y+random.uniform(-3, 3), 0, 0]
    p[2] = p[0]+w*random.uniform(.4, 2.5); p[3] = p[1]+h*random.uniform(.6, 1.6)
    cases.append((p, g))
d = 0.5
worst_iou = 0; worst_giou = 0; giou_cex = None; xywh_gap = 0
for p, g in cases:
    pr = proj(p, g, d)
    if not (pr[2] > pr[0] and pr[3] > pr[1]): continue
    m1, _ = grid_max(iou, p, g, d, n=11)
    gap = m1-iou(p, pr); worst_iou = max(worst_iou, gap)
    m2, a2 = grid_max(giou, p, g, d, n=11)
    gap2 = m2-giou(p, pr)
    if gap2 > worst_giou: worst_giou, giou_cex = gap2, {'pred': p, 'gt': g, 'proj': pr, 'giou_proj': giou(p, pr), 'giou_max_grid': m2, 'argmax': a2}
    m3, _ = grid_max(iou, p, g, d, n=11, xywh=True)
    xywh_gap = max(xywh_gap, abs(m3-m1))
res = {'n_cases': len(cases), 'delta': d,
       'IoU: max over edge-box set minus edge-projection (worst, grid n=11)': worst_iou,
       'GIoU: max over edge-box set minus edge-projection (worst)': worst_giou,
       'GIoU counterexample': giou_cex,
       'IoU: |max over xywh set - max over edge-box set| (worst)': xywh_gap}
json.dump(res, open('367_set_iou_math.json', 'x', encoding='utf-8'), indent=1)
print(json.dumps(res, indent=1))
