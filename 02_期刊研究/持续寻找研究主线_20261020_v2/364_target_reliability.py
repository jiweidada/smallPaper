"""364 只读：IoU^1.5 质量目标的"可靠度"（信号方差占比）。
样本：每个 GT 的 IoU 最大 final300 query，且 raw IoU>=.5（即会成为正样本的那类配对）。
取整噪声：GT 各边原图 U[-0.5,0.5] px（VisDrone 标注全为整数，362）。K=200。
reliability = 1 - E[noise var] / Var(observed target)；相关系数上限 = sqrt(reliability)。
同时给出噪声下"正样本身份"翻转率：P(IoU 跨过 0.5)。按 640 拉伸 GT 宽分档。无模型前向。"""
import json, collections
from pathlib import Path
import torch
torch.set_num_threads(2); torch.manual_seed(20261026)
ROOT = Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文')
CACHE = ROOT/'code/unit_instance_completion_20261021_v1/342_codex_stage3_20261022_v1'
OUT = ROOT/'持续寻找研究主线_20261020_v2/364_target_reliability.json'
def iou_mat(a, b):
    lt = torch.max(a[:, None, :2], b[None, :, :2]); rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb-lt).clamp(min=0); inter = wh[..., 0]*wh[..., 1]
    aa = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]); ab = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    return inter/(aa[:, None]+ab[None]-inter).clamp(min=1e-12)
def iou_pair(a, b):
    lt = torch.max(a[..., :2], b[..., :2]); rb = torch.min(a[..., 2:], b[..., 2:])
    wh = (rb-lt).clamp(min=0); inter = wh[..., 0]*wh[..., 1]
    aa = (a[..., 2]-a[..., 0])*(a[..., 3]-a[..., 1]); ab = (b[..., 2]-b[..., 0])*(b[..., 3]-b[..., 1])
    return inter/(aa+ab-inter).clamp(min=1e-12)
WB = [0, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 1e9]
res = {}
for seed in (1, 2):
    rows = []
    for f in sorted((CACHE/f'342_cache_val_v2_seed{seed}').glob('342_*.pt')):
        d = torch.load(f, map_location='cpu', weights_only=False)
        G = d['gt_boxes_640'].double()
        if len(G) == 0: continue
        ow, oh = d['orig_wh']; s4 = torch.tensor([640/ow, 640/oh, 640/ow, 640/oh], dtype=torch.float64)
        b = d['final300_boxes'].double()*640
        P = torch.cat([b[:, :2]-b[:, 2:]/2, b[:, :2]+b[:, 2:]/2], -1)
        I = iou_mat(G, P); bi, bq = I.max(1)
        m = bi >= .5
        if not m.any(): continue
        Gm, Pm, im = G[m], P[bq[m]], bi[m]
        nz = (torch.rand(200, len(Gm), 4, dtype=torch.float64)-.5)*s4
        In = iou_pair(Pm[None].expand(200, -1, -1), Gm[None]+nz)
        t = In.clamp(min=0)**1.5
        w = Gm[:, 2]-Gm[:, 0]
        for j in range(len(Gm)):
            rows.append((float(w[j]), float(im[j]**1.5), float(t[:, j].var()), float((In[:, j] < .5).double().mean())))
    T = torch.tensor(rows, dtype=torch.float64); W, OBS, NV, FLIP = T.unbind(1)
    R = []
    for lo, hi in zip(WB[:-1], WB[1:]):
        k = (W >= lo) & (W < hi); n = int(k.sum())
        if n < 30: R.append({'w640': f'[{lo},{hi})', 'n': n}); continue
        rel = 1 - float(NV[k].mean())/float(OBS[k].var())
        R.append({'w640': f'[{lo},{"inf" if hi > 1e8 else hi})', 'n': n, 'reliability': round(rel, 4),
                  'corr_ceiling_sqrt_rel': round(max(rel, 0)**.5, 4),
                  'positive_flip_prob_mean': round(float(FLIP[k].mean()), 4)})
    res[f'seed{seed}'] = R
json.dump(res, open(OUT, 'x', encoding='utf-8'), ensure_ascii=False, indent=1)
for s in (1, 2):
    print(f'seed{s}'); [print('  ', r) for r in res[f'seed{s}']]
