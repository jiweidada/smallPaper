"""360 只读：IoU 感知分数的"分辨率拐点"。
对每个 GT 取与其 IoU 最大的 final300 query（任意类别），读 GT 真类别分数。
在 IoU∈[0.5,1] 内，按 GT 宽（高≥8）或 GT 高（宽≥8）细分档，拟合 score~IoU 的斜率与 Spearman。
另报每档 E[IoU_best^1.5]（DEIM MAL 目标 IoU^gamma, gamma=1.5）与 TP 分数中位数对照。无模型前向。"""
import json, collections
from pathlib import Path
import torch
torch.set_num_threads(2)
ROOT = Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文')
CACHE = ROOT/'code/unit_instance_completion_20261021_v1/342_codex_stage3_20261022_v1'
OUT = ROOT/'持续寻找研究主线_20261020_v2/360_quality_resolution_knee.json'
def iou_mat(a, b):
    lt = torch.max(a[:, None, :2], b[None, :, :2]); rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb-lt).clamp(min=0); inter = wh[..., 0]*wh[..., 1]
    aa = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]); ab = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    return inter/(aa[:, None]+ab[None]-inter).clamp(min=1e-12)
BINS = [0, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 1e9]
def rank(x): return torch.argsort(torch.argsort(x)).double()
res = {}
for seed in (1, 2):
    rows = []
    for f in sorted((CACHE/f'342_cache_val_v2_seed{seed}').glob('342_*.pt')):
        d = torch.load(f, map_location='cpu', weights_only=False)
        G = d['gt_boxes_640'].double(); L = d['gt_labels']
        if len(G) == 0: continue
        b = d['final300_boxes'].double()*640
        P = torch.cat([b[:, :2]-b[:, 2:]/2, b[:, :2]+b[:, 2:]/2], -1)
        S = d['final300_logits'].double().sigmoid()
        I = iou_mat(G, P); bi, bq = I.max(1)
        sc = S[bq, L]
        wh = G[:, 2:]-G[:, :2]
        for j in range(len(G)):
            rows.append((float(wh[j, 0]), float(wh[j, 1]), float(bi[j]), float(sc[j])))
    T = torch.tensor(rows, dtype=torch.float64)
    W, H, IO, SC = T.unbind(1)
    out = {}
    for axis, (v, other) in (('width(h>=8)', (W, H)), ('height(w>=8)', (H, W))):
        R = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = (v >= lo) & (v < hi) & (other >= 8)
            mm = m & (IO >= .5)
            n, nm = int(m.sum()), int(mm.sum())
            row = {'bin': f'[{lo},{"inf" if hi > 1e8 else hi})', 'n': n, 'n_iou_ge.5': nm,
                   'E_bestIoU^1.5_all': round(float((IO[m]**1.5).mean()), 4) if n else None}
            if nm >= 30:
                x, y = IO[mm], SC[mm]
                X = torch.stack([torch.ones_like(x), x], 1)
                beta = torch.linalg.lstsq(X, y[:, None]).solution[:, 0]
                row['slope_score_per_IoU'] = round(float(beta[1]), 4)
                row['spearman'] = round(float(torch.corrcoef(torch.stack([rank(x), rank(y)]))[0, 1]), 4)
                row['score_median_iou_ge.5'] = round(float(y.median()), 4)
                row['score_median_iou_ge.8'] = round(float(y[x >= .8].median()), 4) if int((x >= .8).sum()) >= 20 else None
            R.append(row)
        out[axis] = R
    res[f'seed{seed}'] = out
json.dump(res, open(OUT, 'x', encoding='utf-8'), ensure_ascii=False, indent=1)
for s in (1, 2):
    for ax in ('width(h>=8)', 'height(w>=8)'):
        print(f'seed{s}', ax)
        for r in res[f'seed{s}'][ax]:
            print('  ', r['bin'], r['n'], r['n_iou_ge.5'], 'E[IoU^1.5]', r['E_bestIoU^1.5_all'], 'slope', r.get('slope_score_per_IoU'), 'rho', r.get('spearman'), 'med', r.get('score_median_iou_ge.5'), 'med@.8', r.get('score_median_iou_ge.8'))
