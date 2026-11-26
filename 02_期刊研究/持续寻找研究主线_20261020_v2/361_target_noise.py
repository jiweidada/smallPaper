"""361 只读：IoU 质量目标在 tiny 上的标注噪声敏感性，以及模型分数是否更贴近"噪声边际化 IoU"。
对每个 GT 取 IoU 最大的 final300 query（任意类别）及其 GT 真类别分数。
(1) 目标噪声：GT 四条边各自独立加 U{-1,0,+1} 原图像素扰动（换算到 640 平面），9^4 太多，改为 200 次蒙特卡洛，
    报告 IoU^1.5 的标准差 / 均值（目标的变异系数），按 GT 宽分档。
(2) 分数贴近谁：在 IoU>=.5 的样本内，分别计算 score 与 raw IoU^1.5、与噪声边际化 E[IoU^1.5] 的 Spearman。
无模型前向。"""
import json, collections
from pathlib import Path
import torch
torch.set_num_threads(2)
torch.manual_seed(20261026)
ROOT = Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文')
CACHE = ROOT/'code/unit_instance_completion_20261021_v1/342_codex_stage3_20261022_v1'
OUT = ROOT/'持续寻找研究主线_20261020_v2/361_target_noise.json'
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
def rank(x): return torch.argsort(torch.argsort(x)).double()
def rho(x, y): return float(torch.corrcoef(torch.stack([rank(x), rank(y)]))[0, 1])
BINS = [0, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 1e9]
K = 200
res = {}
for seed in (1, 2):
    rows = []
    for f in sorted((CACHE/f'342_cache_val_v2_seed{seed}').glob('342_*.pt')):
        d = torch.load(f, map_location='cpu', weights_only=False)
        G = d['gt_boxes_640'].double(); L = d['gt_labels']
        if len(G) == 0: continue
        ow, oh = d['orig_wh']; sx, sy = 640/ow, 640/oh
        b = d['final300_boxes'].double()*640
        P = torch.cat([b[:, :2]-b[:, 2:]/2, b[:, :2]+b[:, 2:]/2], -1)
        S = d['final300_logits'].double().sigmoid()
        I = iou_mat(G, P); bi, bq = I.max(1)
        keep = bi >= .3
        if not keep.any(): continue
        Gk = G[keep]; Pk = P[bq[keep]]; sc = S[bq[keep], L[keep]]; ik = bi[keep]
        noise = torch.randint(-1, 2, (K, len(Gk), 4)).double() * torch.tensor([sx, sy, sx, sy], dtype=torch.float64)
        Gn = Gk[None] + noise
        Gn = torch.cat([torch.minimum(Gn[..., :2], Gn[..., 2:]-1e-3), Gn[..., 2:]], -1)
        Inz = iou_pair(Pk[None].expand(K, -1, -1), Gn)              # [K, n]
        t = Inz.clamp(min=0)**1.5
        mean_t, std_t = t.mean(0), t.std(0)
        w = (Gk[:, 2]-Gk[:, 0])
        for j in range(len(Gk)):
            rows.append((float(w[j]), float(ik[j]), float(sc[j]), float(mean_t[j]), float(std_t[j])))
    T = torch.tensor(rows, dtype=torch.float64)
    W, IO, SC, MT, SD = T.unbind(1)
    R = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (W >= lo) & (W < hi) & (IO >= .5)
        n = int(m.sum())
        row = {'gt_width_bin': f'[{lo},{"inf" if hi > 1e8 else hi})', 'n_iou_ge.5': n}
        if n >= 30:
            raw = IO[m]**1.5
            row['target_std_median'] = round(float(SD[m].median()), 4)
            row['target_cv_median'] = round(float((SD[m]/MT[m].clamp(min=1e-9)).median()), 4)
            row['rho_score_vs_rawIoU^1.5'] = round(rho(SC[m], raw), 4)
            row['rho_score_vs_noiseMarginal'] = round(rho(SC[m], MT[m]), 4)
            row['rho_rawIoU_vs_noiseMarginal'] = round(rho(raw, MT[m]), 4)
        R.append(row)
    res[f'seed{seed}'] = R
res['_note'] = '扰动=GT 四边各自 U{-1,0,+1} 原图像素，换算到 640 拉伸平面；K=200 次；target=IoU^1.5（DEIM MAL gamma=1.5）；样本=IoU>=.5 的最佳候选'
json.dump(res, open(OUT, 'x', encoding='utf-8'), ensure_ascii=False, indent=1)
for s in (1, 2):
    print(f'seed{s}')
    for r in res[f'seed{s}']:
        print('  ', r)
