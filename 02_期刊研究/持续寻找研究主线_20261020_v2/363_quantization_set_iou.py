"""363 只读：把原图整数像素标注看成一个"可容许框集合" B(annot)：每条边在原图坐标 ±0.5 px 内（取整误差）。
set-IoU = max_{b in B} IoU(pred, b)。用 4 条边各 {-0.5,0,+0.5} 的 81 个角点近似 max（IoU 对单边是分段单调，角点+原点给下界）。
对每个 GT 取 final300 中 IoU 最大的 query（任意类别），按原图宽度分档报告：
  raw IoU>=.5 比例；set-IoU>=.5 比例；Geo0（raw<.5）中 set-IoU>=.5 的比例；
  以及取整噪声（各边 U[-0.5,0.5] 原图像素，200 次）下 IoU^1.5 目标的标准差。无模型前向。"""
import json, itertools, collections
from pathlib import Path
import torch
torch.set_num_threads(2); torch.manual_seed(20261026)
ROOT = Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文')
CACHE = ROOT/'code/unit_instance_completion_20261021_v1/342_codex_stage3_20261022_v1'
OUT = ROOT/'持续寻找研究主线_20261020_v2/363_quantization_set_iou.json'
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
CORNERS = torch.tensor(list(itertools.product([-.5, 0, .5], repeat=4)), dtype=torch.float64)  # [81,4]
WB = [(2, 3), (3, 4), (4, 5), (5, 6), (6, 8), (8, 12), (12, 16), (16, 32), (32, 1e9)]
res = {}
for seed in (1, 2):
    A = collections.defaultdict(lambda: collections.Counter())
    SD = collections.defaultdict(list)
    for f in sorted((CACHE/f'342_cache_val_v2_seed{seed}').glob('342_*.pt')):
        d = torch.load(f, map_location='cpu', weights_only=False)
        G = d['gt_boxes_640'].double()
        if len(G) == 0: continue
        ow, oh = d['orig_wh']; sc4 = torch.tensor([640/ow, 640/oh, 640/ow, 640/oh], dtype=torch.float64)
        b = d['final300_boxes'].double()*640
        P = torch.cat([b[:, :2]-b[:, 2:]/2, b[:, :2]+b[:, 2:]/2], -1)
        I = iou_mat(G, P); bi, bq = I.max(1); Pb = P[bq]
        Gs = G[:, None, :] + CORNERS[None]*sc4                                  # [n,81,4]
        sio = iou_pair(Pb[:, None, :].expand(-1, 81, -1), Gs).max(1).values
        noise = (torch.rand(200, len(G), 4, dtype=torch.float64)-.5)*sc4
        tgt = iou_pair(Pb[None].expand(200, -1, -1), G[None]+noise).clamp(min=0)**1.5
        sd = tgt.std(0)
        worig = (G[:, 2]-G[:, 0])/sc4[0]
        ss640 = (G[:, 2:]-G[:, :2]).min(1).values
        for j in range(len(G)):
            if ss640[j] >= 4: continue          # 只看 tiny（640 拉伸短边<4）
            k = next((i for i, (lo, hi) in enumerate(WB) if lo <= round(float(worig[j])) < hi), None)
            if k is None: continue
            c = A[k]; c['n'] += 1; c['raw_ge.5'] += int(bi[j] >= .5); c['set_ge.5'] += int(sio[j] >= .5)
            if bi[j] < .5: c['geo0'] += 1; c['geo0_set_ge.5'] += int(sio[j] >= .5)
            if bi[j] >= .5: SD[k].append(float(sd[j]))
    R = {}
    for k, (lo, hi) in enumerate(WB):
        c = A[k]
        if not c['n']: continue
        nm = f'orig_w[{lo},{"inf" if hi > 1e8 else hi})'
        R[nm] = {'n_tiny': c['n'], 'raw_iou_ge.5': f"{c['raw_ge.5']}/{c['n']}", 'set_iou_ge.5': f"{c['set_ge.5']}/{c['n']}",
                 'geo0_rescued_by_set': f"{c['geo0_set_ge.5']}/{c['geo0']}",
                 'target_std_rounding_noise_median(iou>=.5)': round(float(torch.tensor(SD[k]).median()), 4) if len(SD[k]) >= 20 else None}
    res[f'seed{seed}'] = R
res['_note'] = 'tiny=640拉伸短边<4；原图宽按整数取档；B(annot)=各边原图±0.5px；set-IoU 取 81 个角点的最大值（下界近似）'
json.dump(res, open(OUT, 'x', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
