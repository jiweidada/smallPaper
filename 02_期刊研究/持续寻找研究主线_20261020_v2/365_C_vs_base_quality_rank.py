"""365 只读：P2 支路（C）是否改善 tiny 的"分数-定位质量"对齐。seed0 baseline vs C，历史 GPU 预测（COCO json）。
对每个 GT：同类别预测中 IoU 最大者，取其 IoU 与分数；在 IoU>=.5 内按 640 拉伸 GT 宽分档算 Spearman 与 score~IoU 斜率。无模型前向。"""
import json, collections
from pathlib import Path
import torch
ROOT = Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文')
GOLD = ROOT/'data/04_冻结权重诊断_20261001'
ANN = Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val\annotations\val.json')
OUT = ROOT/'持续寻找研究主线_20261020_v2/365_C_vs_base_quality_rank.json'
def iou_mat(a, b):
    lt = torch.max(a[:, None, :2], b[None, :, :2]); rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb-lt).clamp(min=0); inter = wh[..., 0]*wh[..., 1]
    aa = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]); ab = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    return inter/(aa[:, None]+ab[None]-inter).clamp(min=1e-12)
def rank(x): return torch.argsort(torch.argsort(x)).double()
ann = json.load(open(ANN, encoding='utf-8')); ims = {i['id']: i for i in ann['images']}
gts = collections.defaultdict(list)
for a in ann['annotations']: gts[a['image_id']].append(a)
WB = [0, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 1e9]
res = {}
for label in ('baseline', 'c'):
    pr = json.load(open(GOLD/f'{label}_640_predictions.json', encoding='utf-8'))
    byi = collections.defaultdict(list)
    for r in pr: byi[r['image_id']].append(r)
    rows = []
    for iid, L in gts.items():
        im = ims[iid]; sx = 640/im['width']
        D = byi.get(iid, [])
        if not D: continue
        Pb = torch.tensor([[r['bbox'][0], r['bbox'][1], r['bbox'][0]+r['bbox'][2], r['bbox'][1]+r['bbox'][3]] for r in D], dtype=torch.float64)
        Pc = torch.tensor([r['category_id'] for r in D]); Ps = torch.tensor([r['score'] for r in D], dtype=torch.float64)
        Gb = torch.tensor([[a['bbox'][0], a['bbox'][1], a['bbox'][0]+a['bbox'][2], a['bbox'][1]+a['bbox'][3]] for a in L], dtype=torch.float64)
        Gc = torch.tensor([a['category_id'] for a in L])
        I = iou_mat(Gb, Pb); I = torch.where(Gc[:, None] == Pc[None], I, torch.zeros_like(I))
        bi, bq = I.max(1)
        for j, a in enumerate(L):
            rows.append((a['bbox'][2]*sx, float(bi[j]), float(Ps[bq[j]])))
    T = torch.tensor(rows, dtype=torch.float64); W, IO, SC = T.unbind(1)
    R = []
    for lo, hi in zip(WB[:-1], WB[1:]):
        k = (W >= lo) & (W < hi) & (IO >= .5); n = int(k.sum())
        row = {'w640': f'[{lo},{"inf" if hi > 1e8 else hi})', 'n_iou_ge.5': n}
        if n >= 30:
            x, y = IO[k], SC[k]
            X = torch.stack([torch.ones_like(x), x], 1)
            row['slope'] = round(float(torch.linalg.lstsq(X, y[:, None]).solution[1, 0]), 4)
            row['spearman'] = round(float(torch.corrcoef(torch.stack([rank(x), rank(y)]))[0, 1]), 4)
        R.append(row)
    res[label] = R
json.dump(res, open(OUT, 'x', encoding='utf-8'), ensure_ascii=False, indent=1)
for i in range(len(WB)-1):
    b, c = res['baseline'][i], res['c'][i]
    print(b['w640'], 'n', b['n_iou_ge.5'], c['n_iou_ge.5'], '| rho', b.get('spearman'), '->', c.get('spearman'), '| slope', b.get('slope'), '->', c.get('slope'))
