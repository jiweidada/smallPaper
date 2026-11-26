"""357 只读：val548 BASE24 双 seed，按 GT 尺寸看"同类最佳候选"的 IoU 与分数关系。无模型前向。
输出 357_score_iou_by_size.json"""
import json, collections
from pathlib import Path
import torch
torch.set_num_threads(2)
ROOT = Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文')
CACHE = ROOT/'code/unit_instance_completion_20261021_v1/342_codex_stage3_20261022_v1'
OUT = ROOT/'持续寻找研究主线_20261020_v2/357_score_iou_by_size.json'
def iou_mat(a, b):
    lt = torch.max(a[:, None, :2], b[None, :, :2]); rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb-lt).clamp(min=0); inter = wh[..., 0]*wh[..., 1]
    aa = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]); ab = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    return inter/(aa[:, None]+ab[None]-inter).clamp(min=1e-12)
SB = [(0, 4), (4, 8), (8, 16), (16, 32), (32, 1e9)]
IB = [(.5, .6), (.6, .7), (.7, .8), (.8, 1.01)]
res = {}
for seed in (1, 2):
    acc = collections.defaultdict(list)
    fp_scores = collections.defaultdict(list)
    for f in sorted((CACHE/f'342_cache_val_v2_seed{seed}').glob('342_*.pt')):
        d = torch.load(f, map_location='cpu', weights_only=False)
        b = d['final300_boxes'].double()*640
        P = torch.cat([b[:, :2]-b[:, 2:]/2, b[:, :2]+b[:, 2:]/2], -1)
        S = d['final300_logits'].double().sigmoid()          # [300,10]
        G = d['gt_boxes_640'].double(); L = d['gt_labels']
        if len(G) == 0: continue
        I = iou_mat(G, P)                                     # [n,300]
        ss = (G[:, 2:]-G[:, :2]).min(1).values
        sc = S[:, L].T                                        # [n,300] 该GT类别下各query分数
        best = I.argmax(1)                                    # IoU 最高的 query（任意类别）
        bi = I.gather(1, best[:, None])[:, 0]; bs = sc.gather(1, best[:, None])[:, 0]
        for j in range(len(G)):
            s = next(k for k, (lo, hi) in enumerate(SB) if lo <= ss[j] < hi)
            acc[f's{s}_maxiou'].append(float(bi[j]))
            if bi[j] >= .5:
                k = next(k for k, (lo, hi) in enumerate(IB) if lo <= bi[j] < hi)
                acc[f's{s}_i{k}_score'].append(float(bs[j]))
        # 每图 FP 参考：与所有 GT IoU<0.1 的 query 在其最高类上的分数
        far = I.max(0).values < .1
        fp_scores['all'] += S[far].max(1).values.tolist()
    def q(x):
        if not x: return {'n': 0}
        t = torch.tensor(x); return {'n': len(x), 'p25': round(float(t.quantile(.25)), 4), 'p50': round(float(t.quantile(.5)), 4), 'p75': round(float(t.quantile(.75)), 4)}
    R = {}
    for s, (lo, hi) in enumerate(SB):
        name = f'short[{lo},{"inf" if hi > 1e8 else hi})'
        mi = acc[f's{s}_maxiou']; t = torch.tensor(mi)
        R[name] = {'n_gt': len(mi), 'frac_maxiou_ge.5': round(float((t >= .5).float().mean()), 4),
                   'frac_maxiou_ge.75': round(float((t >= .75).float().mean()), 4),
                   'score_given_iou': {f'[{a},{min(b,1.0)})': q(acc[f's{s}_i{k}_score']) for k, (a, b) in enumerate(IB)}}
    R['far_from_gt_query_maxscore'] = q(fp_scores['all'])
    res[f'seed{seed}'] = R
res['_note'] = '最佳候选=与该GT IoU最大的final300 query（任意类别）；分数取该GT真类别的sigmoid；far=与所有GT IoU<0.1的query'
json.dump(res, open(OUT, 'x', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
