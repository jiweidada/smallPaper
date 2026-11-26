"""359 只读：tiny 的分数是否被多个重复 query 分摊（score fragmentation）。
对每个 GT：取 final300 中与其 IoU>=.5 的所有 query（任意类别）= 重复组；
报告组大小、组内真类别分数的 max 与 sum、组内 IoU 与分数的 Spearman 相关。按 GT 短边分档。无模型前向。"""
import json, collections
from pathlib import Path
import torch
torch.set_num_threads(2)
ROOT = Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文')
CACHE = ROOT/'code/unit_instance_completion_20261021_v1/342_codex_stage3_20261022_v1'
OUT = ROOT/'持续寻找研究主线_20261020_v2/359_score_fragmentation.json'
def iou_mat(a, b):
    lt = torch.max(a[:, None, :2], b[None, :, :2]); rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb-lt).clamp(min=0); inter = wh[..., 0]*wh[..., 1]
    aa = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]); ab = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    return inter/(aa[:, None]+ab[None]-inter).clamp(min=1e-12)
def rank(x): return torch.argsort(torch.argsort(x)).double()
SB = [(0, 4), (4, 8), (8, 16), (16, 32), (32, 1e9)]
def q(x):
    if not x: return None
    t = torch.tensor(x, dtype=torch.float64); return [round(float(t.quantile(p)), 3) for p in (.25, .5, .75)]
res = {}
for seed in (1, 2):
    A = collections.defaultdict(list)
    for f in sorted((CACHE/f'342_cache_val_v2_seed{seed}').glob('342_*.pt')):
        d = torch.load(f, map_location='cpu', weights_only=False)
        G = d['gt_boxes_640'].double(); L = d['gt_labels']
        if len(G) == 0: continue
        b = d['final300_boxes'].double()*640
        P = torch.cat([b[:, :2]-b[:, 2:]/2, b[:, :2]+b[:, 2:]/2], -1)
        S = d['final300_logits'].double().sigmoid()
        I = iou_mat(G, P); ss = (G[:, 2:]-G[:, :2]).min(1).values
        for j in range(len(G)):
            si = next(k for k, (lo, hi) in enumerate(SB) if lo <= ss[j] < hi)
            grp = torch.nonzero(I[j] >= .5).flatten()
            A[f's{si}_size'].append(len(grp))
            if len(grp) == 0: continue
            sc = S[grp, L[j]]; io = I[j, grp]
            A[f's{si}_max'].append(float(sc.max())); A[f's{si}_sum'].append(float(sc.sum()))
            A[f's{si}_top_is_bestiou'].append(float(int(sc.argmax()) == int(io.argmax())))
            if len(grp) >= 3:
                r1, r2 = rank(io), rank(sc)
                A[f's{si}_spearman'].append(float(torch.corrcoef(torch.stack([r1, r2]))[0, 1]))
    R = {}
    for si, (lo, hi) in enumerate(SB):
        nm = f'short[{lo},{"inf" if hi > 1e8 else hi})'
        sz = torch.tensor(A[f's{si}_size'], dtype=torch.float64)
        R[nm] = {'n_gt': len(sz), 'frac_group_nonempty': round(float((sz > 0).double().mean()), 4),
                 'group_size_q(nonempty)': q([v for v in A[f's{si}_size'] if v > 0]),
                 'max_score_q': q(A[f's{si}_max']), 'sum_score_q': q(A[f's{si}_sum']),
                 'top_score_query_is_best_iou_query': round(sum(A[f's{si}_top_is_bestiou'])/max(1, len(A[f's{si}_top_is_bestiou'])), 4),
                 'spearman_iou_score_within_group(size>=3)': {'n': len(A[f's{si}_spearman']), 'q': q(A[f's{si}_spearman'])}}
    res[f'seed{seed}'] = R
json.dump(res, open(OUT, 'x', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
