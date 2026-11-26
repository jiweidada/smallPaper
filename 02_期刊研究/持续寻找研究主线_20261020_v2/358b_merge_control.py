"""358b 只读：合并签名的对照。对每个 tiny GT，取最近的同类 GT 邻居（不限同格），
看是否存在 final300 框与 union(g, 邻居) IoU>=.5，且该框与 g 自身 IoU<.5。
按 Geo0 / 非Geo0、按邻居距离（以 g 的宽为单位）分组。无模型前向。"""
import json, collections
from pathlib import Path
import torch
torch.set_num_threads(2)
ROOT = Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文')
CACHE = ROOT/'code/unit_instance_completion_20261021_v1/342_codex_stage3_20261022_v1'
OUT = ROOT/'持续寻找研究主线_20261020_v2/358b_merge_control.json'
def iou_mat(a, b):
    lt = torch.max(a[:, None, :2], b[None, :, :2]); rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb-lt).clamp(min=0); inter = wh[..., 0]*wh[..., 1]
    aa = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]); ab = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    return inter/(aa[:, None]+ab[None]-inter).clamp(min=1e-12)
DB = [(0, 1), (1, 2), (2, 4), (4, 1e9)]
res = {}
for seed in (1, 2):
    A = collections.defaultdict(lambda: [0, 0])
    for f in sorted((CACHE/f'342_cache_val_v2_seed{seed}').glob('342_*.pt')):
        d = torch.load(f, map_location='cpu', weights_only=False)
        G = d['gt_boxes_640'].double(); L = d['gt_labels']
        if len(G) < 2: continue
        b = d['final300_boxes'].double()*640
        P = torch.cat([b[:, :2]-b[:, 2:]/2, b[:, :2]+b[:, 2:]/2], -1)
        wh = G[:, 2:]-G[:, :2]; ss = wh.min(1).values; gc = (G[:, :2]+G[:, 2:])/2
        I = iou_mat(G, P); g0 = I.max(1).values < .5
        D = torch.cdist(gc, gc); D.fill_diagonal_(float('inf'))
        D = torch.where(L[None, :] == L[:, None], D, torch.full_like(D, float('inf')))
        nn = D.argmin(1); dn = D.min(1).values
        for j in torch.nonzero(ss < 4).flatten().tolist():
            if not torch.isfinite(dn[j]): continue
            o = int(nn[j])
            U = torch.cat([torch.minimum(G[j, :2], G[o, :2]), torch.maximum(G[j, 2:], G[o, 2:])])[None]
            iu = iou_mat(U, P)[0]
            hit = bool(((iu >= .5) & (I[j] < .5)).any())
            rel = float(dn[j]/wh[j, 0].clamp(min=1e-6))
            di = next(k for k, (lo, hi) in enumerate(DB) if lo <= rel < hi)
            key = f'{"Geo0" if g0[j] else "covered"}_nn_dist_rel_w[{DB[di][0]},{DB[di][1] if DB[di][1] < 1e8 else "inf"})'
            A[key][0] += 1; A[key][1] += int(hit)
    res[f'seed{seed}'] = {k: f'{v[1]}/{v[0]} = {v[1]/v[0]:.3f}' for k, v in sorted(A.items())}
res['_note'] = 'tiny GT；邻居=最近同类 GT（中心距离，以 g 自身宽为单位）；hit=存在框与 union(g,邻居) IoU>=.5 且与 g IoU<.5'
json.dump(res, open(OUT, 'x', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
