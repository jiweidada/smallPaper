"""348d 只读：预测宽度是否有下限（val548 双 seed）。无模型前向。输出 348d_width_floor.json"""
import json, collections
from pathlib import Path
import torch
torch.set_num_threads(2)
ROOT = Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文')
CACHE = ROOT/'code/unit_instance_completion_20261021_v1/342_codex_stage3_20261022_v1'
OUT = ROOT/'持续寻找研究主线_20261020_v2/348d_width_floor.json'
def iou_mat(a, b):
    lt = torch.max(a[:, None, :2], b[None, :, :2]); rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb-lt).clamp(min=0); inter = wh[..., 0]*wh[..., 1]
    aa = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]); ab = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    return inter/(aa[:, None]+ab[None]-inter).clamp(min=1e-12)
def q(x, ps=(.1,.25,.5,.75,.9)):
    x = torch.tensor(x, dtype=torch.float64)
    return {str(p): round(float(x.quantile(p)), 3) for p in ps} | {'n': len(x)} if len(x) else {'n': 0}
res = {}
for seed in (1, 2):
    W = collections.defaultdict(list)
    for f in sorted((CACHE/f'342_cache_val_v2_seed{seed}').glob('342_*.pt')):
        d = torch.load(f, map_location='cpu', weights_only=False)
        b = d['final300_boxes'].double()*640
        P = torch.cat([b[:, :2]-b[:, 2:]/2, b[:, :2]+b[:, 2:]/2], -1)
        sc = d['final300_logits'].sigmoid().max(1).values
        W['pred_w_all'] += b[:, 2].tolist(); W['pred_h_all'] += b[:, 3].tolist()
        top = sc >= .3
        W['pred_w_score_ge.3'] += b[top, 2].tolist()
        G = d['gt_boxes_640'].double()
        if len(G) == 0: continue
        gw = G[:, 2]-G[:, 0]; gh = G[:, 3]-G[:, 1]
        I = iou_mat(G, P); mx, arg = I.max(1)
        gc = (G[:, :2]+G[:, 2:])/2; pc = b[:, :2]
        nn = torch.cdist(gc, pc).argmin(1)
        for lo, hi in ((0, 2), (2, 3), (3, 4), (4, 6), (6, 8), (8, 12)):
            m = (gw >= lo) & (gw < hi)
            W[f'gt_w[{lo},{hi})_nn_pred_w'] += b[nn[m], 2].tolist()
            W[f'gt_w[{lo},{hi})_best_pred_w'] += b[arg[m], 2].tolist()
            W[f'gt_w[{lo},{hi})_gt_h'] += gh[m].tolist()
            W[f'gt_w[{lo},{hi})_nn_pred_h'] += b[nn[m], 3].tolist()
            mm = m & (mx >= .5)
            W[f'gt_w[{lo},{hi})_matched_best_pred_w'] += b[arg[mm], 2].tolist()
            W[f'gt_w[{lo},{hi})_n_maxiou_ge.5'] += [1.0]*int(mm.sum())
        W['gt_w_all'] += gw.tolist(); W['gt_h_all'] += gh.tolist()
    S = {}
    for k, v in W.items():
        S[k] = {'n': len(v)} if k.endswith('ge.5') else q(v)
    res[f'seed{seed}'] = S
json.dump(res, open(OUT, 'x', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
