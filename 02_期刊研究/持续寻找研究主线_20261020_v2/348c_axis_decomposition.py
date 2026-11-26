"""348c 只读：tiny Geo0 的轴向分解（val548 双 seed）。无模型前向、无训练。
对每个 tiny/small GT 取全部 final300 中与其 IoU 最大的候选 q*，以及中心最近候选 q_nn：
  ox = 一维 x 区间 IoU，oy = 一维 y 区间 IoU（q*）
  O_x : 用 GT 的 x 区间替换 q_nn 的 x 区间（y 不变）后 IoU
  O_y : 用 GT 的 y 区间替换 q_nn 的 y 区间（x 不变）后 IoU
  O_size_nn : q_nn 中心 + GT 宽高（只剩中心误差）
另统计 GT/预测宽、高<4 的计数，原图尺寸分布、x/y 缩放比。
输出 348c_axis_decomposition.json
"""
import json, collections
from pathlib import Path
import torch
torch.set_num_threads(2)
ROOT = Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文')
CACHE = ROOT/'code/unit_instance_completion_20261021_v1/342_codex_stage3_20261022_v1'
OUT = ROOT/'持续寻找研究主线_20261020_v2/348c_axis_decomposition.json'

def iou1(a0, a1, b0, b1):
    inter = (torch.min(a1, b1)-torch.max(a0, b0)).clamp(min=0)
    return inter/((a1-a0)+(b1-b0)-inter).clamp(min=1e-12)

def iou_pair(a, b):
    ix = (torch.min(a[..., 2], b[..., 2])-torch.max(a[..., 0], b[..., 0])).clamp(min=0)
    iy = (torch.min(a[..., 3], b[..., 3])-torch.max(a[..., 1], b[..., 1])).clamp(min=0)
    inter = ix*iy
    aa = (a[..., 2]-a[..., 0])*(a[..., 3]-a[..., 1]); ab = (b[..., 2]-b[..., 0])*(b[..., 3]-b[..., 1])
    return inter/(aa+ab-inter).clamp(min=1e-12)

def iou_mat(a, b):
    lt = torch.max(a[:, None, :2], b[None, :, :2]); rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb-lt).clamp(min=0); inter = wh[..., 0]*wh[..., 1]
    aa = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]); ab = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    return inter/(aa[:, None]+ab[None]-inter).clamp(min=1e-12)

res = {}
for seed in (1, 2):
    acc = collections.defaultdict(int); sizes = collections.Counter()
    for f in sorted((CACHE/f'342_cache_val_v2_seed{seed}').glob('342_*.pt')):
        d = torch.load(f, map_location='cpu', weights_only=False)
        if seed == 1: sizes[tuple(d['orig_wh'])] += 1
        b = d['final300_boxes'].double()*640
        P = torch.cat([b[:, :2]-b[:, 2:]/2, b[:, :2]+b[:, 2:]/2], -1)
        pw = b[:, 2]; ph = b[:, 3]
        acc['pred_w_lt4'] += int((pw < 4).sum()); acc['pred_h_lt4'] += int((ph < 4).sum()); acc['pred_total'] += 300
        acc['pred_w_lt6'] += int((pw < 6).sum())
        G = d['gt_boxes_640'].double()
        if len(G) == 0: continue
        gw = G[:, 2]-G[:, 0]; gh = G[:, 3]-G[:, 1]
        acc['gt_w_lt4'] += int((gw < 4).sum()); acc['gt_h_lt4'] += int((gh < 4).sum()); acc['gt_total'] += len(G)
        acc['gt_w_lt6'] += int((gw < 6).sum())
        I = iou_mat(G, P); mx, arg = I.max(1)
        gc = (G[:, :2]+G[:, 2:])/2; pc = (P[:, :2]+P[:, 2:])/2
        nn = torch.cdist(gc, pc).argmin(1)
        Qs = P[arg]; Qn = P[nn]
        ox = iou1(G[:, 0], G[:, 2], Qs[:, 0], Qs[:, 2]); oy = iou1(G[:, 1], G[:, 3], Qs[:, 1], Qs[:, 3])
        Ox = iou_pair(G, torch.stack([G[:, 0], Qn[:, 1], G[:, 2], Qn[:, 3]], -1))
        Oy = iou_pair(G, torch.stack([Qn[:, 0], G[:, 1], Qn[:, 2], G[:, 3]], -1))
        nc = (Qn[:, :2]+Qn[:, 2:])/2
        Os = iou_pair(G, torch.cat([nc-(G[:, 2:]-G[:, :2])/2, nc+(G[:, 2:]-G[:, :2])/2], -1))
        # 中心误差分轴（相对 GT 宽/高）
        dx = (nc[:, 0]-gc[:, 0]).abs()/gw.clamp(min=1e-6); dy = (nc[:, 1]-gc[:, 1]).abs()/gh.clamp(min=1e-6)
        wr = (Qn[:, 2]-Qn[:, 0])/gw.clamp(min=1e-6); hr = (Qn[:, 3]-Qn[:, 1])/gh.clamp(min=1e-6)
        ss = torch.minimum(gw, gh)
        for name, m in (('tiny', ss < 4), ('small4_16', (ss >= 4) & (ss < 16))):
            g0 = m & (mx < .5)
            acc[f'{name}_geo0'] += int(g0.sum())
            acc[f'{name}_geo0_best_ox_lt_oy'] += int((g0 & (ox < oy)).sum())
            acc[f'{name}_geo0_best_ox_lt.5'] += int((g0 & (ox < .5)).sum())
            acc[f'{name}_geo0_best_oy_lt.5'] += int((g0 & (oy < .5)).sum())
            acc[f'{name}_geo0_nn_Ox_fix_ge.5'] += int((g0 & (Ox >= .5)).sum())
            acc[f'{name}_geo0_nn_Oy_fix_ge.5'] += int((g0 & (Oy >= .5)).sum())
            acc[f'{name}_geo0_nn_center_only_ge.5'] += int((g0 & (Os >= .5)).sum())
            acc[f'{name}_geo0_nn_dx_rel_gt.5'] += int((g0 & (dx > .5)).sum())
            acc[f'{name}_geo0_nn_dy_rel_gt.5'] += int((g0 & (dy > .5)).sum())
            acc[f'{name}_geo0_nn_wratio_gt1.5'] += int((g0 & (wr > 1.5)).sum())
            acc[f'{name}_geo0_nn_hratio_gt1.5'] += int((g0 & (hr > 1.5)).sum())
            acc[f'{name}_geo0_nn_wratio_lt.67'] += int((g0 & (wr < 2/3)).sum())
            acc[f'{name}_geo0_nn_hratio_lt.67'] += int((g0 & (hr < 2/3)).sum())
    acc = dict(acc)
    if seed == 1: acc['orig_wh_counts'] = {f'{k[0]}x{k[1]}': v for k, v in sizes.most_common()}
    res[f'seed{seed}'] = acc
res['_note'] = 'q*=全部final300中IoU最大候选；q_nn=中心最近候选；O_x/O_y=把q_nn的x或y区间替换为GT后IoU；均为oracle描述'
json.dump(res, open(OUT, 'x', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
