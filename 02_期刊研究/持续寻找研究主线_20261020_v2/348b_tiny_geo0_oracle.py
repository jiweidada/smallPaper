"""348b 只读：tiny Geo0 成因 oracle 分解（val548 双 seed）。无模型前向、无训练。
对每个 GT，在全部 final300 框上算以下 oracle 的最大 IoU（不论类别/分数）：
  O_size : 保留候选中心，宽高替换为 GT 宽高 -> 只剩中心误差
  O_ctr  : 保留候选宽高，中心替换为 GT 中心 -> 只剩尺寸误差
  O_shrink_k: 候选宽高按 min(1, k*GT边/候选边) 收缩（中心不动），k=1.5
并记录 final300 框短边分布、tiny GT 的短边方向、IoU 阈值敏感性。
输出 348b_tiny_geo0_oracle.json
"""
import json, collections
from pathlib import Path
import torch
torch.set_num_threads(2)
ROOT = Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文')
CACHE = ROOT/'code/unit_instance_completion_20261021_v1/342_codex_stage3_20261022_v1'
OUT = ROOT/'持续寻找研究主线_20261020_v2/348b_tiny_geo0_oracle.json'

def iou(a, b):
    lt = torch.max(a[:, None, :2], b[None, :, :2]); rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb-lt).clamp(min=0); inter = wh[..., 0]*wh[..., 1]
    aa = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]); ab = (b[..., 2]-b[..., 0])*(b[..., 3]-b[..., 1])
    return inter/(aa[:, None]+ab-inter).clamp(min=1e-12)

def mk(c, wh): return torch.cat([c-wh/2, c+wh/2], -1)

res = {}
for seed in (1, 2):
    acc = collections.defaultdict(int)
    pred_ss_bins = collections.Counter(); pred_tot = 0
    short_dir = collections.Counter()
    for f in sorted((CACHE/f'342_cache_val_v2_seed{seed}').glob('342_*.pt')):
        d = torch.load(f, map_location='cpu', weights_only=False)
        b = d['final300_boxes'].double()*640
        pc, pwh = b[:, :2], b[:, 2:]
        P = mk(pc, pwh)
        pss = pwh.min(1).values
        for v in pss.tolist():
            pred_ss_bins['<2' if v < 2 else '[2,4)' if v < 4 else '[4,8)' if v < 8 else '[8,16)' if v < 16 else '>=16'] += 1
        pred_tot += len(pss)
        sc = d['final300_logits'].sigmoid().max(1).values
        top100 = sc.topk(100).indices
        G = d['gt_boxes_640'].double()
        if len(G) == 0: continue
        gc = (G[:, :2]+G[:, 2:])/2; gwh = G[:, 2:]-G[:, :2]
        ss = gwh.min(1).values
        tiny = ss < 4
        I = iou(G, P); mx = I.max(1).values
        # O_size: 候选中心 + GT 宽高
        Bs = mk(pc[None].expand(len(G), -1, -1), gwh[:, None].expand(-1, 300, -1))
        Is = iou(G, Bs).diagonal(dim1=0, dim2=0) if False else None
        # 逐 GT 计算避免巨型张量
        o_size = torch.empty(len(G)); o_ctr = torch.empty(len(G)); o_shr = torch.empty(len(G))
        for j in range(len(G)):
            g = G[j:j+1]
            o_size[j] = iou(g, mk(pc, gwh[j].expand(300, 2))).max()
            o_ctr[j] = iou(g, mk(gc[j].expand(300, 2), pwh)).max()
            shr = torch.minimum(pwh, 1.5*gwh[j])
            o_shr[j] = iou(g, mk(pc, shr)).max()
        mx100 = I[:, top100].max(1).values
        for name, m in (('tiny', tiny), ('small4_16', (ss >= 4) & (ss < 16))):
            geo0 = m & (mx < .5)
            acc[f'{name}_n'] += int(m.sum()); acc[f'{name}_geo0'] += int(geo0.sum())
            acc[f'{name}_geo0_Osize_ge.5'] += int((geo0 & (o_size >= .5)).sum())
            acc[f'{name}_geo0_Octr_ge.5'] += int((geo0 & (o_ctr >= .5)).sum())
            acc[f'{name}_geo0_Oshrink1.5_ge.5'] += int((geo0 & (o_shr >= .5)).sum())
            acc[f'{name}_geo0_neither_Osize_nor_Octr'] += int((geo0 & (o_size < .5) & (o_ctr < .5)).sum())
            acc[f'{name}_geo0_both_Osize_and_Octr'] += int((geo0 & (o_size >= .5) & (o_ctr >= .5)).sum())
            acc[f'{name}_maxiou_ge.3'] += int((m & (mx >= .3)).sum())
            acc[f'{name}_maxiou_ge.5'] += int((m & (mx >= .5)).sum())
            acc[f'{name}_maxiou_ge.75'] += int((m & (mx >= .75)).sum())
            acc[f'{name}_top100score_maxiou_ge.5'] += int((m & (mx100 >= .5)).sum())
        tw = gwh[tiny]
        short_dir['width_is_short'] += int((tw[:, 0] <= tw[:, 1]).sum()); short_dir['height_is_short'] += int((tw[:, 0] > tw[:, 1]).sum())
    acc = dict(acc)
    acc['final300_pred_shortside_px640'] = dict(pred_ss_bins); acc['final300_pred_total'] = pred_tot
    acc['tiny_gt_short_side_direction'] = dict(short_dir)
    res[f'seed{seed}'] = acc
res['_note'] = 'Geo0=全部final300任意类 IoU<.5；O_size=候选中心+GT宽高；O_ctr=GT中心+候选宽高；Oshrink1.5=候选宽高截到<=1.5×GT宽高、中心不动；全部为oracle几何上界，非部署'
json.dump(res, open(OUT, 'x', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
