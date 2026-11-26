"""348 只读分析：val548 tiny GT 的 final300 覆盖分解。无模型前向、无训练。
输入：342_cache_val_v2_seed{1,2}/*.pt（final300_boxes, final300_logits, gt_boxes_640, gt_labels, m0_cand_owner, fullp_query_index）
       VisDrone2019-DET-val/annotations/val.json（原图尺寸、原始 bbox）
输出：348_tiny_coverage_analysis.json
"""
import json, sys, collections
from pathlib import Path
import torch
torch.set_num_threads(2)
ROOT = Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文')
CACHE = ROOT/'code/unit_instance_completion_20261021_v1/342_codex_stage3_20261022_v1'
ANN = Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val\annotations\val.json')
OUT = ROOT/'持续寻找研究主线_20261020_v2/348_tiny_coverage_analysis.json'
CLS = ['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor']

ann = json.load(open(ANN, encoding='utf-8'))
imgs = {im['id']: im for im in ann['images']}

def iou(a, b):  # a [N,4], b [M,4] xyxy
    lt = torch.max(a[:, None, :2], b[None, :, :2]); rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb-lt).clamp(min=0); inter = wh[..., 0]*wh[..., 1]
    aa = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]); ab = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    return inter/(aa[:, None]+ab[None]-inter).clamp(min=1e-12)

def cnt(xs): return dict(sorted(collections.Counter(xs).items(), key=lambda kv: str(kv[0])))

def bin_(v, edges, names):
    for e, n in zip(edges, names):
        if v < e: return n
    return names[-1]

res = {}
per_seed_geo0 = {}
for seed in (1, 2):
    rows = []
    for f in sorted((CACHE/f'342_cache_val_v2_seed{seed}').glob('342_*.pt')):
        d = torch.load(f, map_location='cpu', weights_only=False)
        iid = d['image_id']; im = imgs[iid]
        b = d['final300_boxes'].double()*640
        P = torch.stack([b[:, 0]-b[:, 2]/2, b[:, 1]-b[:, 3]/2, b[:, 0]+b[:, 2]/2, b[:, 1]+b[:, 3]/2], -1)
        pscore = d['final300_logits'].sigmoid().max(1).values.double()
        pcls = d['final300_logits'].argmax(1)
        G = d['gt_boxes_640'].double(); L = d['gt_labels']
        n = len(G)
        if n == 0: continue
        ss = torch.minimum(G[:, 2]-G[:, 0], G[:, 3]-G[:, 1])
        I = iou(G, P)                     # [n,300]
        mx, arg = I.max(1)
        gc = (G[:, :2]+G[:, 2:])/2; pc = (P[:, :2]+P[:, 2:])/2
        dist = torch.cdist(gc, pc)        # [n,300] 640像素
        dmin, darg = dist.min(1)
        gsz = ((G[:, 2]-G[:, 0])*(G[:, 3]-G[:, 1])).sqrt()
        psz = ((P[:, 2]-P[:, 0])*(P[:, 3]-P[:, 1])).sqrt()
        inside = ((pc[None, :, 0] >= G[:, None, 0]) & (pc[None, :, 0] <= G[:, None, 2]) &
                  (pc[None, :, 1] >= G[:, None, 1]) & (pc[None, :, 1] <= G[:, None, 3]))
        # 同类候选
        same = (pcls[None, :] == L[:, None])
        Isame = torch.where(same, I, torch.zeros_like(I)).max(1).values
        # GT 局部密度：中心 16/32 像素(640)内其它 GT 数
        gd = torch.cdist(gc, gc)
        dens16 = (gd < 16).sum(1)-1; dens32 = (gd < 32).sum(1)-1
        # GT-GT 最大 IoU（重叠标注）
        gg = iou(G, G); gg.fill_diagonal_(0); ggmax = gg.max(1).values
        # 中心 8 像素内候选数（局部候选供给）
        near8 = (dist < 8).sum(1)
        owner = d['m0_cand_owner']        # 对 fullp 行
        q_owned = torch.zeros(300, dtype=torch.bool)
        fq = d['fullp_query_index']
        if (owner >= 0).any(): q_owned[fq[owner >= 0]] = True
        # 原图尺度
        oxy = {a['id']: a['bbox'] for a in []}
        sx = 640/im['width']; sy = 640/im['height']
        for j in range(n):
            rows.append(dict(
                img=iid, gid=int(d['gt_annotation_ids'][j]), cls=int(L[j]), ss=float(ss[j]), sz=float(gsz[j]),
                maxiou=float(mx[j]), maxiou_same=float(Isame[j]),
                dmin=float(dmin[j]), dmin_rel=float(dmin[j]/gsz[j].clamp(min=1e-6)),
                nn_size_ratio=float(psz[darg[j]]/gsz[j].clamp(min=1e-6)),
                best_size_ratio=float(psz[arg[j]]/gsz[j].clamp(min=1e-6)),
                nn_score=float(pscore[darg[j]]), nn_owned=bool(q_owned[darg[j]]),
                any_center_inside=bool(inside[j].any()), near8=int(near8[j]),
                dens16=int(dens16[j]), dens32=int(dens32[j]), ggmax=float(ggmax[j]),
                n_gt_img=n, n_tiny_img=int((ss < 4).sum()),
                orig_ss=float(ss[j]/min(sx, sy)) if False else float(min((G[j, 2]-G[j, 0])/sx, (G[j, 3]-G[j, 1])/sy)),
                scale_min=float(min(sx, sy)), aspect_distort=float(sx/sy)))
    tiny = [r for r in rows if r['ss'] < 4]
    geo0 = [r for r in tiny if r['maxiou'] < 0.5]
    per_seed_geo0[seed] = {r['gid'] for r in geo0}
    def frac(sub, tot): return f'{len(sub)}/{len(tot)}'
    S = {}
    S['n_gt_all'] = len(rows); S['n_tiny'] = len(tiny); S['n_tiny_geo0'] = len(geo0)
    # 各尺寸段 Geo0 率（对照）
    sb = {}
    for lo, hi in [(0, 2), (2, 3), (3, 4), (4, 6), (6, 8), (8, 12), (12, 16), (16, 32), (32, 1e9)]:
        sub = [r for r in rows if lo <= r['ss'] < hi]
        sb[f'[{lo},{hi})'] = dict(n=len(sub), geo0=sum(r['maxiou'] < .5 for r in sub),
                                  maxiou_lt_0p1=sum(r['maxiou'] < .1 for r in sub))
    S['shortside_bins_all_gt'] = sb
    S['tiny_by_class'] = {CLS[c]: dict(n=sum(r['cls'] == c for r in tiny), geo0=sum(r['cls'] == c for r in geo0)) for c in range(10)}
    S['tiny_geo0_maxiou_bins'] = cnt(bin_(r['maxiou'], [1e-9, .1, .2, .3, .4, .5], ['0', '(0,.1)', '[.1,.2)', '[.2,.3)', '[.3,.4)', '[.4,.5)']) for r in geo0)
    S['tiny_geo0_dmin_px640_bins'] = cnt(bin_(r['dmin'], [1, 2, 4, 8, 16, 32, 1e9], ['<1', '[1,2)', '[2,4)', '[4,8)', '[8,16)', '[16,32)', '>=32']) for r in geo0)
    S['tiny_geo0_dmin_rel_bins'] = cnt(bin_(r['dmin_rel'], [.25, .5, 1, 2, 4, 1e9], ['<.25', '[.25,.5)', '[.5,1)', '[1,2)', '[2,4)', '>=4']) for r in geo0)
    S['tiny_geo0_nn_size_ratio_bins'] = cnt(bin_(r['nn_size_ratio'], [.5, 1.5, 2, 3, 5, 1e9], ['<.5', '[.5,1.5)', '[1.5,2)', '[2,3)', '[3,5)', '>=5']) for r in geo0)
    S['tiny_geo0_any_cand_center_inside_gt'] = sum(r['any_center_inside'] for r in geo0)
    S['tiny_geo0_nn_owned_by_other_gt_M0'] = sum(r['nn_owned'] for r in geo0)
    S['tiny_geo0_nn_score_bins'] = cnt(bin_(r['nn_score'], [.01, .05, .1, .2, .3, .5, 2], ['<.01', '[.01,.05)', '[.05,.1)', '[.1,.2)', '[.2,.3)', '[.3,.5)', '>=.5']) for r in geo0)
    S['tiny_geo0_near8_cands_bins'] = cnt(bin_(r['near8'], [1, 2, 4, 8, 1e9], ['0', '1', '2-3', '4-7', '>=8']) for r in geo0)
    S['tiny_geo0_dens32_bins'] = cnt(bin_(r['dens32'], [1, 3, 6, 11, 21, 1e9], ['0', '1-2', '3-5', '6-10', '11-20', '>20']) for r in geo0)
    S['tiny_all_dens32_bins'] = cnt(bin_(r['dens32'], [1, 3, 6, 11, 21, 1e9], ['0', '1-2', '3-5', '6-10', '11-20', '>20']) for r in tiny)
    S['tiny_geo0_gtgt_maxiou_ge_0p3'] = sum(r['ggmax'] >= .3 for r in geo0)
    S['tiny_geo0_img_ngt_bins'] = cnt(bin_(r['n_gt_img'], [100, 200, 300, 1e9], ['<100', '[100,200)', '[200,300)', '>=300']) for r in geo0)
    S['tiny_geo0_img_ntiny_bins'] = cnt(bin_(r['n_tiny_img'], [5, 20, 50, 100, 1e9], ['<5', '[5,20)', '[20,50)', '[50,100)', '>=100']) for r in geo0)
    S['tiny_geo0_orig_shortside_bins'] = cnt(bin_(r['orig_ss'], [4, 6, 8, 10, 12, 1e9], ['<4', '[4,6)', '[6,8)', '[8,10)', '[10,12)', '>=12']) for r in geo0)
    S['tiny_all_orig_shortside_bins'] = cnt(bin_(r['orig_ss'], [4, 6, 8, 10, 12, 1e9], ['<4', '[4,6)', '[6,8)', '[8,10)', '[10,12)', '>=12']) for r in tiny)
    S['tiny_geo0_scale_min_bins'] = cnt(bin_(r['scale_min'], [.33, .5, .7, 1e9], ['<.33', '[.33,.5)', '[.5,.7)', '>=.7']) for r in geo0)
    S['tiny_geo0_aspect_distort_bins'] = cnt(bin_(r['aspect_distort'], [.7, .9, 1.1, 1e9], ['<.7', '[.7,.9)', '[.9,1.1)', '>=1.1']) for r in geo0)
    # 图像集中度：tiny Geo0 最多的 10% 图占多少
    per_img = collections.Counter(r['img'] for r in geo0)
    vals = sorted(per_img.values(), reverse=True)
    S['tiny_geo0_images_with_any'] = len(per_img)
    S['tiny_geo0_top10pct_images_share'] = f'{sum(vals[:55])}/{len(geo0)}'
    S['tiny_geo0_top20_images_share'] = f'{sum(vals[:20])}/{len(geo0)}'
    # 简单"几何可达"上界：若 nn 候选中心距离 < GT尺寸/3 且尺寸可调 -> 仅尺寸回归问题
    S['tiny_geo0_center_within_third_size'] = sum(r['dmin_rel'] < 1/3 for r in geo0)
    S['tiny_geo0_center_within_one_size'] = sum(r['dmin_rel'] < 1 for r in geo0)
    S['tiny_geo0_same_class_maxiou_lt_0p1'] = sum(r['maxiou_same'] < .1 for r in geo0)
    # 非 tiny small(4-16) 对照的同口径
    sm = [r for r in rows if 4 <= r['ss'] < 16]; smg = [r for r in sm if r['maxiou'] < .5]
    S['small4_16_geo0'] = frac(smg, sm)
    S['small4_16_geo0_dmin_rel_bins'] = cnt(bin_(r['dmin_rel'], [.25, .5, 1, 2, 4, 1e9], ['<.25', '[.25,.5)', '[.5,1)', '[1,2)', '[2,4)', '>=4']) for r in smg)
    S['small4_16_geo0_nn_size_ratio_bins'] = cnt(bin_(r['nn_size_ratio'], [.5, 1.5, 2, 3, 5, 1e9], ['<.5', '[.5,1.5)', '[1.5,2)', '[2,3)', '[3,5)', '>=5']) for r in smg)
    # 原框尺寸分布：全部 300 框最小短边
    res[f'seed{seed}'] = S
res['tiny_geo0_both_seeds_intersection'] = len(per_seed_geo0[1] & per_seed_geo0[2])
res['tiny_geo0_union'] = len(per_seed_geo0[1] | per_seed_geo0[2])
res['_note'] = 'tiny=640输入短边<4；Geo0=final300全部300框(不论类别/分数)与GT IoU<0.5；坐标为非等比640 resize；无模型前向'
json.dump(res, open(OUT, 'x', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
