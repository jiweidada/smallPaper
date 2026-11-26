"""358 只读：stride 碰撞假设。一个 P3 格（stride 8，640 拉伸平面）里同时有 >=2 个 GT 中心时，
标准 DETR 系 query 生成（每个 encoder token 一个 anchor）只给这一格一个初始位置。
检验：(1) 碰撞与否对 Geo0 率的影响（控制尺寸）；(2) Geo0 是否呈"一个框盖住两个 GT"的合并签名；
(3) 碰撞的方向（x 向还是 y 向）；(4) C(P2) 相对 baseline 的救回是否集中在碰撞 GT。无模型前向。"""
import json, collections
from pathlib import Path
import torch
torch.set_num_threads(2)
ROOT = Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文')
CACHE = ROOT/'code/unit_instance_completion_20261021_v1/342_codex_stage3_20261022_v1'
PAIR = ROOT/'code/c_axis_precheck_20261025_v2_用户修订/352_353_historical_iou50_paired_GT.json'
ANN = Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val\annotations\val.json')
OUT = ROOT/'持续寻找研究主线_20261020_v2/358_stride_collision.json'
def iou_mat(a, b):
    lt = torch.max(a[:, None, :2], b[None, :, :2]); rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb-lt).clamp(min=0); inter = wh[..., 0]*wh[..., 1]
    aa = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]); ab = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    return inter/(aa[:, None]+ab[None]-inter).clamp(min=1e-12)
def cell_collide(c, s):
    k = torch.floor(c/s).long(); key = k[:, 0]*100000+k[:, 1]
    cnt = collections.Counter(key.tolist())
    return torch.tensor([cnt[v] >= 2 for v in key.tolist()])
SB = [(0, 4), (4, 8), (8, 16), (16, 1e9)]
res = {}
for seed in (1, 2):
    A = collections.defaultdict(int)
    for f in sorted((CACHE/f'342_cache_val_v2_seed{seed}').glob('342_*.pt')):
        d = torch.load(f, map_location='cpu', weights_only=False)
        G = d['gt_boxes_640'].double()
        if len(G) == 0: continue
        b = d['final300_boxes'].double()*640
        P = torch.cat([b[:, :2]-b[:, 2:]/2, b[:, :2]+b[:, 2:]/2], -1)
        gc = (G[:, :2]+G[:, 2:])/2; wh = G[:, 2:]-G[:, :2]; ss = wh.min(1).values
        I = iou_mat(G, P); mx = I.max(1).values; g0 = mx < .5
        col8 = cell_collide(gc, 8.0); col4 = cell_collide(gc, 4.0); col16 = cell_collide(gc, 16.0)
        # 合并签名：对每个 GT g，取同一 P3 格内的其它 GT g'，看是否有预测框与 union(g,g') IoU>=.5
        k = torch.floor(gc/8).long(); key = (k[:, 0]*100000+k[:, 1]).tolist()
        groups = collections.defaultdict(list)
        for j, v in enumerate(key): groups[v].append(j)
        merged = torch.zeros(len(G), dtype=torch.bool); dirx = torch.zeros(len(G), dtype=torch.bool)
        for j, v in enumerate(key):
            if len(groups[v]) < 2: continue
            others = [o for o in groups[v] if o != j]
            # 碰撞方向：最近的同格邻居在 x 上更近还是 y 上更近（按各自尺寸归一）
            o = min(others, key=lambda o: float((gc[o]-gc[j]).abs().sum()))
            dx = float((gc[o, 0]-gc[j, 0]).abs()); dy = float((gc[o, 1]-gc[j, 1]).abs())
            dirx[j] = dx >= dy
            U = torch.stack([torch.cat([torch.minimum(G[j, :2], G[o2, :2]), torch.maximum(G[j, 2:], G[o2, 2:])]) for o2 in others])
            merged[j] = bool((iou_mat(U, P).max() >= .5))
        for si, (lo, hi) in enumerate(SB):
            m = (ss >= lo) & (ss < hi)
            for name, col in (('col4', col4), ('col8', col8), ('col16', col16)):
                for flag in (True, False):
                    sel = m & (col == flag)
                    A[f's{si}_{name}_{int(flag)}_n'] += int(sel.sum()); A[f's{si}_{name}_{int(flag)}_geo0'] += int((sel & g0).sum())
            sel = m & col8 & g0
            A[f's{si}_col8_geo0_merged'] += int((sel & merged).sum())
            A[f's{si}_col8_geo0_dirx'] += int((sel & dirx).sum())
            A[f's{si}_col8_all_dirx'] += int((m & col8 & dirx).sum())
    R = {}
    for si, (lo, hi) in enumerate(SB):
        nm = f'short[{lo},{"inf" if hi > 1e8 else hi})'
        r = {}
        for name in ('col4', 'col8', 'col16'):
            for flag in (1, 0):
                n = A[f's{si}_{name}_{flag}_n']; g = A[f's{si}_{name}_{flag}_geo0']
                r[f'{name}={"yes" if flag else "no"}'] = f'{g}/{n} = {g/n:.3f}' if n else '0/0'
        n8 = A[f's{si}_col8_1_geo0']
        r['col8_geo0_with_merged_box'] = f'{A[f"s{si}_col8_geo0_merged"]}/{n8}'
        r['col8_geo0_nearest_neighbor_along_x'] = f'{A[f"s{si}_col8_geo0_dirx"]}/{n8}'
        r['col8_all_nearest_neighbor_along_x'] = f'{A[f"s{si}_col8_all_dirx"]}/{A[f"s{si}_col8_1_n"]}'
        R[nm] = r
    res[f'seed{seed}_BASE24'] = R
# (4) seed0 baseline vs C（历史 GPU 预测，352/353 已配对的逐 GT TP）
ann = json.load(open(ANN, encoding='utf-8')); ims = {i['id']: i for i in ann['images']}
cen = {}
for a in ann['annotations']:
    im = ims[a['image_id']]; x, y, w, h = a['bbox']
    cen[a['id']] = (a['image_id'], (x+w/2)*640/im['width'], (y+h/2)*640/im['height'])
byimg = collections.defaultdict(list)
for gid, (iid, cx, cy) in cen.items(): byimg[iid].append((gid, cx, cy))
col = {}
for iid, L in byimg.items():
    cnt = collections.Counter((int(cx//8), int(cy//8)) for _, cx, cy in L)
    for gid, cx, cy in L: col[gid] = cnt[(int(cx//8), int(cy//8))] >= 2
pg = json.load(open(PAIR, encoding='utf-8'))
C = collections.defaultdict(lambda: [0, 0, 0, 0])
for r in pg:
    s = min(r['w640'], r['h640']); si = next(i for i, (lo, hi) in enumerate(SB) if lo <= s < hi)
    k = (si, col[r['gt_id']]); C[k][0] += 1; C[k][1] += int(r['baseline_tp']); C[k][2] += int(r['c_tp'] and not r['baseline_tp']); C[k][3] += int(r['baseline_tp'] and not r['c_tp'])
R4 = {}
for (si, c), (n, b, resc, lost) in sorted(C.items()):
    lo, hi = SB[si]
    R4[f'short[{lo},{"inf" if hi > 1e8 else hi}) col8={"yes" if c else "no"}'] = {
        'n': n, 'baseline_recall': round(b/n, 4), 'C_rescued': resc, 'C_lost': lost,
        'net_delta_recall_pp': round(100*(resc-lost)/n, 3),
        'rescue_rate_of_baseline_misses': round(resc/(n-b), 4) if n > b else None}
res['seed0_baseline_vs_C_historical_GPU'] = R4
res['_note'] = 'col8=同一 8px 格（640 拉伸平面）内另有 GT 中心；merged=存在 final300 框与 union(g, 同格邻居) IoU>=.5；dirx=最近同格邻居 |dx|>=|dy|'
json.dump(res, open(OUT, 'x', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
