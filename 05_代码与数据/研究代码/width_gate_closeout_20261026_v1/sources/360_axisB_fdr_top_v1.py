
import os, sys, json, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
from pathlib import Path
import torch
torch.set_num_threads(4)

WORK   = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
NATIVE = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\dn_pre_frozen_diagnostic_20261020_v1\native"
CODE   = Path(WORK) / "342_codex_stage3_20261022_v1"
SPEC   = Path(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2")
VAL    = Path(r"C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val")
OUT    = Path(r"C:\Users\jiwei\Desktop\小论文投稿\tmp")

sys.path.insert(0, WORK); sys.path.insert(0, NATIVE)
import engine.deim.dfine_decoder as dec
from engine.deim.dfine_decoder import Integral
import stage1_cache_v3 as S
from PIL import Image

CAP = {}
_orig = dec.distance2bbox
def _d2b(points, distance, reg_scale):
    CAP['ref'] = points.detach(); CAP['dist'] = distance.detach(); CAP['rs'] = float(reg_scale)
    return _orig(points, distance, reg_scale)
dec.distance2bbox = _d2b

def _hook(m, inputs, output):
    CAP['x'] = inputs[0].detach(); CAP['proj'] = inputs[1].detach()

def xyxy(b):
    x, y, w, h = b[:,0]*640, b[:,1]*640, b[:,2]*640, b[:,3]*640
    return torch.stack([x-w/2, y-h/2, x+w/2, y+h/2], dim=1)

def iou_mat(a, b):
    ax1, ay1, ax2, ay2 = a[:,0:1], a[:,1:2], a[:,2:3], a[:,3:4]
    bx1, by1, bx2, by2 = b[:,0], b[:,1], b[:,2], b[:,3]
    iw = (torch.min(ax2, bx2) - torch.max(ax1, bx1)).clamp(min=0)
    ih = (torch.min(ay2, by2) - torch.max(ay1, by1)).clamp(min=0)
    inter = iw * ih
    aa = (ax2-ax1)*(ay2-ay1); ab = (bx2-bx1)*(by2-by1)
    return inter / (aa + ab - inter + 1e-12)

def qq(t, ps=(10,25,50,75,90)):
    t = t.double().flatten(); t = t[t == t]
    if t.numel() == 0: return None
    v = torch.quantile(t, torch.tensor([p/100 for p in ps], dtype=torch.float64))
    return {f"p{p}": round(float(x), 4) for p, x in zip(ps, v)}

def newgrp():
    return {'m': [], 'b0': [], 'bR': [], 'argmax': [], 'refw': [], 'refh': [], 'dl': [], 'dr': [], 'n': 0, 'top': 0}

res = {}
for seed in (1, 2):
    cache = sorted((CODE / f"342_cache_val_v2_seed{seed}").glob("342_*.pt"))
    idxj = json.loads((SPEC / "327_训练实施输入索引.json").read_text(encoding='utf-8'))
    cp = next(c for c in idxj['checkpoints'] if c['seed'] == seed)
    model = S.load_real_model(cp['path'])
    mods = [m for _, m in model.named_modules() if isinstance(m, Integral)]
    assert len(mods) == 1
    hk = mods[0].register_forward_hook(_hook)

    G = {'g0': newgrp(), 'cov': newgrp()}
    n_gt = n_tiny = n_geo0 = 0
    max_pb_diff = 0.0; max_recon = 0.0
    R = None; proj_last = None; refw_le4 = 0; ratio_n = 0
    t0 = time.perf_counter()
    for k, f in enumerate(cache, 1):
        e = torch.load(f, map_location='cpu', weights_only=True)
        pb_c = e["final300_boxes"].double()
        gb = e["gt_boxes_640"].double(); gl = e["gt_labels"].long()
        ts = e["gt_short_side_640_all"].double()
        with Image.open(VAL / "images" / e["file_name"]) as im:
            pil = im.convert('RGB')
        pb, lg = S.global_forward_640(model, pil)
        pb = pb.double(); lg = lg.double()
        assert 'x' in CAP and 'ref' in CAP, 'hook missed'
        max_pb_diff = max(max_pb_diff, float((pb - pb_c).abs().max()))

        x = CAP['x'][0]; ref = CAP['ref'][0].double(); dist = CAP['dist'][0].double()
        proj = CAP['proj'].double(); proj_last = proj; rs = CAP['rs']
        R = x.shape[-1] // 4 - 1   # reg_max: x is (300, 4*(reg_max+1))
        pmass = torch.softmax(x.view(-1, 4, R + 1).double(), dim=-1)
        rwn = ref[:,2]*640; rhn = ref[:,3]*640
        # integrity: rebuild cxcywh from ref+dist, compare to model output
        cx = ref[:,0]; cy = ref[:,1]; w = ref[:,2]; h = ref[:,3]
        x1 = cx - (0.5*rs + dist[:,0]) * (w/rs); x2 = cx + (0.5*rs + dist[:,2]) * (w/rs)
        y1 = cy - (0.5*rs + dist[:,1]) * (h/rs); y2 = cy + (0.5*rs + dist[:,3]) * (h/rs)
        recon = torch.stack([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], 1)
        max_recon = max(max_recon, float((recon - pb).abs().max()))

        pw = pb[:,2]*640; pcx = pb[:,0]*640; pcy = pb[:,1]*640
        pcls = lg.argmax(dim=1)
        iou = iou_mat(xyxy(pb), gb); mx = iou.max(dim=0).values
        gw = gb[:,2]-gb[:,0]; gh = gb[:,3]-gb[:,1]
        gcx = (gb[:,0]+gb[:,2])/2; gcy = (gb[:,1]+gb[:,3])/2
        tiny = ts < 4; geo0 = mx < 0.5
        n_gt += int(ts.numel()); n_tiny += int(tiny.sum()); n_geo0 += int((tiny & geo0).sum())
        g0 = (tiny & geo0).nonzero().flatten(); cov = (tiny & ~geo0).nonzero().flatten()
        if g0.numel() == 0 and cov.numel() == 0: continue
        cls_ok = (pcls[:,None] == gl[None,:])
        cd = torch.sqrt((pcx[:,None]-gcx[None,:])**2 + (pcy[:,None]-gcy[None,:])**2)
        cd = torch.where(cls_ok, cd, torch.full_like(cd, 1e9))
        d1, j1 = cd.min(dim=0)
        ms = pmass[:,:,0] + pmass[:,:,-1]     # (300,4) end-bin mass per side

        for grp, sel in (('g0', g0), ('cov', cov)):
            if sel.numel() == 0: continue
            ok = d1[sel] < 1e8
            qs = j1[sel][ok]; gs = sel[ok]
            if qs.numel() == 0: continue
            A = G[grp]
            A['n'] += int(qs.numel())
            A['top'] += int((ms[qs].max(dim=1).values >= 0.5).sum())
            A['m'].append(ms[qs]); A['b0'].append(pmass[qs,:,0]); A['bR'].append(pmass[qs,:,-1])
            A['argmax'].append(pmass[qs].argmax(dim=-1))
            A['dl'].append(dist[qs,0]); A['dr'].append(dist[qs,2])
            if grp == 'g0':
                rwv = rwn[qs] / gw[gs]; rhv = rhn[qs] / gh[gs]
                A['refw'].append(rwv); A['refh'].append(rhv)
                ratio_n += int(rwv.numel()); refw_le4 += int((rwn[qs] <= 4).sum())
        if k % 64 == 0 or k == len(cache):
            print(f"seed={seed} {k}/{len(cache)} geo0={n_geo0} m={G['g0']['n']} top={G['g0']['top']} t={time.perf_counter()-t0:.0f}s", flush=True)
    hk.remove()

    out = {"files": len(cache), "n_gt": n_gt, "tiny": n_tiny, "tiny_geo0": n_geo0,
           "reg_max": int(R), "project": [round(float(v), 6) for v in proj_last.tolist()],
           "max_pb_diff_vs_cache": max_pb_diff, "max_box_recon_err": max_recon,
           "refw_le4px_frac": round(refw_le4 / max(ratio_n, 1), 4)}
    for grp in ('g0', 'cov'):
        A = G[grp]
        M = torch.cat(A['m']) if A['m'] else None
        d = {"matched": A['n'], "top": A['top'], "P_top": round(A['top'] / max(A['n'], 1), 4)}
        if M is not None:
            d["m_pct"] = qq(M)
            d["side_rate_L"] = round(float((M[:,0] >= 0.5).double().mean()), 4)
            d["side_rate_R"] = round(float((M[:,2] >= 0.5).double().mean()), 4)
            d["side_rate_T"] = round(float((M[:,1] >= 0.5).double().mean()), 4)
            d["side_rate_B"] = round(float((M[:,3] >= 0.5).double().mean()), 4)
            d["sides_top_mean"] = round(float((M >= 0.5).sum(dim=1).double().mean()), 3)
            d["mass_b0_mean"] = round(float(torch.cat(A['b0']).mean()), 4)
            d["mass_bR_mean"] = round(float(torch.cat(A['bR']).mean()), 4)
            am = torch.cat(A['argmax'])
            d["argmax_bin_hist"] = [round(float(v), 4) for v in (torch.bincount(am.flatten().long(), minlength=R+1).double() / am.numel()).tolist()]
            d["dl_pct"] = qq(torch.cat(A['dl'])); d["dr_pct"] = qq(torch.cat(A['dr']))
        if grp == 'g0' and A['refw']:
            d["ratio_refw_pct"] = qq(torch.cat(A['refw']))
            d["ratio_refw_le1"] = round(float((torch.cat(A['refw']) <= 1).double().mean()), 4)
        out[grp] = d
    res[seed] = out
    print("SEED", seed, json.dumps(out, ensure_ascii=False), flush=True)

(OUT / "360_axisB_fdr_top_v1.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("DONE", (OUT / "360_axisB_fdr_top_v1.json").stat().st_size)
