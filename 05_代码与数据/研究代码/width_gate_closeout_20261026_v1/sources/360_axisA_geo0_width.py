
import torch, json, time
from pathlib import Path
torch.set_num_threads(4)
CODE = Path(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\342_codex_stage3_20261022_v1")
OUT  = Path(r"C:\Users\jiwei\Desktop\小论文投稿\tmp")

def xyxy(b):
    x,y,w,h = b[:,0]*640, b[:,1]*640, b[:,2]*640, b[:,3]*640
    return torch.stack([x-w/2, y-h/2, x+w/2, y+h/2], dim=1)

def iou_mat(a, b):
    ax1,ay1,ax2,ay2 = a[:,0:1],a[:,1:2],a[:,2:3],a[:,3:4]
    bx1,by1,bx2,by2 = b[:,0],b[:,1],b[:,2],b[:,3]
    iw = (torch.min(ax2,bx2)-torch.max(ax1,bx1)).clamp(min=0)
    ih = (torch.min(ay2,by2)-torch.max(ay1,by1)).clamp(min=0)
    inter = iw*ih
    aa = (ax2-ax1)*(ay2-ay1); ab = (bx2-bx1)*(by2-by1)
    return inter/(aa+ab-inter+1e-12)

def q(t, ps=(10,25,50,75,90)):
    if t.numel()==0: return None
    v = torch.quantile(t.double(), torch.tensor([p/100 for p in ps], dtype=torch.float64))
    return {f"p{p}": round(float(x),4) for p,x in zip(ps,v)}

res = {}
for seed in (1,2):
    files = sorted((CODE/f"342_cache_val_v2_seed{seed}").glob("342_*.pt"))
    acc = {"c1_aw":[], "c1_rel":[], "c1_cd":[], "c2_aw":[], "c2_rel":[], "c2_cd":[],
           "ref_aw":[], "mod4_1":[], "mod4_2":[], "w_gt1":[], "w_pred1":[], "w_pred2":[]}
    n_gt=0; n_tiny=0; n_geo0=0; n_m1=0; n_m2=0; n_ref=0
    wmin=1e9; hist_small=None
    t0=time.perf_counter()
    for f in files:
        e = torch.load(f, map_location="cpu", weights_only=True)
        pb = e["final300_boxes"].double(); lg = e["final300_logits"].double()
        gb = e["gt_boxes_640"].double(); gl = e["gt_labels"].long()
        ts = e["gt_short_side_640_all"].double(); ow, oh = e["orig_wh"]
        pw = pb[:,2]*640; ph = pb[:,3]*640
        pcx = pb[:,0]*640; pcy = pb[:,1]*640
        pcls = lg.argmax(dim=1)
        wmin = min(wmin, float(pw.min()))
        if hist_small is None:
            hist_small = torch.zeros(20, dtype=torch.float64)
        hist_small += torch.histc(pw, bins=20, min=0, max=5).double()
        iou = iou_mat(xyxy(pb), gb)
        mx = iou.max(dim=0).values
        gw = (gb[:,2]-gb[:,0]); gh = (gb[:,3]-gb[:,1])
        gcx = (gb[:,0]+gb[:,2])/2; gcy = (gb[:,1]+gb[:,3])/2
        tiny = ts < 4
        geo0 = mx < 0.5
        n_gt += int(ts.numel()); n_tiny += int(tiny.sum()); n_geo0 += int((tiny&geo0).sum())
        idx = (tiny & geo0).nonzero().flatten()
        ref = (tiny & ~geo0).nonzero().flatten()
        if idx.numel()==0 and ref.numel()==0: continue
        cls_ok = (pcls[:,None]==gl[None,:])
        # criterion 1: same class, nearest center
        cd = ((pcx[:,None]-gcx[None,:])**2 + (pcy[:,None]-gcy[None,:])**2).sqrt()
        cd = torch.where(cls_ok, cd, torch.full_like(cd, 1e9))
        d1, j1 = cd.min(dim=0)
        # criterion 2: same class, IoU>=0.1, max iou
        iou_ok = torch.where(cls_ok & (iou>=0.1), iou, torch.full_like(iou, -1.0))
        v2, j2 = iou_ok.max(dim=0)
        for grp, nm in ((idx,"c1"),(idx,"c2"),(ref,"ref")):
            pass
        # group 1: geo0 via criterion1
        if idx.numel():
            cand = j1[idx]; ok = d1[idx] < 1e8
            j = cand[ok]; g = idx[ok]
            if j.numel():
                acc["c1_aw"].append((pw[j]-gw[g]).abs())
                acc["c1_rel"].append(((pw[j]-gw[g])/gw[g]))
                acc["c1_cd"].append(d1[g])
                acc["w_gt1"].append(gw[g]); acc["w_pred1"].append(pw[j])
                acc["mod4_1"].append(pw[j] % 4)
            n_m1 += int(ok.sum())
            ok2 = v2[idx] >= 0
            j2s = j2[idx][ok2]; g2 = idx[ok2]
            if j2s.numel():
                acc["c2_aw"].append((pw[j2s]-gw[g2]).abs())
                acc["c2_rel"].append((pw[j2s]-gw[g2])/gw[g2])
                acc["c2_cd"].append(((pcx[j2s]-gcx[g2])**2+(pcy[j2s]-gcy[g2])**2).sqrt())
                acc["w_pred2"].append(pw[j2s])
                acc["mod4_2"].append(pw[j2s] % 4)
            n_m2 += int(ok2.sum())
        if ref.numel():
            g = ref; j = j1[ref]; ok = d1[ref] < 1e8
            if ok.any():
                acc["ref_aw"].append((pw[j[ok]]-gw[g[ok]]).abs()); n_ref += int(ok.sum())
    out = {"files":len(files), "n_gt":n_gt, "tiny":n_tiny, "tiny_geo0":n_geo0,
           "matched_c1":n_m1, "matched_c2":n_m2, "ref_tiny_covered":n_ref,
           "pred_w_min":round(wmin,4), "elapsed_s":round(time.perf_counter()-t0,1)}
    def cat(k): 
        return torch.cat(acc[k]) if acc[k] else torch.empty(0)
    for k in ("c1_aw","c1_rel","c1_cd","c2_aw","c2_rel","c2_cd","ref_aw","w_gt1","w_pred1","w_pred2"):
        t = cat(k); out[k] = q(t)
        if k in ("c1_aw","c2_aw","ref_aw") and t.numel():
            out[k+"_le1"] = round(float((t<=1).double().mean()),4)
            out[k+"_le2"] = round(float((t<=2).double().mean()),4)
    for k in ("mod4_1","mod4_2"):
        t = cat(k)
        if t.numel():
            h = torch.histc(t, bins=8, min=0, max=4).double()/t.numel()
            out[k] = {f"{i*0.5:.1f}": round(float(h[i]),4) for i in range(8)}
            out[k+"_lt0.1"] = round(float((t%4<0.1).double().mean()),4)
            out[k+"_n"] = int(t.numel())
    out["small_hist_0_5px"] = [round(float(x),3) for x in hist_small.tolist()]
    res[seed] = out
    print(seed, json.dumps(out, ensure_ascii=False), flush=True)
(OUT/"360_axisA_geo0_width.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("DONE")
