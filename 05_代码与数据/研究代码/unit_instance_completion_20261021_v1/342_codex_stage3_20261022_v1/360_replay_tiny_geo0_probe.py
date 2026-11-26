
import torch, json, time
from pathlib import Path
torch.set_num_threads(2)
HERE = Path(r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1\342_codex_stage3_20261022_v1")

def xyxy(b):
    x,y,w,h = b[:,0]*640, b[:,1]*640, b[:,2]*640, b[:,3]*640
    return torch.stack([x-w/2, y-h/2, x+w/2, y+h/2], dim=1)

def iou(a, b):
    ax1,ay1,ax2,ay2 = a[:,0:1],a[:,1:2],a[:,2:3],a[:,3:4]
    bx1,by1,bx2,by2 = b[:,0],b[:,1],b[:,2],b[:,3]
    iw = (torch.min(ax2,bx2)-torch.max(ax1,bx1)).clamp(min=0)
    ih = (torch.min(ay2,by2)-torch.max(ay1,by1)).clamp(min=0)
    inter = iw*ih
    aa = (ax2-ax1)*(ay2-ay1); ab = (bx2-bx1)*(by2-by1)
    return inter/(aa+ab-inter+1e-12)

bins = [i*0.25 for i in range(41)]  # 0..10 px
res = {}
for seed in (1,2):
    folder = HERE/f"342_cache_val_v2_seed{seed}"
    files = sorted(folder.glob("342_*.pt"))
    hist = torch.zeros(len(bins)-1, dtype=torch.float64)
    ngt=0; ntiny=0; ngeo0=0; n_tiny_expl=0; n_geo0_expl=0
    n_tiny_expl_ge1 = 0
    wmin = 1e9; n_lt4=0; n_pred=0
    t0=time.perf_counter()
    for f in files:
        e = torch.load(f, map_location="cpu", weights_only=True)
        pb = e["final300_boxes"]; gb = e["gt_boxes_640"]
        pw = (pb[:,2]*640)
        n_pred += pw.numel(); n_lt4 += int((pw<4).sum())
        wmin = min(wmin, float(pw.min()))
        h = torch.histc(pw, bins=len(bins)-1, min=bins[0], max=bins[-1])
        hist += h.double()
        gi = iou(xyxy(pb), gb)           # (300, N)
        mx = gi.max(dim=0).values        # per GT max IoU
        ts = e["gt_short_side_640_all"]
        tiny = ts < 4
        geo0 = mx < 0.5
        expl = e["m0_gt_explained"]
        ngt += int(ts.numel()); ntiny += int(tiny.sum())
        ngeo0 += int((tiny & geo0).sum())
        n_tiny_expl += int((tiny & expl).sum())
        n_geo0_expl += int((tiny & geo0 & expl).sum())
    res[seed] = dict(files=len(files), n_gt=ngt, tiny=ntiny, tiny_geo0=ngeo0,
                     tiny_explained=n_tiny_expl, tiny_geo0_explained=n_geo0_expl,
                     n_pred=n_pred, pred_w_lt4=n_lt4, pred_w_min=round(wmin,4),
                     elapsed=round(time.perf_counter()-t0,1),
                     hist={f"{bins[i]:.2f}": int(hist[i]) for i in range(len(hist)) if hist[i]})
    print(seed, json.dumps({k:v for k,v in res[seed].items() if k!='hist'}, ensure_ascii=False), flush=True)
Path(HERE/"360_replay_tiny_geo0_probe.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("done")
