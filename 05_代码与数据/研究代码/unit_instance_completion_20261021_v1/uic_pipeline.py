import torch, math, hashlib, json

def select_window(centers_xy, grid=64, win_px=640, win_size=64):
    """317 §2: on 640 plane 64^2 grid, pick cell with max count of fullP(final300) prediction centers;
    row-major tie-break (smallest row then col). Return (r,c) top-left cell index (0-based)."""
    # cell size in px
    cs = win_px/grid
    counts = torch.zeros(grid,grid, dtype=torch.long)
    for x,y in centers_xy:
        ci=int(x//cs); cj=int(y//cs)
        ci=max(0,min(grid-1,ci)); cj=max(0,min(grid-1,cj))
        counts[ci,cj]+=1
    flat=counts.flatten()
    maxv=flat.max()
    if maxv==0:
        return (0,0)  # no centers -> top-left, deterministic
    idx=int(torch.nonzero(flat==maxv)[0])  # first in row-major order
    return (idx//grid, idx%grid)

def window_box_px(rc, grid=64, win_px=640):
    cs=win_px/grid; r,c=rc
    x0=c*cs; y0=r*cs
    return (x0,y0,x0+cs if False else x0+win_px/grid, y0+win_px/grid)  # 64^2 square

def development_set(sample_plan_development_ids, expected_sha, ids_bytes_for_sha):
    """Verify development ids SHA against 327 fdea615d..., return set."""
    h=hashlib.sha256(ids_bytes_for_sha).hexdigest()
    return {"sha":h,"match":h==expected_sha,"count":len(sample_plan_development_ids)}

def heldout_guard(path):
    """Write paths must not enter heldout3236."""
    bad = ("heldout3236" in str(path)) or ("heldout" in str(path).lower() and "3236" in str(path))
    return {"path":str(path),"is_heldout":bad,"safe_to_write":not bad}

def run_arithmetic(n_dev=3235, batch=16, epochs=6):
    steps=math.ceil(n_dev/batch)  # drop_last=False
    return {"steps_per_epoch":steps,"total_steps":steps*epochs,"crop_exposures":n_dev*epochs,
            "expected_317":{"203":steps==203,"1218":steps*epochs==1218,"19410":n_dev*epochs==19410}}

def resume_state_keys():
    """317 §6: resume restores complete run state: model,optimizer,lr_sched,epoch,step,RNG."""
    return ["model","optimizer","lr_warmup_scheduler","lr_scheduler","epoch","last_epoch",
            "best_stat","scaler","ema","rng_state_torch","rng_state_cuda","commit"]
