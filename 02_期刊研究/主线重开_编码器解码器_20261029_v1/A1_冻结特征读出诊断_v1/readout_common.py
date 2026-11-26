"""A1 deterministic sampling and finite linear-probe math. NumPy only."""
import hashlib
import numpy as np

VIEWS = ('P2', 'P2_down', 'P3_before', 'P3_after')
RING = ((-2,-2), (-2,0), (-2,2), (0,-2), (0,2), (2,-2), (2,0), (2,2))

def stable_seed(text, seed=20261030):
    return (int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) + seed) % (2**32)

def center_field(xy, centers):
    if len(centers) == 0:
        return np.zeros(len(xy), dtype=np.float64)
    distance2 = ((xy[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    return np.exp(-distance2.min(axis=1) / 8.0)  # sigma2 input pixels

def sample_points(boxes, ignore_boxes, file_name):
    """Same fine-grid input-coordinate locations for every feature view."""
    rr, cc = np.meshgrid(np.arange(160), np.arange(160), indexing='ij')
    xy_all = np.stack([cc.ravel()*4.0+2.0, rr.ravel()*4.0+2.0], -1)
    valid = (rr.ravel() >= 2) & (rr.ravel() <= 157) & (cc.ravel() >= 2) & (cc.ravel() <= 157)
    for x1,y1,x2,y2 in ignore_boxes:
        valid &= ~((xy_all[:,0] >= x1-4) & (xy_all[:,0] <= x2+4) &
                   (xy_all[:,1] >= y1-4) & (xy_all[:,1] <= y2+4))
    centers = (boxes[:, :2] + boxes[:, 2:]) * .5
    center_ij = np.floor(centers[:, ::-1] / 4).astype(np.int64)  # row,col
    selected = set()
    for row,col in center_ij:
        for dr,dc in ((0,0),) + RING:
            r,c = int(row+dr), int(col+dc)
            if 0 <= r < 160 and 0 <= c < 160 and valid[r*160+c]:
                selected.add(r*160+c)
    rng = np.random.default_rng(stable_seed(file_name))
    remaining = np.array(sorted(set(np.where(valid)[0]) - selected), dtype=np.int64)
    uniform = rng.choice(remaining, size=min(512, len(remaining)), replace=False)
    ids = np.array(sorted(selected | set(uniform.tolist())), dtype=np.int64)
    xy = xy_all[ids]
    target = center_field(xy, centers)
    inverse = {int(flat): i for i,flat in enumerate(ids)}
    ci = np.full(len(boxes), -1, dtype=np.int64)
    ri = np.full((len(boxes), 8), -1, dtype=np.int64)
    for g,(row,col) in enumerate(center_ij):
        if 0 <= row < 160 and 0 <= col < 160:
            ci[g] = inverse.get(int(row*160+col), -1)
        if ci[g] < 0:
            continue
        for j,(dr,dc) in enumerate(RING):
            r,c = int(row+dr), int(col+dc)
            k = inverse.get(r*160+c, -1) if 0 <= r < 160 and 0 <= c < 160 else -1
            if k >= 0 and target[k] <= .1 * target[ci[g]]:
                ri[g,j] = k
    eligible = (ci >= 0) & ((ri >= 0).sum(1) >= 4)
    return dict(fine_flat_ids=ids, xy=xy.astype(np.float32), target=target,
                gt_center_index=ci, gt_ring_indices=ri, gt_probe_eligible=eligible)

def max_iou(gt, boxes_xyxy):
    if len(gt) == 0:
        return np.zeros(0, dtype=np.float64)
    left = np.maximum(gt[:,None,:2], boxes_xyxy[None,:,:2])
    right = np.minimum(gt[:,None,2:], boxes_xyxy[None,:,2:])
    wh = np.maximum(right-left, 0)
    inter = wh[...,0]*wh[...,1]
    ga = np.prod(np.maximum(gt[:,2:]-gt[:,:2],0),1)
    ba = np.prod(np.maximum(boxes_xyxy[:,2:]-boxes_xyxy[:,:2],0),1)
    return (inter / np.maximum(ga[:,None]+ba[None,:]-inter,1e-12)).max(1)

def cxcywh_to_xyxy(b):
    return np.concatenate([b[:,:2]-b[:,2:]*.5, b[:,:2]+b[:,2:]*.5],-1)*640.

def fit_readouts(rows, view, ridge=.01, fold=0):
    """Two prespecified equal-capacity fits: real and within-image shuffled labels."""
    x = np.concatenate([r['features_'+view].astype(np.float64) for r in rows])
    y = np.concatenate([r['target'] for r in rows])
    weights = np.concatenate([np.full(len(r['target']),1./len(r['target'])/len(rows)) for r in rows])
    permuted = []
    for row in rows:
        rng = np.random.default_rng(stable_seed('label-null:'+str(fold)+':'+str(row['file_name'])))
        permuted.append(row['target'][rng.permutation(len(row['target']))])
    targets = np.column_stack([y, np.concatenate(permuted)])
    mean = (x * weights[:,None]).sum(0)
    std = np.sqrt(((x-mean)**2 * weights[:,None]).sum(0))
    std = np.where(std > 1e-6, std, 1.0)
    z = np.column_stack([(x-mean)/std, np.ones(len(x))])
    gram = z.T @ (z * weights[:,None])
    penalty = np.eye(z.shape[1]) * ridge
    penalty[-1,-1] = 0.0  # intercept is not penalized
    coef = np.linalg.solve(gram+penalty, z.T @ (targets*weights[:,None]))
    assert coef.shape == (193,2) and np.isfinite(coef).all()
    return dict(mean=mean, std=std, coef=coef)

def predict(row, view, state):
    x = row['features_'+view].astype(np.float64)
    z = np.column_stack([(x-state['mean'])/state['std'], np.ones(len(x))])
    return z @ state['coef']

def gt_pair_wins(row, scores):
    wins = np.full((len(row['gt_boxes']), scores.shape[1]), np.nan)
    for g in np.where(row['gt_probe_eligible'])[0]:
        c = row['gt_center_index'][g]
        ring = row['gt_ring_indices'][g]
        ring = ring[ring >= 0]
        delta = scores[c][None,:] - scores[ring]
        wins[g] = ((delta > 1e-7) + .5*(np.abs(delta) <= 1e-7)).mean(0)
    return wins

def paired_group_ci(image_rows, key, seed=20261030, iterations=2000):
    rows = [r for r in image_rows if r.get(key) is not None]
    if not rows:
        return dict(mean=None, low=None, high=None, images=0, groups=0)
    groups = sorted({r['group'] for r in rows})
    grouped = [[r[key] for r in rows if r['group']==g] for g in groups]
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(iterations):
        draw = rng.integers(0,len(groups),size=len(groups))
        bootstrap.append(np.mean([v for index in draw for v in grouped[index]]))
    return dict(mean=float(np.mean([r[key] for r in rows])),
                low=float(np.quantile(bootstrap,.025)), high=float(np.quantile(bootstrap,.975)),
                images=len(rows), groups=len(groups))

def selfcheck():
    field = center_field(np.array([[10.,10.],[18.,10.]]),np.array([[10.,10.]]))
    assert np.allclose(field,[1.,np.exp(-8.)],rtol=1e-12,atol=1e-12)
    sample = sample_points(np.array([[16.,16.,20.,20.]]),np.empty((0,4)),'synthetic')
    idx = sample['gt_center_index'][0]
    assert np.array_equal(sample['xy'][idx],[18.,18.]) and sample['target'][idx] == 1.
    assert (sample['gt_ring_indices'][0] >= 0).sum() == 8
    assert sample['gt_probe_eligible'][0]
    border = sample_points(np.array([[8.,8.,12.,12.]]),np.empty((0,4)),'synthetic-border')
    assert (border['gt_ring_indices'][0] >= 0).sum() == 3 and not border['gt_probe_eligible'][0]
    scores = sample['target'][:,None]
    assert gt_pair_wins(dict(sample,gt_boxes=np.zeros((1,4))),scores)[0,0] == 1.
    equal = np.ones_like(scores)
    assert gt_pair_wins(dict(sample,gt_boxes=np.zeros((1,4))),equal)[0,0] == .5
    assert np.isclose(max_iou(np.array([[0.,0.,2.,2.]]),np.array([[0.,0.,2.,2.]]))[0],1.)
    return 'PASS known center-field values,xy-axis/bin mapping,eligible ring mask,tie handling,and identity IoU'

if __name__ == '__main__':
    print(selfcheck())
