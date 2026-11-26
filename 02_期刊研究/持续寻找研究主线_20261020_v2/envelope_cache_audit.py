"""Development-cache audit of the roundtable's fixed-33-node envelope proposal.

No detector construction/forward, no optimizer, no heldout, no remote access.
The only autograd is a small synthetic conservation test before reading data.
"""
import os
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import datetime, hashlib, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
sys.path.insert(0, str(WORK/'英文小论文/code/dn_pre_frozen_diagnostic_20261020_v1'))
import common as native
from engine.deim.dfine_utils import weighting_function
torch, np = native.torch, native.np
DATA = WORK/'英文小论文/data/08_匹配聚合覆盖探针_20261019_v1/isolated_v3'
OUT = ROOT/'envelope_cache_audit'


def save(name, value):
    native.save_json(OUT/name, value)


def softmax(z):
    e = np.exp(z-z.max(-1, keepdims=True))
    return e/e.sum(-1, keepdims=True)


def xyxy(b):
    return np.concatenate([b[..., :2]-b[..., 2:]/2, b[..., :2]+b[..., 2:]/2], -1)


def iou(a, b):
    inter = np.maximum(0, np.minimum(a[..., 2:], b[..., 2:])-np.maximum(a[..., :2], b[..., :2])).prod(-1)
    union = np.maximum(0, a[..., 2:]-a[..., :2]).prod(-1)+np.maximum(0, b[..., 2:]-b[..., :2]).prod(-1)-inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union>0)


def nodes(ref, bins, scale):
    c, wh = ref[..., :2], ref[..., 2:]
    left = c[..., :, None]-wh[..., :, None]*(.5+bins/scale)
    right = c[..., :, None]+wh[..., :, None]*(.5+bins/scale)
    # left/top native bin order is descending physical coordinate.
    return np.concatenate([left[..., ::-1], right], -2)


def probabilities(corners):
    p = softmax(corners.reshape(-1, 4, 33))
    return np.concatenate([p[:, :2, ::-1], p[:, 2:]], 1)


def envelope(old, local):
    x = local.copy()
    x[..., 0] = np.minimum(old[..., 0], local[..., 0])
    x[..., -1] = np.maximum(old[..., -1], local[..., -1])
    assert np.all(np.diff(x, axis=-1)>0)
    return x


def reachable(x, gt):
    return ((gt>=x[..., 0]) & (gt<=x[..., -1])).all(-1)


def project(p, old, new):
    p, old, new = [v.reshape(-1, 33) for v in (p, old, new)]
    # old nodes are inside the new envelope: no clipping allowed here.
    assert np.all(old[:, 0]>=new[:, 0]-1e-14) and np.all(old[:, -1]<=new[:, -1]+1e-14)
    hi = (new[:, None, :]<=old[:, :, None]).sum(-1).clip(1, 32)
    lo = hi-1
    rr = np.arange(len(p))[:, None]
    t = (old-new[rr, lo])/(new[rr, hi]-new[rr, lo])
    q = np.zeros_like(p)
    np.add.at(q, (rr, lo), p*(1-t))
    np.add.at(q, (rr, hi), p*t)
    return q


def maxent(x, mean):
    x = x.reshape(-1, 33)
    mean = mean.reshape(-1)
    width = x[:, -1]-x[:, 0]
    y = (x-x[:, :1])/width[:, None]
    target = (mean-x[:, 0])/width
    assert np.all((target>0) & (target<1)), 'endpoint/invalid mean needs separate NA handling'
    low, high = np.full(len(x), -16384.), np.full(len(x), 16384.)
    for _ in range(75):
        lam = (low+high)/2
        u = softmax(lam[:, None]*y)
        below = (u*y).sum(-1)<target
        low = np.where(below, lam, low)
        high = np.where(below, high, lam)
    u = softmax(((low+high)/2)[:, None]*y)
    assert float(np.abs((u*x).sum(-1)-mean).max())<2e-11
    return u


class MaxEntImplicit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mean):
        u = torch.from_numpy(maxent(x.detach().numpy(), mean.detach().numpy())).to(mean)
        ctx.save_for_backward(x, mean, u)
        return u

    @staticmethod
    def backward(ctx, grad):
        x, mean, u = ctx.saved_tensors
        centered = x-mean[:, None]
        var = (u*centered.square()).sum(-1)
        assert bool((var>0).all())
        du_dm = u*centered/var[:, None]
        return None, (grad*du_dm).sum(-1)


def synthetic_tests():
    torch.manual_seed(20261021)
    old = torch.tensor([[-2., -.4, 0., .7, 2.]], dtype=torch.float64)
    # Test the generic maxent derivative with 33 physical nodes.
    x = torch.linspace(-3., 4., 33, dtype=torch.float64)[None]
    z = torch.randn(1, 33, dtype=torch.float64, requires_grad=True)
    oldx = torch.linspace(-2., 2., 33, dtype=torch.float64)[None]
    p = z.softmax(-1); mean = (p*oldx).sum(-1)
    transport = torch.from_numpy(project(p.detach().numpy(), oldx.numpy(), x.numpy()))
    # Constant transfer matrix; retain p gradients instead of detaching its value.
    basis = np.eye(33)
    matrix = project(basis, np.repeat(oldx.numpy(), 33, 0), np.repeat(x.numpy(), 33, 0))
    moved = p@torch.from_numpy(matrix)
    u = MaxEntImplicit.apply(x, mean)
    q = .9*moved+.1*u
    original_grad = torch.autograd.grad(mean.sum(), z, retain_graph=True)[0]
    checks = {}
    for key, v in [('transport', moved), ('mean_reset', u), ('mixture', q)]:
        value = (v*x).sum(-1)
        grad = torch.autograd.grad(value.sum(), z, retain_graph=True)[0]
        checks[key] = {'mass_error': float((v.sum(-1)-1).abs().max()),
                       'mean_error': float((value-mean).abs().max()),
                       'geometry_gradient_error': float((grad-original_grad).abs().max())}
        assert max(checks[key].values())<2e-12
    assert torch.autograd.gradcheck(lambda m: MaxEntImplicit.apply(x, m), (mean.detach().requires_grad_(),), eps=1e-6, atol=1e-6)
    # Physical ordering: a zero-offset PMF decodes to the reference box edges.
    ref = np.array([[.4, .6, .1, .2]])
    bins = np.linspace(-4., 4., 33)
    assert np.max(np.abs(nodes(ref, bins, 4.)[..., 16]-xyxy(ref)))<1e-14
    # Target equal to the endpoint remains inside, not a floating-point penalty.
    testx = nodes(ref, bins, 4.)
    assert reachable(testx, testx[..., 0])[0]
    return {'conservation_and_geometry_gradients': checks, 'implicit_derivative_gradcheck': True,
            'left_top_sorted_and_zero_offset_decoding': True, 'closed_support_endpoint': True}


def target_ce(x, gt, prior):
    n = len(x)
    valid = (gt>=x[:, 0]) & (gt<=x[:, -1])
    hi = (x<=gt[:, None]).sum(-1).clip(1, 32); lo = hi-1
    rr = np.arange(n)
    width = x[rr, hi]-x[rr, lo]
    t = ((gt-x[rr, lo])/width).clip(0, 1)
    ce = -(1-t)*np.log(prior[rr, lo].clip(1e-300))-t*np.log(prior[rr, hi].clip(1e-300))
    ce[~valid] = np.nan
    width[~valid] = np.nan
    return ce, width, valid


def stats(v):
    v = np.asarray(v, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    return {'n': len(v), 'mean': float(v.mean()) if len(v) else None,
            'q25': float(np.quantile(v, .25)) if len(v) else None,
            'median': float(np.median(v)) if len(v) else None,
            'q75': float(np.quantile(v, .75)) if len(v) else None}


def main():
    started = time.time()
    OUT.mkdir(exist_ok=False)
    resource = native.resource_limits()
    protocol = {
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'candidate': 'Roundtable fixed33 envelope, once before layer2; coordinates stop-gradient; mean-preserving PMF transfer vs same-mean maximum-entropy reset',
        'data': 'Existing same-forward GPU BASE24 development00-07, A seed1/B seed2 only. Never heldout. Exposure counts, not independent GT or final AP.',
        'fixed': {'epsilon': .1, 'maxent': 'exp(lambda*x), mean equal to old physical mean, implicit d u / d mean',
                  'new_nodes': '[min(old0,g0),g1,...,g31,max(old32,g32)]',
                  'reference': 'saved first distribution-layer pred_boxes; shift-only and scale-only preserve original size and center respectively'},
        'primary': 'final native union, IoU==0, GT exact box outside own original support; separate other-query-reachable vs all-query-unreachable',
        'additional': 'IoU<.1 outside, good IoU>=.5, all; cross capacity N<=Q/N>Q and input shortside<4 vs all',
        'decision': 'If math tests fail reject this implementation. If both seeds have zero newly exact-reachable primary exposures, no training for the range-rescue rationale. Positive reachability alone is not permission to train or proof of PMF-history value.',
        'descriptive_only': ['shift/scale decomposition has overlapping sets, not additive causal attribution',
                             'CE(migrated mixture)-CE(reset) in nats, eligible physical target edges only; not a training-benefit surrogate or mandatory causal gate',
                             'old vs new GT bracketing widths at640; continuous soft expectations have no hard pixel precision floor',
                             'IoU-weighted first-layer FGL CE difference and zero-weight counts; native L1/GIoU/DN still present'],
        'historical': 'Oct5 transport and Oct6 reanchor prototypes already exist; this audit completes a specified variant, does not establish novelty.',
        'script_sha256': native.sha(__file__), 'resource': resource,
        'new_detector_forwards': 0, 'new_detector_backwards': 0, 'optimizer_steps': 0,
    }
    save('00_protocol.json', protocol)
    save('01_synthetic_tests.json', synthetic_tests())
    torch.set_grad_enabled(False)
    rows, sources = [], []
    checks = {'decode_first_box_max_abs': 0., 'transport_mass_error': 0., 'transport_mean_error': 0., 'maxent_mean_error': 0., 'min_q': 1., 'zero_u_entries': 0}
    for side, seed in [('A', 1), ('B', 2)]:
        for batch in range(8):
            ap = DATA/side/f'development/batch{batch:02d}_analysis.json'
            pp = DATA/side/f'frozen_predictions/development/batch{batch:02d}_predictions.pt'
            analysis = json.loads(ap.read_text(encoding='utf8'))
            digest = native.sha(pp)
            assert digest == analysis['prediction_sha256']
            sources.append({'analysis': str(ap), 'analysis_sha256': native.sha(ap), 'prediction': str(pp), 'prediction_sha256': digest})
            data = torch.load(pp, map_location='cpu', weights_only=False)
            outputs = data['outputs']; first = outputs['aux_outputs'][0]
            scale = abs(float(outputs['reg_scale']))
            bins = weighting_function(32, outputs['up'], outputs['reg_scale']).numpy().astype(float)
            for image, item in enumerate(analysis['inputs']):
                match = item['branches']['final']['native_union']
                qs = np.asarray(match['pair_query'], dtype=int); gs = np.asarray(match['pair_GT'], dtype=int)
                ref_all = outputs['ref_points'][image].double().numpy()
                b1_all = first['pred_boxes'][image].double().numpy()
                assert np.all(ref_all[:, 2:]>0) and np.all(b1_all[:, 2:]>0)
                old_all = nodes(ref_all, bins, scale)
                old = old_all[qs]
                gtbox = data['targets'][image]['boxes'][gs].double().numpy(); gt = xyxy(gtbox)
                pred = xyxy(outputs['pred_boxes'][image, qs].double().numpy())
                final_iou = iou(pred, gt)
                assert np.max(np.abs(final_iou-np.asarray(match['pair_iou'])))<3e-5
                first_iou = iou(xyxy(b1_all[qs]), gt)
                old_reach = reachable(old, gt)
                # GT may repeat among union queries: these are exposures, not unique objects.
                any_reach = ((gt[:, None, :]>=old_all[None, :, :, 0]) &
                             (gt[:, None, :]<=old_all[None, :, :, -1])).all(-1).any(-1)
                p = probabilities(first['pred_corners'][image, qs].double().numpy())
                means = (p*old).sum(-1)
                checks['decode_first_box_max_abs'] = max(checks['decode_first_box_max_abs'], float(np.max(np.abs(means-xyxy(b1_all[qs])))))
                grids = {}
                for arm in ['shift_only', 'scale_only', 'full']:
                    ref = ref_all[qs].copy()
                    if arm in ['shift_only', 'full']: ref[:, :2] = b1_all[qs, :2]
                    if arm in ['scale_only', 'full']: ref[:, 2:] = b1_all[qs, 2:]
                    grids[arm] = envelope(old, nodes(ref, bins, scale))
                new = grids['full']
                moved = project(p, old, new)
                flatx, flatmean = new.reshape(-1, 33), means.reshape(-1)
                u = maxent(flatx, flatmean)
                q = .9*moved+.1*u
                checks['transport_mass_error'] = max(checks['transport_mass_error'], float(np.max(np.abs(moved.sum(-1)-1))))
                checks['transport_mean_error'] = max(checks['transport_mean_error'], float(np.max(np.abs((moved*flatx).sum(-1)-flatmean))))
                checks['maxent_mean_error'] = max(checks['maxent_mean_error'], float(np.max(np.abs((u*flatx).sum(-1)-flatmean))))
                checks['min_q'] = min(checks['min_q'], float(q.min()))
                checks['zero_u_entries'] += int((u==0).sum())
                ce_q, width_new, eligible = target_ce(flatx, gt.reshape(-1), q)
                ce_u, _, _ = target_ce(flatx, gt.reshape(-1), u)
                _, width_old, old_edge_eligible = target_ce(old.reshape(-1, 33), gt.reshape(-1), p.reshape(-1, 33))
                for j, (query, g) in enumerate(zip(qs, gs)):
                    edges = slice(4*j, 4*j+4)
                    row = {'seed': seed, 'batch': batch, 'image': image, 'query': int(query), 'GT': int(g),
                           'image_GT': item['GT'], 'Q': len(ref_all), 'shortside640': float(gtbox[j, 2:].min()*640),
                           'final_IoU': float(final_iou[j]), 'first_IoU': float(first_iou[j]),
                           'old_reachable': bool(old_reach[j]), 'any_query_old_reachable': bool(any_reach[j]),
                           'new_reachable': {a: bool(reachable(x[j:j+1], gt[j:j+1])[0]) for a, x in grids.items()},
                           'CE_delta_q_minus_u_edges': [float(v) if np.isfinite(v) else None for v in (ce_q-ce_u)[edges]],
                           'eligible_edges': int(eligible[edges].sum()),
                           'GT_new_interval_width_px': [float(v*640) if np.isfinite(v) else None for v in width_new[edges]],
                           'GT_old_interval_width_px': [float(v*640) if np.isfinite(v) else None for v in width_old[edges]]}
                    rows.append(row)
            print(json.dumps({'seed': seed, 'batch': batch, 'exposures_so_far': len(rows), 'elapsed': round(time.time()-started, 2)}), flush=True)
    assert checks['decode_first_box_max_abs']<2e-6
    assert checks['transport_mean_error']<2e-11
    groups = []
    for seed in [1, 2]:
        for capacity in ['all', 'N<=Q', 'N>Q']:
            for size in ['all', 'shortside<4']:
                for cohort in ['all', 'IoU0_outside', 'IoU<.1_outside', 'IoU>=.5']:
                    for source in ['all', 'other_query_reachable', 'all_queries_unreachable']:
                        selected = [r for r in rows if r['seed']==seed and
                                    (capacity=='all' or (r['image_GT']<=r['Q'])==(capacity=='N<=Q')) and
                                    (size=='all' or r['shortside640']<4) and
                                    (cohort=='all' or (cohort=='IoU0_outside' and r['final_IoU']==0 and not r['old_reachable']) or
                                     (cohort=='IoU<.1_outside' and r['final_IoU']<.1 and not r['old_reachable']) or
                                     (cohort=='IoU>=.5' and r['final_IoU']>=.5)) and
                                    (source=='all' or r['any_query_old_reachable']==(source=='other_query_reachable'))]
                        n = len(selected)
                        ce = [v for r in selected for v in r['CE_delta_q_minus_u_edges'] if v is not None]
                        weighted_num = sum(r['first_IoU']*sum(v for v in r['CE_delta_q_minus_u_edges'] if v is not None) for r in selected)
                        weighted_den = sum(r['first_IoU']*r['eligible_edges'] for r in selected)
                        newonly = [r for r in selected if not r['old_reachable'] and r['new_reachable']['full']]
                        widths = [b/a for r in selected for a,b in zip(r['GT_old_interval_width_px'],r['GT_new_interval_width_px']) if a is not None and b is not None]
                        groups.append({'seed': seed, 'capacity': capacity, 'size': size, 'cohort': cohort, 'query_support_source': source,
                                       'exposures': n, 'old_reachable': sum(r['old_reachable'] for r in selected),
                                       'newly_reachable': {a: sum(not r['old_reachable'] and r['new_reachable'][a] for r in selected) for a in grids},
                                       'full_new_only_not_scale': sum(not r['old_reachable'] and r['new_reachable']['full'] and not r['new_reachable']['scale_only'] for r in selected),
                                       'full_new_only_not_shift': sum(not r['old_reachable'] and r['new_reachable']['full'] and not r['new_reachable']['shift_only'] for r in selected),
                                       'CE_delta_q_minus_u': stats(ce),
                                       'newly_reachable_CE_delta': stats([v for r in newonly for v in r['CE_delta_q_minus_u_edges'] if v is not None]),
                                       'weighted_CE_delta_numerator': weighted_num, 'weighted_CE_delta_denominator': weighted_den,
                                       'weighted_CE_delta': weighted_num/weighted_den if weighted_den else None,
                                       'first_IoU_zero_exposures': sum(r['first_IoU']==0 for r in selected),
                                       'new_to_old_bracket_width_ratio': stats(widths),
                                       'new_GT_bracket_width_px': stats([v for r in selected for v in r['GT_new_interval_width_px'] if v is not None])})
    primary = [g for g in groups if g['capacity']=='all' and g['size']=='all' and g['cohort']=='IoU0_outside' and g['query_support_source']=='all']
    decision = 'zero_new_reach_both_seeds_no_range_rescue_training' if all(g['newly_reachable']['full']==0 for g in primary) else 'nonzero_range_scope_only_history_value_and_training_justification_unresolved'
    result = {'protocol_sha256': native.sha(OUT/'00_protocol.json'), 'sources': sources, 'checks': checks,
              'groups': groups, 'primary': primary, 'decision': decision, 'seconds': time.time()-started,
              'limits': 'Native fixed-coordinate historical outputs. New coordinates, PMF transfer and reset have not been trained. Reachability is exact GT feasibility, not IoU.5 recall. CE is a descriptive prior, not detector/AP evidence. No hard pixel quantization floor.'}
    save('02_results.json', result)
    save('03_exposure_rows.json', {'rows': rows})
    save('04_manifest.json', {p.name: native.sha(p) for p in OUT.iterdir() if p.is_file()})
    print(json.dumps({'checks': checks, 'primary': primary, 'decision': decision}, ensure_ascii=False, indent=2), flush=True)


if __name__=='__main__':
    main()
