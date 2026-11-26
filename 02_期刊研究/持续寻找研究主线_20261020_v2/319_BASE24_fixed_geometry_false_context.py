"""318: fixed geometry, oracle-mass false-context deletion audit. Root runs only.
No torch/scipy/model/PT/image/solver. Synthetic runs before real JSON/NPZ reads.
"""
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
sys.dont_write_bytecode = True
import argparse
import ctypes
import gc
import hashlib
import json
import pathlib
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parent
PROTOCOL = ROOT / '318_B_固定几何消耗无合法边条件_事前协议.txt'
PROTOCOL_SHA = '1a5592415f5ee00fa5624f92ebb05515e6b64802cf0302ef60bc0d33f77ee368'
D263 = ROOT / '263_BASE24_residual_window_probe'
D284 = ROOT / '284_BASE24_explanation_context_probe'
D300 = ROOT / '300_BASE24_oracle_residual_transport_polished'
WINDOWS = ('A_density', 'B_residual')
COHORTS = ('all', 'primary_Geo0', 'geometry_covered', 'primary_Geo0_tiny', 'geometry_covered_tiny')
GRID, REPEATS, RANDOM_SEED = 16, 32, 2026102105
PARITY_TOL, ALG_TOL, EFFECT_TOL = 2e-8, 1e-12, 1e-8
ANCHORS = {
 '299_BASE24_oracle_residual_transport_polished.py': 'c3ba865bd804e7257b6f379bc237035d56e1c5e4d65fcd97d7fb1e2a163b6526',
 '263_BASE24_residual_window_probe/00_manifest_and_synthetic.json': 'fc6cd224c14c4f9f9e35cb195b9f41240d723cb15a0039e82417a64151c0f7c8',
 '263_BASE24_residual_window_probe/01_results.json': '56b7bb86d3f9927bd501df48a2df9be7aab82a0157942b5da47c2513797530b3',
 '263_BASE24_residual_window_probe/02_output_sha256.json': '8e200dee1ffe99616aa0207e5b80222f2370a9904c529c37a7c5f42f9535d934',
 '284_BASE24_explanation_context_probe/00_sources_and_synthetic.json': '9afae6ba9a933575357a78e99fdd6bd505d44edebafc3136d470c4fe1ae792ef',
 '284_BASE24_explanation_context_probe/01_results.json': 'da0d380c27fdc76bf935955fad17edf3745a510d536208f2fc8927df6a1f3578',
 '284_BASE24_explanation_context_probe/02_output_sha256.json': 'ade4a6a8a104a27c4cc2e06f02c6b24dfbee4b64cffb9e50de5b9b0365c39809',
 '300_BASE24_oracle_residual_transport_polished/00_sources_and_synthetic.json': '5d13f97c269436ba30530dd907e41592d7264b66e6d58d0b7b2eb0854c496022',
 '300_BASE24_oracle_residual_transport_polished/01_results.json': '3578581c438d5ee0fe39d6008e2fbab45709b27284489c5e80ede7f4931915d3',
 '300_BASE24_oracle_residual_transport_polished/02_output_sha256.json': '089733fb15fba0a15654f0f8d146e8013ec2405fe48cab4815bb34b3ce0daa7c',
}
COUNTS = dict(npz_batches=0, images=0, windows=0, random_deletions=0,
              PT_loads=0, model_builds=0, checkpoint_loads=0, image_reads=0,
              forwards=0, gradients=0, optimizer_steps=0, solver_calls=0, heldout_reads=0)
np = None


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    with pathlib.Path(path).open(encoding='utf-8-sig') as f:
        return json.load(f)


def save(path, value):
    with pathlib.Path(path).open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def memory_available():
    class Status(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ('total_phys', 'avail_phys', 'total_page', 'avail_page', 'total_virtual', 'avail_virtual', 'avail_extended')]
    s = Status()
    s.length = ctypes.sizeof(s)
    need(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)), 'GlobalMemoryStatusEx')
    return s.avail_phys / 2**30


def resources():
    need(os.name == 'nt', 'this frozen protocol targets Windows')
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = ctypes.c_void_p
    handle = k.GetCurrentProcess()
    k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    k.GetPriorityClass.argtypes = [ctypes.c_void_p]
    k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    k.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    need(k.SetPriorityClass(handle, 0x4000), 'BelowNormal failed')
    mask = (1 << 13) | (1 << 15)
    need(k.SetProcessAffinityMask(handle, mask), 'affinity [13,15] failed')
    actual, system = ctypes.c_size_t(), ctypes.c_size_t()
    need(k.GetProcessAffinityMask(handle, ctypes.byref(actual), ctypes.byref(system)), 'read affinity')
    need(actual.value == mask and k.GetPriorityClass(handle) == 0x4000, 'resource state')
    available = memory_available()
    need(available > 8., 'available RAM must exceed 8 GiB')
    return dict(threads=2, interop='not_applicable_no_torch', affinity=[13, 15],
                priority='BelowNormal', available_RAM_GiB=available, executable=sys.executable,
                MKL_THREADING_LAYER=os.environ['MKL_THREADING_LAYER'])


def source_manifest():
    files = {}
    def add(path, digest):
        p = pathlib.Path(path).resolve()
        need(p.is_relative_to(ROOT) and 'heldout' not in p.as_posix().lower(), 'source path outside approved scope')
        need(sha(p) == digest, 'source SHA changed: ' + str(p))
        need(str(p) not in files or files[str(p)] == digest, 'conflicting SHA')
        files[str(p)] = digest
    add(PROTOCOL, PROTOCOL_SHA)
    add(pathlib.Path(__file__), sha(__file__))
    for relative, digest in ANCHORS.items():
        add(ROOT / relative, digest)
    for folder in (D263, D284, D300):
        result, index = read(folder / '01_results.json'), read(folder / '02_output_sha256.json')
        unchanged = result.get('sources_unchanged', result.get('source_unchanged'))
        need(result['state'] == 'complete' and isinstance(unchanged, dict)
             and len(unchanged) > 0 and all(v is True for v in unchanged.values()), 'producer incomplete/changed')
        for letter in ('A', 'B'):
            for batch in range(8):
                stem = f'{letter}_batch{batch:02d}'
                names = [stem + '.json'] + ([stem + '_arrays.npz'] if folder == D300 else [])
                for name in names:
                    digest = result['artifact_sha256'][name]
                    need(index[name] == digest, 'producer artifact index differs')
                    add(folder / name, digest)
    meta = read(D300 / '00_sources_and_synthetic.json')
    par = meta['parameters']
    need((par['grid'], par['epsilon'], par['kappa'], par['ABS']) == (16, 1., 0., 2e-8), '300 scientific parameters')
    inherited = {str(pathlib.Path(p).resolve()): h for p, h in meta['source_sha256'].items()}
    for p, h in files.items():
        if pathlib.Path(p).parent in (D263, D284):
            need(inherited.get(p) == h, '300 inherited identity differs')
    return files, dict(inherited_300_source_sha256=meta['source_sha256'],
                      inherited_upstream_files_reopened=False,
                      actual_reads='318/319/299 source and fixed 263/284/300 JSON; 300 sixteen NPZ only')


def geometry_cost(points, boxes):
    cost = np.full((len(points), len(boxes)), np.inf, dtype=np.float64)
    good = np.isfinite(boxes).all(1) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    for j in np.flatnonzero(good):
        outside = np.maximum(np.maximum(boxes[j, :2] - points, points - boxes[j, 2:]), 0.)
        cost[:, j] = (outside / (boxes[j, 2:] - boxes[j, :2])).sum(1)
    return cost


def class_sum(p, labels, selected=None):
    if selected is None:
        selected = np.arange(len(labels))
    u = np.zeros((10, p.shape[0]), dtype=np.float64)
    for c in range(10):
        ids = selected[labels[selected] == c]
        u[c] = p[:, ids].sum(1)
    return u


def independent(mass, cost, labels):
    need(mass.shape == (10, cost.shape[0]) and cost.shape[1] == len(labels), 'allocation shape')
    need(np.isfinite(mass).all() and (mass >= 0).all() and not np.isnan(cost).any()
         and (cost >= 0).all() and ((labels >= 0) & (labels < 10)).all(), 'allocation inputs')
    w = mass[labels].T * np.exp(-cost)
    den = 1. + w.sum(0)
    p, z = w / den, 1. / den
    u = class_sum(p, labels)
    factor = np.minimum(1., np.divide(mass, u, out=np.ones_like(mass), where=u > 0))
    capped = p * factor[labels].T
    null = z + (p - capped).sum(0)
    r = np.maximum(mass - u, 0.)
    tol = ALG_TOL * (1 + mass.sum() + len(labels))
    need(np.max(np.abs(p.sum(0) + z - 1), initial=0) <= tol, 'independent columns')
    need(np.max(np.abs(capped.sum(0) + null - 1), initial=0) <= tol, 'cap columns')
    need(np.max(np.abs(r - (mass - class_sum(capped, labels))), initial=0) <= tol, 'one-pass clip identity')
    return p, u, r, null


def deleted(mass, p, labels, ids, u, base):
    need(len(np.unique(ids)) == len(ids) and ((ids >= 0) & (ids < len(labels))).all(), 'deletion indices')
    removed = class_sum(p, labels, ids)
    result = np.maximum(mass - u + removed, 0.)
    keep = np.setdiff1d(np.arange(len(labels)), ids, assume_unique=True)
    keep_sum = class_sum(p, labels, keep)
    check = np.maximum(mass - keep_sum, 0.)
    delta = result - base
    tol = ALG_TOL * (1 + mass.sum() + len(labels))
    need(np.max(np.abs(result - check), initial=0) <= tol, 'delete/recompute equivalence')
    need(np.min(delta, initial=0) >= -tol and np.max(delta - removed, initial=0) <= tol
         and np.max(result - mass, initial=0) <= tol, 'deletion monotonicity/bounds')
    factors = np.minimum(1., np.divide(mass, keep_sum, out=np.ones_like(mass), where=keep_sum > 0))
    kept_p = p[:, keep]
    capped = kept_p * factors[labels[keep]].T
    kept_null = 1. - kept_p.sum(0) + (kept_p - capped).sum(0)
    need(np.max(np.abs(capped.sum(0) + kept_null - 1.), initial=0) <= tol
         and np.max(class_sum(capped, labels[keep]) - mass, initial=0) <= tol
         and np.min(kept_null, initial=0) >= -tol, 'deleted cap row/column/null contract')
    return result, delta


def bad_ids(flat, neighbors):
    need(all(str(int(f)) in neighbors for f in flat), 'missing global legal neighbors')
    return np.array([j for j, f in enumerate(flat) if len(neighbors[str(int(f))]) == 0], dtype=np.int64)


def random_ids(labels, bad, seed, batch, image, window, rep):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [RANDOM_SEED, seed, batch, image, window, rep])))
    out = []
    for c in range(10):
        n = int(np.sum(labels[bad] == c))
        pool = np.flatnonzero(labels == c)
        if n:
            out.extend(rng.choice(pool, size=n, replace=False).tolist())
    ids = np.array(sorted(out), dtype=np.int64)
    need(np.array_equal(np.bincount(labels[ids], minlength=10), np.bincount(labels[bad], minlength=10)), 'random class counts')
    need(len(np.unique(ids)) == len(ids), 'random without replacement')
    return ids


def stats(values):
    valid = [float(v) for v in values if v is not None]
    need(all(np.isfinite(v) for v in valid), 'finite statistics')
    if not valid:
        return dict(n=0, NA=len(values), sum=None, mean=None, median=None, min=None, max=None,
                    q05=None, q25=None, q75=None, q95=None)
    a = np.array(valid)
    q = np.quantile(a, [.05, .25, .75, .95], method='linear')
    return dict(n=len(valid), NA=len(values)-len(valid), sum=float(a.sum()), mean=float(a.mean()),
                median=float(np.median(a)), min=float(a.min()), max=float(a.max()),
                **{k: float(v) for k, v in zip(('q05', 'q25', 'q75', 'q95'), q)})


def summarize(rows):
    result = {}
    for cohort in COHORTS:
        rr = [r for r in rows if cohort == 'all' or r[cohort]]
        by_image = {}
        for r in rr:
            by_image.setdefault((r['batch'], r['image_index']), []).append(r)
        metrics = {}
        for name in ('raw_delta', 'relative_delta', 'raw_bad_minus_random_mean', 'relative_bad_minus_random_mean'):
            im_values = [stats([r[name] for r in group])['mean'] for group in by_image.values()]
            metrics[name] = dict(pooled_GT=stats([r[name] for r in rr]), input_mean=stats(im_values))
        random_panel = {}
        for name in ('raw', 'relative'):
            rnd = [[r['random_' + name][k] for r in rr] for k in range(REPEATS)]
            pooled_means = [stats(v)['mean'] for v in rnd]
            input_means = [stats([stats([r['random_' + name][k] for r in group])['mean']
                                 for group in by_image.values()])['mean'] for k in range(REPEATS)]
            bad_pooled = metrics[name + '_delta']['pooled_GT']['mean']
            bad_input = metrics[name + '_delta']['input_mean']['mean']
            random_panel[name] = dict(pooled_means_32=pooled_means, input_means_32=input_means,
                pooled_distribution=stats(pooled_means), input_distribution=stats(input_means),
                paired_bad_minus_random_pooled_32=[None if v is None else bad_pooled-v for v in pooled_means],
                paired_bad_minus_random_input_32=[None if v is None else bad_input-v for v in input_means])
        affected = [r for r in rr if r['affected']]
        result[cohort] = dict(GT_exposures=len(rr), unique_GT_exposures=len({(r['batch'], r['image_index'], r['GT']) for r in rr}),
            inputs=len(by_image), affected_unique_GT=len({(r['batch'], r['image_index'], r['GT']) for r in affected}),
            affected_inputs=len({(r['batch'], r['image_index']) for r in affected}), metrics=metrics, random=random_panel,
            max_raw_delta_any_window=max((r.get('max_raw_delta_across_windows', r['raw_delta']) for r in rr), default=None))
    return result


def merge_windows(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r['batch'], r['image_index'], r['GT']), []).append(r)
    out = []
    for group in groups.values():
        first = group[0]
        need(all(all(r[k] == first[k] for k in ('label', 'actual_input_sub4') + COHORTS[1:]) for r in group), 'cohort differs across windows')
        row = {k: first[k] for k in ('batch', 'image_index', 'GT', 'label', 'actual_input_sub4') + COHORTS[1:]}
        for name in ('raw_delta', 'relative_delta', 'raw_bad_minus_random_mean', 'relative_bad_minus_random_mean'):
            row[name] = stats([r[name] for r in group])['mean']
        for name in ('raw', 'relative'):
            row['random_' + name] = [stats([r['random_' + name][k] for r in group])['mean'] for k in range(REPEATS)]
        row.update(affected=any(r['affected'] for r in group),
                   max_raw_delta_across_windows=max(r['raw_delta'] for r in group),
                   windows=[r['window'] for r in group], cells=[r['cell'] for r in group])
        out.append(row)
    return out


def probe_image(im, ctx, prior, ar, batch, seed):
    index = im['image_index']
    need(index == ctx['image_index'] == prior['image_index'] and im['GT'] == ctx['GT'] == prior['GT'], 'image identity')
    post = im['post']
    flat_all = np.asarray(post['P_flat'], dtype=np.int64)
    natural, keep = np.asarray(post['natural_top300'], dtype=np.int64), np.asarray(post['threshold_kept'], dtype=bool)
    need(natural.shape == keep.shape == (300,) and np.array_equal(flat_all, natural[keep])
         and len(set(flat_all.tolist())) == len(flat_all), 'native fullflatten P identity')
    scores = np.asarray(post['P_scores'], dtype=float)
    boxes_all = np.asarray(post['P_boxes_xyxy'], dtype=float).reshape(-1, 4)
    need(scores.shape == flat_all.shape and (scores > .001).all() and len(boxes_all) == len(flat_all), 'fullP threshold')
    neighbors = ctx['thresholds']['0.5']['all_P_legal_neighbors']
    need(set(neighbors) == {str(int(f)) for f in flat_all}, 'all global P neighbor keys')
    need(all(all(0 <= g < im['GT'] for g in v) for v in neighbors.values()), 'global GT neighbor IDs')
    need(flat_all[bad_ids(flat_all, neighbors)].tolist() == ctx['thresholds']['0.5']['full_image_no_legal_P'], 'global no-legal identity')
    lookup = {int(f): j for j, f in enumerate(flat_all)}
    primary = set(ctx['thresholds']['0.5']['definitions']['primary_G_geo0']['full_image_zero_degree_GT'])
    result = {}
    for wi, arm in enumerate(WINDOWS):
        old, cw = prior['windows'][arm], ctx['thresholds']['0.5']['definitions']['primary_G_geo0']['windows'][arm]
        if old['NA']:
            need(cw['NA'] and im['windows'][arm]['cell'] is None, 'window NA identity')
            result[arm] = dict(NA=True, GT_rows=[])
            continue
        need(not cw['NA'] and old['cell'] == cw['cell'] == im['windows'][arm]['cell'], 'frozen window cell')
        stem = f'i{index}_{arm}'
        mass, weights = ar[stem + '_mass'], ar[stem + '_GT_weights']
        gtids, flat = ar[stem + '_GT_ids'], ar[stem + '_condition_flat']
        need(mass.dtype == weights.dtype == np.float64 and mass.shape == (10, 256)
             and weights.shape == (len(gtids), 256), 'mass/weights schema')
        need(gtids.tolist() == old['complete_GT'] == cw['complete_GT']
             and flat.tolist() == old['condition_flat'] == cw['supplies'], 'exact GT and centre-supply IDs')
        need(np.isfinite(weights).all() and (weights >= 0).all()
             and np.max(np.abs(weights.sum(1)-1), initial=0) <= 1e-14, 'unit GT kernels')
        oldrows = old['GT_rows']
        need([r['GT'] for r in oldrows] == gtids.tolist(), 'GT row identity')
        labels = np.asarray([r['label'] for r in oldrows], dtype=np.int64)
        need(((labels >= 0) & (labels < 10)).all(), 'GT label range')
        rebuilt = np.zeros_like(mass)
        for k, c in enumerate(labels):
            rebuilt[c] += weights[k]
        need(np.array_equal(rebuilt, mass), 'oracle mass equals fixed kernel sum')
        xy = np.asarray(old['xyxy'], dtype=float)
        need(np.array_equal(xy, cw['window_xyxy']) and np.array_equal(xy, im['windows'][arm]['xyxy_pixels']), 'window xyxy')
        points = np.array([(xy[0]+(x+.5)*(xy[2]-xy[0])/16, xy[1]+(y+.5)*(xy[3]-xy[1])/16)
                           for y in range(16) for x in range(16)], dtype=float)
        boxes = boxes_all[[lookup[int(f)] for f in flat]]
        plabels = flat % 10
        cost = geometry_cost(points, boxes)
        p, u, base, null = independent(mass, cost, plabels)
        base_error = float(np.max(np.abs(base - ar[stem + '_I_cap_residual']), initial=0))
        null_error = float(np.max(np.abs(null - ar[stem + '_I_cap_null']), initial=0))
        need(base_error <= PARITY_TOL and null_error <= PARITY_TOL, '300 Icap residual/null reproduction')
        bad = bad_ids(flat, neighbors)
        after, delta = deleted(mass, p, plabels, bad, u, base)
        random_raw, deletion_lists = [], []
        for rep in range(REPEATS):
            ids = random_ids(plabels, bad, seed, batch, index, wi, rep)
            _, dr = deleted(mass, p, plabels, ids, u, base)
            random_raw.append(np.array([weights[k] @ dr[c] for k, c in enumerate(labels)]))
            deletion_lists.append(flat[ids].tolist())
        random_raw = np.asarray(random_raw).reshape(REPEATS, len(gtids))
        rows = []
        for k, g in enumerate(gtids):
            oldrow, c = oldrows[k], int(labels[k])
            need(bool(oldrow['primary_Geo0']) == (int(g) in primary), 'primary Geo0 scope')
            raw0, raw1 = float(weights[k] @ base[c]), float(weights[k] @ after[c])
            raw = float(weights[k] @ delta[c])
            den = float(weights[k] @ mass[c])
            need(np.isfinite(den), 'finite GT projection denominator')
            ratio = raw / den if den > 0 else None
            rand = random_raw[:, k].tolist()
            rel = [v/den for v in rand] if den > 0 else [None] * REPEATS
            need(raw >= -ALG_TOL and raw <= den + ALG_TOL*(1+den), 'GT projection bounds')
            row = {name: oldrow[name] for name in ('GT', 'label', 'actual_input_sub4') + COHORTS[1:]}
            row.update(batch=batch, image_index=index, window=arm, cell=int(old['cell']),
                d_g=den, denominator_status='resolved' if den > 0 else 'nonpositive_NA',
                original_projection=raw0, deletion_projection=raw1, raw_delta=raw, relative_delta=ratio,
                affected=bool(raw > EFFECT_TOL), random_raw=rand, random_relative=rel,
                raw_bad_minus_random_mean=raw-float(np.mean(rand)),
                relative_bad_minus_random_mean=None if ratio is None else ratio-float(np.mean(rel)))
            rows.append(row)
        result[arm] = dict(NA=False, cell=old['cell'], xyxy=old['xyxy'], mass_total=float(mass.sum()),
            condition_flat=flat.tolist(), Ubad_flat=flat[bad].tolist(),
            Ubad_class_counts=np.bincount(plabels[bad], minlength=10).tolist(),
            random_deleted_flat_32=deletion_lists, GT_rows=rows,
            original_residual_total=float(base.sum()), deletion_residual_total=float(after.sum()),
            raw_release_total=float(delta.sum()), independent_p_sha256=hashlib.sha256(p.tobytes(order='C')).hexdigest(),
            independent_p_shape=list(p.shape), independent_p_dtype=str(p.dtype),
            parity=dict(residual_maxabs=base_error, null_maxabs=null_error), cohorts=summarize(rows))
        COUNTS['windows'] += 1
        COUNTS['random_deletions'] += REPEATS
    return dict(image_index=index, GT=im['GT'], windows=result)


def aggregate(batches):
    result = {}
    for letter in ('A', 'B'):
        bb = [b for b in batches if b['letter'] == letter]
        images = [im for b in bb for im in b['images']]
        panels = {arm: summarize([r for im in images for r in im['windows'][arm]['GT_rows']]) for arm in WINDOWS}
        allrows = [r for im in images for arm in WINDOWS for r in im['windows'][arm]['GT_rows']]
        panels['two_windows_GT_deduplicated'] = summarize(merge_windows(allrows))
        panel = panels['two_windows_GT_deduplicated']['primary_Geo0_tiny']
        valid = panel['GT_exposures'] > 0 and panel['metrics']['relative_delta']['pooled_GT']['NA'] == 0
        no_effect = valid and panel['affected_inputs'] == 0 and panel['max_raw_delta_any_window'] <= EFFECT_TOL
        result[letter] = dict(inputs=len(images), panels=panels,
            no_primary_scope=not valid, zero_primary_action_scope=bool(no_effect),
            nonzero_primary_action_scope=bool(valid and panel['affected_inputs'] > 0),
            same_cell_two_windows=sum(not im['windows'][WINDOWS[0]]['NA'] and not im['windows'][WINDOWS[1]]['NA']
                                     and im['windows'][WINDOWS[0]]['cell'] == im['windows'][WINDOWS[1]]['cell'] for im in images))
    return result


def synthetic():
    m = np.zeros((10, 2)); m[0] = [1, 1]
    labels = np.array([0], dtype=np.int64)
    p, u, base, z = independent(m, np.zeros((2, 1)), labels)
    need(np.allclose(p, 1/3, rtol=0, atol=1e-15) and abs(z[0]-1/3) < 1e-15, 'single condition analytic')
    after, dr = deleted(m, p, labels, np.array([0]), u, base)
    need(np.allclose(after, m) and np.allclose(dr[0], 1/3), 'delete all analytic')
    zero = np.zeros((10, 1)); lab4 = np.zeros(4, dtype=np.int64)
    pp, uu, rr, zz = independent(zero, np.zeros((1, 4)), lab4)
    need(not pp.any() and not rr.any() and np.array_equal(zz, np.ones(4)), 'zero mass null')
    zero[0, 0] = .25
    pp, uu, rr, zz = independent(zero, np.zeros((1, 4)), lab4)
    need(rr[0, 0] == 0 and deleted(zero, pp, lab4, np.array([0]), uu, rr)[1][0, 0] == 0, 'overload deletion can have no release')
    need(deleted(zero, pp, lab4, np.arange(4), uu, rr)[0][0, 0] == .25, 'full overloaded deletion')
    p0, u0, r0, z0 = independent(m, np.zeros((2, 0)), np.array([], dtype=np.int64))
    need(np.array_equal(r0, m) and not deleted(m, p0, np.array([], dtype=np.int64), np.array([], dtype=np.int64), u0, r0)[1].any(), 'no supply')
    pts = np.array([[1., 1.], [3., 1.]])
    boxes = np.array([[0., 0., 2., 2.], [1., 1., 0., 0.]])
    cost = geometry_cost(pts, boxes)
    need(np.array_equal(cost[:, 0], [0., .5]) and np.isinf(cost[:, 1]).all()
         and np.array_equal(cost, geometry_cost(pts*3, boxes*3)), 'geometry and scale')
    pi, ui, ri, zi = independent(m, cost, np.array([0, 1]))
    need(not pi[:, 1].any() and zi[1] == 1 and not ui[1].any(), 'invalid/null/class isolation')
    need(bad_ids(np.array([0, 10, 21]), {'0': [], '10': [999], '21': []}).tolist() == [0, 2], 'global neighbor outside local prevents deletion')
    labels2 = np.array([0, 0, 0, 1, 1, 2]); bad = np.array([0, 3, 5])
    ids = random_ids(labels2, bad, 1, 0, 0, 0, 0)
    need(np.array_equal(ids, random_ids(labels2, bad, 1, 0, 0, 0, 0)), 'random determinism')
    need(len(random_ids(labels2, np.array([], dtype=np.int64), 1, 0, 0, 0, 0)) == 0, 'empty bad random')
    need(np.array_equal(random_ids(labels2, np.arange(6), 1, 0, 0, 0, 0), np.arange(6)), 'all bad random')
    weights = np.array([[.25, .75], [1., 0.]])
    mm = weights.sum(0)
    need(np.array_equal(weights.sum(1), [1., 1.]) and (weights @ mm > 0).all(), 'unit kernels positive denominator')
    row = dict(batch=0, image_index=0, GT=0, label=0, actual_input_sub4=True,
               primary_Geo0=True, geometry_covered=False, primary_Geo0_tiny=True, geometry_covered_tiny=False,
               window='a', cell=0, affected=False, raw_delta=0., relative_delta=0.,
               random_raw=[.2]*32, random_relative=[.2]*32,
               raw_bad_minus_random_mean=-.2, relative_bad_minus_random_mean=-.2)
    second = dict(row, window='b', affected=True, raw_delta=.1, relative_delta=.1,
                  raw_bad_minus_random_mean=-.1, relative_bad_minus_random_mean=-.1)
    merged = merge_windows([row, second])
    need(len(merged) == 1 and merged[0]['affected'] and merged[0]['raw_delta'] == .05, 'two window dedup and affected union')
    ss = summarize(merged)['primary_Geo0_tiny']
    need(ss['affected_unique_GT'] == ss['affected_inputs'] == 1
         and ss['metrics']['raw_bad_minus_random_mean']['pooled_GT']['mean'] < 0, 'negative paired accepted')
    need(stats([])['mean'] is None and stats([None])['NA'] == 1, 'NA not zero')
    return dict(single_condition=True, zero_mass=True, zero_supply=True, overcapacity=True,
                geometry_scale=True, invalid_box_null=True, class_isolation=True,
                deletion_clip_identity=True, global_legal_neighbors=True, random_matched_counts=True,
                random_reproducible=True, unit_kernel=True, two_window_dedup=True, NA_and_negative_paired=True)


def main():
    global np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic-only', action='store_true')
    parser.add_argument('--out', type=pathlib.Path, default=ROOT/'322_BASE24_fixed_geometry_false_context')
    args = parser.parse_args()
    started = time.monotonic()
    limits = resources()
    import numpy as _np
    np = _np
    limits['numpy'] = np.__version__
    need(sha(PROTOCOL) == PROTOCOL_SHA, '318 protocol changed')
    tests = synthetic()
    if args.synthetic_only:
        print(json.dumps(dict(synthetic=tests, resources=limits, counts=COUNTS), ensure_ascii=False))
        return
    out = args.out.resolve()
    need(out.parent == ROOT and out.name.startswith('322_') and not out.exists(), 'new322 output only')
    out.mkdir()
    stage, files, batches, artifacts = 'source_seal', {}, [], {}
    try:
        files, lineage = source_manifest()
        save(out/'00_sources_and_synthetic.json', dict(source_sha256=files, lineage=lineage, synthetic=tests,
            resources=limits, parameters=dict(grid=16, epsilon=1, kappa=0, random_seed=RANDOM_SEED,
                                            repeats=32, parity_tolerance=PARITY_TOL, effect_raw_tolerance=EFFECT_TOL)))
        for letter, seed in (('A', 1), ('B', 2)):
            for batch in range(8):
                need(memory_available() > 8., 'RAM below 8 GiB before batch')
                stage = f'{letter}_batch{batch:02d}'
                old, ctx, prior = read(D263/(stage+'.json')), read(D284/(stage+'.json')), read(D300/(stage+'.json'))
                for key in ('letter', 'seed', 'batch', 'shape', 'input_sha256', 'prediction_sha256', 'source_ids'):
                    need(old[key] == ctx[key] == prior[key], 'batch source identity: '+key)
                need(old['letter'] == letter and old['seed'] == seed and old['batch'] == batch
                     and old['shape'][:2] == [4, 3] and old['native_postprocess_exact'] is True, 'scope/order')
                need(old['postprocess']['top'] == 300 and old['postprocess']['score_threshold'] == .001, 'native fullP')
                need([im['image_index'] for im in old['images']] == [im['image_index'] for im in ctx['images']]
                     == [im['image_index'] for im in prior['images']] == list(range(4)), 'image order')
                with np.load(D300/(stage+'_arrays.npz'), allow_pickle=False) as ar:
                    images = [probe_image(o, c, p, ar, batch, seed) for o, c, p in zip(old['images'], ctx['images'], prior['images'])]
                report = {key: old[key] for key in ('letter', 'seed', 'batch', 'shape', 'input_sha256', 'prediction_sha256', 'source_ids')}
                report['images'] = images
                save(out/(stage+'.json'), report)
                artifacts[stage+'.json'] = sha(out/(stage+'.json'))
                batches.append(report)
                COUNTS['npz_batches'] += 1
                COUNTS['images'] += 4
                del old, ctx, prior
                gc.collect()
                print(stage, 'complete', flush=True)
        need(COUNTS['npz_batches'] == 16 and COUNTS['images'] == 64, 'exact 16 development batches')
        summaries = aggregate(batches)
        stop = all(summaries[a]['zero_primary_action_scope'] for a in ('A', 'B'))
        if any(summaries[a]['no_primary_scope'] for a in ('A', 'B')):
            interpretation = 'NA_primary_scope_not_a_zero_effect'
        elif stop:
            interpretation = 'stop_only_current_no_legal_condition_consumption_motivation'
        elif all(summaries[a]['nonzero_primary_action_scope'] for a in ('A', 'B')):
            interpretation = 'both_seeds_nonzero_descriptive_scope_only'
        else:
            interpretation = 'mixed_descriptive_scope_only'
        unchanged = {p: sha(p) == h for p, h in files.items()}
        need(all(unchanged.values()), 'source changed during run')
        need(all(v == 0 for k, v in COUNTS.items() if k not in ('npz_batches', 'images', 'windows', 'random_deletions')), 'zero model/PT/solver scope')
        need('torch' not in sys.modules and 'scipy' not in sys.modules, 'no torch/scipy imports')
        save(out/'01_results.json', dict(state='complete', summaries=summaries, counts=COUNTS, resources=limits,
            interpretation=interpretation, stop_current_motivation=stop, sources_unchanged=unchanged,
            artifact_sha256=artifacts, seconds=time.monotonic()-started, oracle_mass=True,
            fixed_initial_geometry_kappa0_only=True, Gcal_effect=False, AP=False,
            learnability_proven=False, training_gate=False, training_authorized=False))
        save(out/'02_output_sha256.json', {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    except BaseException:
        save(out/'FAILED.json', dict(state='implementation_failure_no_method_conclusion', stage=stage,
                                    counts=COUNTS, traceback=traceback.format_exc(), seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()
