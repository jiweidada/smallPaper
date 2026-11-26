"""282: frozen-window explanation-context premise counts. No model or training.
Root reviews/runs; --synthetic-only does not read real result/cache contents.
"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
sys.dont_write_bytecode = True
import argparse, gc, hashlib, importlib.util, json, pathlib, time, traceback
ROOT = pathlib.Path(__file__).resolve().parent
PROTOCOL = ROOT / '282_B_条件过度解释现实前提_事前协议.txt'
ANCHORS = {
 '262_BASE24_residual_window_probe.py': 'ba695c5627e1120ed4bf2f7bd9bef274c94b39ccf8ba1d183e185c3e277ee065',
 '243_BASE24_cached_displacement_floor_probe_fixed.py': '0840b94e8528a461902456b7d0985060216386b795abd954062a4848bc93db0e',
 '263_BASE24_residual_window_probe/00_manifest_and_synthetic.json': 'fc6cd224c14c4f9f9e35cb195b9f41240d723cb15a0039e82417a64151c0f7c8',
 '263_BASE24_residual_window_probe/01_results.json': '56b7bb86d3f9927bd501df48a2df9be7aab82a0157942b5da47c2513797530b3',
 '263_BASE24_residual_window_probe/02_output_sha256.json': '8e200dee1ffe99616aa0207e5b80222f2370a9904c529c37a7c5f42f9535d934',
 '244_BASE24_cached_displacement_floor_probe_fixed/00_inputs_and_synthetic.json': '626ce41b80efcb07ae7d400312b41c153ff8591f7d4ba8a716db765c8340c56d',
 '244_BASE24_cached_displacement_floor_probe_fixed/01_results.json': '9dfbf71fc5ea60820b529c4e1e0fb0c0412fd96af034a40f439a3e98618160b9',
 '244_BASE24_cached_displacement_floor_probe_fixed/02_output_sha256.json': '1fb64644422ac499eb3bc5b5afa22f1d5e7856409514c4fcfd2266f38d0f4233',
}
REUSE = ROOT / '262_BASE24_residual_window_probe.py'
if hashlib.sha256(REUSE.read_bytes()).hexdigest() != ANCHORS[REUSE.name]:
    raise RuntimeError('262 source changed')
_spec = importlib.util.spec_from_file_location('_probe283_reuse262', REUSE)
R = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(R)
P, H, np, torch = R.P, R.H, R.np, R.torch
need, sha, save = R.need, R.sha, R.save
WINDOWS = ('A_density', 'B_residual')
DEFS = ('primary_G_geo0', 'secondary_D0_P')
TYPES = ('wrong_class_geometry', 'same_class_localization', 'no_overlap', 'other_class_low_overlap')
KINDS = ('a_legal_contains_missing', 'b_shared_GT_same_window', 'b_singleton_duplicate_same_window',
         'c_no_legal_contains_missing') + tuple('c_' + s for s in TYPES)
COUNTS = dict(npz_batches=0, images=0, prediction_PT_loads=0, native_matcher_calls=0,
              model_builds=0, checkpoint_loads=0, model_forwards=0, criterion_forwards=0,
              autograd_calls=0, model_backwards=0, model_vjps=0, optimizer_steps=0, heldout_reads=0)


def read(path):
    with pathlib.Path(path).open(encoding='utf-8-sig') as f: return json.load(f)


def valid_boxes(x):
    return np.isfinite(x).all(-1) & (x[:, 2] > x[:, 0]) & (x[:, 3] > x[:, 1])


def contains(boxes, centers):
    return (valid_boxes(boxes)[:, None] & np.isfinite(centers).all(-1)[None, :]
            & (centers[None, :, 0] >= boxes[:, None, 0])
            & (centers[None, :, 1] >= boxes[:, None, 1])
            & (centers[None, :, 0] <= boxes[:, None, 2])
            & (centers[None, :, 1] <= boxes[:, None, 3]))


def classify_unmatched(max_any, max_same, threshold):
    if max_any >= threshold: return TYPES[0]
    if max_same > 0: return TYPES[1]
    if max_any == 0: return TYPES[2]
    return TYPES[3]


def event_sets(rows):
    missing, conditions, explained, incidence = set(), set(), set(), set()
    for row in rows:
        missing.update(row['missing_GT']); conditions.update(row['conditions'])
        explained.update(row['explained_GT'])
        incidence.update((p, g) for p in row['conditions'] for g in row['missing_GT'])
    return missing, conditions, explained, incidence


def event_summary(rows, tiny):
    missing, conditions, explained, incidence = event_sets(rows)
    return dict(witness_groups=len(rows), inputs_with_event=int(bool(rows)),
                unique_missing_GT=len(missing), unique_missing_actual_sub4=sum(bool(tiny[g]) for g in missing),
                unique_conditions=len(conditions), unique_queries=len({p // 10 for p in conditions}),
                unique_explainable_GT=len(explained), condition_missing_incidence=len(incidence))


def audit_window(cell, grid, p_cells, p_xy, p_flat, gt_xy, gt_centers, gt_cells,
                 tiny, legal, missing, max_any, max_same, threshold):
    events = {key: [] for key in KINDS}
    if cell is None:
        return dict(NA=True, cell=None, supplies=[], missing_GT=[], missing_actual_sub4=[],
                    events=events, counts={key: event_summary([], tiny) for key in KINDS})
    window = grid[cell]
    full, center, _ = R.visibility(window, gt_xy, gt_cells, cell)
    supplies = np.flatnonzero(p_cells == cell)
    miss = np.flatnonzero(full & missing)
    inside = contains(p_xy, gt_centers)
    neighbors = [np.flatnonzero(legal[p]).tolist() for p in range(len(p_flat))]
    for p in supplies:
        covered_missing = miss[inside[p, miss]].tolist()
        own_centers = [g for g in neighbors[p] if inside[p, g]]
        if neighbors[p] and own_centers and covered_missing:
            events[KINDS[0]].append(dict(conditions=[int(p_flat[p])], explained_GT=own_centers,
                missing_GT=covered_missing, all_legal_neighbors=neighbors[p]))
        if not neighbors[p] and covered_missing:
            kind = classify_unmatched(float(max_any[p]), float(max_same[p]), threshold)
            row = dict(conditions=[int(p_flat[p])], explained_GT=[], missing_GT=covered_missing,
                       error_type=kind, max_any_IoU=float(max_any[p]), max_same_IoU=float(max_same[p]))
            events['c_no_legal_contains_missing'].append(row)
            events['c_' + kind].append(row)
    if len(miss):
        for g in range(len(gt_xy)):
            shared = [int(p) for p in supplies if legal[p, g]]
            singleton = [p for p in shared if neighbors[p] == [g]]
            for kind, pp in (('b_shared_GT_same_window', shared), ('b_singleton_duplicate_same_window', singleton)):
                if len(pp) >= 2:
                    events[kind].append(dict(conditions=[int(p_flat[p]) for p in pp], explained_GT=[g],
                        missing_GT=miss.tolist(), explained_GT_complete=bool(full[g]),
                        explained_GT_center_in_window=bool(center[g]),
                        condition_all_legal_neighbors={str(int(p_flat[p])): neighbors[p] for p in pp}))
    return dict(NA=False, cell=int(cell), window_xyxy=window.tolist(), supplies=p_flat[supplies].tolist(),
                invalid_supplies=p_flat[supplies[~valid_boxes(p_xy)[supplies]]].tolist(),
                complete_GT=np.flatnonzero(full).tolist(), missing_GT=miss.tolist(),
                missing_actual_sub4=miss[tiny[miss]].tolist(), events=events,
                counts={key: event_summary(rows, tiny) for key, rows in events.items()})


def image_probe(im, archive, hw, box, tv):
    index = im['image_index']; prefix = f'i{index}_final'
    arrays = {k: archive[prefix + '_' + k] for k in ('cached_boxes', 'GT', 'GT_labels', 'actual_input_sub4')}
    boxes64, gt64 = arrays['cached_boxes'], arrays['GT']
    need(boxes64.shape == (300, 4) and gt64.ndim == 2 and gt64.shape[1] == 4, 'final geometry shapes')
    need(boxes64.dtype == gt64.dtype == np.float64, '244 FP64 copies')
    need(np.array_equal(boxes64.astype(np.float32).astype(np.float64), boxes64)
         and np.array_equal(gt64.astype(np.float32).astype(np.float64), gt64), 'lossless original FP32 recovery')
    labels = arrays['GT_labels']; need(labels.dtype == np.int64 and labels.shape == (len(gt64),), 'GT labels')
    need(((labels >= 0) & (labels < 10)).all(), 'GT class range')
    boxes = torch.from_numpy(boxes64.astype(np.float32)); gt = torch.from_numpy(gt64.astype(np.float32))
    gt_xy_t = R.pixels(gt, hw, tv); xy_t = R.pixels(boxes, hw, tv)
    gt_xy, xy = gt_xy_t.numpy().astype(np.float64), xy_t.numpy().astype(np.float64)
    need(valid_boxes(gt_xy).all(), 'invalid GT')
    wh32 = torch.tensor([hw[1], hw[0]], dtype=torch.float32)
    centers = (boxes[:, :2] * wh32).numpy().astype(np.float64)
    gt_centers = (gt[:, :2] * wh32).numpy().astype(np.float64)
    tiny = (gt64[:, 2:] * np.array([hw[1], hw[0]])).min(-1) < 4
    need(np.array_equal(tiny, arrays['actual_input_sub4']) and len(gt64) == im['GT']
         and int(tiny.sum()) == im['actual_sub4'], 'GT/tiny source identity')
    post = im['post']; flat = np.asarray(post['P_flat'], dtype=np.int64)
    scores = np.asarray(post['P_scores'], dtype=np.float64); q = flat // 10; p_labels = flat % 10
    natural = np.asarray(post['natural_top300'], dtype=np.int64)
    kept = np.asarray(post['threshold_kept'], dtype=bool)
    need(natural.shape == kept.shape == (300,) and len(set(natural.tolist())) == 300, 'natural flat identity')
    need(((natural >= 0) & (natural < 3000)).all() and np.array_equal(flat, natural[kept]), 'full thresholded P')
    need(scores.shape == flat.shape and (scores > .001).all() and (scores <= 1).all()
         and np.isfinite(scores).all() and (np.diff(scores) <= 0).all(), 'full P score contract')
    p_xy = np.asarray(post['P_boxes_xyxy'], dtype=np.float64).reshape(-1, 4)
    need(np.array_equal(p_xy, xy[q]) and len(flat) == post['P_count']
         and len(set(q.tolist())) == post['P_unique_queries'], 'P/query exact box identity')
    iou_all = R.pair_iou(xy_t, gt_xy_t, box)
    iou_p = iou_all[q]
    same = p_labels[:, None] == labels[None, :]
    max_any = iou_p.max(1) if len(labels) else np.zeros(len(flat), dtype=np.float32)
    max_same = np.where(same, iou_p, 0).max(1) if len(labels) else np.zeros(len(flat), dtype=np.float32)
    grid = R.make_grid(hw); gt_cells = R.cell_ids(gt_centers, hw); p_cells = R.cell_ids(centers[q], hw)
    need(np.array_equal(grid, np.asarray(im['grid_xyxy'])), 'same original 64 grid')
    chosen = {}
    for arm in WINDOWS:
        old = im['windows'][arm]; cell = old['cell']
        need(bool(old['NA']) == (cell is None), 'window NA consistency')
        if cell is not None:
            need(0 <= cell < len(grid) and np.array_equal(grid[cell], old['xyxy_pixels']), 'frozen original window')
            full, center, fraction = R.visibility(grid[cell], gt_xy, gt_cells, cell)
            need(int(full.sum()) == old['visibility']['complete']['GT'], 'same complete GT count')
            need(sorted(r['GT'] for r in old['GT_rows'] if r['complete']) == np.flatnonzero(full).tolist(), 'same complete GT IDs')
            for row in old['GT_rows']:
                g = row['GT']
                need(row['label'] == int(labels[g]) and row['actual_sub4'] == bool(tiny[g])
                     and row['complete'] == bool(full[g]) and row['center'] == bool(center[g]), 'old window GT identity')
        chosen[arm] = cell
    result = dict(image_index=index, GT=len(labels), actual_sub4=int(tiny.sum()), shape_hw=list(hw),
                  full_P=len(flat), full_P_unique_queries=len(set(q.tolist())), invalid_final_queries=int((~valid_boxes(xy)).sum()),
                  P_invalid_center=flat[p_cells < 0].tolist(), frozen_cells=chosen, thresholds={})
    for threshold in (.5, .75):
        legal = same & (iou_p >= threshold)
        definitions = {DEFS[0]: ~(iou_all >= threshold).any(0), DEFS[1]: ~legal.any(0)}
        need(np.all(~definitions[DEFS[0]] | definitions[DEFS[1]]), 'primary subset of secondary')
        panels = {}
        for name, missing in definitions.items():
            ww = {arm: audit_window(cell, grid, p_cells, p_xy, flat, gt_xy, gt_centers, gt_cells,
                    tiny, legal, missing, max_any, max_same, threshold) for arm, cell in chosen.items()}
            union = {kind: event_summary([row for arm in WINDOWS for row in ww[arm]['events'][kind]], tiny) for kind in KINDS}
            # Group count may duplicate across windows; unique sets/incidence above are de-duplicated.
            panels[name] = dict(full_image_zero_degree_GT=np.flatnonzero(missing).tolist(),
                full_image_zero_degree_actual_sub4=np.flatnonzero(missing & tiny).tolist(), windows=ww,
                two_window_union_counts=union)
        result['thresholds'][str(threshold)] = dict(definitions=panels,
            all_P_legal_neighbors={str(int(flat[p])): np.flatnonzero(legal[p]).tolist() for p in range(len(flat))},
            full_image_no_legal_P=flat[~legal.any(1)].tolist())
    return result


def synthetic(box):
    # No real cache/manifest reads. Geometry inputs here are artificial.
    grid = R.make_grid((128, 128))
    need(R.cell_ids(np.array([[64, 0], [128, 128], [-1, 1.]]), (128, 128)).tolist() == [1, 3, -1], 'cell boundaries')
    need(R.make_grid((544, 544))[-1].tolist() == [512, 512, 544, 544], 'remainder cell')
    gt = np.array([[1, 1, 3, 3], [4, 1, 6, 3], [80, 80, 82, 82]], dtype=float)
    centers = (gt[:, :2] + gt[:, 2:]) / 2; tiny = np.ones(3, dtype=bool)
    # Conditions 0 and 1 explain g0; condition 2 has a legal neighbor outside W.
    p = np.array([[0, 0, 7, 4], [0, 0, 7, 4], [0, 0, 7, 4], [3, 0, 7, 4]], dtype=float)
    flat = np.array([0, 10, 20, 31]); legal = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
    args = (0, grid, np.zeros(4, dtype=int), p, flat, gt, centers, R.cell_ids(centers, (128, 128)), tiny, legal,
            np.array([0, 1, 0], dtype=bool), np.array([.7, .7, .6, .6]), np.array([.7, .7, .6, 0.]), .5)
    r = audit_window(*args)
    need(r['counts'][KINDS[0]]['unique_conditions'] == 2, 'legal area covers another missing center')
    need(r['counts']['b_singleton_duplicate_same_window']['unique_conditions'] == 2, 'singleton duplicates')
    need(r['events']['c_no_legal_contains_missing'][0]['conditions'] == [31], 'global outside-window neighbor not FP')
    need(r['counts']['c_wrong_class_geometry']['unique_missing_GT'] == 1, 'wrong-class relation')
    changed = legal.copy(); changed[1, 2] = True
    aa = list(args); aa[9] = changed
    rr = audit_window(*aa)
    need(rr['counts']['b_shared_GT_same_window']['witness_groups'] == 2
         and rr['counts']['b_singleton_duplicate_same_window']['witness_groups'] == 0, 'ambiguous differs singleton')
    need([classify_unmatched(a, s, .5) for a, s in ((.5, 0), (.4, .2), (0, 0), (.2, 0))] == list(TYPES), 'exhaustive error classes')
    need(contains(np.array([[0., 0, 1, 1], [1, 1, 0, 0]]), np.array([[1., 1]])).tolist() == [[True], [False]], 'containment boundary/invalid')
    row = dict(conditions=[0, 1], explained_GT=[0], missing_GT=[1])
    es = event_summary([row, row], tiny)
    need(es['unique_conditions'] == 2 and es['unique_queries'] == 1 and es['condition_missing_incidence'] == 2, 'same query classes and union dedup')
    many = np.arange(101, dtype=np.int64) * 10
    empty_legal = np.zeros((101, 3), dtype=bool)
    many_r = audit_window(0, grid, np.zeros(101, dtype=int), np.tile([3., 0, 7, 4], (101, 1)), many,
        gt, centers, R.cell_ids(centers, (128, 128)), tiny, empty_legal, np.array([0, 1, 0], bool),
        np.zeros(101), np.zeros(101), .5)
    need(len(many_r['supplies']) == 101 and many_r['counts']['c_no_overlap']['unique_conditions'] == 101, 'no perclass100 cap')
    zero = audit_window(None, *args[1:])
    need(zero['NA'] and all(s['unique_conditions'] == 0 for s in zero['counts'].values()), 'NA stays NA')
    xy = torch.tensor([[0., 0, 2, 2], [3, 0, 5, 2]])
    ii = R.pair_iou(xy, xy, box)
    geo0 = ~(ii >= .5).any(0)
    # Full P contains q0 under the wrong class, not q1; both GTs are secondary-only omissions.
    d0 = ~((ii[[0]] >= .5) & np.array([[False, False]])).any(0)
    need(not geo0.any() and d0.all(), 'all300 geometry vs full-P class zero degree')
    need((R.pair_iou(torch.tensor([[0., 0, 2, 2]]), torch.tensor([[0., 0, 1, 2]]), box) >= .5).item(), 'inclusive IoU threshold')
    return dict(cell_boundary=True, remainder=True, area_over_explanation=True, global_GT_neighbors=True,
                singleton_vs_ambiguous=True, exhaustive_error_types=True, multiclass_query=True,
                no_P100=True, union_dedup=True, primary_secondary_distinct=True, threshold_inclusive=True, NA=True)


def source_manifest():
    sources = {}
    def add(path, digest):
        path = pathlib.Path(path).resolve(); key = str(path)
        need('heldout' not in path.as_posix().lower(), 'heldout source forbidden')
        need(key not in sources or sources[key] == digest, 'conflicting source SHA')
        sources[key] = digest
    for relative, digest in ANCHORS.items():
        path = ROOT / relative; need(sha(path) == digest, 'anchor changed: ' + relative); add(path, digest)
    for dirname, manifest_name in (('263_BASE24_residual_window_probe', '00_manifest_and_synthetic.json'),
                                    ('244_BASE24_cached_displacement_floor_probe_fixed', '00_inputs_and_synthetic.json')):
        folder = ROOT / dirname; manifest = read(folder / manifest_name); result = read(folder / '01_results.json')
        need(result['state'] == 'complete' and all(result['sources_unchanged'].values()), 'producer did not complete unchanged')
        outputs = read(folder / '02_output_sha256.json')
        for key, path in manifest['paths'].items(): add(path, manifest['sha256'][key])
        for letter in ('A', 'B'):
            for batch in range(8):
                stem = f'{letter}_batch{batch:02d}'
                names = [stem + '.json'] + ([stem + '_arrays.npz'] if dirname.startswith('244') else [])
                for name in names:
                    digest = result['artifact_sha256'][name]
                    need(outputs[name] == digest, 'artifact index agrees')
                    add(folder / name, digest)
    add(PROTOCOL, sha(PROTOCOL)); add(pathlib.Path(__file__), sha(pathlib.Path(__file__)))
    for path, digest in sources.items(): need(sha(pathlib.Path(path)) == digest, 'source seal: ' + path)
    return sources


def aggregate(batches):
    summaries = {}
    for letter in ('A', 'B'):
        ims = [im for b in batches if b['letter'] == letter for im in b['images']]
        if not ims: continue
        panels = {}
        for threshold in ('0.5', '0.75'):
            panels[threshold] = {}
            for definition in DEFS:
                per_arm = {}
                for arm in WINDOWS + ('two_window_union',):
                    by_kind = {}
                    for kind in KINDS:
                        counts = [im['thresholds'][threshold]['definitions'][definition]
                            ['two_window_union_counts'][kind] if arm == 'two_window_union' else
                            im['thresholds'][threshold]['definitions'][definition]['windows'][arm]['counts'][kind] for im in ims]
                        by_kind[kind] = {key: sum(c[key] for c in counts) for key in counts[0]}
                    per_arm[arm] = by_kind
                denominators = {}
                for arm in WINDOWS:
                    ws = [im['thresholds'][threshold]['definitions'][definition]['windows'][arm] for im in ims]
                    denominators[arm] = dict(selected_inputs=sum(not w['NA'] for w in ws),
                        supplied_conditions=sum(len(w['supplies']) for w in ws),
                        supplied_queries=sum(len({p // 10 for p in w['supplies']}) for w in ws),
                        complete_GT=sum(len(w.get('complete_GT', [])) for w in ws),
                        missing_GT=sum(len(w['missing_GT']) for w in ws),
                        missing_actual_sub4=sum(len(w['missing_actual_sub4']) for w in ws))
                denominator_union = dict(missing_GT=0, missing_actual_sub4=0, supplied_conditions=0, supplied_queries=0)
                for im in ims:
                    ws = im['thresholds'][threshold]['definitions'][definition]['windows']
                    for key, field in (('missing_GT', 'missing_GT'), ('missing_actual_sub4', 'missing_actual_sub4'), ('supplied_conditions', 'supplies')):
                        denominator_union[key] += len({v for arm in WINDOWS for v in ws[arm][field]})
                    denominator_union['supplied_queries'] += len({p // 10 for arm in WINDOWS for p in ws[arm]['supplies']})
                denominators['two_window_union'] = denominator_union
                panels[threshold][definition] = dict(event_counts=per_arm, denominators=denominators)
        summaries[letter] = dict(inputs=len(ims), GT=sum(im['GT'] for im in ims), actual_sub4=sum(im['actual_sub4'] for im in ims),
                                thresholds=panels)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic-only', action='store_true')
    parser.add_argument('--out', type=pathlib.Path, default=ROOT / '284_BASE24_explanation_context_probe')
    args = parser.parse_args(); started = time.monotonic(); resources = H.resource_limits()
    need(resources['priority'] == 'below_normal' and torch.version.cuda is None, 'CPU resource contract')
    for key, path in (('box_ops', P.A.NATIVE / 'box_ops.py'), ('dfine_utils', P.A.NATIVE / 'dfine_utils.py'), ('postprocessor', R.POST)):
        need(sha(path) == P.A.EXPECTED[key], 'native math source: ' + key)
    native = H.load_native_math(); box = sys.modules[native.__package__ + '.box_ops']
    _, _, tv = R.load_post_api()
    tests = synthetic(box)
    if args.synthetic_only:
        print(json.dumps(dict(tests=tests,resources=resources,counts=COUNTS), ensure_ascii=False)); return
    out = args.out.resolve(); need(out.parent == ROOT and out.name.startswith('284_') and not out.exists(), 'new284 only')
    out.mkdir(); stage = 'sources'; sources = {}; batches = []; artifacts = {}
    try:
        sources = source_manifest()
        save(out / '00_sources_and_synthetic.json', dict(source_sha256=sources, resources=resources, synthetic=tests,
            inherited_native_postprocess_exact=True, raw_logits_deserialized=False))
        for letter, seed in (('A', 1), ('B', 2)):
            for batch in range(8):
                P.M.guard(); stage = f'{letter}_batch{batch:02d}'
                old = read(ROOT / '263_BASE24_residual_window_probe' / (stage + '.json'))
                origin = read(ROOT / '244_BASE24_cached_displacement_floor_probe_fixed' / (stage + '.json'))
                for key in ('letter', 'seed', 'batch', 'shape', 'input_sha256', 'prediction_sha256', 'source_ids'):
                    need(old[key] == origin[key], 'common batch identity: ' + key)
                need(old['letter'] == letter and old['seed'] == seed and old['batch'] == batch
                     and old['shape'][:2] == [4, 3] and old['native_postprocess_exact'] is True, 'frozen batch contract')
                need(old['postprocess']['top'] == 300 and old['postprocess']['score_threshold'] == .001, 'full native selection')
                need([im['image_index'] for im in old['images']] == list(range(4)), 'original image order')
                hw = tuple(old['shape'][2:])
                with np.load(ROOT / '244_BASE24_cached_displacement_floor_probe_fixed' / (stage + '_arrays.npz'), allow_pickle=False) as archive:
                    images = [image_probe(im, archive, hw, box, tv) for im in old['images']]
                COUNTS['npz_batches'] += 1; COUNTS['images'] += len(images)
                report = dict(letter=letter, seed=seed, batch=batch, shape=old['shape'], input_sha256=old['input_sha256'],
                              prediction_sha256=old['prediction_sha256'], source_ids=old['source_ids'], images=images,
                              full_native_P_inherited_and_query_boxes_exact=True, no_GT_condition_filter=True)
                report['summary'] = aggregate([report])[letter]
                save(out / (stage + '.json'), report); artifacts[stage + '.json'] = sha(out / (stage + '.json')); batches.append(report)
                del old, origin; gc.collect(); print(stage, 'complete', flush=True)
        need(COUNTS['npz_batches'] == 16 and COUNTS['images'] == 64, 'exact development budget')
        unchanged = {path: sha(pathlib.Path(path)) == digest for path, digest in sources.items()}
        need(all(unchanged.values()), 'source changed')
        need(all(v == 0 for k, v in COUNTS.items() if k not in ('npz_batches', 'images')), 'zero model/gradient/PT/heldout budget')
        save(out / '01_results.json', dict(state='complete', summaries=aggregate(batches), counts=COUNTS, resources=resources,
            sources_unchanged=unchanged, artifact_sha256=artifacts, seconds=time.monotonic()-started,
            premise_counts_only=True, independent_normalize_null_clip_not_tested=True,
            joint_transport_necessary_proven=False, learnability_proven=False, AP=False, training_authorized=False))
        save(out / '02_output_sha256.json', {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    except BaseException:
        save(out / 'FAILED.json', dict(state='error_no_method_conclusion', stage=stage, counts=COUNTS,
                                       traceback=traceback.format_exc(), seconds=time.monotonic()-started)); raise


if __name__ == '__main__': main()
