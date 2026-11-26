"""Read-only DN geometry description from cache188; root executes after review."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
sys.dont_write_bytecode = True
import argparse, collections, csv, hashlib, importlib.util, json, pathlib, time
ROOT = pathlib.Path(__file__).resolve().parent
COMMON = ROOT.parent / 'code/dn_pre_frozen_diagnostic_20261020_v1'
sys.path.insert(0, str(COMMON / 'runtime'))
import torch, psutil
CACHE = ROOT / '188_C0_true_GO_support_cache'
CPATH = CACHE / '01_cache.pt'
CSHA = '8a100f920dff9c961cc13ac5814e338dd37fc0bf5f390cc453a607f546b0ced9'
MSHA = 'b655c054be540e0332256db51027433e00b876182b2e36bea6b55bff3d1bb6ba'
BOX = COMMON / 'native/engine/deim/box_ops.py'
DENOISE = BOX.with_name('denoising.py')
GROUPS = ('all', 'original_sub4', 'current_target_sub4')
ZERO = dict(model_forwards=0, checkpoint_loads=0, backward=0, VJP=0, cache_autograd=0, updates=0)
def need(ok, message):
    if not bool(ok): raise RuntimeError(message)
def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()
need(sha(BOX) == '398fc107fc003b0fabaf36154594d50a8e0aa54195056b343031960c0aff0975', 'native box source changed')
need(sha(DENOISE) == '7b69610d9fee63fb7939df9301b76cbca35d4a3f4b6f4d270b0f13930d96a489', 'DN source layout changed')
spec = importlib.util.spec_from_file_location('_native_box197', BOX)
box = importlib.util.module_from_spec(spec); spec.loader.exec_module(box)
def save(path, value):
    with pathlib.Path(path).open('x', encoding='utf-8') as f: json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
def resources():
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.set_grad_enabled(False)
    p = psutil.Process(); p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    need({13, 15}.issubset(p.cpu_affinity()), 'affinity unavailable'); p.cpu_affinity([13, 15])
    need(sorted(p.cpu_affinity()) == [13, 15] and torch.version.cuda is None and torch.get_num_threads() == 2 and torch.get_num_interop_threads() == 1, 'CPU/thread contract')
    need(psutil.virtual_memory().available / 2**30 > 8, 'RAM guard')
    return dict(threads=2, interop=1, affinity=p.cpu_affinity(), priority='BelowNormal', available_RAM_GiB=psutil.virtual_memory().available / 2**30)
def ious(pred, target):
    need(pred.dtype == target.dtype == torch.float32 and pred.shape == target.shape, 'FP32 paired box interface')
    values = torch.cat([box.box_iou(box.box_cxcywh_to_xyxy(pred[k:k+128]), box.box_cxcywh_to_xyxy(target[k:k+128]))[0].diag() for k in range(0, len(pred), 128)]) if len(pred) else torch.empty(0)
    need(bool(torch.isfinite(values).all()) and not values.requires_grad, 'invalid IoU/no-grad contract')
    return values
def stats(values):
    x = torch.tensor(values, dtype=torch.float64); count = len(values)
    return dict(n=count, ge05=int((x >= .5).sum()), ge075=int((x >= .75).sum()),
                rate05=float((x >= .5).double().mean()) if count else None, rate075=float((x >= .75).double().mean()) if count else None,
                mean=float(x.mean()) if count else None, quantiles=dict(zip(('min', 'p10', 'p25', 'p50', 'p75', 'p90', 'max'), torch.quantile(x, torch.tensor([0, .1, .25, .5, .75, .9, 1], dtype=torch.float64)).tolist())) if count else None)
def best_rows(rows, population):
    bins = collections.defaultdict(list)
    for row in rows: bins[(row['image_index'], row['GT'])].append(row['IoU'])
    return [{**gt, 'positive_slots': len(bins[(gt['image_index'], gt['GT'])]), 'best_IoU': max(bins[(gt['image_index'], gt['GT'])]) if bins[(gt['image_index'], gt['GT'])] else None} for gt in population]
def describe(rows, population):
    best = best_rows(rows, population); values = [r['best_IoU'] for r in best if r['best_IoU'] is not None]
    s = stats(values); n = len(population)
    return dict(raw_GT=n, positive_slots=stats([r['IoU'] for r in rows]), unique_represented_GT=len(values), GT_without_positive=n-len(values),
                GT_best_oracle_represented_only=s, GT_best_ge05_over_raw_GT=s['ge05']/n if n else None, GT_best_ge075_over_raw_GT=s['ge075']/n if n else None,
                GT_positive_multiplicity_histogram=dict(collections.Counter(r['positive_slots'] for r in best)))
def group_ids(q, g, number, groups, split, positive):
    expected_q = (torch.arange(groups)[:, None] * (2 * number) + torch.arange(number)).flatten()
    valid = split == [2 * number * groups, 300] and torch.equal(q, positive) and torch.equal(q, expected_q) and torch.equal(g, torch.arange(number).repeat(groups))
    return (q // (2 * number)).tolist() if valid else None
def synthetic():
    target = torch.tensor([[.5, .5, 1, 1]] * 4, dtype=torch.float32)
    pred = torch.tensor([[.5, .5, 1, 1], [.25, .5, .5, 1], [.375, .5, .75, 1], [2, .5, 1, 1]], dtype=torch.float32)
    z = ious(pred, target); need(torch.equal(z, torch.tensor([1., .5, .75, 0.])), 'native exact IoU boundary')
    need(stats(z.tolist())['ge05'] == 3 and stats(z.tolist())['ge075'] == 2, 'inclusive thresholds')
    rows = [dict(image_index=0, GT=g, IoU=v) for g, v in ((0, .2), (0, .8), (1, .5))]
    population = [dict(image_index=0, GT=g) for g in (0, 1, 2)]
    d = describe(rows, population); need(d['positive_slots']['n'] == 3 and d['unique_represented_GT'] == 2 and d['GT_without_positive'] == 1, 'repeated-GT aggregation')
    need(abs(d['GT_best_oracle_represented_only']['mean']-.65) < 1e-12 and d['GT_best_ge05_over_raw_GT'] == 2/3, 'oracle denominator')
    q = torch.tensor([0, 1, 4, 5]); g = torch.tensor([0, 1, 0, 1])
    need(group_ids(q, g, 2, 2, [8, 300], q) == [0, 0, 1, 1], 'native DN source grouping')
    need(group_ids(q, g.flip(0), 2, 2, [8, 300], q) is None and stats([])['mean'] is None, 'no guessed grouping/empty statistic')
    return dict(native_FP32_boundaries=True, repeated_GT_and_missing_GT=True, exact_group_index_and_no_guess=True)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--synthetic-only', action='store_true'); ap.add_argument('--out', type=pathlib.Path, default=ROOT/'198_C0_cached_DN_descriptive'); args = ap.parse_args()
    start = time.monotonic(); resource = resources()
    with torch.no_grad(): tests = synthetic()
    if args.synthetic_only: print(json.dumps(dict(synthetic=tests, resources=resource, counts=ZERO))); return
    out = args.out.resolve(); need(out.parent == ROOT and out.name.startswith('198_') and not out.exists(), 'new 198 output required')
    need(sha(CACHE/'05_output_sha256.json') == MSHA, '188 manifest changed')
    manifest = json.loads((CACHE/'05_output_sha256.json').read_text(encoding='utf-8'))
    need(all(sha(CACHE/name) == digest for name, digest in manifest.items()) and manifest['01_cache.pt'] == CSHA, '188 output/cache SHA mismatch')
    sources = {str(CPATH): CSHA, str(CACHE/'05_output_sha256.json'): MSHA, str(BOX): sha(BOX), str(DENOISE): sha(DENOISE), str(pathlib.Path(__file__)): sha(__file__), str(ROOT/'196_DN既有缓存描述_事前口径.txt'): sha(ROOT/'196_DN既有缓存描述_事前口径.txt')}
    out.mkdir(); save(out/'00_inputs_and_synthetic.json', dict(sources=sources, synthetic=tests, resources=resource))
    data = torch.load(CPATH, map_location='cpu', weights_only=False)
    need(data['schema'] == 'true-GO-train-ref-v1' and data['weight_key'] == 'model_online' and data['GT'] == 902 and data['original_sub4'] == 453 and len(data['records']) == 4, '188 identity')
    need([r['image_index'] for r in data['records']] == [0, 1, 2, 3] and [len(r['target']['boxes']) for r in data['records']] == [146, 280, 418, 58], 'development image identity')
    pairs, population, metadata = [], [], []
    with torch.no_grad():
        for record in data['records']:
            need(psutil.virtual_memory().available / 2**30 > 8, 'RAM guard')
            index = record['image_index']; target = record['target']; boxes = target['boxes']; number = len(boxes); native = record['native_outputs']; dn = native['dn_meta']
            original = record['original_sub4'].bool(); short = boxes[:, 2:].min(-1).values * 640; current = short < 4
            pop = [dict(image_index=index, GT=g, GT_class=int(target['labels'][g]), original_sub4=bool(original[g]), current_target_sub4=bool(current[g]), current_shortside_px=float(short[g])) for g in range(number)]; population.extend(pop)
            stages = [('ordinary_final_own', 'final', 'mal', native), ('ordinary_final_GO', 'final', 'boxes', native), ('DN_pre', 'dn_pre_outputs', 'boxes', native['dn_pre_outputs'])] + [(f'DN_FDR_{j}', f'dn_outputs.{j}', 'boxes', value) for j, value in enumerate(native['dn_outputs'])]
            meta = dict(image_index=index, event=record['event'], input_sha256=record['input_sha256'], raw_GT=number, original_sub4=int(original.sum()), current_target_sub4=int(current.sum()), sub4_intersection=int((original & current).sum()), original_only=int((original & ~current).sum()), current_only=int((current & ~original).sum()), DN_positive_groups=int(dn['dn_num_group']), stages=[])
            for name, stage, loss, output in stages:
                calls = [c for c in record['loss_calls'] if c['stage'] == stage and c['loss'] == loss]; need(len(calls) == 1, 'unique native loss call required')
                call = calls[0]; need(len(call['indices']) == 1, 'single-image indices'); q, g = call['indices'][0]
                need(q.dtype == g.dtype == torch.int64 and q.numel() == g.numel() and q.unique().numel() == q.numel(), 'native positive index contract')
                need(bool((q >= 0).all() and (q < output['pred_boxes'].shape[1]).all() and (g >= 0).all() and (g < number).all()), 'positive index range')
                is_dn = name.startswith('DN_'); groups = group_ids(q, g, number, int(dn['dn_num_group']), list(dn['dn_num_split']), dn['dn_positive_idx'][0]) if is_dn else None
                if not is_dn:
                    expected = record['matching']['indices_own_final' if loss == 'mal' else 'indices_go'][0]
                    need(torch.equal(q, expected[0]) and torch.equal(g, expected[1]), 'ordinary index parity')
                vals = ious(output['pred_boxes'][0, q], boxes[g]); stage_rows = []
                for p in range(len(q)):
                    row = {**pop[int(g[p])], 'stage': name, 'query': int(q[p]), 'DN_group': groups[p] if groups is not None else None, 'IoU': float(vals[p]), 'ge05': bool(vals[p] >= .5), 'ge075': bool(vals[p] >= .75)}; stage_rows.append(row)
                pairs.extend(stage_rows)
                cohorts = {cohort: describe([r for r in stage_rows if cohort == 'all' or r[cohort]], [r for r in pop if cohort == 'all' or r[cohort]]) for cohort in GROUPS}
                first = {cohort: describe([r for r in stage_rows if r['DN_group'] == 0 and (cohort == 'all' or r[cohort])], [r for r in pop if cohort == 'all' or r[cohort]]) for cohort in GROUPS} if groups is not None else None
                meta['stages'].append(dict(stage=name, native_call_stage=stage, native_loss_denominator=call['denominator'], positive_slots=len(q), unique_GT=int(g.unique().numel()), positive_groups=int(dn['dn_num_group']) if is_dn else None, exact_group_source_verified=groups is not None if is_dn else None, cohorts=cohorts, first_DN_group=first))
            metadata.append(meta)
    need(len(population) == 902 and sum(r['original_sub4'] for r in population) == 453, 'cohort totals changed')
    stage_names = list(dict.fromkeys(r['stage'] for r in pairs)); pooled, gt_table = {}, []
    for name in stage_names:
        rows = [r for r in pairs if r['stage'] == name]; pooled[name] = {cohort: describe([r for r in rows if cohort == 'all' or r[cohort]], [r for r in population if cohort == 'all' or r[cohort]]) for cohort in GROUPS}
        gt_table.extend(dict(stage=name, **r) for r in best_rows(rows, population))
    for filename, rows in [('02_pairs.csv', pairs), ('03_GT_best.csv', gt_table)]:
        with (out/filename).open('x', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    save(out/'04_summary.json', dict(per_image=metadata, pooled=pooled, raw_GT=902, original_sub4=453, current_target_sub4=sum(r['current_target_sub4'] for r in population), metric='assigned-pair geometry only; per-GT best is oracle', method_gate=None, causal_claim=False, AP=False))
    unchanged = {path: sha(path) == digest for path, digest in sources.items()}; need(all(unchanged.values()), 'source changed')
    save(out/'05_execution.json', dict(state='complete', resources=resource, counts=ZERO, sources_unchanged=unchanged, heldout_reads=False, seconds=time.monotonic()-start, new_training=False))
    save(out/'06_output_sha256.json', {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(dict(state='complete', pair_rows=len(pairs), GT_best_rows=len(gt_table), counts=ZERO)))
if __name__ == '__main__': main()
