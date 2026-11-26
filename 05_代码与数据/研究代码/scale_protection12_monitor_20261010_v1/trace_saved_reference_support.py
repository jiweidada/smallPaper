"""Local numerical stage diagnostic using existing augmented BASE12 arrays only."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path.cwd()
DATA = ROOT / 'outputs/native_augmented_supervision_20261010_v1'
PAIRS = ROOT / 'outputs/saved_fgl_coefficient_analysis_20261010_v2'
OUT = PAIRS / 'reference_stage_trace_v1'
CURRENT = ROOT / 'outputs/scale_protection12_20261010_v1/terminal_collection/seed1_target_protect_v1/extracted'
read = lambda p: json.loads(Path(p).read_text(encoding='utf8'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, v):
    with Path(p).open('x', encoding='utf8') as stream:
        json.dump(v, stream, ensure_ascii=False, indent=2, allow_nan=False)


def required_distances(refs, gt):
    refs, gt = refs.astype(np.float64), gt.astype(np.float64)
    assert refs.shape == (300, 4) and (refs[:, 2:] > 0).all()
    assert gt.ndim == 2 and gt.shape[1] == 4 and (gt[:, 2:] > 0).all()
    xy = np.concatenate((gt[:, :2] - .5 * gt[:, 2:], gt[:, :2] + .5 * gt[:, 2:]), 1)
    cx, cy, w, h = (refs[None, :, i] for i in range(4))
    ds = np.stack((4 * (cx - xy[:, 0, None]) / w - 2,
                   4 * (cy - xy[:, 1, None]) / h - 2,
                   4 * (xy[:, 2, None] - cx) / w - 2,
                   4 * (xy[:, 3, None] - cy) / h - 2), -1)
    assert np.isfinite(ds).all()
    return np.abs(ds).max(-1)


def certified_max_matching(edges):
    """Exact augmenting paths plus equal-size vertex cover certificate."""
    n, q = edges.shape
    neighbors = [np.flatnonzero(row).tolist() for row in edges]
    owner = np.full(q, -1, dtype=np.int32)

    def visit(g, seen):
        for query in neighbors[g]:
            if seen[query]:
                continue
            seen[query] = True
            if owner[query] < 0 or visit(int(owner[query]), seen):
                owner[query] = g
                return True
        return False

    for g in range(n):
        visit(g, np.zeros(q, dtype=bool))
    assigned = np.full(n, -1, dtype=np.int32)
    for query in np.flatnonzero(owner >= 0):
        assigned[owner[query]] = query
    matched = assigned >= 0
    assert len(np.unique(assigned[matched])) == int(matched.sum())
    assert edges[np.flatnonzero(matched), assigned[matched]].all()
    # Alternating reachability from unmatched GT, GT--query nonmatching edges.
    reach_g = ~matched
    reach_q = np.zeros(q, dtype=bool)
    pending = np.flatnonzero(reach_g).tolist()
    while pending:
        g = pending.pop()
        for query in neighbors[g]:
            if assigned[g] == query or reach_q[query]:
                continue
            reach_q[query] = True
            if owner[query] >= 0 and not reach_g[owner[query]]:
                reach_g[owner[query]] = True
                pending.append(int(owner[query]))
    cover_g, cover_q = ~reach_g, reach_q
    assert not (edges & ~cover_g[:, None] & ~cover_q[None, :]).any()
    assert int(cover_g.sum() + cover_q.sum()) == int(matched.sum())
    return assigned, cover_g, cover_q


parent = PAIRS / 'range_and_assignment_pool'
manifest = read(parent / 'manifest.json')
assert all(sha(parent / n) == row['sha256'] for n, row in manifest['files'].items())
with np.load(parent / 'BASE12_FGL_pairs_with_range_and_assignment.npz', allow_pickle=False) as z:
    final_code = read(PAIRS / 'report.json')['stage_codes']['model']
    mask = (z['stage_code'] == final_code) & z['image_GT_le300'] & (z['actual_input_shortside'] < 4)
    pairs = {k: z[k][mask].copy() for k in z.files}
assert len(pairs['seed']) == 4049
pair_init_max = np.full(4049, np.nan)
pair_init_pool = np.zeros(4049, dtype=np.int16)
pair_pre_pool = np.zeros(4049, dtype=np.int16)
summary, gt_rows, sources, graph_certificates = [], [], {}, []
for seed in (1, 2):
    identity = read(DATA / f'collection_seed{seed}/capture_v1/source_identity.json')
    assert identity['before_and_after_equal']
    assert identity['native_files_sha256']['engine/deim/dfine_decoder.py'] == sha(CURRENT / 'native/engine/deim/dfine_decoder.py')
    constants_path = ROOT / f'outputs/sod_fdr_range_20261010_v1/FDR_constants_seed{seed}.json'
    constants = read(constants_path)
    assert constants['last_epoch'] == 11
    assert constants['constants']['decoder.up'] == [.5] and constants['constants']['decoder.reg_scale'] == [4.]
    sources[str(constants_path.relative_to(ROOT))] = sha(constants_path)
    for batch in range(16):
        folder = DATA / f'collection_seed{seed}/capture_v1/endpoint12/batch_{batch:03d}'
        fm = read(folder / 'manifest.json')
        assert all(sha(folder / name) == digest for name, digest in fm['files'].items())
        sources[str((folder / 'manifest.json').relative_to(ROOT))] = sha(folder / 'manifest.json')
        ev = read(folder / 'evidence.json')
        assert ev['metadata']['endpoint'] == 12 and not ev['metadata']['AMP'] and not ev['metadata']['backward_executed']
        _, _, height, width = ev['metadata']['input_shape']
        mal = [e for e in ev['events'] if e.get('kind') == 'loss_call' and e.get('loss') == 'mal' and e['stage'] == 'model']
        assert len(mal) == 1
        with np.load(folder / 'arrays.npz', allow_pickle=False) as z:
            enc, pre = z['model.enc_aux_outputs.0.pred_boxes'], z['model.pre_outputs.pred_boxes']
            assert enc.shape == pre.shape == (4, 300, 4)
            for slot in range(4):
                gt = z[f'target.{slot}.boxes']
                if len(gt) > 300:
                    continue
                short = (gt[:, 2:].astype(np.float64) * np.array([width, height])).min(1)
                tiny = np.flatnonzero(short < 4)
                initial_max = required_distances(enc[slot], gt)
                pre_max = required_distances(pre[slot], gt)
                initial_ok, pre_ok = initial_max < 4 - 1e-6, pre_max < 4 - 1e-6
                init_count, pre_count = initial_ok.sum(1), pre_ok.sum(1)
                prefix = mal[0]['prefix'] + f'.batch{slot}'
                qs, gs = z[prefix + '.query_index'], z[prefix + '.GT_row_index']
                assert len(gs) == len(np.unique(gs)) and len(qs) == len(np.unique(qs))
                mal_query_for_gt = np.full(len(gt), -1, dtype=int)
                mal_query_for_gt[gs] = qs
                assert (mal_query_for_gt >= 0).all(), 'GT<=300 native ordinary MAL coverage changed; preserve failure'
                pre_match_ok = pre_ok[np.arange(len(gt)), mal_query_for_gt]
                init_match_ok = initial_ok[np.arange(len(gt)), mal_query_for_gt]
                positions = np.flatnonzero((pairs['seed'] == seed) & (pairs['batch'] == batch) & (pairs['slot'] == slot))
                gr, qr = pairs['GT_row'][positions], pairs['query_row'][positions]
                pair_init_max[positions] = initial_max[gr, qr]
                pair_init_pool[positions] = init_count[gr]
                pair_pre_pool[positions] = pre_count[gr]
                assert np.allclose(pre_max[gr, qr], pairs['maximum_abs_required_edge_distance'][positions], atol=1e-12, rtol=0)
                assert np.array_equal(pre_count[gr], pairs['strict_FDR_support_query_count_in_all300_pool'][positions])
                assigned, cover_g, cover_q = certified_max_matching(pre_ok[tiny])
                graph_certificates.append(dict(seed=seed, batch=batch, slot=slot, tiny_GT_rows=tiny.tolist(),
                    matched_query_rows=assigned.tolist(), minimum_vertex_cover_GT_rows=tiny[cover_g].tolist(),
                    minimum_vertex_cover_query_rows=np.flatnonzero(cover_q).tolist(), graph_sha256=hashlib.sha256(pre_ok[tiny].tobytes()).hexdigest()))
                source_ids = z[f'target.{slot}.verified_source_annotation_ids']
                for g in tiny:
                    gt_rows.append(dict(seed=seed, batch=batch, slot=slot, GT_row=int(g), source_annotation_id=int(source_ids[g]),
                        actual_input_shortside=float(short[g]), image_GT=len(gt),
                        encoder_ref_proxy_reachable_queries=int(init_count[g]), native_pre_ref_reachable_queries=int(pre_count[g]),
                        native_MAL_query_row=int(mal_query_for_gt[g]), native_MAL_query_pre_reachable=bool(pre_match_ok[g]),
                        native_MAL_query_encoder_proxy_reachable=bool(init_match_ok[g])))
                summary.append(dict(seed=seed, batch=batch, slot=slot, tiny_GT=len(tiny),
                    encoder_proxy_any=int((init_count[tiny] > 0).sum()), pre_any=int((pre_count[tiny] > 0).sum()),
                    pool_lost_at_pre=int(((init_count[tiny] > 0) & (pre_count[tiny] == 0)).sum()),
                    pool_gained_at_pre=int(((init_count[tiny] == 0) & (pre_count[tiny] > 0)).sum()),
                    neither_pool=int(((init_count[tiny] == 0) & (pre_count[tiny] == 0)).sum()),
                    native_MAL_query_pre_reachable=int(pre_match_ok[tiny].sum()),
                    native_MAL_query_pre_unreachable_but_other_reachable=int((~pre_match_ok[tiny] & (pre_count[tiny] > 0)).sum()),
                    certified_max_distinct_queries_tiny_only=int((assigned >= 0).sum())))
    print(json.dumps(dict(progress='READ_SAVED_BASE12_REFERENCES', seed=seed)), flush=True)
assert np.isfinite(pair_init_max).all()
pairs['encoder_reference_proxy_max_abs_distance'] = pair_init_max
pairs['encoder_reference_proxy_pool_count'] = pair_init_pool
pairs['native_pre_reference_pool_count_rechecked'] = pair_pre_pool
totals, primary = [], []
for seed in (1, 2):
    rows = [r for r in summary if r['seed'] == seed]
    totals.append(dict(seed=seed, sampled_images_GTle300=len(rows), **{k: sum(r[k] for r in rows) for k in rows[0] if k not in ('seed', 'batch', 'slot')}))
    mask = pairs['seed'] == seed
    zero = mask & (pairs['recomputed_FP32_IoU'] == 0)
    outside = zero & ~pairs['in_closed_fixed_FDR_support']
    init_ok = pair_init_max < 4 - 1e-6
    primary.append(dict(seed=seed, final_tiny_FGL_pair_exposures=int(mask.sum()), zero_IoU=int(zero.sum()),
        zero_outside_own_pre_support=int(outside.sum()),
        zero_outside_own_pre_but_own_encoder_proxy_inside=int((outside & init_ok).sum()),
        zero_no_pre_query=int((zero & (pair_pre_pool == 0)).sum()),
        zero_no_pre_query_but_encoder_proxy_pool_available=int((zero & (pair_pre_pool == 0) & (pair_init_pool > 0)).sum()),
        zero_no_query_in_either_reference_pool=int((zero & (pair_pre_pool == 0) & (pair_init_pool == 0)).sum())))
OUT.mkdir(exist_ok=False)
np.savez_compressed(OUT / 'final_under4_FGL_pair_reference_transitions.npz', **pairs)
with (OUT / 'distinct_augmented_GT_reference_transitions.csv').open('x', encoding='utf-8-sig', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(gt_rows[0])); writer.writeheader(); writer.writerows(gt_rows)
write(OUT / 'tiny_query_matching_certificates.json', graph_certificates)
value = dict(status='SAVED_AUGMENTED_BASE12_REFERENCE_AND_PHYSICAL_ASSIGNMENT_TRACED',
    distinct_GT_cohort_totals=totals, actual_FGL_final_pair_totals=primary, per_image=summary,
    parent_pool_manifest_sha256=sha(parent / 'manifest.json'), source_files=sources,
    exact_saved_encoder_and_pre_rows_have_native_same_query_order=True,
    native_FDR_reference='pre_outputs.pred_boxes, shared by all FDR stages',
    encoder_box_span_comparison='Counterfactual FDR support proxy only. Native encoder and pre-head box updates use sigmoid residuals, not this FDR range.',
    graph='GT-assisted strict native pre-reference continuous FDR support; class-agnostic tiny-only graph, no other-size GT competition or costs.',
    graph_exact_optimality_certificates=len(graph_certificates),
    maximum_matching_is_not_native_Hungarian_reassignment_or_prediction_AP=True,
    neither_reference_proxy_pool_does_not_imply_encoder_cannot_correct_in_future=True,
    old_augmented_fixture_not_actual_training_history_or_population_estimate=True,
    DN_predictions_unavailable_BASE132_unanalyzed=True,
    no_new_acceptance_gate_or_training_configuration=True,
    first_error_root_and_effectiveness_not_claimed=True,
    model_forwards=0, model_backwards=0, evaluator_calls=0, optimizer_steps=0, training_epochs=0)
assert 'torch' not in sys.modules
write(OUT / 'report.json', value)
write(OUT / 'manifest.json', dict(files={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in OUT.iterdir() if p.is_file()}, source_sha256=sha(__file__)))
print(json.dumps(dict(status=value['status'], totals=totals, actual_pair_totals=primary,
    graph_certificates=len(graph_certificates), manifest_sha256=sha(OUT / 'manifest.json'))), flush=True)
