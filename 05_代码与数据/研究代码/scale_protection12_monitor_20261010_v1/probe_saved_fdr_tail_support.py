"""A fixed twofold tail-range counterfactual, using downloaded arrays only."""
import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path.cwd()
OUT = ROOT/'outputs/sod_fdr_tail_support_20261010_v1'
TRACE = ROOT/'outputs/saved_fgl_coefficient_analysis_20261010_v2/reference_stage_trace_v1'
RANGE = ROOT/'outputs/sod_fdr_range_20261010_v1/analysis'
HELPER = Path(__file__).with_name('trace_saved_reference_support.py')
read = lambda p: json.loads(Path(p).read_text(encoding='utf8'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, v):
    with Path(p).open('x', encoding='utf8') as stream:
        json.dump(v, stream, ensure_ascii=False, indent=2, allow_nan=False)

def prepare():
    fm = read(TRACE/'manifest.json')
    assert fm['source_sha256'] == sha(HELPER)
    old = read(RANGE/'report.json')
    grid = np.asarray(old['weights'], dtype=np.float64)
    assert grid.shape == (33,) and grid[0] == -4 and grid[-1] == 4 and (np.diff(grid)>0).all()
    alternative = grid.copy(); alternative[[0,-1]] = [-8,8]
    assert np.array_equal(grid[1:-1], alternative[1:-1])
    sources = [TRACE/'manifest.json',TRACE/'report.json',TRACE/'tiny_query_matching_certificates.json',
        TRACE/'final_under4_FGL_pair_reference_transitions.npz', RANGE/'manifest.json',RANGE/'report.json',
        RANGE/'seed1_per_GT.npz',RANGE/'seed2_per_GT.npz',HELPER,
        ROOT/'英文小论文/文献_相关原始论文_20261004/105_D_FINE_ICLR2025_Publisher.pdf',
        ROOT/'英文小论文/data/00_实验计划与证据索引/FDR重锚定候选独立复核_20261006_v1/独立复核裁决_数学统计与文献_20261006.txt']
    value = dict(schema='saved_fdr_tail_support_probe_v1',frozen_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sha(__file__), baseline_bound=4., alternative_bound=8., ordinary_reg_scale=4.,
        saved_train_strict_margin=1e-6, saved_val_strict_margin=1e-7,
        old_grid=grid.tolist(),counterfactual_grid=alternative.tolist(), central_31_positions_unchanged=True,
        hypothesis='Fixed pre-reference ranges may exclude actual small-object regression responsibilities. Test geometric reachability and tiny-only query competition with fixed endpoints+-8, keeping all31 interior mathematical supports unchanged.',
        selection='One fixed factor2 counterfactual; no parameter search, no efficacy or training acceptance gate.',
        scopes=['Existing BASE12 augmented FP32 fixtures: final FGL actual union pairs and allunder4 GT in imagesGT<=300.',
            'Existing full548 BASE12 validation algebraic range arrays: any-query support only, no new joint matching or predictor.'],
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        paper='D-FINE ICLR2025 official Eq4/Table9 already addresses range and curvature. Changing the range alone is not a novelty claim.',
        paper_url='https://proceedings.iclr.cc/paper_files/paper/2025/file/6cf58a87e3097e7d1f9be3e8693a93de-Paper-Conference.pdf',
        previous_reanchor_route_not_restarted=True,
        predictions_unchanged_in_this_probe=True, GT_assisted_not_deployable=True,
        no_new_model_configuration_or_training=True, model_forwards=0, evaluator_calls=0, optimizer_steps=0, epochs_added=0)
    OUT.mkdir(exist_ok=False);write(OUT/'protocol.json',value)
    print(json.dumps(dict(status='FIXED_TAIL_RANGE_NUMERICAL_PROBE_FROZEN',protocol_sha256=sha(OUT/'protocol.json'))),flush=True)

def analyze():
    p = read(OUT/'protocol.json')
    assert p['source_sha256'] == sha(__file__)
    assert all(sha(ROOT/name) == digest for name,digest in p['sources'].items())
    tm, rm = read(TRACE/'manifest.json'),read(RANGE/'manifest.json')
    assert all(sha(TRACE/name) == row['sha256'] for name,row in tm['files'].items())
    assert all(sha(RANGE/name) == digest for name,digest in rm['files'].items())
    nodes = [n for n in ast.parse(HELPER.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef)
        and n.name in ('required_distances','certified_max_matching')]
    assert len(nodes) == 2
    namespace={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(HELPER),'exec'),namespace)
    distance,maximum=namespace['required_distances'],namespace['certified_max_matching']
    prior = read(TRACE/'report.json'); old_certs=read(TRACE/'tiny_query_matching_certificates.json')
    certs, rows, source_manifests = [], [], {}
    for old in prior['per_image']:
        seed,batch,slot=(old[k] for k in ('seed','batch','slot'))
        folder=ROOT/f'outputs/native_augmented_supervision_20261010_v1/collection_seed{seed}/capture_v1/endpoint12/batch_{batch:03d}'
        fm=read(folder/'manifest.json');assert all(sha(folder/name)==digest for name,digest in fm['files'].items())
        source_manifests[str((folder/'manifest.json').relative_to(ROOT))]=sha(folder/'manifest.json')
        ev=read(folder/'evidence.json');assert ev['metadata']['endpoint']==12 and not ev['metadata']['backward_executed']
        _,_,h,w=ev['metadata']['input_shape']
        with np.load(folder/'arrays.npz',allow_pickle=False) as z:
            gt=z[f'target.{slot}.boxes'];tiny=np.flatnonzero((gt[:,2:].astype(np.float64)*[w,h]).min(1)<4)
            assert len(gt)<=300 and len(tiny)==old['tiny_GT']
            required=distance(z['model.pre_outputs.pred_boxes'][slot],gt)[tiny]
            e4=required<p['baseline_bound']-p['saved_train_strict_margin']
            e8=required<p['alternative_bound']-p['saved_train_strict_margin']
            assert not (e4 & ~e8).any()
            matching,cg,cq=maximum(e8)
            old_cert=next(c for c in old_certs if (c['seed'],c['batch'],c['slot'])==(seed,batch,slot))
            assert hashlib.sha256(e4.tobytes()).hexdigest()==old_cert['graph_sha256']
            assert int(e4.any(1).sum())==old['pre_any']
            old_assign=np.asarray(old_cert['matched_query_rows']);good=old_assign>=0
            assert e4[np.flatnonzero(good),old_assign[good]].all()
            assert int(good.sum())==old['certified_max_distinct_queries_tiny_only']
            cover_gt=np.isin(tiny,old_cert['minimum_vertex_cover_GT_rows'])
            cover_q=np.isin(np.arange(300),old_cert['minimum_vertex_cover_query_rows'])
            assert not (e4 & ~cover_gt[:,None] & ~cover_q[None,:]).any()
            assert int(cover_gt.sum()+cover_q.sum())==int(good.sum())
        row=dict(seed=seed,batch=batch,slot=slot,tiny_GT=len(tiny),range4_any=int(e4.any(1).sum()),range8_any=int(e8.any(1).sum()),
            range4_max_distinct=int(good.sum()),range8_max_distinct=int((matching>=0).sum()))
        rows.append(row)
        certs.append(dict(seed=seed,batch=batch,slot=slot,tiny_GT_rows=tiny.tolist(),matched_query_rows=matching.tolist(),
            cover_GT_rows=tiny[cg].tolist(),cover_query_rows=np.flatnonzero(cq).tolist(),graph_sha256=hashlib.sha256(e8.tobytes()).hexdigest()))
    totals=[]
    for seed in (1,2):
        rr=[row for row in rows if row['seed']==seed]
        totals.append(dict(seed=seed,images=len(rr),**{k:sum(row[k] for row in rr) for k in
            ('tiny_GT','range4_any','range8_any','range4_max_distinct','range8_max_distinct')}))
    pair_totals=[]
    with np.load(TRACE/'final_under4_FGL_pair_reference_transitions.npz',allow_pickle=False) as z:
        for seed in (1,2):
            selected=z['seed']==seed;zero=selected & (z['recomputed_FP32_IoU']==0)
            d=z['maximum_abs_required_edge_distance']
            s4=d<p['baseline_bound']-p['saved_train_strict_margin'];s8=d<p['alternative_bound']-p['saved_train_strict_margin']
            pair_totals.append(dict(seed=seed,actual_FGL_pair_exposures=int(selected.sum()),zero_IoU=int(zero.sum()),
                own_ref_range4=int((selected&s4).sum()),own_ref_range8=int((selected&s8).sum()),
                zero_IoU_own_ref_range4=int((zero&s4).sum()),zero_IoU_own_ref_range8=int((zero&s8).sum()),
                zero_IoU_newly_expressible_in_own_reference=int((zero&~s4&s8).sum())))
    validation=[]
    for seed in (1,2):
        with np.load(RANGE/f'seed{seed}_per_GT.npz',allow_pickle=False) as z:
            tiny=z['input_shortside']<4;minimum=z['minimum_max_abs_distance']
            s4=minimum<p['baseline_bound']-p['saved_val_strict_margin'];s8=minimum<p['alternative_bound']-p['saved_val_strict_margin']
            assert int(tiny.sum())==5096 and np.array_equal(s4,z['any_strict_interior'])
            failed=tiny & (z['max_iou'][:,2]<.75)
            validation.append(dict(seed=seed,full_validation_images=548,tiny_GT=int(tiny.sum()),
                range4_any=int((tiny&s4).sum()),range8_any=int((tiny&s8).sum()),
                newly_any_expressible=int((tiny&~s4&s8).sum()),
                native_final_geometry_failed75=int(failed.sum()),failed75_newly_any_expressible=int((failed&~s4&s8).sum())))
    dst=OUT/'analysis_v1';dst.mkdir(exist_ok=False)
    write(dst/'range8_matching_certificates.json',certs)
    value=dict(status='FIXED_FDR_TAIL_RANGE_GEOMETRIC_COUNTERFACTUAL_COMPLETED',protocol_sha256=sha(OUT/'protocol.json'),
        augmented_tiny_only_competition=totals,actual_FGL_pairs=pair_totals,full_validation_any_query_only=validation,
        augmented_per_image=rows,source_batch_manifests=source_manifests,
        exact_matching_optimality_certificates=len(certs),central_31_mathematical_support_values_unchanged=True,
        model_corner_probabilities_missing_predictive_damage_not_measured=True,
        normal_validation_any_support_is_not_joint_assignment=True,
        GT_assisted_geometry_not_accuracy_or_AP_bound=True,
        ordinary_L1_GIoU_and_DN_supervision_not_missing=True,
        literature_precludes_claiming_simple_range_change_is_new=True,
        no_new_configuration_promoted_or_trained=True,model_forwards=0,backwards=0,evaluator_calls=0,optimizer_steps=0,epochs_added=0)
    assert 'torch' not in sys.modules
    write(dst/'report.json',value)
    write(dst/'manifest.json',dict(files={q.name:dict(sha256=sha(q),bytes=q.stat().st_size) for q in dst.iterdir() if q.is_file()},source_sha256=sha(__file__)))
    print(json.dumps(dict(status=value['status'],train=totals,pairs=pair_totals,validation=validation,
        manifest_sha256=sha(dst/'manifest.json'))),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');args=ap.parse_args()
    prepare() if args.prepare else analyze()
