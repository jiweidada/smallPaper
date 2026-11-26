"""Six fixed12 records: actual selector observations joined to verified saved GT stages.

This consumer does not load a model or call an evaluator. Formal terminal-reader
passes and exact diagnostic replays are required; native primary logs alone fail.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys
import numpy as np
from selection_core import image_description, verify_grid_centers, RADII

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
KINDS = {'original_stem': 'full', 'native_stem_scratch': 'reference', 'highfreq_rgb': 'control'}
read = lambda p: json.loads(Path(p).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def npz(path):
    with np.load(path, allow_pickle=False) as z: return {n: z[n].copy() for n in z.files}


def manifest(folder, expected=None):
    m = read(folder / 'manifest.json')
    if expected is not None: assert set(m['files']) == set(expected)
    for name, digest in m['files'].items():
        p = Path(name)
        assert name and not p.is_absolute() and '..' not in p.parts and ':' not in name and '\\' not in name
        assert sha(folder / p) == digest, name
    return m


def validate_job_set(jobs):
    assert type(jobs) is list and len(jobs) == 6, 'Exactly three arms times two seeds required'
    required = {(arm, seed) for arm in KINDS for seed in (1, 2)}
    seen = set()
    for j in jobs:
        assert j['arm'] in KINDS and j['kind'] == KINDS[j['arm']]
        assert type(j['seed']) is int and j['seed'] in (1, 2)
        key = (j['arm'], j['seed']); assert key not in seen, 'Duplicate arm/seed'
        seen.add(key)
    assert seen == required


def prepared():
    p = read(HERE / 'protocol.json')
    assert p['schema'] == 'actual_selection_offline_analysis_v5'
    for n, h in p['files'].items(): assert sha(HERE / n) == h, n
    for n, h in p['dependencies'].items(): assert sha(ROOT / n) == h, n
    stage = module('selection_v5_saved_stages', ROOT / p['stage_analysis_source'])
    stage_parts = stage.prepared()
    old = module('selection_v5_pinned_aggregate', ROOT / p['aggregate_source'])
    return p, stage, stage_parts, old.aggregate


def group_masks(values, annotation):
    anns = {int(a['id']): a for a in annotation['annotations']}
    ims = {int(i['id']): i for i in annotation['images']}
    short = np.array([min(640 * anns[int(g)]['bbox'][2] / ims[int(anns[int(g)]['image_id'])]['width'],
                          640 * anns[int(g)]['bbox'][3] / ims[int(anns[int(g)]['image_id'])]['height'])
                      for g in values['annotation_id']])
    # Native evaluator small area range is closed [0, 1024]. This is a GT
    # descriptive stratum; TP flags below still come from native all-area matching.
    return {'all': np.ones(len(short), bool), 'COCO_small': values['original_area'] <= 1024,
            'input_under4': short < 4, 'input_4_to_under8': (short >= 4) & (short < 8),
            'input_8_to_under16': (short >= 8) & (short < 16), 'input_ge16': short >= 16}


def validate_replay(v, done, identity, checkpoint, training_completion_sha, p):
    assert v['status'] == 'PASS_SELECTION_DIAGNOSTIC_EXACT_SAVED_EMA_REPLAY'
    assert v['identity'] == identity and v['arm'] == identity['arm'] and v['seed'] == identity['seed']
    assert v['checkpoint_sha256'] == checkpoint
    assert v['original_training_completion_sha256'] == training_completion_sha
    assert v['diagnostic_protocol_sha256'] == p['diagnostic_protocol_sha256']
    assert v['verification_script_sha256'] == p['diagnostic_replay_sha256']
    assert v['stats_max_abs_fraction_difference'] == v['stats_tolerance'] == 0
    assert v['original_stats'] == v['reloaded_stats']
    assert v['native_GT_hits_exact'] is True and v['all548_stage_arrays_exact'] is True
    assert all(x == 0 for x in v['stage_max_abs_difference'].values())
    assert v['candidate_model_forwards'] == v['actual_selector_calls'] == v['existing_selected_bbox_head_calls'] == 137
    assert v['images'] == v['selection_images'] == 548
    assert v['additional_bbox_head_or_topk_calls'] == v['baseline_model_forwards'] == v['optimizer_updates'] == v['epochs_added'] == 0
    assert v['automatic_retry'] is False and v['all_parameters_frozen_like_native_EMA'] is True
    assert done['exit_code'] == 0 and done['timed_out'] is False and done['automatic_retry'] is False
    assert done['baseline_model_forwards'] == done['epochs_added'] == 0



def checkpoint_anchor_reference(job, trained):
    """An exported actual savedEMA buffer, never a freshly generated proxy."""
    assert sha(job['anchor_audit_report'])==job['anchor_audit_report_sha256']
    report=read(job['anchor_audit_report']);assert report['status'] in ['PASS_CHECKPOINT_BOUND_EMA_MODEL_AND_NATIVE_GENERATED_ANCHOR_TENSOR_AUDIT','VERIFIED_ACTUAL_EMA_EXPORT_AND_FEATURE_CELLS_CPU_MODEL_GENERATOR_DIFFERENCE_RECORDED']
    assert report['seed']==job['seed'] and report['model_forwards']==report['training_epochs']==report['optimizer_updates']==0 and report['CUDA_initialized'] is False
    assert report['native_generator_source_sha256']=='b31408e49d4961e1e82692703b8f9fc3f07f1e280fd13ee12f896cb337f9bae5'
    assert report['EMA_source_sha256']=='aba8ace1dacfe68b2b24e51bdb908ea17c0c2b2e1ebf2e0496cd1499a16a2cff'
    case=report['cases'][job['arm']];assert case['checkpoint_sha256']==trained['last_checkpoint_sha256']
    assert sha(job['anchor_export'])==job['anchor_export_sha256']==case['export_sha256']
    values=npz(job['anchor_export']);assert set(values)=={'EMA_anchors','model_anchors','generated_anchors'}
    a=values['EMA_anchors'];assert a.shape==(8400,4) and a.dtype==np.float32 and hashlib.sha256(a.tobytes()).hexdigest()==case['array_sha256']
    # EMA is bound exactly to its actual checkpoint export. Freshly generated
    # CPU model anchors are supplementary provenance, not a proxy for that EMA.
    m,g=values['model_anchors'],values['generated_anchors']
    assert m.shape==g.shape==a.shape and m.dtype==g.dtype==a.dtype
    assert case['model_equal_generated']==bool(np.array_equal(m,g))
    good=np.isfinite(a).all(1)
    assert np.array_equal(np.isfinite(m),np.isfinite(a)) and np.array_equal(np.isfinite(g),np.isfinite(a))
    assert float(np.max(np.abs(m[good].astype('float64')-g[good].astype('float64'))))==case['model_generated_max_abs_difference']
    verify_grid_centers(dict(dense_anchor_logits=m),m)
    verify_grid_centers(dict(dense_anchor_logits=g),g)
    verify_grid_centers(dict(dense_anchor_logits=a),a)
    return a


def binding(job, p, reader, base):
    packet, folder = Path(job['packet']), Path(job['diagnostic'])
    assert sha(job['report']) == job['report_sha256']
    report = read(job['report']); identity = report['identity']; rp = reader.local_protocol()
    assert report['status'] == reader.PASS and report['fixed_endpoint'] == p['fixed_endpoint'] == rp['endpoint']
    assert identity['arm'] == job['arm'] and identity['seed'] == job['seed']
    assert identity['implementation'] == rp['implementation_sha256']
    assert report['verification_script_sha256'] == rp['reader_files_sha256']['verify_terminal_local.py']
    assert report['local_reader_protocol_sha256'] == sha(reader.PROTOCOL)
    assert report['all548_stage_arrays_reloaded_exact'] is True and report['first100_inputs_and_pre_forward_RNG_exact'] is True
    assert report['new_model_forwards'] == report['new_evaluator_calls'] == report['new_optimizer_updates'] == report['new_training_epochs'] == 0
    if job['kind'] != 'full': assert packet.resolve() == Path(report['downloaded_packet']).resolve()
    assert sha(packet / 'terminal_manifest.json') == report['terminal_manifest_sha256']
    tm = read(packet / 'terminal_manifest.json'); assert tm['identity'] == identity
    for n, item in tm['files'].items():
        rel = Path(n); assert not rel.is_absolute() and '..' not in rel.parts and ':' not in n and '\\' not in n
        actual = packet / rel
        assert actual.stat().st_size == item['bytes'] and sha(actual) == item['sha256'], n
    for key in ('common_init_sha256', 'baseline_checkpoint_sha256', 'baseline_first100_sha256', 'val_annotation_sha256', 'evaluator_binding'):
        assert identity[key] == base['summary']['identity'][key], key
    assert sha(base['packet'] / 'reference/val_annotations.json') == identity['val_annotation_sha256']
    run = packet / 'run'; trained = read(run / 'training_completion.json')
    assert trained['status'] == 'NATIVE_FIT_FINISHED_EXACT12' and trained['identity'] == identity
    assert trained['completed_epochs'] == 12 and trained['entered_epochs'] == list(range(12))
    assert trained['automatic_retry'] is False and sha(run / 'last.pth') == trained['last_checkpoint_sha256']
    assert sha(run / 'stages_epoch12/manifest.json') == trained['stage_manifest_sha256']
    assert sha(run / 'endpoint_epoch12/manifest.json') == trained['endpoint_manifest_sha256']
    assert sha(folder / 'verification.json') == job['verification_sha256']
    assert sha(folder / 'selection_epoch12/manifest.json') == job['selection_manifest_sha256']
    v = read(folder / 'verification.json')
    validate_replay(v, read(folder / 'completion.json'), identity, trained['last_checkpoint_sha256'], sha(run / 'training_completion.json'), p)
    assert v['selection_manifest_sha256'] == job['selection_manifest_sha256']
    sm = manifest(folder / 'selection_epoch12', {'summary.json'} | {f'image_{i:06d}.npz' for i in range(548)})
    assert sm['trace_code_sha256'] == p['selection_stage_trace_sha256']
    ss = read(folder / 'selection_epoch12/summary.json')
    assert ss['status'] == 'NATIVE_SELECTION_AND_EXISTING_STAGE_BATCH_IDS_JOINED' and ss['mode'] == 'native_runtime'
    assert ss['identity'] == identity and ss['images'] == 548 and ss['image_ids'] == list(range(548))
    assert ss['epoch_zero_based'] == 11 and ss['evaluated_weights'] == 'EMA'
    assert ss['dense_locations'] == 8400 and ss['selected_queries'] == 300 and ss['input_hw'] == [640, 640]
    assert ss['additional_head_calls'] == ss['additional_topk_calls'] == 0 and ss['cross_run_query_identity_assumed'] is False
    manifest(folder / 'stages_epoch12', {'summary.json'} | {f'image_{i:06d}.npz' for i in range(548)})
    assert read(folder / 'stages_epoch12/summary.json')['identity'] == identity
    manifest(folder / 'endpoint_epoch12', {'summary.json', 'precision_recall.npz', 'GT_hits.npz', 'matching_all_area.json.gz'})
    original, original_hits, _ = reader.native_endpoint(run / 'endpoint_epoch12', base['annotation'])
    replay, replay_hits, _ = reader.native_endpoint(folder / 'endpoint_epoch12', base['annotation'])
    assert original['identity'] == replay['identity'] == identity
    assert original['native_stats'] == replay['native_stats'] == v['original_stats']
    assert original['overall_percent'] == report['candidate_overall_percent']
    assert all(np.array_equal(original_hits[k], replay_hits[k]) for k in original_hits)
    checkpoint_anchor_reference(job,trained)
    return identity


def describe_selection(job, values, annotation, geometry, aggregate, output):
    folder, run = Path(job['diagnostic']), Path(job['packet']) / 'run'
    anchor_reference=checkpoint_anchor_reference(job,read(run/'training_completion.json'))
    loc = {int(g): i for i, g in enumerate(values['annotation_id'])}
    assert set(loc) == {int(a['id']) for a in annotation['annotations']}
    joined = {k: values[k].copy() for k in ('annotation_id', 'category_id', 'image_id', 'original_area', 'native_TP', 'max_iou')}
    by_image = {i: [] for i in range(548)}
    for a in annotation['annotations']: by_image[int(a['image_id'])].append(a)
    ims = {int(im['id']): im for im in annotation['images']}; audits = []; seen = set()
    for iid in range(548):
        selection = npz(folder / f'selection_epoch12/image_{iid:06d}.npz')
        stage = npz(folder / f'stages_epoch12/image_{iid:06d}.npz')
        primary = npz(run / f'stages_epoch12/image_{iid:06d}.npz')
        assert set(stage) == set(primary) and all(stage[k].dtype == primary[k].dtype and np.array_equal(stage[k], primary[k]) for k in stage)
        assert int(selection['image_id']) == iid and np.array_equal(selection['original_hw'], stage['original_hw'])
        assert selection['original_hw'].tolist() == [ims[iid]['height'], ims[iid]['width']]
        targets = sorted(by_image[iid], key=lambda a: int(a['id']))
        gt = np.array([a['bbox'] for a in targets], float).reshape(-1, 4)
        boxes = stage['encoder_initial_boxes']; valid = (boxes[:, 2:] > 0).all(1)
        iou = np.zeros((len(targets), 300), float)
        iou[:, valid] = geometry.iou_matrix(gt, geometry.pixel_xywh(boxes[valid], selection['original_hw']))
        result, audit = image_description(selection, targets, iou)
        audit['image_id'] = iid; audit['finite_observed_anchor_center_max_abs_grid_error'] = verify_grid_centers(selection,anchor_reference)
        audits.append(audit)
        positions = np.array([loc[int(a['id'])] for a in targets], np.int64)
        assert np.array_equal(values['image_id'][positions], np.full(len(targets), iid))
        assert np.array_equal(values['category_id'][positions], np.array([a['category_id'] for a in targets], np.int64))
        if len(targets): assert np.array_equal(iou.max(1), values['max_iou'][positions, 0])
        for key, array in result.items():
            if key not in joined: joined[key] = np.empty((len(loc), *array.shape[1:]), dtype=array.dtype)
            joined[key][positions] = array
        seen.update(int(a['id']) for a in targets)
    assert seen == set(loc)
    masks = group_masks(joined, annotation); assert len(loc) == 38759 and masks['COCO_small'].sum() == 26586
    groups = {g: dict(all=aggregate(joined, m), native_miss_IoU50=aggregate(joined, m & ~joined['native_TP'][:, 0]),
                      native_miss_IoU75=aggregate(joined, m & ~joined['native_TP'][:, 5])) for g, m in masks.items()}
    dest = output / (job['arm'] + '_seed' + str(job['seed']))
    np.savez_compressed(dest / 'selection_per_GT.npz', **joined)
    d = dict(groups=groups, image_audits=audits, terminal_report_sha256=job['report_sha256'],
             diagnostic_verification_sha256=job['verification_sha256'], selection_manifest_sha256=job['selection_manifest_sha256'],
             actual_GT=len(loc), all548_images=True, normalized_radius_windows_are_spatial_proxies=True,
             native_TP_source='Native all-area matching, stratified by GT area/input size')
    write(dest / 'selection_description.json', d)
    return joined, d


def selection_pair(left, right, annotation):
    for k in ('annotation_id', 'category_id', 'image_id', 'original_area'): assert np.array_equal(left[k], right[k])
    out = {}
    for group, mask in group_masks(left, annotation).items():
        table = {}
        for ri, radius in enumerate(RADII):
            av = left['selected_neighborhood_sites'][:, ri, 3] > 0; bv = right['selected_neighborhood_sites'][:, ri, 3] > 0
            gain, loss = ~av & bv, av & ~bv
            row = dict(GT=int(mask.sum()), gained_nearby_selection=int((mask & gain).sum()), lost_nearby_selection=int((mask & loss).sum()))
            for t, ti in (('50', 0), ('75', 5)):
                new_tp = ~left['native_TP'][:, ti] & right['native_TP'][:, ti]
                lost_tp = left['native_TP'][:, ti] & ~right['native_TP'][:, ti]
                row['native_TP' + t] = dict(gained=int((mask & new_tp).sum()), lost=int((mask & lost_tp).sum()),
                    gained_with_nearby_selection_gain=int((mask & new_tp & gain).sum()), lost_with_nearby_selection_loss=int((mask & lost_tp & loss).sum()))
            table[str(radius)] = row
        out[group] = table
    return out


def analyze(spec_path, output):
    p, stage_helper, parts, aggregate = prepared()
    stage_p, geometry, onset, rank, readers = parts
    spec = read(spec_path); jobs = spec['jobs']; validate_job_set(jobs)
    output = Path(output); assert not output.exists()
    bases = {s: stage_helper.cached_baseline(s, stage_p, readers['full']) for s in (1, 2)}
    assert bases[1]['annotation'] == bases[2]['annotation']
    # All six positive terminal bindings are mandatory before output creation.
    for j in jobs: binding(j, p, readers[j['kind']], bases[j['seed']])
    output.mkdir(); stages = {}; selected = {}; descriptions = {}
    for j in jobs:
        key = (j['arm'], j['seed']); annotation = bases[j['seed']]['annotation']
        v, sd = stage_helper.inspect(j, stage_p, geometry, onset, rank, readers, bases[j['seed']], output)
        assert sd['native_TP_proxy_conflicts'] == 0 and not sd['score_mismatch_images']
        joined, d = describe_selection(j, v, annotation, geometry, aggregate, output)
        stages[key], selected[key] = v, joined
        descriptions[j['arm'] + '_seed' + str(j['seed'])] = dict(stages=sd, selection=d)
    pairs = {}
    for seed in (1, 2):
        annotation = bases[seed]['annotation']; full = ('original_stem', seed); pairs[str(seed)] = {}
        for arm in ('native_stem_scratch', 'highfreq_rgb'):
            left = (arm, seed)
            pairs[str(seed)][arm + '_to_original_stem'] = dict(
                actual_selection_association=selection_pair(selected[left], selected[full], annotation),
                saved_geometry_boundaries=stage_helper.pair(stages[left], stages[full], annotation, geometry))
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    report = dict(status='ANALYZED_SIX_FIXED12_ACTUAL_TOPK_AND_SAVED_GEOMETRY_DESCRIPTIONS', pairs=pairs,
        descriptions=descriptions, arms=list(KINDS), seeds=[1, 2], spec_sha256=sha(spec_path), protocol_sha256=sha(HERE / 'protocol.json'),
        baseline_sources={str(s): dict(source_report_sha256=b['source_report_sha256'], stage_cache_manifest_sha256=b['cache_stage_manifest_sha256'],
                                     rank_cache_manifest_sha256=b['cache_rank_manifest_sha256']) for s, b in bases.items()},
        causal_root_proven=False, novelty_proven=False, actual_model_efficacy_inferred_here=False,
        new_model_forwards=0, new_evaluator_calls=0, new_optimizer_updates=0, new_training_epochs=0, scope=p['scope'])
    write(output / 'report.json', report)
    write(output / 'manifest.json', dict(files={x.relative_to(output).as_posix(): sha(x) for x in output.rglob('*') if x.is_file()}))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--spec', required=True); parser.add_argument('--output', required=True)
    a = parser.parse_args(); print(analyze(a.spec, a.output)['status'])
