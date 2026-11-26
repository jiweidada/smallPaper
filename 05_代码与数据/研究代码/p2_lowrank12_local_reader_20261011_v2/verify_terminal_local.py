"""Verify new rank16 P2 fixed12 terminal packets; all saved EMA arrays exact. No Torch."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CORE = ROOT / '英文小论文/code/p2_lowrank12_20261011_v3'
PROTOCOL = HERE / 'local_reader_protocol.json'
PASS = 'PASS_LOW_RANK_P2_FIXED12_DOWNLOADED_EVIDENCE_AND_EXACT_SAVED_EMA_RELOAD'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf8').splitlines()]


def local_protocol():
    p = read(PROTOCOL)
    assert p['schema'] == 'p2_lowrank12_local_reader_v2'
    for name, h in p['reader_files_sha256'].items():
        assert sha(HERE / name) == h, name
    for name, h in p['core_files_sha256'].items():
        assert sha(CORE / name) == h, name
    assert p['endpoint'] == dict(epochs=12, zero_based_epoch=11, weights='EMA')
    assert p['acceptance'] == dict(per_seed_APs_delta_pp='strictly_positive',
        per_seed_AP_delta_pp='nonnegative', required_seeds=[1, 2])
    return p


def pure_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, CORE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_metadata(meta):
    assert meta['status'] == 'STEM_FIXED12_TRAIN_RELOAD_COLLECTION_FINISHED'
    assert type(meta['seed']) is int and meta['seed'] in (1, 2)
    assert meta['baseline_forwards'] == meta['extra_epochs'] == 0
    assert type(meta['archive_entries']) is int and meta['archive_entries'] > 0
    assert type(meta['bytes']) is int and meta['bytes'] > 0
    for key in ('sha256', 'manifest_sha256'):
        assert isinstance(meta[key], str) and len(meta[key]) == 64
        assert set(meta[key]) <= set('0123456789abcdef')


def validate_names(names):
    assert len(names) == len(set(names)) == len({n.casefold() for n in names})
    for name in names:
        p = PurePosixPath(name)
        assert name and not p.is_absolute() and '..' not in p.parts
        assert '\\' not in name and ':' not in name and p.as_posix() == name


def extract(archive, meta, destination):
    validate_metadata(meta)
    assert sha(archive) == meta['sha256'] and Path(archive).stat().st_size == meta['bytes']
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        validate_names(names)
        assert all(not x.is_dir() for x in z.infolist())
        assert len(names) == meta['archive_entries'] and z.testzip() is None
        data = z.read('terminal_manifest.json')
        assert hashlib.sha256(data).hexdigest() == meta['manifest_sha256']
        manifest = json.loads(data)
        assert manifest['schema'] == 'p2_lowrank12_terminal_v1'
        assert manifest['identity']['seed'] == meta['seed']
        assert manifest['training_epochs'] == 12 and manifest['baseline_model_forwards_added'] == 0
        assert manifest['automatic_retry'] is False
        assert set(names) == set(manifest['files']) | {'terminal_manifest.json'}
        assert all((destination / n).resolve().is_relative_to(destination.resolve()) for n in names)
        if destination.exists():
            assert set(names) == {p.relative_to(destination).as_posix()
                for p in destination.rglob('*') if p.is_file()}
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            assert shutil.disk_usage(destination.parent).free > sum(x.file_size for x in z.infolist()) + 64*1024**2
            destination.mkdir()
            z.extractall(destination)
    for name, item in manifest['files'].items():
        p = destination / name
        assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], name
    assert sha(destination / 'terminal_manifest.json') == meta['manifest_sha256']
    return manifest


def validate_identity(identity, plan, protocol):
    assert identity['schema'] == plan['schema'] == 'p2_lowrank12_training_v1'
    assert identity['arm'] == plan['arm'] == 'p2_lowrank16'
    assert type(identity['seed']) is int and identity['seed'] == plan['seed'] in (1, 2)
    assert identity['implementation'] == protocol['implementation_sha256']
    assert identity['protocol_sha256'] == protocol['core_files_sha256']['training_protocol.json']
    assert identity['native_parameter_count'] == 10231997 and identity['added_parameter_count'] == 6176
    assert identity['baseline_training_epochs_added'] == identity['baseline_model_forwards_added'] == 0
    assert plan['max_epochs'] == 12 and plan['recipe_epochs'] == 132 and plan['extra_parameters'] == 6176 and plan['expected_parameters']==10238173
    assert plan['fixed_endpoint'] == 'native_epoch12_EMA'


def reuse_sources(identity, protocol):
    seed = identity['seed']
    a = protocol['reused_local_evidence'][str(seed)]
    packet = (ROOT / a['packet_relative_path']).resolve()
    report_path = (ROOT / a['report_relative_path']).resolve()
    assert packet.is_relative_to(ROOT) and report_path.is_relative_to(ROOT)
    assert sha(report_path) == a['report_sha256']
    old = read(report_path)
    assert old['status'] == 'PASS_FIXED12_DOWNLOADED_EVIDENCE_AND_SAVED_WEIGHT_RELOAD'
    assert old['identity']['seed'] == seed
    assert sha(packet / 'manifest.json') == a['manifest_sha256']
    source = read(CORE / 'source_binding.json')
    assert len(source['native_files_sha256']) == 468
    for name, h in source['native_files_sha256'].items():
        assert sha(packet / 'native' / name) == h, name
    for name, h in source['legacy_constructor_files'].items():
        assert sha(CORE / name) == h, name
    assert identity['evaluator_binding'] == old['identity']['evaluator_binding']
    for name, h in identity['evaluator_binding']['files'].items():
        assert sha(packet / 'evaluator/pycocoeval' / name) == h, name
    assert len(identity['evaluator_binding']['files']) == 7
    for name, key in [('baseline_epoch12.pth', 'baseline_checkpoint_sha256'),
        ('baseline_first100.jsonl', 'baseline_first100_sha256'),
        ('train_annotations.json', 'train_ann_sha256'), ('val_annotations.json', 'val_ann_sha256')]:
        assert sha(packet / 'reference' / name) == identity[key] == old['identity'][key], name
    common = (ROOT / protocol['common_init_relative_path']).resolve()
    assert common.is_relative_to(ROOT) and sha(common) == identity['common_init_sha256'] == old['identity']['common_init_sha256']
    manifest = packet / f'deployment/image_manifest_seed{seed}.json'
    assert sha(manifest) == identity['image_manifest_sha256']
    return packet


def native_endpoint(path, annotations):
    result_module = pure_module('stem12_results_reader', 'results.py')
    summary, flags, manifest = result_module.load_endpoint(path)
    assert summary['mode'] == manifest['mode'] == 'native_runtime'
    assert summary['epoch_zero_based'] == 11 and summary['epoch_one_based'] == 12
    assert summary['evaluated_weights'] == 'EMA' and summary['GT_count'] == 38759
    # The reused BASE packet keeps its historical exporter identity; the new
    # candidate/reload exporters are bound separately by validate_identity.
    assert manifest['capture_code_sha256'] == summary['identity']['implementation']['results.py']
    gt = {int(x['id']): x for x in annotations['annotations']}
    assert len(gt) == len(annotations['annotations']) == 38759
    ids = sorted(gt)
    assert np.array_equal(flags['annotation_id'], ids)
    assert np.array_equal(flags['category_id'], [gt[g]['category_id'] for g in ids])
    assert flags['hits'].shape == (38759, 10) and flags['hits'].dtype == np.dtype(bool)
    p = summary['protocol']
    assert p['iou_type'] == 'bbox' and p['max_dets'] == [1, 10, 100]
    assert p['area_ranges'] == [[0, 1e10], [0, 1024], [1024, 9216], [9216, 1e10]]
    assert p['area_labels'] == ['all', 'small', 'medium', 'large']
    assert p['category_ids'] == list(range(10)) and p['use_categories'] == 1
    assert p['image_ids'] == sorted(int(x['id']) for x in annotations['images']) == list(range(548))
    assert np.allclose(p['iou_thresholds'], np.linspace(.5, .95, 10), atol=1e-12, rtol=0)
    assert np.allclose(p['recall_thresholds'], np.linspace(0, 1, 101), atol=1e-12, rtol=0)
    assert [x['class_id'] for x in summary['categories_percent']] == list(range(10))
    with np.load(path / 'precision_recall.npz', allow_pickle=False) as z:
        precision, recall = z['precision'], z['recall']
    assert precision.shape == (10, 101, 10, 4, 3) and recall.shape == (10, 10, 4, 3)
    for v in (precision, recall):
        assert np.isfinite(v).all() and ((v >= -1) & (v <= 1)).all()
    def mean(v):
        good = v[v > -1]
        return float(good.mean()) if good.size else -1.
    stats = [mean(precision[:, :, :, 0, 2]), mean(precision[0, :, :, 0, 2]), mean(precision[5, :, :, 0, 2])]
    stats += [mean(precision[:, :, :, a, 2]) for a in (1, 2, 3)]
    stats += [mean(recall[:, :, 0, m]) for m in (0, 1, 2)]
    stats += [mean(recall[:, :, a, 2]) for a in (1, 2, 3)]
    assert np.allclose(stats, summary['native_stats'], atol=1e-12, rtol=0)
    metrics = ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl')
    for k, v in zip(metrics, stats[:6]):
        assert abs(100*v - summary['overall_percent'][k]) <= 1e-9
    assert abs(100*mean(precision[5, :, :, 1, 2]) - summary['overall_percent']['small_AP75']) <= 1e-9
    for c in summary['categories_percent']:
        k = c['class_id']
        values = [mean(precision[:, :, k, 0, 2]), mean(precision[0, :, k, 0, 2]), mean(precision[5, :, k, 0, 2])]
        values += [mean(precision[:, :, k, a, 2]) for a in (1, 2, 3)]
        for field, v in zip(metrics, values):
            assert abs(100*v - c[field]) <= 1e-9
    index = {g: i for i, g in enumerate(ids)}
    hits, fp, seen, cells = np.zeros_like(flags['hits']), np.zeros(10, np.int64), set(), set()
    with gzip.open(path / 'matching_all_area.json.gz', 'rt', encoding='utf8') as f:
        records = json.load(f)
    for row in records:
        assert row['aRng'] == [0, 1e10] and row['maxDet'] == 100 and len(row['dtIds']) <= 100
        cell = (row['image_id'], row['category_id'])
        assert cell not in cells and cell[0] in p['image_ids'] and cell[1] in p['category_ids']
        cells.add(cell)
        assert len(row['gtIgnore']) == len(row['gtIds']) and not any(row['gtIgnore'])
        for g in row['gtIds']:
            assert g in gt and g not in seen
            assert (gt[g]['image_id'], gt[g]['category_id']) == cell
            seen.add(g)
        match, ignore = np.asarray(row['dtMatches']), np.asarray(row['dtIgnore'], dtype=bool)
        assert match.shape == ignore.shape == (10, len(row['dtIds']))
        assert np.isfinite(match).all() and (match == np.floor(match)).all()
        for t in range(10):
            fp[t] += int(((match[t] == 0) & ~ignore[t]).sum())
            for g in match[t][(match[t] != 0) & ~ignore[t]].astype(np.int64):
                assert int(g) in row['gtIds'] and not hits[index[int(g)], t]
                hits[index[int(g)], t] = True
    assert seen == set(gt) and np.array_equal(hits, flags['hits'])
    assert hits.sum(0).tolist() == summary['TP_by_IoU'] and fp.tolist() == summary['FP_by_IoU']
    return summary, flags, dict(native_PR12_reconstructed=True, native_GT_flags_and_FP_reconstructed=True,
        GT=len(gt), matching_cells=len(cells))


def config_check(receipt, plan, destination):
    from pathlib import PurePosixPath
    cfg=deepcopy(plan['historical_resolved_config'])
    assert cfg['model']=='ResidualGeometryDEIM' and cfg['ResidualGeometryDEIM']['arm']=='baseline'
    assert cfg['epoches']==132 and cfg['flat_epoch']==64 and cfg['no_aug_epoch']==12
    assert cfg['train_dataloader']['total_batch_size']==4
    assert cfg['train_dataloader']['dataset']['transforms']['policy']['epoch']==[4,64,120]
    assert cfg['seed']==plan['seed'] and cfg['use_amp'] and cfg['use_ema']
    assert not cfg.get('use_amp_fallback_fp32',False)
    monitor=cfg.get('baseline_log_monitor')
    assert monitor is None or (isinstance(monitor,dict) and monitor.get('enabled') is False)
    runtime=pure_module('lowrank12_pure_configuration','runtime.py')
    expected=runtime.resolved_config(plan,{k:PurePosixPath(v) for k,v in receipt['paths'].items()})
    assert read(destination/'bundle/resolved.yml')==expected
    assert read(destination/'run/args.json')['yaml_cfg']==expected


def stages(folder, identity, annotations):
    path = folder / 'stages_epoch12'
    m, s = read(path / 'manifest.json'), read(path / 'summary.json')
    assert m['mode'] == s['mode'] == 'native_runtime'
    assert m['trace_code_sha256'] == sha(CORE / 'stage_trace.py')
    assert s['identity'] == identity and s['evaluated_weights'] == 'EMA' and s['epoch_zero_based'] == 11
    assert s['images'] == 548 and s['image_ids'] == list(range(548)) and s['query_count'] == 300
    assert s['native_target_orig_size_order'] == 'width_height' and s['stored_original_hw_order'] == 'height_width'
    assert s['additional_model_or_head_calls'] == s['additional_IoU_or_matcher_calls'] == 0
    assert s['cross_run_query_identity_assumed'] is False and s['model_decoder_final_equal'] is True
    expected = {'summary.json'} | {f'image_{i:06d}.npz' for i in range(548)}
    assert set(m['files']) == expected
    image = {int(x['id']): x for x in annotations['images']}
    fields = dict(encoder_initial_boxes=(300, 4), pre_after_first_layer_boxes=(300, 4),
        pre_after_first_layer_logits=(300, 10), final_boxes=(300, 4), final_logits=(300, 10), original_hw=(2,))
    for name, h in m['files'].items():
        assert sha(path / name) == h, name
    for i in range(548):
        with np.load(path / f'image_{i:06d}.npz', allow_pickle=False) as z:
            assert set(z.files) == set(fields)
            for k, shape in fields.items():
                assert z[k].shape == shape and np.isfinite(z[k]).all()
                if k != 'original_hw':
                    assert z[k].dtype == np.dtype('float32')
            assert np.array_equal(z['original_hw'], [image[i]['height'], image[i]['width']])
    return s


def training_snapshots(run):
    expected={(e,a) for e in (0,4,8,11) for a in (1,101,501,1001,1501)}
    records=rows(run/'sampled_training_target_sizes.jsonl')
    assert len(records)==20
    seen=set()
    for r in records:
        key=(r['epoch_zero_based'],r['epoch_attempt'])
        assert key in expected and key not in seen;seen.add(key)
        assert r['samples_shape'][:2]==[4,3]
        assert len(r['target_histogram'])==6 and all(type(n) is int and n>=0 for n in r['target_histogram'])
        assert sum(r['target_histogram'])==r['target_exposures']
        assert r['bands']==['under2','2_to_under4','4_to_under8','8_to_under16','16_to_under32','ge32']
    assert seen==expected
    return dict(scheduled_size_histograms=20,structural_count_checks_passed=True,
        scope='Passive target-size counts, not evidence for a causal mechanism.')


def verify(archive, metadata, destination, report):
    destination, report = Path(destination).resolve(), Path(report).resolve()
    assert not report.exists()
    protocol = local_protocol()
    meta = read(metadata)
    manifest = extract(archive, meta, destination)
    receipt, plan = read(destination / 'bundle/receipt.json'), read(destination / 'bundle/plan.json')
    identity = receipt['identity']
    validate_identity(identity, plan, protocol)
    assert manifest['identity'] == identity and identity['seed'] == meta['seed']
    assert sha(destination / 'bundle/plan.json') == identity['plan_sha256']
    assert sha(destination / 'bundle/resolved.yml') == receipt['resolved_sha256']
    for name, h in protocol['core_files_sha256'].items():
        assert sha(destination / 'code' / name) == h, name
    old = reuse_sources(identity, protocol)
    native = read(CORE / 'training_protocol.json')
    reference = native['references'][str(identity['seed'])]
    for k in ('train_ann_sha256', 'val_ann_sha256', 'common_init_sha256', 'baseline_checkpoint_sha256',
        'baseline_first100_sha256', 'image_manifest_sha256', 'evaluator_binding'):
        assert identity[k] == reference[k], k
    assert identity['train_annotation_sha256'] == identity['train_ann_sha256']
    assert identity['val_annotation_sha256'] == identity['val_ann_sha256']
    for key in ('image_manifest', 'image_verification'):
        assert sha(destination / f'bundle/{key}.json') == identity[key+'_sha256']
    images = read(destination / 'bundle/image_manifest.json')
    image_check = read(destination / 'bundle/image_verification.json')
    assert (len(images['train']), len(images['val'])) == (6471, 548)
    assert image_check['status'] == 'ALL_CURRENT_IMAGES_MATCH_FROZEN_DATASET'
    assert image_check['files_verified'] == 7019 and image_check['image_manifest_sha256'] == identity['image_manifest_sha256']
    assert image_check['new_model_forwards'] == image_check['new_training_updates'] == 0
    reuse = read(destination / 'screen/baseline_reuse.json')
    assert reuse['status'] == 'REUSE_VERIFIED_NATIVE_BASE12'
    assert reuse['new_model_forwards'] == reuse['new_optimizer_steps'] == reuse['new_epochs'] == 0
    assert reuse['original_source'] == native['baseline_reuse'][str(identity['seed'])]['source']
    for name, h in native['baseline_reuse'][str(identity['seed'])]['pinned_files'].items():
        assert sha(destination / 'baseline_eval' / name) == h == sha(old / 'baseline_eval' / name), name
    baseline_done = read(destination / 'baseline_eval/completion.json')
    assert baseline_done['checkpoint_sha256'] == identity['baseline_checkpoint_sha256']
    assert baseline_done['images'] == 548 and baseline_done['optimizer_steps'] == 0
    assert baseline_done['historical_log_reproduction_pass'] and baseline_done['all_parameters_frozen_like_native_EMA']
    actual, baseline_inputs = rows(destination / 'run/first100_training_inputs.jsonl'), rows(old / 'reference/baseline_first100.jsonl')
    assert len(actual) == len(baseline_inputs) == 100 and actual == baseline_inputs
    assert read(destination / 'run/first100_parity.json')['all100_rows_exact'] is True
    run, replay, screen = destination / 'run', destination / 'candidate_reload_v1', destination / 'screen'
    for name in ('input_parity_failure.json', 'controller_failure.json'):
        assert not (run / name).exists() and not (screen / name).exists()
    done, runtime = read(run / 'training_completion.json'), read(run / 'runtime_start.json')
    assert done['status'] == 'NATIVE_FIT_FINISHED_EXACT12'
    assert done['identity'] == runtime['identity'] == identity
    assert done['completed_epochs'] == 12 and done['entered_epochs'] == list(range(12)) and done['automatic_retry'] is False
    assert sha(run / 'last.pth') == done['last_checkpoint_sha256']
    init=runtime['initialization']
    assert init['parameters']==10238173 and init['extra_parameters']==6176
    assert init['exact_shared_tensors']==794 and init['state_tensors']==802
    assert init['new_state_tensors']==8 and init['removed_native_tensors']==0
    assert init['removed_common_prefixes']==[] and init['new_state_prefixes']==['backbone.p2_skip.']
    assert init['user_arm']=='p2_lowrank16' and init['common_init_sha256']==identity['common_init_sha256']
    assert init['strict_load'] and init['all_shared_tensors_exact'] and init['new_state_unchanged']
    assert init==read(destination/'bundle/cpu_initialization.json')['initialization']
    assert runtime['optimizer_coverage']==dict(trainable_tensors=493,all_native_parameter_groups_unchanged=True,
        all_parameters_exactly_once=True,added_parameters=6176)
    assert runtime['user_source']==read(CORE/'user_sources.json')
    assert runtime['added_parameters']==6176 and 'matching_cost' not in runtime
    preflight_binding = protocol['preflight_sidecars'][str(identity['seed'])]
    preflight_path = (ROOT / preflight_binding['relative_path']).resolve()
    assert preflight_path.is_relative_to(ROOT)
    assert sha(preflight_path) == preflight_binding['sha256']
    assert preflight_path.stat().st_size == preflight_binding['bytes']
    capture_path = (ROOT / preflight_binding['source_capture_relative_path']).resolve()
    assert capture_path.is_relative_to(ROOT) and sha(capture_path) == preflight_binding['source_capture_sha256']
    capture = read(capture_path)['turn7_lowrank_local_reader_actual_runtime_inputs_seed'+str(identity['seed'])]
    assert hashlib.sha256(capture['payload'].encode()).hexdigest() == capture['transport']['sha256']
    captured_amp = json.loads(capture['payload'])['files']['AMP']
    assert captured_amp['sha256'] == preflight_binding['sha256'] and captured_amp['report'] == read(preflight_path)
    preflight=read(preflight_path)
    assert preflight['status']=='PASS_LOW_RANK_CONFIG_FULL_AMP_LOSS_BACKWARD_NO_UPDATES'
    assert preflight['identity']==identity and preflight['initialization']==init
    assert preflight['optimizer_coverage']==runtime['optimizer_coverage']
    assert preflight['candidate_detector_forwards']==2 and preflight['BASE_detector_forwards']==0
    assert preflight['optimizer_updates']==preflight['training_epochs']==preflight['evaluator_calls']==0
    assert preflight['states_discarded'] and [x['side'] for x in preflight['cases']]==[640,832]
    assert all(x['finite_gradients'] and all(x['branch'][k] for k in ('zero_restore_maintained','restore_gradient_active','upstream_bridge_gradients_initially_zero')) for x in preflight['cases'])
    assert preflight['preflight_source_sha256']==sha(CORE/'preflight.py')
    launch=read(screen/'detector_preflight_dispatch.json');completion=read(screen/'detector_preflight_completion.json')
    prefix={1:'/25085400120',2:'/24085403014'}[identity['seed']]
    python=prefix+('/envs/deim/bin/python' if identity['seed']==1 else '/deim_migration_runtime/envs/deim/bin/python')
    assert launch['command']==[python,'-B',prefix+'/DEIM/experiments/p2_lowrank12_20261011_v3/code/preflight.py','--bundle',receipt['paths']['screen_root']+'/bundle']
    assert completion['exit_code']==completion['training_epochs']==completion['optimizer_updates']==0
    assert runtime['resume'] is None and runtime['automatic_retry'] is False
    assert runtime['native_augmentation_and_supervision_weights_unchanged'] is True
    assert runtime['evaluator_runtime']['binding'] == identity['evaluator_binding']
    config_check(receipt, plan, destination)
    audit = rows(run / 'training_audit_epochs.jsonl')
    assert len(audit) == 12 and [x['epoch_zero_based'] for x in audit] == list(range(12))
    previous = 0
    for e, row in enumerate(audit):
        assert row['attempts_total'] == row['EMA_updates'] == 1617*(e+1)
        assert previous <= row['actual_optimizer_steps_total'] <= row['attempts_total']
        assert row['attempts_without_optimizer_step'] == row['attempts_total'] - row['actual_optimizer_steps_total']
        assert row['user_backbone_arm']=='p2_lowrank16' and row['candidate_active_from_epoch_zero'] is True
        assert row['no_new_model_parameters'] is False and row['added_parameters']==6176
        previous = row['actual_optimizer_steps_total']
    snapshots = training_snapshots(run)
    log = rows(run / 'log.txt')
    assert len(log) == 12 and [int(x['epoch']) for x in log] == list(range(12))
    train_exit, controller = read(screen / 'candidate_completion.json'), read(screen / 'controller_completion.json')
    assert train_exit['exit_code'] == 0 and train_exit['timed_out'] is False and train_exit['automatic_retry'] is False
    assert controller['status'] == 'COMPLETE_FIXED12' and controller['automatic_retry'] is False
    assert controller['automatic_extension'] is False and controller['automatic_control_launch'] is False
    replay_exit, proof = read(replay / 'completion.json'), read(replay / 'verification.json')
    assert replay_exit['exit_code'] == 0 and replay_exit['timed_out'] is False and replay_exit['automatic_retry'] is False
    assert replay_exit['baseline_model_forwards'] == replay_exit['epochs_added'] == 0
    closeout = read(CORE / 'closeout_protocol.json')
    assert closeout['schema'] == 'p2_lowrank12_closeout_v1' and closeout['stats_fraction_tolerance'] == 0.0
    assert proof['status'] == 'PASS_SAVED_CANDIDATE_RELOAD' and proof['identity'] == identity
    assert proof['checkpoint_sha256'] == done['last_checkpoint_sha256']
    assert proof['original_training_completion_sha256'] == sha(run / 'training_completion.json')
    assert proof['strict_loaded_tensors'] == 802 and proof['parameters'] == 10238173
    assert proof['all_parameters_frozen_like_native_EMA'] is True
    assert proof['verification_script_sha256'] == closeout['files']['verify_candidate_reload.py']
    assert proof['closeout_protocol_sha256'] == sha(CORE / 'closeout_protocol.json')
    assert proof['candidate_model_forwards'] == 137 and proof['images'] == 548
    assert proof['baseline_model_forwards'] == proof['optimizer_updates'] == proof['epochs_added'] == 0
    assert proof['automatic_retry'] is False and proof['native_GT_hits_exact'] is True and proof['all548_stage_arrays_exact'] is True
    assert proof['stats_tolerance'] == proof['stats_max_abs_fraction_difference'] == 0.0
    assert all(v == 0.0 for v in proof['stage_max_abs_difference'].values())
    annotations = read(old / 'reference/val_annotations.json')
    base, bh, bc = native_endpoint(destination / 'baseline_eval/endpoint_epoch12', annotations)
    cand, ch, cc = native_endpoint(run / 'endpoint_epoch12', annotations)
    loaded, lh, lc = native_endpoint(replay / 'endpoint_epoch12', annotations)
    evaluator = pure_module('stem12_evaluator_reader', 'evaluator_binding.py')
    evaluator.same_result_evaluator(base, cand)
    evaluator.same_result_evaluator(cand, loaded)
    assert base['identity']['arm'] == 'baseline' and base['identity']['seed'] == identity['seed']
    for key in ('common_init_sha256', 'baseline_checkpoint_sha256', 'baseline_first100_sha256',
        'val_annotation_sha256', 'evaluator_binding'):
        assert base['identity'][key] == identity[key], key
    assert cand['identity'] == loaded['identity'] == identity
    assert base['native_stats'] == plan['historical_baseline_stats']
    assert log[-1]['test_coco_eval_bbox'] == cand['native_stats']
    assert cand['native_stats'] == loaded['native_stats'] == proof['original_stats'] == proof['reloaded_stats']
    assert all(np.array_equal(ch[k], lh[k]) for k in ch)
    assert sha(run / 'endpoint_epoch12/manifest.json') == done['endpoint_manifest_sha256']
    assert sha(run / 'stages_epoch12/manifest.json') == done['stage_manifest_sha256']
    for folder, ident in ((destination / 'baseline_eval', base['identity']), (run, identity), (replay, identity)):
        stages(folder, ident, annotations)
    for i in range(548):
        name = f'image_{i:06d}.npz'
        with np.load(run / 'stages_epoch12' / name, allow_pickle=False) as a, np.load(replay / 'stages_epoch12' / name, allow_pickle=False) as b:
            assert set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
    comparison = pure_module('stem12_pair_reader', 'results.py').pair(base, cand, bh, ch)
    saved = read(run / 'fixed12_comparison.json')
    for k, v in comparison.items():
        assert saved[k] == v, k
    delta = comparison['delta_AP_metrics_pp']
    passed = delta['APs'] > 0 and delta['AP'] >= 0
    assert saved['seed'] == identity['seed'] and saved['arm'] == 'p2_lowrank16'
    assert saved['native_gate_passed'] == done['native_gate_passed'] == passed
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    result = dict(status=PASS, identity=identity, fixed_endpoint=dict(epochs=12, zero_based_epoch=11, weights='EMA'),
        archive_sha256=sha(archive), metadata_sha256=sha(metadata), terminal_manifest_sha256=meta['manifest_sha256'],
        files_verified=len(manifest['files'])+1, native_source_files_reused_and_verified=468, evaluator_files_reused_and_verified=7,
        baseline_source_packet=str(old), baseline_source_report_sha256=protocol['reused_local_evidence'][str(identity['seed'])]['report_sha256'],
        baseline_overall_percent=base['overall_percent'], candidate_overall_percent=cand['overall_percent'],
        delta_AP_metrics_pp=delta, native_gate_passed_this_seed=passed, paired_native_results=comparison,
        endpoint_checks=dict(baseline=bc, candidate=cc, reloaded=lc), all548_stage_arrays_reloaded_exact=True,
        first100_inputs_and_pre_forward_RNG_exact=True, full_training_trajectory_equivalence_not_claimed=True,
        actual_training_snapshots=snapshots,
        final_training_counts=audit[-1], training_elapsed_seconds=train_exit['elapsed_seconds'],
        new_model_forwards=0, new_evaluator_calls=0, new_optimizer_updates=0, new_training_epochs=0,
        verification_script_sha256=sha(__file__), local_reader_protocol_sha256=sha(PROTOCOL),
        preflight_evidence_path=str(preflight_path), preflight_evidence_sha256=sha(preflight_path),
        claim_boundary='One-seed fixed12 result only. No novelty, causal root, significance or mature-model improvement claim. Both-seed gate remains required.')
    write(report, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('archive', 'metadata', 'destination', 'report'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    value = verify(a.archive, a.metadata, a.destination, a.report)
    print(json.dumps(dict(status=value['status'], seed=value['identity']['seed'],
        delta_AP_metrics_pp=value['delta_AP_metrics_pp'], native_gate_passed_this_seed=value['native_gate_passed_this_seed'])), flush=True)
