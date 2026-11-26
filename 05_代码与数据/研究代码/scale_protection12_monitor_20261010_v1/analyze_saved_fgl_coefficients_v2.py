"""FGL IoU coefficients from saved union-regression pairs, not MAL assignments."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path.cwd()
INPUT = ROOT / 'outputs/native_augmented_supervision_20261010_v1'
OUT = ROOT / 'outputs/saved_fgl_coefficient_analysis_20261010_v2'
CURRENT = ROOT / 'outputs/scale_protection12_20261010_v1/terminal_collection/seed1_target_protect_v1/extracted'
read = lambda p: json.loads(Path(p).read_text(encoding='utf8'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def aligned_iou(a, b, dtype):
    a, b = np.asarray(a, dtype=dtype), np.asarray(b, dtype=dtype)
    aa = np.concatenate((a[:, :2] - a[:, 2:] * dtype(.5), a[:, :2] + a[:, 2:] * dtype(.5)), axis=1)
    bb = np.concatenate((b[:, :2] - b[:, 2:] * dtype(.5), b[:, :2] + b[:, 2:] * dtype(.5)), axis=1)
    wh = np.maximum(dtype(0), np.minimum(aa[:, 2:], bb[:, 2:]) - np.maximum(aa[:, :2], bb[:, :2]))
    intersection = wh[:, 0] * wh[:, 1]
    area_a = (aa[:, 2] - aa[:, 0]) * (aa[:, 3] - aa[:, 1])
    area_b = (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])
    return intersection / (area_a + area_b - intersection)


def band(shortside):
    return np.select([shortside < 4, shortside < 8, shortside < 16], ['under4', '4_to_under8', '8_to_under16'], default='ge16')


reference_gt = np.array([[0, 0, 2, 2]] * 3, dtype=np.float32)
reference_dt = np.array([[0, 0, 2, 2], [5, 5, 2, 2], [0, 0, 1, 1]], dtype=np.float32)
assert np.array_equal(aligned_iou(reference_dt, reference_gt, np.float32), np.array([1, 0, .25], dtype=np.float32))
download = read(INPUT / 'download_verification.json')
assert download['status'] == 'PASS_DUAL_SEED_ARCHIVES_AND_ALL_CAPTURE_FILES'
assert all(row['completion']['observer_plain_outputs_losses_BN_RNG_exact'] for row in download['seeds'])
criterion = CURRENT / 'native/engine/deim/deim_criterion.py'
box_ops = CURRENT / 'native/engine/deim/box_ops.py'
config = read(CURRENT / 'prepared/resolved.yml')['DEIMCriterion']
assert config['weight_dict']['loss_fgl'] == .15 and config.get('boxes_weight_format') is None
assert config['gamma'] == 1.5 and config['losses'] == ['mal', 'boxes', 'local']
OUT.mkdir(exist_ok=False)
groups = defaultdict(list)
source_files, pair_arrays, call_count, numeric_max_error, zero_disagreement = {}, [], 0, 0., 0
stage_codes, unavailable_calls = {}, []
for seed in (1, 2):
    root = INPUT / f'collection_seed{seed}/capture_v1'
    original_manifest = read(root / 'manifest.json')
    identity_path = root / 'source_identity.json'
    assert sha(identity_path) == original_manifest['files']['source_identity.json']
    identity = read(identity_path)
    assert identity['before_and_after_equal']
    assert identity['native_files_sha256']['engine/deim/deim_criterion.py'] == sha(criterion)
    assert identity['native_files_sha256']['engine/deim/box_ops.py'] == sha(box_ops)
    source_files[str(identity_path.relative_to(ROOT))] = sha(identity_path)
    for endpoint in (12, 132):
        batches = sorted((root / f'endpoint{endpoint}').glob('batch_*'))
        assert len(batches) == 16
        for batch_index, folder in enumerate(batches):
            capture_manifest = read(folder / 'manifest.json')
            for name, digest in capture_manifest['files'].items():
                assert sha(folder / name) == digest
                source_files[str((folder / name).relative_to(ROOT))] = digest
            evidence = read(folder / 'evidence.json')
            metadata = evidence['metadata']
            assert metadata['endpoint'] == endpoint and metadata['seed'] == seed and metadata['batch'] == batch_index
            assert metadata['dtype'] == 'CUDA_FP32' and metadata['AMP'] is False and metadata['backward_executed'] is False
            assert not evidence['errors'] and evidence['actual_optimizer_steps_in_sample'] == 0
            _, _, height, width = metadata['input_shape']
            with np.load(folder / 'arrays.npz', allow_pickle=False) as arrays:
                for event in evidence['events']:
                    if event.get('kind') != 'loss_call' or event.get('loss') != 'local' or 'loss_fgl' not in event['native_unweighted_loss_scalars']:
                        continue
                    stage, prefix = event['stage'], event['prefix']
                    if stage + '.pred_boxes' not in arrays.files:
                        assert event['is_DN'], 'Missing ordinary stage boxes; preserve failure'
                        unavailable_calls.append(dict(seed=seed, endpoint=endpoint, batch=batch_index,
                            stage=stage, reason='DN predicted boxes were not saved in this existing capture',
                            matched_pair_exposures=sum(row['pairs'] for row in event['batches'])))
                        continue
                    assert not event['is_DN']
                    call_count += 1
                    stage_codes.setdefault(stage, len(stage_codes))
                    normalizer = float(event['num_boxes_normalizer'])
                    assert normalizer > 0 and event['weight_dict']['loss_fgl'] == .15
                    for batch in event['batches']:
                        slot = batch['batch']
                        key = prefix + f'.batch{slot}'
                        gt_rows = arrays[key + '.GT_row_index']
                        query_rows = arrays[key + '.query_index']
                        pred = arrays[stage + '.pred_boxes'][slot, query_rows]
                        gt = arrays[f'target.{slot}.boxes'][gt_rows]
                        source_ids = arrays[f'target.{slot}.verified_source_annotation_ids'][gt_rows]
                        assert pred.dtype == gt.dtype == np.float32 and pred.shape == gt.shape == (batch['pairs'], 4)
                        assert np.array_equal(gt, arrays[f'target.{slot}.boxes'][gt_rows])
                        assert len(query_rows) == len(gt_rows) == len(source_ids)
                        q = aligned_iou(pred, gt, np.float32)
                        q64 = aligned_iou(pred, gt, np.float64)
                        assert np.isfinite(q).all() and (q >= 0).all() and (q <= 1 + 1e-6).all()
                        if len(q):
                            numeric_max_error = max(numeric_max_error, float(np.max(np.abs(q.astype(np.float64) - q64))))
                            zero_disagreement += int(((q == 0) != (q64 == 0)).sum())
                        short = (gt[:, 2:] * np.array([width, height], dtype=np.float64)).min(1)
                        assert np.isfinite(short).all() and (short > 0).all()
                        tags = band(short)
                        per_edge_coefficient = .15 * q.astype(np.float64) / normalizer
                        for size in ('under4', '4_to_under8', '8_to_under16', 'ge16'):
                            mask = tags == size
                            group = (seed, endpoint, stage, bool(event['is_DN']), batch['GT_rows'] <= 300, size)
                            for pos in np.flatnonzero(mask):
                                groups[group].append((float(q[pos]), float(per_edge_coefficient[pos]), batch_index, slot, int(gt_rows[pos])))
                        n = len(q)
                        pair_arrays.append(dict(seed=np.full(n, seed, dtype=np.int16), endpoint=np.full(n, endpoint, dtype=np.int16),
                            stage_code=np.full(n, stage_codes[stage], dtype=np.int16), batch=np.full(n, batch_index, dtype=np.int16),
                            slot=np.full(n, slot, dtype=np.int8), is_DN=np.full(n, event['is_DN'], dtype=bool),
                            image_GT_le300=np.full(n, batch['GT_rows'] <= 300, dtype=bool), query_row=query_rows, GT_row=gt_rows,
                            source_annotation_id=source_ids, actual_input_shortside=short, recomputed_FP32_IoU=q,
                            native_loss_normalizer=np.full(n, normalizer), FGL_per_edge_direct_coefficient=per_edge_coefficient))
    print(json.dumps(dict(progress='SAVED_COEFFICIENTS_READ', seed=seed, component_calls=call_count)), flush=True)
assert numeric_max_error <= 1e-4, 'Inspect numerical source reconstruction; not an efficacy gate'
summaries = []
for key, values in sorted(groups.items()):
    q = np.array([row[0] for row in values])
    coefficients = np.array([row[1] for row in values])
    identities = {(row[2], row[3], row[4]) for row in values}
    seed, endpoint, stage, dn, le300, size = key
    summaries.append(dict(seed=seed, endpoint=endpoint, stage=stage, is_DN=dn, image_GT_le300=le300, input_size_band=size,
        matched_loss_pair_exposures=len(q), distinct_augmented_GT_rows_in_this_stage=len(identities),
        mean_recomputed_FP32_IoU=float(q.mean()), median_recomputed_FP32_IoU=float(np.median(q)),
        zero_IoU_pairs=int((q == 0).sum()), IoU_lt_point1_pairs=int((q < .1).sum()),
        IoU_lt_point25_pairs=int((q < .25).sum()), IoU_ge_point75_pairs=int((q >= .75).sum()),
        sum_four_edge_direct_FGL_coefficients=float(4 * coefficients.sum()),
        bound_sum_abs_FGL_leaf_logits_gradients=float(8 * coefficients.sum()),
        actual_model_gradient_norm_not_measured=True))
data = {name: np.concatenate([row[name] for row in pair_arrays]) for name in pair_arrays[0]}
np.savez_compressed(OUT / 'per_FGL_loss_pair.npz', **data)
with (OUT / 'coefficient_groups.csv').open('x', encoding='utf-8-sig', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
    writer.writeheader()
    writer.writerows(summaries)
main = [row for row in summaries if row['stage'] == 'model' and row['image_GT_le300'] and row['input_size_band'] == 'under4']
report = dict(status='SAVED_UNION_REGRESSION_FGL_QUALITY_COEFFICIENTS_DESCRIBED',
    source_files=source_files, current_criterion_sha256=sha(criterion), current_box_ops_sha256=sha(box_ops),
    current_config_sha256=sha(CURRENT / 'prepared/resolved.yml'), stage_codes=stage_codes,
    actual_saved_FGL_loss_calls_described=call_count, matched_loss_pair_exposures=len(data['seed']), groups=summaries,
    DN_FGL_calls_not_described_missing_saved_predictions=unavailable_calls,
    DN_FGL_quality_not_inferred=True,
    final_ordinary_GTle300_under4=main,
    coefficient_formula='For each of four FGL edges: 0.15 * detach(aligned_box_IoU) / actual_native_loss_normalizer.',
    mathematical_gradient_bound='Direct FGL softmax-CE leaf-logit L1 norm <= 2 * coefficient per edge. This is an upper bound, not a measured model gradient.',
    mathematical_zero_statement='If recomputed IoU is exactly zero, the detached IoU-weighted direct FGL component has zero leaf-logit gradient. L1, GIoU, classification, other stages and DN remain separate.',
    numeric_FP32_vs_independent_FP64_max_abs_difference=numeric_max_error,
    numeric_zero_classification_disagreements=zero_disagreement,
    scope='Existing 64 CUDA-FP32 frozen-EMA augmented batches; endpoints12/132 separated, two seeds, actual epoch4 augmentation. Not current target/random optimization history, AMP gradient evidence or validation population.',
    actual_regression_union_pairs_used_instead_of_MAL_assignment_pairs=True,
    predicted_boxes_gathered_from_saved_same_stage_outputs_using_local_loss_actual_query_indices=True,
    actual_FGL_component_losses_or_gradients_not_recalculated=True,
    duplicated_DN_and_cross_stage_loss_exposures_not_claimed_unique_GT=True,
    all_model_gradient_zero_or_FGL_root_cause_not_claimed=True,
    no_loss_bug_or_efficacy_or_innovation_claim=True,
    model_forwards=0, model_backwards=0, evaluator_calls=0, optimizer_updates=0, training_epochs_added=0,
    current_random_training_protocol_and_gates_unchanged=True)
assert 'torch' not in sys.modules
write(OUT / 'report.json', report)
write(OUT / 'manifest.json', dict(files={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in OUT.iterdir() if p.is_file()}, analysis_source_sha256=sha(__file__)))
print(json.dumps(dict(status=report['status'], ordinary_final_under4_GTle300=main,
    pair_exposures=len(data['seed']), component_calls=call_count, numerical_max_error=numeric_max_error,
    manifest_sha256=sha(OUT / 'manifest.json')), ensure_ascii=False), flush=True)
