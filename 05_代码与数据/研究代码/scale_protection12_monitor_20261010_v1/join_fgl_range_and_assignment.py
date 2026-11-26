"""Existing BASE12 union pairs: FDR range and same-query MAL GT agreement."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path.cwd()
P = ROOT / 'outputs/saved_fgl_coefficient_analysis_20261010_v2'
INPUT = ROOT / 'outputs/native_augmented_supervision_20261010_v1'
OUT = P / 'range_and_assignment'
CURRENT = ROOT / 'outputs/scale_protection12_20261010_v1/terminal_collection/seed1_target_protect_v1/extracted'
read = lambda p: json.loads(Path(p).read_text(encoding='utf8'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


report = read(P / 'report.json')
manifest = read(P / 'manifest.json')
assert all(sha(P / name) == row['sha256'] for name, row in manifest['files'].items())
with np.load(P / 'per_FGL_loss_pair.npz', allow_pickle=False) as arrays:
    data = {name: arrays[name].copy() for name in arrays.files}
indices = np.flatnonzero(data['endpoint'] == 12)
selected = {name: value[indices] for name, value in data.items()}
codes = {code: stage for stage, code in report['stage_codes'].items()}
count = len(indices)
maximum_distance = np.full(count, np.nan)
mal_same_GT = np.zeros(count, dtype=bool)
mal_query_matched = np.zeros(count, dtype=bool)
sources, roundtrip_error = {}, 0.
for seed in (1, 2):
    root = INPUT / f'collection_seed{seed}'
    capture = root / 'capture_v1'
    identity = read(capture / 'source_identity.json')
    for relative in ('engine/deim/dfine_utils.py', 'engine/deim/dfine_decoder.py'):
        assert identity['native_files_sha256'][relative] == sha(CURRENT / 'native' / relative)
    protocol = read(root / 'protocol.json')
    code = root / 'code/capture_coverage.py'
    assert sha(code) == protocol['code_sha256']['capture_coverage.py']
    tree = ast.parse(code.read_text(encoding='utf8'))
    constants_node = next(node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'CHECKPOINTS' for target in node.targets))
    checkpoint_ids = ast.literal_eval(constants_node.value)
    constant_path = ROOT / f'outputs/sod_fdr_range_20261010_v1/FDR_constants_seed{seed}.json'
    constants = read(constant_path)
    assert constants['checkpoint_sha256'] == checkpoint_ids[seed][12] and constants['last_epoch'] == 11
    scale, up = constants['constants']['decoder.reg_scale'][0], constants['constants']['decoder.up'][0]
    assert scale == 4 and up == .5
    sources[str(constant_path.relative_to(ROOT))] = sha(constant_path)
    sources[str(code.relative_to(ROOT))] = sha(code)
    for batch in range(16):
        folder = capture / 'endpoint12' / f'batch_{batch:03d}'
        evidence = read(folder / 'evidence.json')
        box_manifest = read(folder / 'manifest.json')
        assert sha(folder / 'arrays.npz') == box_manifest['files']['arrays.npz']
        assert sha(folder / 'evidence.json') == box_manifest['files']['evidence.json']
        with np.load(folder / 'arrays.npz', allow_pickle=False) as arrays:
            references = arrays['model.pre_outputs.pred_boxes']
            assert references.shape == (4, 300, 4) and (references[..., 2:] > 0).all()
            for stage_code, stage in codes.items():
                matching = [event for event in evidence['events'] if event.get('kind') == 'loss_call'
                    and event.get('loss') == 'mal' and event['stage'] == stage]
                assert len(matching) == 1
                prefix = matching[0]['prefix']
                for slot in range(4):
                    mask = (selected['seed'] == seed) & (selected['batch'] == batch) & (selected['slot'] == slot) & (selected['stage_code'] == stage_code)
                    positions = np.flatnonzero(mask)
                    if not len(positions):
                        continue
                    query_rows, gt_rows = selected['query_row'][mask], selected['GT_row'][mask]
                    points = references[slot, query_rows].astype(np.float64)
                    gt = arrays[f'target.{slot}.boxes'][gt_rows].astype(np.float64)
                    xyxy = np.concatenate((gt[:, :2] - gt[:, 2:] * .5, gt[:, :2] + gt[:, 2:] * .5), axis=1)
                    cx, cy, w, h = points.T
                    distances = np.stack((scale * (cx - xyxy[:, 0]) / w - scale * .5,
                        scale * (cy - xyxy[:, 1]) / h - scale * .5,
                        scale * (xyxy[:, 2] - cx) / w - scale * .5,
                        scale * (xyxy[:, 3] - cy) / h - scale * .5), axis=1)
                    maximum_distance[positions] = np.abs(distances).max(1)
                    decoded = np.stack((cx - (.5 * scale + distances[:, 0]) * w / scale,
                        cy - (.5 * scale + distances[:, 1]) * h / scale,
                        cx + (.5 * scale + distances[:, 2]) * w / scale,
                        cy + (.5 * scale + distances[:, 3]) * h / scale), axis=1)
                    roundtrip_error = max(roundtrip_error, float(np.abs(decoded - xyxy).max()))
                    mal_q = arrays[prefix + f'.batch{slot}.query_index']
                    mal_g = arrays[prefix + f'.batch{slot}.GT_row_index']
                    assert len(mal_q) == len(np.unique(mal_q))
                    lookup = np.full(300, -1, dtype=np.int64)
                    lookup[mal_q] = mal_g
                    mal_same_GT[positions] = lookup[query_rows] == gt_rows
                    mal_query_matched[positions] = lookup[query_rows] >= 0
assert np.isfinite(maximum_distance).all() and roundtrip_error < 1e-10
selected['maximum_abs_required_edge_distance'] = maximum_distance
selected['strictly_inside_fixed_FDR_support'] = maximum_distance < 4 - 1e-6
selected['in_closed_fixed_FDR_support'] = maximum_distance <= 4
selected['same_query_MAL_and_local_loss_GT_agree'] = mal_same_GT
selected['same_query_has_MAL_match'] = mal_query_matched
OUT.mkdir(exist_ok=False)
np.savez_compressed(OUT / 'BASE12_FGL_pairs_with_range_and_assignment.npz', **selected)
summary = []
for seed in (1, 2):
    for stage_code, stage in codes.items():
        base = (selected['seed'] == seed) & (selected['stage_code'] == stage_code) & selected['image_GT_le300'] & (selected['actual_input_shortside'] < 4)
        zero = base & (selected['recomputed_FP32_IoU'] == 0)
        inside, agree = selected['strictly_inside_fixed_FDR_support'], mal_same_GT
        summary.append(dict(seed=seed, stage=stage, matched_loss_pair_exposures=int(base.sum()),
            zero_IoU_pairs=int(zero.sum()), zero_IoU_strict_range_pairs=int((zero & inside).sum()),
            zero_IoU_strict_range_and_MAL_GT_agree_pairs=int((zero & inside & agree).sum()),
            zero_IoU_outside_closed_range_pairs=int((zero & ~selected['in_closed_fixed_FDR_support']).sum()),
            all_pairs_same_query_MAL_GT_agree=int((base & agree).sum())))
value = dict(status='SAVED_BASE12_FGL_UNION_PAIR_RANGE_AND_ASSIGNMENT_JOINED', groups=summary, sources=sources,
    normalized_box_inverse_roundtrip_max_abs_error=roundtrip_error, matched_loss_pair_exposures=count,
    references='Exact saved model.pre_outputs.pred_boxes; source identifies ref_points_initial=pre_bboxes.detach for every ordinary FDR stage.',
    support='33-bin softmax expectation, actual verified BASE12 reg_scale4/up0.5, support[-4,4]; strict interior margin1e-6.',
    GT_assisted_representability_not_a_deployable_prediction=True,
    FGL_same_stage_actual_union_pairs_not_MAL_pair_substitution=True,
    MAL_agreement_is_actual_same_query_GT_row_identity=True,
    DN_and_BASE132_not_joined_or_inferred=True,
    classification_regression_conflict_root_not_claimed=True,
    nonzero_other_losses_and_model_gradient_paths_preserved_in_interpretation=True,
    loss_floor_is_untrained_hypothesis_not_accepted=True,
    model_forwards=0, backwards=0, evaluator_calls=0, optimizer_updates=0, training_epochs_added=0)
assert 'torch' not in sys.modules
write(OUT / 'report.json', value)
write(OUT / 'manifest.json', dict(files={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in OUT.iterdir() if p.is_file()}, source_sha256=sha(__file__)))
print(json.dumps(dict(status=value['status'], final_ordinary_under4_GTle300=[row for row in summary if row['stage'] == 'model'],
    roundtrip_error=roundtrip_error, manifest_sha256=sha(OUT / 'manifest.json'))), flush=True)
