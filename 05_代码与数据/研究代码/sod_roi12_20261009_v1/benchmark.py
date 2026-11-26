"""Fixed ROI12 end-to-end cost audit, after both immutable epoch12 results.

Exactly the first32 sorted full-validation IDs, batch4, two warmup batches,
three complete repeats. Compressed image bytes are memory-resident; each timed
batch decodes/resizes them again and runs a fresh detector forward. No GT or
cached predictions enter inference. Timing excludes loading weights, disk I/O,
provenance checks, and diagnostic assertions. It is NOT an accuracy evaluation.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def under(root, name):
    path = (root / name).resolve()
    require(root in path.parents, 'Path escapes experiment root: ' + str(name))
    return path


def state_sha(model):
    # Same model digest as train.py, including tensor dtype and shape.
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def verify(root):
    from cache_native import CHECKPOINT_SHA, ANNOTATIONS, verify_sources
    from supervise import validate_lock
    from data import _records

    require(root.name == 'sod_roi12_20261009_v1' and root.parent.name == 'experiments',
            'Only the isolated experiments/sod_roi12_20261009_v1 root is allowed')
    require(Path(__file__).resolve().parent == root / 'code', 'Run the locked root/code script')
    plan_path = under(root, 'metadata/locked_plan.json')
    plan = read(plan_path)
    bindings_path = under(root, 'metadata/source_bindings_proposed.json')
    _, lock_sha, _ = validate_lock(root, bindings_path)
    require(Path(plan['source_bindings']).resolve() == bindings_path, 'Different source bindings')
    require(plan['arms'] == ['raw', 'degraded'] and plan['maximum_epochs'] == 12
            and plan['R_passes'] == 8 and plan['G_passes'] == 4, 'Different paired12 plan')
    require(plan['autocast'] is False and plan['tf32'] is False, 'Head precision differs')
    require(plan['threshold_grid'] == [0, .05, .10, .20, .30, .50, 1], 'Threshold grid differs')
    expected_roi = {'size': 64, 'channels': 4, 'context': 2,
                    'min_extent_original_px': 32, 'eligible_short_side_at640_lt': 16,
                    'max_edge_fraction': .2}
    for key, value in expected_roi.items():
        require(plan['ROI'][key] == value, 'Different ROI contract: ' + key)
    actual = {p.name for p in (root / 'code').glob('*.py')}
    require(actual == set(plan['code_sha256']), 'Plan does not bind the complete code set')
    for name, digest in plan['code_sha256'].items():
        require(Path(name).name == name and sha(under(root, 'code/' + name)) == digest,
                'Code differs from plan: ' + name)
    seeds = [b for b in read(bindings_path)['seeds'] if b['seed'] == 1]
    require(len(seeds) == 1, 'Missing unique seed1 binding')
    binding = seeds[0]
    native = Path(binding['reusable_native_snapshot']['remote']).resolve()
    checkpoint = Path(binding['frozen_detector_proposed']['remote_checkpoint']).resolve()
    source_binding = under(root, 'metadata/training_source_binding.json')
    verify_sources(native, source_binding)
    require(sha(checkpoint) == CHECKPOINT_SHA == binding['frozen_detector_proposed']['sha256'],
            'Wrong formal132 EMA checkpoint')
    manifest_path = under(root, 'cache/val/manifest.json')
    require(Path(plan['val_manifest']).resolve() == manifest_path
            and sha(manifest_path) == plan['val_manifest_sha256'], 'Val manifest changed')
    manifest = read(manifest_path)
    require(manifest['status'] == 'COMPLETE' and not manifest['engineering_subset']
            and manifest['images_cached'] == manifest['dataset_images'] == 548,
            'Requires the full548 native validation cache')
    require(manifest['checkpoint_sha256'] == CHECKPOINT_SHA
            and Path(manifest['checkpoint']).resolve() == checkpoint
            and Path(manifest['training_source_root']).resolve() == native,
            'Native binding differs from val cache')
    require(manifest['source_binding_sha256'] == sha(source_binding), 'Source binding changed')
    config = under(root, 'metadata/baseline_seed1.json')
    require(Path(manifest['input_config']).resolve() == config
            and sha(config) == manifest['input_config_sha256'], 'Native input config changed')
    annotation = Path(binding['data']['val']['ann']).resolve()
    require(Path(manifest['annotation']).resolve() == annotation
            and sha(annotation) == manifest['annotation_sha256'] == ANNOTATIONS['val'][0],
            'Validation annotation hash changed')
    # Hash annotation bytes for provenance only; never parse GT or select by GT.
    records = _records(manifest)
    require(len(records) == 548 and sorted(records) == [r['image_id'] for r in manifest['images']],
            'Full-val native sorted image population changed')
    image_root = Path(binding['data']['val']['images']).resolve()
    require(Path(manifest['image_root']).resolve() == image_root, 'Different original images')
    population = []
    for image_id in sorted(records)[:32]:
        row = records[image_id]
        path = (image_root / row['file_name']).resolve()
        require(image_root in path.parents, 'Image escapes bound directory')
        compressed = path.read_bytes()
        require(hashlib.sha256(compressed).hexdigest() == row['image_sha256'], 'Image hash mismatch')
        population.append({'image_id': image_id, 'file_name': row['file_name'],
                           'width': row['width'], 'height': row['height'],
                           'image_sha256': row['image_sha256'], 'bytes': compressed})
    arm_meta = {}
    for arm in plan['arms']:
        directory = under(root, 'arms/' + arm)
        result = read(directory / 'result.json')
        selection = read(directory / 'evaluation_epoch12/threshold_selection.json')
        require(result['status'] == 'COMPLETE_12_PASSES' and result['epochs_completed'] == 12
                and result['arm'] == arm, 'Arm has no complete epoch12 result: ' + arm)
        tau = result['threshold']
        require(tau in plan['threshold_grid'] and selection['official_val_used'] is False
                and selection['selected']['threshold'] == tau, 'Threshold not selected on C')
        require(read(directory / 'launch.json')['plan_sha256'] == sha(plan_path), 'Arm plan changed')
        require(read(directory / 'plan.json') == plan, 'Arm uses a different locked plan')
        arm_meta[arm] = {'threshold': tau, 'checkpoint_sha256': sha(directory / 'epoch12.pth'),
                         'result_sha256': sha(directory / 'result.json'),
                         'threshold_selection_sha256': sha(directory / 'evaluation_epoch12/threshold_selection.json')}
    return plan, manifest, binding, population, arm_meta, {
        'locked_plan_sha256': sha(plan_path), 'launch_lock_sha256': lock_sha,
        'bindings_sha256': sha(bindings_path), 'native_checkpoint_sha256': CHECKPOINT_SHA,
        'native_source_binding_sha256': sha(source_binding), 'native_bound_files': 468,
        'val_manifest_sha256': sha(manifest_path), 'config_sha256': sha(config),
        'annotation_sha256': manifest['annotation_sha256'], 'code_sha256': plan['code_sha256']}


def set_precision(torch, manifest, head=False):
    torch.backends.cuda.matmul.allow_tf32 = False if head else manifest['tf32_matmul']
    torch.backends.cudnn.allow_tf32 = False if head else manifest['tf32_cudnn']
    torch.backends.cudnn.benchmark = False if head else manifest['cudnn_benchmark']
    torch.backends.cudnn.deterministic = head
    torch.use_deterministic_algorithms(head)


def load_models(root, plan, manifest, binding, mode, torch, device):
    from cache_native import load_recipe
    from engine.core import YAMLConfig
    from refiner import ROIRefiner

    recipe = load_recipe(root / 'metadata/baseline_seed1.json')
    resolved = root / 'cache/val/resolved_eval_config.yaml'
    expected = copy.deepcopy(recipe)
    expected['val_dataloader']['dataset']['ann_file'] = manifest['annotation']
    expected['val_dataloader']['dataset']['img_folder'] = manifest['image_root']
    require(sha(resolved) == manifest['resolved_config_sha256'] and read(resolved) == expected,
            'Resolved native recipe changed')
    cfg = YAMLConfig(str(resolved))
    model = cfg.model.float().eval()
    checkpoint = torch.load(binding['frozen_detector_proposed']['remote_checkpoint'],
                            map_location='cpu', weights_only=True)
    require(checkpoint['last_epoch'] == 131, 'Not complete epoch132')
    model.load_state_dict(checkpoint['ema']['module'], strict=True)
    require(all(torch.equal(value, checkpoint['ema']['module'][key])
                for key, value in model.state_dict().items()), 'EMA state load differs')
    del checkpoint
    model.requires_grad_(False).to(device)
    pp = cfg.postprocessor.to(device).eval()
    require(pp.num_top_queries == 300 and pp.num_classes == 10 and pp.use_focal_loss
            and not pp.remap_mscoco_category and pp.score_threshold == .001, 'Postprocessor differs')
    head = None
    if mode != 'BASE':
        checkpoint = torch.load(root / 'arms' / mode / 'epoch12.pth', map_location='cpu', weights_only=True)
        require(checkpoint['epoch'] == 12 and checkpoint['phase'] == 'gate'
                and checkpoint['plan_sha256'] == sha(root / 'metadata/locked_plan.json'),
                'Head checkpoint is not locked epoch12')
        head = ROIRefiner(initialization_seed=plan['seed']).float().eval()
        head.load_state_dict(checkpoint['model'], strict=True)
        require(state_sha(head) == checkpoint['row']['model_state_sha256'], 'Head state SHA mismatch')
        # Keep stage flags: all-ineligible forward expects one grad-enabled gate
        # parameter. inference_mode below prevents all graph construction.
        head.set_stage('gate')
        head.to(device)
    return model, pp, head


def batch_forward(rows, model, pp, head, capture, mode, threshold, manifest, torch, np, device,
                  audit=False):
    from PIL import Image
    from refiner import prepare_rois, apply_refinement
    from engine.deim.postprocessor import scale_boxes_to_original_image

    set_precision(torch, manifest)
    low_images, originals = [], []
    for row in rows:
        with Image.open(io.BytesIO(row['bytes'])) as opened:
            pil = opened.convert('RGB')
            require(pil.size == (row['width'], row['height']), 'Decoded image dimensions differ')
            low = pil.resize((640, 640), resample=Image.Resampling.BILINEAR)
            low_images.append(torch.from_numpy(np.array(low, copy=True)).permute(2, 0, 1).float() / 255.)
            if head is not None:
                originals.append(torch.from_numpy(np.array(pil, copy=True)).permute(2, 0, 1).float() / 255.)
    samples = torch.stack(low_images).to(device)
    sizes = torch.tensor([[r['width'], r['height']] for r in rows], device=device)
    if capture is not None:
        capture.reset()
    outputs = model(samples)
    native = pp(outputs, sizes, for_eval=True)
    features = capture.finish(outputs, torch) if capture is not None else None
    if head is not None:
        flat_scores, flat_index = outputs['pred_logits'].sigmoid().flatten(1).topk(300, dim=-1)
        query_ids = flat_index // 10
    result = []
    for j, (row, pred) in enumerate(zip(rows, native)):
        base = {key: value.detach().cpu() for key, value in pred.items()}
        final = dict(base)
        unique, eligible, accepted_count = 0, 0, 0
        if head is not None:
            keep = flat_scores[j] > pp.score_threshold
            q = query_ids[j][keep]
            q_cpu = q.cpu()
            if audit:
                require(torch.equal(flat_scores[j][keep], pred['scores'])
                        and torch.equal((flat_index[j] % 10)[keep], pred['labels']),
                        'Native flat query/class reconstruction differs')
                scaled = scale_boxes_to_original_image(outputs['pred_boxes'], sizes, None)
                require(torch.equal(scaled[j, q], pred['boxes']), 'Native selected boxes differ')
            rois = prepare_rois(originals[j], base['boxes'], q_cpu, arm=mode,
                                input_hw=(640, 640), degraded_image=low_images[j] if mode == 'degraded' else None)
            unique, eligible = len(rois.roi_query_ids), int(rois.eligible.sum())
            gpu_rois = type(rois)(**{f.name: (getattr(rois, f.name).to(device)
                                            if isinstance(getattr(rois, f.name), torch.Tensor)
                                            else getattr(rois, f.name)) for f in dataclasses.fields(rois)})
            set_precision(torch, manifest, head=True)
            proposed = head(gpu_rois, pred['boxes'], pred['labels'], features[j, q])
            boxes, accepted = apply_refinement(pred['boxes'], proposed['boxes1'], proposed['gain'],
                                               proposed['eligible'], policy='gate', threshold=threshold)
            final['boxes'] = boxes.cpu()
            accepted_count = int(accepted.sum().cpu())
            final['query_ids'] = q_cpu
            if audit:
                reject, mask = apply_refinement(pred['boxes'], proposed['boxes1'], proposed['gain'],
                                                proposed['eligible'], policy='gate', threshold=1.)
                require(not bool(mask.any()) and torch.equal(reject, pred['boxes']), 'tau=1 changed boxes')
        # labels/scores stay the original tensors and index order; only boxes change.
        result.append({'image_id': row['image_id'], 'base': base, 'final': final,
                       'output_items': len(base['scores']), 'unique_ROIs': unique,
                       'eligible_output_items': eligible, 'accepted_output_items': accepted_count})
    if capture is not None:
        capture.reset()
    return result


def check_outputs(values, reference, torch):
    for value in values:
        image_id = value['image_id']
        for key in ('labels', 'scores', 'boxes'):
            require(torch.equal(value['base'][key], reference[image_id][key]),
                    f'Fresh native output differs: {image_id}/{key}')
        for key in ('labels', 'scores'):
            require(torch.equal(value['final'][key], value['base'][key]), 'Refiner changed labels/scores/order')
        require(value['final']['boxes'].shape == value['base']['boxes'].shape, 'Output count changed')
        require(bool(torch.isfinite(value['final']['boxes']).all()), 'Nonfinite final boxes')


def run(root):
    plan, manifest, binding, population, arm_meta, hashes = verify(root)
    require('engine' not in sys.modules, 'Start a fresh interpreter for native binding')
    sys.path.insert(0, binding['reusable_native_snapshot']['remote'])
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
    import numpy as np
    import torch
    from PIL import Image
    import PIL
    from torchvision.transforms.v2 import Resize
    from engine.data.transforms._transforms import ConvertPILImage
    from cache_native import FinalFeatures, verify_sources

    require(torch.cuda.is_available(), 'Benchmark requires CUDA0; never report CPU as GPU cost')
    torch.set_num_threads(2)
    torch.manual_seed(plan['seed'])
    device = torch.device('cuda:0')
    output = root / 'benchmark.json'
    require(not output.exists() and not output.is_symlink(), 'Refuse to replace a completed benchmark')
    # Prove the explicit decode/resize path equals native transforms before timing.
    native_convert = ConvertPILImage(dtype='float32', scale=True)
    for row in population:
        with Image.open(io.BytesIO(row['bytes'])) as opened:
            pil = opened.convert('RGB')
            explicit = torch.from_numpy(np.array(pil.resize((640, 640), Image.Resampling.BILINEAR), copy=True))
            explicit = explicit.permute(2, 0, 1).float() / 255.
            require(torch.equal(explicit, native_convert(Resize([640, 640])(pil))), 'Native transform mismatch')
    batches = [population[i:i + 4] for i in range(0, 32, 4)]
    baseline = {}
    results = {}
    for mode in ('BASE', 'raw', 'degraded'):
        gc.collect(); torch.cuda.empty_cache()
        set_precision(torch, manifest)
        model, pp, head = load_models(root, plan, manifest, binding, mode, torch, device)
        before_model, before_head = state_sha(model), state_sha(head) if head is not None else None
        capture = FinalFeatures(model) if head is not None else None
        threshold = arm_meta[mode]['threshold'] if head is not None else None
        times, counts = [], []
        try:
            with torch.inference_mode():
                # One untimed audited batch proves native q mapping and actual tau1 fallback.
                audit = batch_forward(batches[0], model, pp, head, capture, mode, threshold,
                                      manifest, torch, np, device, audit=True)
                if mode != 'BASE':
                    check_outputs(audit, baseline, torch)
                del audit
                for rows in batches[:2]:
                    warm = batch_forward(rows, model, pp, head, capture, mode, threshold,
                                         manifest, torch, np, device)
                    del warm
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                for repeat in range(3):
                    repeat_counts = []
                    for batch_index, rows in enumerate(batches):
                        torch.cuda.synchronize(device)
                        start = time.perf_counter()
                        values = batch_forward(rows, model, pp, head, capture, mode, threshold,
                                               manifest, torch, np, device)
                        torch.cuda.synchronize(device)
                        elapsed = (time.perf_counter() - start) * 1000.
                        times.append({'repeat': repeat + 1, 'batch': batch_index,
                                      'image_ids': [r['image_id'] for r in rows], 'ms': elapsed})
                        # CPU-only assertions/reference storage occur after stop timestamp.
                        if mode == 'BASE' and repeat == 0:
                            baseline.update({v['image_id']: v['base'] for v in values})
                        check_outputs(values, baseline, torch)
                        repeat_counts.extend({k: v[k] for k in ('image_id', 'output_items', 'unique_ROIs',
                                                               'eligible_output_items', 'accepted_output_items')}
                                             for v in values)
                        del values
                    counts.append(repeat_counts)
                require(counts[0] == counts[1] == counts[2], 'Repeated output/ROI counts changed')
                require(state_sha(model) == before_model and
                        (head is None or state_sha(head) == before_head), 'Inference changed model state')
                elapsed = np.array([t['ms'] for t in times], dtype=np.float64)
                results[mode] = {
                    'threshold': threshold, 'passive_feature_hook': capture is not None,
                    'unique_images': 32, 'measured_image_presentations': 96, 'measured_batches': 24,
                    'ms_per_image': float(elapsed.sum() / 96),
                    'median_ms_per_image_batch_amortized': float(np.median(elapsed) / 4),
                    'median_batch_ms': float(np.median(elapsed)), 'p95_batch_ms': float(np.percentile(elapsed, 95)),
                    'repeat_ms_per_image': [sum(t['ms'] for t in times if t['repeat'] == k) / 32 for k in (1, 2, 3)],
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                    'detector_parameters': sum(p.numel() for p in model.parameters()),
                    'added_parameters': sum(p.numel() for p in head.parameters()) if head is not None else 0,
                    'detector_state_sha256': before_model, 'head_state_sha256': before_head,
                    'output_items_per_32_images': sum(v['output_items'] for v in counts[0]),
                    'unique_ROIs_per_32_images': sum(v['unique_ROIs'] for v in counts[0]),
                    'unique_ROIs_measured_total': sum(v['unique_ROIs'] for v in counts[0]) * 3,
                    'eligible_output_items_per_32_images': sum(v['eligible_output_items'] for v in counts[0]),
                    'accepted_output_items_per_32_images': sum(v['accepted_output_items'] for v in counts[0]),
                    'counts_per_image': counts[0], 'batch_measurements': times,
                    'fixed_output_identity_checked_every_batch': True,
                    'tau1_exact_untimed_audit': head is not None}
                print(json.dumps({'event': 'benchmark_mode_complete', 'mode': mode,
                                  'ms_per_image': results[mode]['ms_per_image']}), flush=True)
        finally:
            if capture is not None:
                capture.close(); capture.reset()
            del capture, head, pp, model
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
    require(results['raw']['detector_state_sha256'] == results['BASE']['detector_state_sha256'] ==
            results['degraded']['detector_state_sha256'], 'Arms did not use identical detector states')
    require([v['unique_ROIs'] for v in results['raw']['counts_per_image']] ==
            [v['unique_ROIs'] for v in results['degraded']['counts_per_image']], 'ROI populations differ')
    # Recheck immutable input receipts and file hashes after all measured work.
    _, _, _, _, final_arms, final_hashes = verify(root)
    require(final_hashes == hashes and final_arms == arm_meta, 'Inputs changed during benchmarking')
    report = {
        'schema': 'roi12_fixed32_end_to_end_benchmark_v1', 'status': 'COMPLETE',
        'scope': 'Resident compressed bytes -> PIL RGB decode/bilinear640 -> GPU detector/native postprocessor -> CPU originalRGB ROI grids -> GPU head/gate -> final CPU outputs',
        'excluded': ['disk I/O', 'model loading', 'provenance verification', 'untimed diagnostic assertions'],
        'no_AP_claim': True, 'no_GT_inference_input': True, 'cached_predictions_used': False,
        'selection': 'first32 sorted full-val IDs, independent of GT',
        'population': [{k: v for k, v in row.items() if k != 'bytes'} for row in population],
        'batch_size': 4, 'warmup_batches_per_mode': 2, 'additional_untimed_audit_batches_per_mode': 1,
        'repeats': 3, 'mode_order': ['BASE', 'raw', 'degraded'], 'cpu_threads': 2,
        'dtype': 'float32', 'autocast': False, 'cublas_workspace_config': ':4096:8',
        'detector_flags': {k: manifest[k] for k in ('tf32_matmul', 'tf32_cudnn', 'cudnn_benchmark')},
        'detector_unrecorded_flags_assumption': 'Native fresh-process defaults: cudnn.deterministic=False; deterministic_algorithms=False. Fresh outputs checked bitwise across modes/repeats.',
        'head_flags': {'tf32_matmul': False, 'tf32_cudnn': False, 'cudnn_benchmark': False,
                       'cudnn_deterministic': True, 'deterministic_algorithms': True},
        'peak_memory_scope': 'CUDA allocator peak including resident detector/head parameters; reset after warmup; allocated and reserved reported separately',
        'timing_summary': '24 complete batch latencies per mode; p95 uses NumPy linear interpolation; per-image values amortize batch4',
        'limitations': 'Single GPU/process, fixed32 images, serial CPU ROI preprocessing, fixed mode order; no disk or concurrency cost and no population-wide latency claim.',
        'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                        'numpy': np.__version__, 'Pillow': PIL.__version__,
                        'gpu': torch.cuda.get_device_name(device)},
        'hashes': hashes, 'arms': arm_meta, 'modes': results}
    from supervise import atomic_json
    atomic_json(output, report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args(argv)
    run(Path(args.root).resolve())


if __name__ == '__main__':
    main()
