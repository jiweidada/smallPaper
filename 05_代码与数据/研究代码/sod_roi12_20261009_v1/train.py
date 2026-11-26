"""One immutable ROI12 arm: eight R passes, then four G passes, no resume.

The detector is frozen and cached. Official validation never trains the heads
or selects thresholds/checkpoints. Any failed run retains its completed/partial
pass ledger; this entry point cannot silently restart or extend it.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ROIDataset, single_image_collate, worker_init
from evaluate import (evaluate_predictions, assert_bound_baseline,
                      assert_frozen_identity, compare_evaluations)
from refiner import (ROIRefiner, refinement_loss, utility_targets,
                     gain_loss, apply_refinement)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def state_sha(model, prefixes=None):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        if prefixes is not None and not k.startswith(tuple(prefixes)):
            continue
        v = v.detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(v.dtype).encode())
        h.update(str(tuple(v.shape)).encode()); h.update(v.numpy().tobytes())
    return h.hexdigest()


def move(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=False)
    if dataclasses.is_dataclass(obj):
        return type(obj)(**{f.name: move(getattr(obj, f.name), device)
                           for f in dataclasses.fields(obj)})
    if isinstance(obj, dict):
        return {k: move(v, device) for k, v in obj.items()}
    return obj


def loader(dataset, workers, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=1, shuffle=shuffle, generator=generator,
                      num_workers=workers, collate_fn=single_image_collate,
                      worker_init_fn=worker_init, persistent_workers=workers > 0)


def plain_predictions(manifest_path, ids=None):
    manifest_path = Path(manifest_path)
    manifest = read(manifest_path)
    selected = None if ids is None else set(map(int, ids))
    result = {}
    for record in manifest['images']:
        image_id = int(record['image_id'])
        if selected is not None and image_id not in selected:
            continue
        with np.load(manifest_path.parent / record['cache_file'], allow_pickle=False) as arr:
            result[image_id] = {k: arr[k].copy() for k in ('boxes', 'scores', 'labels', 'query_ids')}
            result[image_id]['rowids'] = arr['output_index'].copy()
    if selected is not None and set(result) != selected:
        raise ValueError('Missing cache images')
    return result


def subset_annotation(annotation, ids, destination):
    source = read(annotation)
    ids = set(map(int, ids))
    source['images'] = [x for x in source['images'] if int(x['id']) in ids]
    source['annotations'] = [x for x in source['annotations'] if int(x['image_id']) in ids]
    if {int(x['id']) for x in source['images']} != ids:
        raise ValueError('Calibration image list not covered by annotation')
    write(destination, source)


@torch.no_grad()
def proposals(model, dataset, device, workers, destination):
    """One fixed model pass; all later threshold candidates reuse these proposals."""
    model.eval()
    result = {}
    image_records = []
    t0 = time.perf_counter()
    unique_rois = 0
    Path(destination).mkdir(parents=True, exist_ok=False)
    for cpu in loader(dataset, workers):
        batch = move(cpu, device)
        out = model(batch['rois'], batch['boxes0'], batch['labels'], batch['features'])
        image_id = int(batch['image_id'])
        record = {'boxes0': cpu['boxes0'].numpy(), 'boxes1': out['boxes1'].cpu().numpy(),
                  'scores': cpu['scores'].numpy(), 'labels': cpu['labels'].numpy(),
                  'query_ids': cpu['query_ids'].numpy(), 'rowids': cpu['output_index'].numpy(),
                  'gain': out['gain'].cpu().numpy(), 'eligible': out['eligible'].cpu().numpy()}
        unique_rois += len(cpu['rois'].roi_query_ids)
        result[image_id] = record
        path = Path(destination) / f'{image_id:06d}.npz'
        np.savez_compressed(path, **record)
        image_records.append({'image_id': image_id, 'file': path.name, 'sha256': sha(path)})
    elapsed = time.perf_counter() - t0
    write(Path(destination) / 'manifest.json', {
        'images': image_records, 'seconds': elapsed, 'unique_ROIs': unique_rois,
        'scope': 'cached detector output + original image decode/ROI/head; NOT end-to-end detector latency',
        'model_state_sha256': state_sha(model)})
    return result


def from_proposals(records, policy, threshold):
    output = {}
    accepted_total = 0
    for image_id, r in records.items():
        boxes, accepted = apply_refinement(
            torch.from_numpy(r['boxes0']), torch.from_numpy(r['boxes1']),
            torch.from_numpy(r['gain']), torch.from_numpy(r['eligible']),
            policy=policy, threshold=threshold)
        accepted_total += int(accepted.sum())
        output[image_id] = {'boxes': boxes.numpy(), **{k: r[k] for k in ('scores', 'labels', 'query_ids', 'rowids')}}
    return output, accepted_total


def evaluate_stage(model, plan, arm, outdir, device, val_dataset, cal_dataset,
                   val_base, cal_base, epoch):
    workers = plan['loader_workers']
    stage_dir = outdir / f'evaluation_epoch{epoch:02d}'
    stage_dir.mkdir(exist_ok=False)
    threshold = None
    if epoch == 12:
        cal_ann = stage_dir / 'calibration_annotations.json'
        subset_annotation(read(plan['train_manifest'])['annotation'], cal_dataset.image_ids, cal_ann)
        cal_summary = evaluate_predictions(cal_ann, cal_base, stage_dir / 'calibration_baseline')
        if cal_summary['stats']['APs'] < 0:
            raise ValueError('Calibration split has no valid small-object AP population')
        cal_records = proposals(model, cal_dataset, device, workers, stage_dir / 'calibration_proposals')
        candidates = []
        for tau in plan['threshold_grid']:
            predictions, accepted = from_proposals(cal_records, 'gate', tau)
            assert_frozen_identity(cal_base, predictions, require_query_ids=True,
                                   require_same_boxes=tau == 1)
            summary = evaluate_predictions(cal_ann, predictions, stage_dir / f'calibration_tau{tau:.2f}')
            metrics = summary['stats']
            candidates.append({'threshold': tau, 'metrics': metrics, 'accepted': accepted,
                               'admissible': metrics['AP'] >= cal_summary['stats']['AP']})
        choices = [x for x in candidates if x['admissible']]
        if not choices or not next(x for x in candidates if x['threshold'] == 1)['admissible']:
            raise AssertionError('Reject-all must exactly reproduce calibration baseline')
        selected = max(choices, key=lambda x: (x['metrics']['APs'], x['threshold']))
        threshold = selected['threshold']
        write(stage_dir / 'threshold_selection.json', {
            'rule': 'max C APs subject to C AP>=C baseline; exact ties choose higher threshold',
            'official_val_used': False, 'candidates': candidates, 'selected': selected,
            'note': 'C is internal official-train data; frozen detector has seen its images'})
    val_records = proposals(model, val_dataset, device, workers, stage_dir / 'val_proposals')
    always, accepted_all = from_proposals(val_records, 'all', 0)
    assert_frozen_identity(val_base, always, require_query_ids=True)
    annotation = read(plan['val_manifest'])['annotation']
    all_summary = evaluate_predictions(annotation, always, stage_dir / 'always_refine')
    diagnostic = compare_evaluations(plan['baseline_eval'], stage_dir / 'always_refine', epoch=epoch)
    diagnostic['screening_decision'].update(passed=None, status='ALWAYS_REFINE_DIAGNOSTIC_NOT_PRIMARY')
    write(stage_dir / 'always_refine_comparison.json', diagnostic)
    if epoch != 12:
        write(stage_dir / 'status.json', {'epoch': epoch, 'diagnostic_only': True,
                                        'always_refine': all_summary['overall_percent'], 'accepted': accepted_all})
        return
    primary, accepted = from_proposals(val_records, 'gate', threshold)
    assert_frozen_identity(val_base, primary, require_query_ids=True, require_same_boxes=threshold == 1)
    primary_summary = evaluate_predictions(annotation, primary, stage_dir / 'primary_gated')
    comparison = compare_evaluations(plan['baseline_eval'], stage_dir / 'primary_gated',
                                    stage_dir / 'primary_comparison.json', epoch=12)
    write(outdir / 'result.json', {
        'status': 'COMPLETE_12_PASSES', 'arm': arm, 'epochs_completed': 12,
        'R_passes': 8, 'G_passes': 4, 'threshold': threshold, 'accepted': accepted,
        'primary': primary_summary['overall_percent'], 'comparison': comparison,
        'always_refine_diagnostic': all_summary['overall_percent'],
        'maximum_epochs': 12, 'extension_allowed': False,
        'interpretation': 'exploratory fixed-endpoint screen, not evidence of multi-seed stability or novelty'})


def verify_plan(plan, plan_path):
    if plan['maximum_epochs'] != 12 or plan['R_passes'] != 8 or plan['G_passes'] != 4:
        raise ValueError('12-pass budget changed')
    if plan['threshold_grid'] != [0, .05, .10, .20, .30, .50, 1]:
        raise ValueError('Threshold grid changed')
    if (plan['gate_utility']['kind'] != 'single_final_item_replacement_full_diagnostic_rematching'
            or plan['gate_utility']['lambda_harm'] != 1.):
        raise ValueError('Do not silently substitute local-IoU utility for rematching utility')
    for name, digest in plan['code_sha256'].items():
        if Path(name).name != name or sha(Path(__file__).parent / name) != digest:
            raise ValueError('Code differs from immutable plan: ' + name)
    for split in ('train', 'val'):
        path = Path(plan[split + '_manifest'])
        if sha(path) != plan[split + '_manifest_sha256']:
            raise ValueError('Manifest differs from locked plan')
        manifest = read(path)
        if manifest['status'] != 'COMPLETE':
            raise ValueError('Training requires complete detector caches')
    baseline = read(Path(plan['baseline_eval']) / 'summary.json')
    assert_bound_baseline(baseline, plan['source_bindings'], seed=1)
    if sha(plan['splits']) != plan['splits_sha256']:
        raise ValueError('Split list differs from locked plan')
    return sha(plan_path)


def run(args):
    os.environ['ROI12_SOURCE_BINDINGS'] = str(Path(args.bindings).resolve())
    plan = read(args.plan)
    plan_hash = verify_plan(plan, args.plan)
    if Path(plan['source_bindings']).resolve() != Path(args.bindings).resolve():
        raise ValueError('Different source binding')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    write(root / 'plan.json', plan)
    write(root / 'launch.json', {'arm': args.arm, 'plan_sha256': plan_hash,
                               'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                               'pid': os.getpid(), 'device': args.device})
    completed, optimizer_steps = 0, 0
    current_epoch, current_images = 0, 0
    start = time.perf_counter()
    try:
        random.seed(plan['seed']); np.random.seed(plan['seed']); torch.manual_seed(plan['seed'])
        torch.set_num_threads(plan['cpu_threads'])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # CUBLAS_WORKSPACE_CONFIG is set by supervisor before CUDA initialization.
        torch.use_deterministic_algorithms(True)
        device = torch.device(args.device)
        if device.type != 'cuda':
            raise ValueError('Real training is GPU-only; CPU engineering must not train real images')
        split = read(plan['splits'])
        split_ids = {k: split['splits'][source]['image_ids']
                     for k, source in (('R', 'A'), ('G', 'B'), ('C', 'C'))}
        cache_dir = str(Path(plan['train_manifest']).parent)
        binding_dir = str(Path(plan['splits']).parent / 'fixed_bindings')
        r_data = ROIDataset(cache_dir, split_ids['R'], args.arm, bindings_cache_dir=binding_dir)
        g_data = ROIDataset(cache_dir, split_ids['G'], args.arm, bindings_cache_dir=binding_dir)
        c_data = ROIDataset(cache_dir, split_ids['C'], args.arm, bindings_cache_dir=binding_dir)
        val_manifest = read(plan['val_manifest'])
        v_data = ROIDataset(str(Path(plan['val_manifest']).parent),
                            [x['image_id'] for x in val_manifest['images']], args.arm)
        val_base = plain_predictions(plan['val_manifest'])
        cal_base = plain_predictions(plan['train_manifest'], split_ids['C'])
        assert_frozen_identity(val_base, Path(plan['baseline_eval']) / 'predictions.json',
                               require_same_boxes=True, require_query_ids=True)
        model = ROIRefiner(initialization_seed=plan['seed']).to(device)
        initial = state_sha(model)
        if initial != plan['initial_model_state_sha256']:
            raise ValueError('Paired initialization mismatch')
        torch.save({'model': model.state_dict(), 'epoch': 0, 'state_sha256': initial}, root / 'initial.pth')
        write(root / 'model.json', {'initial_state_sha256': initial,
                                  'parameters': sum(p.numel() for p in model.parameters()),
                                  'model_recipe': {'num_classes': 10, 'query_dim': 256, 'stats_dim': 0,
                                                   'max_edge_fraction': .2, 'initialization_seed': plan['seed']}})
        generator = torch.Generator()
        r_loader = loader(r_data, plan['loader_workers'], True, generator)
        g_loader = loader(g_data, plan['loader_workers'], True, generator)
        optimizer = None
        frozen_hash = None
        gate_labels = root / 'gate_labels'
        gate_label_manifest = None
        ledger = root / 'epochs.jsonl'
        for epoch in range(1, 13):
            current_epoch, current_images = epoch, 0
            phase = 'refiner' if epoch <= 8 else 'gate'
            if epoch in (1, 9):
                model.set_stage(phase)
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                              lr=plan['optimizer'][phase]['lr'],
                                              weight_decay=plan['optimizer']['weight_decay'])
                if epoch == 9:
                    frozen_hash = state_sha(model, ('encoder.', 'geometry.'))
                    gate_labels.mkdir(exist_ok=False)
                    gate_label_manifest = {
                        'status': 'GENERATING', 'frozen_refiner_state_sha256': frozen_hash,
                        'utility': plan['gate_utility'],
                        'refiner_code_sha256': plan['code_sha256']['refiner.py'],
                        'train_manifest_sha256': plan['train_manifest_sha256'],
                        'scope': 'each eligible final item replaced alone, all-output diagnostic rematching; not native COCO AP or joint accepted-set gain',
                        'files': {}}
            generator.manual_seed(plan['seed'] + epoch)
            model.train()
            if phase == 'gate':
                model.encoder.eval()
                model.geometry.eval()
            phase_loader = r_loader if epoch <= 8 else g_loader
            expected = set(split_ids['R' if epoch <= 8 else 'G'])
            seen = set()
            sums = {}; counts = {'eligible': 0, 'bound_eligible': 0, 'old_success_eligible': 0,
                                  'gate_positive': 0, 'gate_zero': 0, 'gate_negative': 0}
            t0 = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            for cpu in phase_loader:
                image_id = int(cpu['image_id'])
                if image_id in seen or image_id not in expected:
                    raise ValueError('Repeated/unregistered image in pass')
                seen.add(image_id)
                batch = move(cpu, device)
                optimizer.zero_grad(set_to_none=True)
                out = model(batch['rois'], batch['boxes0'], batch['labels'], batch['features'])
                eligible = out['eligible']
                counts['eligible'] += int(eligible.sum())
                counts['bound_eligible'] += int((eligible & (batch['bindings'].det_to_gt >= 0)).sum())
                counts['old_success_eligible'] += int((eligible & batch['bindings'].old_success).sum())
                if phase == 'refiner':
                    terms = refinement_loss(batch['boxes0'], out['boxes1'], batch['gt_boxes'],
                                            batch['bindings'], eligible, **plan['loss_weights'])
                    loss = terms['loss']
                else:
                    label_path = gate_labels / f'{image_id:06d}.npz'
                    if epoch == 9:
                        if label_path.exists():
                            raise FileExistsError('Refuse to replace fixed gate labels')
                        targets = utility_targets(batch['boxes0'], out['boxes1'], batch['labels'],
                                                   batch['gt_boxes'], batch['gt_labels'], batch['bindings'],
                                                   eligible, scores=batch['scores'])
                        arrays = {k: v.detach().cpu().numpy() for k, v in targets.items()}
                        arrays.update(boxes0=batch['boxes0'].detach().cpu().numpy(),
                                      boxes1=out['boxes1'].detach().cpu().numpy())
                        np.savez_compressed(label_path, **arrays)
                        gate_label_manifest['files'][label_path.name] = sha(label_path)
                    else:
                        if sha(label_path) != gate_label_manifest['files'][label_path.name]:
                            raise ValueError('Fixed gate target cache changed')
                        with np.load(label_path, allow_pickle=False) as values:
                            targets = {k: torch.from_numpy(values[k].copy()).to(device)
                                       for k in values.files if k not in ('boxes0', 'boxes1')}
                            for name, expected_boxes in (('boxes0', batch['boxes0']), ('boxes1', out['boxes1'])):
                                if not torch.equal(torch.from_numpy(values[name].copy()).to(device), expected_boxes):
                                    raise AssertionError('Frozen gate proposal changed: ' + name)
                    loss = gain_loss(out['gain'], targets, eligible, balance_groups=True)
                    terms = {'loss': loss}
                    y = targets['target_y'][eligible]
                    for name, mask in [('positive', y > 0), ('zero', y == 0), ('negative', y < 0)]:
                        counts['gate_' + name] += int(mask.sum())
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite loss')
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                      plan['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                optimizer_steps += 1; current_images += 1
                for k, value in terms.items():
                    sums[k] = sums.get(k, 0.) + float(value.detach())
                if current_images % 200 == 0:
                    print(json.dumps({'event': 'progress', 'epoch': epoch, 'phase': phase,
                                      'images': current_images, 'total': len(expected),
                                      'loss': sums['loss'] / current_images,
                                      'optimizer_steps': optimizer_steps}), flush=True)
            if seen != expected:
                raise ValueError('Incomplete phase image pass')
            if phase == 'gate' and state_sha(model, ('encoder.', 'geometry.')) != frozen_hash:
                raise AssertionError('Gate training changed frozen refiner/encoder')
            if epoch == 9:
                gate_label_manifest['status'] = 'COMPLETE'
                gate_label_manifest['images'] = len(seen)
                write(gate_labels / 'manifest.json', gate_label_manifest)
            completed = epoch
            row = {'epoch': epoch, 'phase': phase, 'phase_images': len(seen),
                   'image_list_sha256': hashlib.sha256(json.dumps(sorted(seen)).encode()).hexdigest(),
                   'loss': {k: v / len(seen) for k, v in sums.items()}, 'counts': counts,
                   'seconds': time.perf_counter() - t0, 'optimizer_steps_total': optimizer_steps,
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                   'model_state_sha256': state_sha(model), 'refiner_frozen_state_sha256': frozen_hash}
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'epoch': epoch, 'phase': phase, 'plan_sha256': plan_hash,
                        'optimizer_steps': optimizer_steps, 'row': row}, root / f'epoch{epoch:02d}.pth')
            with ledger.open('a', encoding='utf-8') as f:
                f.write(json.dumps(row, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno())
            write(root / 'status.json', {'status': 'RUNNING', 'completed_epochs': completed,
                                         'optimizer_steps': optimizer_steps, 'last': row})
            print(json.dumps({'event': 'epoch_complete', **row}), flush=True)
            if epoch in (8, 12):
                evaluate_stage(model, plan, args.arm, root, device, v_data, c_data,
                               val_base, cal_base, epoch)
        write(root / 'status.json', {'status': 'COMPLETE', 'completed_epochs': 12,
                                     'optimizer_steps': optimizer_steps,
                                     'seconds_including_evaluations': time.perf_counter() - start,
                                     'result_sha256': sha(root / 'result.json'),
                                     'last_checkpoint_sha256': sha(root / 'epoch12.pth')})
    except BaseException as error:
        write(root / 'failure.json', {'error': repr(error), 'traceback': traceback.format_exc(),
                                      'completed_epochs': completed, 'current_epoch': current_epoch,
                                      'images_in_current_epoch': current_images,
                                      'optimizer_steps': optimizer_steps,
                                      'automatic_restart_allowed': False})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--bindings', required=True)
    parser.add_argument('--arm', choices=['raw', 'degraded'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
