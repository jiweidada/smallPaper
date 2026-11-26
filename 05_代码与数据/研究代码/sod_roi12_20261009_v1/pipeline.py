"""Run an immutable paired screen only after the resource supervisor permits it."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def command(root, name, argv):
    log = root / 'logs' / f'{name}.log'
    log.parent.mkdir(exist_ok=True)
    print(json.dumps({'event': 'start', 'name': name, 'argv': argv}), flush=True)
    start = time.perf_counter()
    with log.open('x', encoding='utf-8') as stream:
        proc = subprocess.Popen(argv, cwd=root / 'code', stdout=stream, stderr=subprocess.STDOUT,
                                env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'))
        write(root / 'pipeline_status.json', {'status': 'RUNNING', 'stage': name,
                                             'child_pid': proc.pid, 'argv': argv})
        code = proc.wait()
    receipt = {'stage': name, 'exit_code': code, 'seconds': time.perf_counter() - start,
               'log': str(log), 'log_sha256': sha(log), 'automatic_retry': False}
    write(root / 'receipts' / f'{name}.json', receipt)
    if code:
        raise RuntimeError(f'{name} exited {code}; no automatic retry; see {log}')
    return receipt


def run(args):
    root = Path(args.root).resolve()
    if root.name != 'sod_roi12_20261009_v1' or root.parent.name != 'experiments':
        raise ValueError('Unexpected experiment namespace')
    os.environ['ROI12_SOURCE_BINDINGS'] = str(root / 'metadata/source_bindings_proposed.json')
    from supervise import validate_lock, observe
    validate_lock(root, root / 'metadata/source_bindings_proposed.json')
    current_resources = observe(root)
    if not current_resources['idle']:
        raise RuntimeError('Resources changed before pipeline start: ' + repr(current_resources['reasons']))
    launch = read(root / 'launch_lock.json')
    for rel, digest in launch['code_sha256'].items():
        path = (root / rel).resolve()
        if root not in path.parents or sha(path) != digest:
            raise ValueError('Code lock mismatch: ' + rel)
    binding_path = root / 'metadata/source_bindings_proposed.json'
    if sha(binding_path) != launch['bindings_sha256']:
        raise ValueError('Source binding differs from launch lock')
    binding = read(binding_path)['seeds'][0]
    if binding['seed'] != 1:
        raise ValueError('Expected same seed1 detector for both arms')
    # One-shot marker also forbids direct invocation restarting a failed run.
    with (root / 'pipeline_started.json').open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'launch_lock_sha256': sha(root / 'launch_lock.json')}, stream)
    start = time.perf_counter()
    try:
        command(root, 'synthetic_cuda_preflight',
                [sys.executable, str(root / 'code/synthetic_device_test.py'), '--device', 'cuda:0',
                 '--output', str(root / 'preflight/synthetic_cuda.json')])
        from data import make_splits
        from refiner import ROIRefiner
        from train import plain_predictions, state_sha
        from evaluate import evaluate_predictions, assert_bound_baseline
        cache = root / 'cache'
        common = [sys.executable, str(root / 'code/cache_native.py'),
                  '--native-root', binding['reusable_native_snapshot']['remote'],
                  '--source-binding', str(root / 'metadata/training_source_binding.json'),
                  '--config', str(root / 'metadata/baseline_seed1.json'),
                  '--checkpoint', binding['frozen_detector_proposed']['remote_checkpoint'],
                  '--out', str(cache), '--device', 'cuda:0']
        for split in ('val', 'train'):
            source = binding['data'][split]
            argv = common + ['--split', split, '--' + split + '-ann', source['ann'],
                             '--' + split + '-images', source['images']]
            command(root, 'cache_' + split, argv)
            manifest_path = cache / split / 'manifest.json'
            manifest = read(manifest_path)
            if (manifest['status'] != 'COMPLETE' or manifest['images_cached'] != source['count']
                    or manifest['checkpoint_sha256'] != binding['frozen_detector_proposed']['sha256']
                    or manifest['annotation_sha256'] != source['ann_sha256']
                    or not manifest['native_postprocessor_exact_all_items']
                    or not manifest['first_batch_hook_outputs_exact'] or not manifest['model_unchanged']):
                raise ValueError('Incomplete/unbound native cache')
            if split == 'val':
                # Full GPU baseline must pass before train caching or ANY optimizer update.
                baseline = evaluate_predictions(source['ann'], plain_predictions(manifest_path),
                                                root / 'baseline_eval')
                proof = assert_bound_baseline(baseline, binding_path, seed=1)
                proof.update(status='PASS', cache_manifest_sha256=sha(manifest_path),
                             checkpoint_sha256=manifest['checkpoint_sha256'],
                             baseline_summary_sha256=sha(root / 'baseline_eval/summary.json'))
                write(root / 'receipts/baseline_reproduction.json', proof)
        split_path = root / 'metadata/splits.json'
        splits = make_splits(cache / 'train/manifest.json', cache / 'val/manifest.json', split_path,
                             seed=20261009)
        for name in ('A', 'B', 'C'):
            if not splits['splits'][name]['image_ids']:
                raise ValueError('Empty locked internal split')
        model = ROIRefiner(initialization_seed=20261009)
        # No optimizer is constructed in this orchestration process.
        plan = {
            'schema': 'roi12_locked_paired_plan_v1', 'seed': 20261009,
            'arms': ['raw', 'degraded'], 'maximum_epochs': 12, 'R_passes': 8, 'G_passes': 4,
            'epoch_definition': 'one full R image pass for epochs1..8; one full G image pass for9..12',
            'threshold_grid': [0, .05, .10, .20, .30, .50, 1],
            'threshold_selection': 'C max APs subject to AP>=C baseline, exact ties higher tau',
            'primary_endpoint': 'epoch12 gated APs>full-val baseline AND AP>=full-val baseline',
            'baseline_eval': str(root / 'baseline_eval'), 'source_bindings': str(binding_path),
            'baseline_reproduction_receipt_sha256': sha(root / 'receipts/baseline_reproduction.json'),
            'splits': str(split_path), 'splits_sha256': sha(split_path),
            'initial_model_state_sha256': state_sha(model),
            'train_manifest': str(cache / 'train/manifest.json'),
            'train_manifest_sha256': sha(cache / 'train/manifest.json'),
            'val_manifest': str(cache / 'val/manifest.json'),
            'val_manifest_sha256': sha(cache / 'val/manifest.json'),
            'optimizer': {'type': 'AdamW', 'weight_decay': .0001,
                          'refiner': {'lr': .0003}, 'gate': {'lr': .001}},
            'scheduler': 'none', 'image_batch_size': 1, 'gradient_accumulation': 1,
            'gradient_clip_norm': 1.0, 'loader_workers': 2, 'cpu_threads': 2,
            'autocast': False, 'tf32': False,
            'loss_weights': {'giou_weight': 1., 'iou_weight': 1., 'l1_weight': 1., 'harm_weight': 2.},
            'gate_loss': 'equal mean weight over present positive/zero/negative target groups',
            'gate_utility': {'kind': 'single_final_item_replacement_full_diagnostic_rematching',
                             'thresholds': [.50, .55, .60, .65, .70, .75, .80, .85, .90, .95],
                             'lambda_harm': 1., 'ordering': 'original score descending stable; later GT index for equal IoU',
                             'population': 'all final outputs and native-trainable GT; no COCO max100 truncation',
                             'target': 'mean(new GT count - lost GT count); invalid identity gets rejection target -1',
                             'cache': 'generate once at epoch9 with epoch8 frozen proposals, verify/reuse through epoch12',
                             'not_claimed': 'AP increment, native COCO exact target, or additive joint acceptance guarantee'},
            'GT_policy': 'geometry/identity bindings: all-output baseline075 greedy then remaining050, fixed after binding; train GT clamped. Utility labels separately rematch all outputs for each single-item proposal; neither training matching is native max100 matching',
            'output_policy': 'classes/scores/query IDs/order/count frozen, boxes only; no added clipping/NMS',
            'ROI': {'size': 64, 'channels': 4, 'context': 2, 'min_extent_original_px': 32,
                    'eligible_short_side_at640_lt': 16, 'max_edge_fraction': .2,
                    'degraded': 'native PIL bilinear640 then same physical ROI grid'},
            'code_sha256': {p.name: sha(p) for p in sorted((root / 'code').glob('*.py'))},
            'limitations': ['one frozen detector seed and one head init', '12-pass exploratory screen',
                            'filename prefix is not verified sequence identity',
                            'C unseen by new heads, seen by frozen detector',
                            'no official validation threshold tuning/checkpoint selection',
                            'old_success is a training proxy, not guaranteed preservation of native COCO successes'],
        }
        plan_path = root / 'metadata/locked_plan.json'
        if plan_path.exists():
            raise FileExistsError('Never replace a locked plan')
        write(plan_path, plan)
        del model
        results = {}
        for arm in plan['arms']:
            command(root, 'train_' + arm, [sys.executable, str(root / 'code/train.py'),
                                          '--plan', str(plan_path), '--bindings', str(binding_path),
                                          '--arm', arm, '--output', str(root / 'arms' / arm),
                                          '--device', 'cuda:0'])
            result = read(root / 'arms' / arm / 'result.json')
            if result['epochs_completed'] != 12:
                raise ValueError('Missing complete fixed endpoint')
            results[arm] = result
        raw, degraded = results['raw'], results['degraded']
        command(root, 'end_to_end_benchmark',
                [sys.executable, str(root / 'code/benchmark.py'), '--root', str(root)])
        raw_pass = raw['comparison']['screening_decision']['passed']
        difference = {k: raw['primary'][k] - degraded['primary'][k] for k in ('AP', 'APs', 'AP75')}
        mechanism_supported = raw_pass and difference['APs'] > 0 and difference['AP'] >= 0
        report = {
            'status': 'COMPLETE_PAIRED_12_PASS_SCREEN', 'results': results,
            'raw_minus_degraded_pp': difference,
            'raw_image_information_mechanism_passed_screen': mechanism_supported,
            'conclusion': ('RAW_INFORMATION_SHORT_SCREEN_SUPPORTED' if mechanism_supported else
                           'REFINEMENT_PASSED_RAW_INFORMATION_NOT_ISOLATED' if raw_pass else
                           'STOP_RAW_ROI_ROUTE_NO_EXTENSION'),
            'seconds_including_cache_and_evaluation': time.perf_counter() - start,
            'maximum_training_passes_per_arm': 12, 'automatic_additional_training_allowed': False,
            'latency_status': 'See benchmark.json for the defined measured scope and fixed image population',
            'benchmark_sha256': sha(root / 'benchmark.json'),
            'scope': 'Exploratory point estimates only; not proof of stability, originality, or publishability'}
        write(root / 'paired_result.json', report)
        write(root / 'pipeline_status.json', {'status': 'COMPLETE',
                                             'paired_result_sha256': sha(root / 'paired_result.json'),
                                             'training_complete': True, 'full_goal_complete': False,
                                             'remaining': ['local result archive and final literature comparison']})
    except BaseException as error:
        write(root / 'pipeline_failure.json', {'error': repr(error), 'traceback': traceback.format_exc(),
                                               'automatic_retry': False})
        write(root / 'pipeline_status.json', {'status': 'FAILED', 'error': repr(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    run(parser.parse_args())
