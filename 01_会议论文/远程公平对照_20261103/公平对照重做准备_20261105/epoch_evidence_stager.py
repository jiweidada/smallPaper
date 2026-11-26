"""Prepare pre/post-rollback epoch evidence without importing a model.

An actual trainer must supply checkpoint-writing callbacks, capture RNG states
and call the acknowledgement barrier before proceeding.  This module is not
a training driver and cannot establish checkpoint contents or authorization.
"""
from pathlib import Path
import copy
import csv
import gzip
import io
import math
import os
import shutil

import epoch_backup_barrier as barrier
import epoch_backup_guard as guard

METRICS = {'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl'}
IDENTITY_FIELDS = {'configuration_sha256', 'environment_sha256', 'runner_sha256',
                   'dataset_manifest_sha256', 'initialization_sha256'}
RNG_FIELDS = {'python', 'numpy', 'torch_cpu', 'torch_cuda_all',
              'train_loader_generator', 'train_sampler_generator', 'validation_loader_generator'}


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def bytes_list(value):
    return (isinstance(value, (list, tuple)) and 0 < len(value) <= 200000 and
            all(type(x) is int and 0 <= x <= 255 for x in value))


def validate_rng(rng):
    guard.require(isinstance(rng, dict) and set(rng) == RNG_FIELDS, 'Incomplete RNG snapshot')
    py = rng['python']
    guard.require(isinstance(py, (list, tuple)) and len(py) == 3 and type(py[0]) is int and py[0] == 3 and
                  isinstance(py[1], (list, tuple)) and len(py[1]) == 625 and
                  all(type(x) is int and 0 <= x <= 0xffffffff for x in py[1]) and 0 <= py[1][-1] <= 624 and
                  (py[2] is None or finite_number(py[2])), 'Invalid Python RNG state')
    n = rng['numpy']
    guard.require(isinstance(n, dict) and set(n) == {'name', 'values', 'pos', 'has_gauss', 'cached_gaussian'} and
                  n['name'] == 'MT19937' and isinstance(n['values'], (list, tuple)) and len(n['values']) == 624 and
                  all(type(x) is int and 0 <= x <= 0xffffffff for x in n['values']) and
                  type(n['pos']) is int and 0 <= n['pos'] <= 624 and type(n['has_gauss']) is int and
                  n['has_gauss'] in (0, 1) and finite_number(n['cached_gaussian']), 'Invalid NumPy RNG state')
    for name in ('torch_cpu', 'train_loader_generator', 'train_sampler_generator', 'validation_loader_generator'):
        guard.require(bytes_list(rng[name]), 'Invalid generator state: ' + name)
    cuda = rng['torch_cuda_all']
    guard.require(isinstance(cuda, (list, tuple)) and len(cuda) == 1 and bytes_list(cuda[0]),
                  'Expected one visible CUDA RNG state')
    return copy.deepcopy(rng)


def validate_control(value, epoch):
    keys = {'last_epoch', 'ema_updates', 'ema_decay', 'best_epoch', 'best_ap'}
    guard.require(isinstance(value, dict) and set(value) == keys, 'Incomplete control state')
    guard.require(type(value['last_epoch']) is int and -1 <= value['last_epoch'] <= epoch, 'Invalid last_epoch')
    guard.require(type(value['ema_updates']) is int and value['ema_updates'] >= 0, 'Invalid EMA update count')
    guard.require(finite_number(value['ema_decay']) and 0 <= value['ema_decay'] <= 1, 'Invalid EMA decay')
    guard.require(type(value['best_epoch']) is int and -1 <= value['best_epoch'] <= epoch, 'Invalid best epoch')
    guard.require((value['best_epoch'] == -1 and value['best_ap'] is None) or
                  (value['best_epoch'] >= 0 and finite_number(value['best_ap'])), 'Invalid best AP')
    guard.require(epoch == 0 or value['best_epoch'] >= 0, 'A prior completed epoch must have a best result')
    return dict(value)


def learning_rate_bytes(epoch, rows):
    guard.require(isinstance(rows, (list, tuple)) and len(rows) == guard.BATCHES, 'Wrong learning-rate row count')
    group_count = None
    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')
    for batch, row in enumerate(rows):
        guard.require(isinstance(row, (list, tuple)) and len(row) == 4, 'Invalid learning-rate row')
        attempt, actual_epoch, actual_batch, rates = row
        guard.require(type(attempt) is int and attempt == epoch * guard.BATCHES + batch and
                      type(actual_epoch) is int and actual_epoch == epoch and
                      type(actual_batch) is int and actual_batch == batch, 'Learning-rate counter mismatch')
        guard.require(isinstance(rates, (list, tuple)) and 0 < len(rates) <= 100 and
                      all(finite_number(x) and x >= 0 for x in rates), 'Invalid learning rate')
        if group_count is None:
            group_count = len(rates)
            writer.writerow(['attempt', 'actual_epoch', 'batch'] + [f'group{i}' for i in range(group_count)])
        guard.require(len(rates) == group_count, 'Learning-rate group count changed')
        writer.writerow([attempt, actual_epoch, actual_batch] + [repr(x) for x in rates])
    return output.getvalue().encode('utf-8')


def write_checkpoint(path, writer):
    guard.require(callable(writer), 'Checkpoint writer is required')
    with Path(path).open('xb') as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    guard.regular_file(path)
    return guard.digest_file(path)


def copy_evidence(source, destination, expected_sha):
    before = guard.regular_file(source)
    with Path(source).open('rb') as src, Path(destination).open('xb') as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    after = guard.regular_file(source)
    guard.require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                  'Source changed during copy')
    guard.require(guard.digest_file(source) == expected_sha and guard.digest_file(destination) == expected_sha,
                  'Evaluation evidence digest mismatch')


def stage_directory(run_root, arm, epoch):
    root_path = Path(run_root).absolute()
    guard.require(not root_path.is_symlink() and not getattr(root_path, 'is_junction', lambda: False)(),
                  'Linked run root')
    root = root_path.resolve(strict=True)
    guard.require(root.is_dir(), 'Run root is not a directory')
    current = root
    for part in ('runs', arm):
        current = current / part
        guard.require(current.is_dir() and not current.is_symlink() and
                      not getattr(current, 'is_junction', lambda: False)(), 'Invalid arm directory')
    epochs = current / 'epochs'
    guard.require(not epochs.is_symlink() and not getattr(epochs, 'is_junction', lambda: False)(), 'Linked epoch directory')
    epochs.mkdir(exist_ok=True)
    stage = epochs / ('epoch_%04d' % epoch)
    guard.require(stage.resolve().is_relative_to(root) and not stage.is_symlink(), 'Unsafe stage directory')
    stage.mkdir(exist_ok=False)
    return current, stage


def begin(run_root, run_id, arm, protocol_sha256, epoch, identities, metrics, prediction_sha,
          arrays_sha, evaluated_tensor_sha, rng_before, control_before, learning_rates, checkpoint_writer):
    """Capture the evaluated state before any optional late-stage restoration."""
    guard.validate_identity(run_id, arm, protocol_sha256)
    guard.require(type(epoch) is int and 0 <= epoch < guard.EPOCHS, 'Invalid epoch')
    guard.require(isinstance(identities, dict) and set(identities) == IDENTITY_FIELDS and
                  all(isinstance(v, str) and guard.HEX64.fullmatch(v) for v in identities.values()),
                  'Incomplete source or data identity')
    guard.require(isinstance(metrics, dict) and set(metrics) == METRICS and
                  all(finite_number(v) for v in metrics.values()), 'Invalid evaluation metrics')
    for value in (prediction_sha, arrays_sha, evaluated_tensor_sha):
        guard.require(isinstance(value, str) and guard.HEX64.fullmatch(value), 'Invalid evidence digest')
    rng = validate_rng(rng_before)
    control = validate_control(control_before, epoch)
    guard.require(control['best_epoch'] < epoch, 'Pre-evaluation best must precede the current epoch')
    lr_bytes = learning_rate_bytes(epoch, learning_rates)
    guard.require(callable(checkpoint_writer), 'Checkpoint writer is required')
    out, stage = stage_directory(run_root, arm, epoch)
    evaluated_sha = write_checkpoint(stage / 'evaluated.pt', checkpoint_writer)
    copy_evidence(out / 'latest.predictions.json', stage / 'predictions.json', prediction_sha)
    copy_evidence(out / 'latest.evaluation_arrays.npz', stage / 'evaluation_arrays.npz', arrays_sha)
    with (stage / 'learning_rates.csv.gz').open('xb') as stream:
        with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as compressed:
            compressed.write(lr_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    return {'stage': stage, 'run_id': run_id, 'arm': arm, 'protocol_sha256': protocol_sha256,
            'epoch_index': epoch, 'identities': dict(identities), 'metrics': dict(metrics),
            'evaluated_checkpoint_sha256': evaluated_sha, 'prediction_sha256': prediction_sha,
            'evaluation_arrays_sha256': arrays_sha, 'evaluated_tensor_sha256': evaluated_tensor_sha,
            'rng_before': rng, 'control_before': control}


def finish(handle, restored, continuation_tensor_sha, rng_after, control_after, rng_probe,
           continuation_writer=None, nonce=None):
    """Publish a manifest after the training loop has handled its restore rule."""
    guard.require(isinstance(handle, dict) and type(restored) is bool, 'Invalid evidence handle')
    epoch = handle['epoch_index']
    guard.require(not restored or epoch >= 296, 'Unexpected post-evaluation restoration')
    guard.require(isinstance(continuation_tensor_sha, str) and guard.HEX64.fullmatch(continuation_tensor_sha),
                  'Invalid continuation tensor digest')
    after_rng = validate_rng(rng_after)
    after_control = validate_control(control_after, epoch)
    guard.require(callable(rng_probe), 'RNG probe is required')
    guard.require(after_rng == handle['rng_before'], 'RNG changed while collecting epoch evidence')
    before = handle['control_before']
    improved = before['best_ap'] is None or handle['metrics']['AP'] > before['best_ap']
    guard.require(restored == (epoch >= 296 and not improved), 'Restore flag disagrees with the frozen selection rule')
    if improved:
        guard.require(after_control['best_epoch'] == epoch and after_control['best_ap'] == handle['metrics']['AP'],
                      'Updated best result does not match this evaluation')
    else:
        guard.require(after_control['best_epoch'] == before['best_epoch'] and after_control['best_ap'] == before['best_ap'],
                      'Best result changed on a non-improving epoch')
    if not restored:
        guard.require(continuation_writer is None and continuation_tensor_sha == handle['evaluated_tensor_sha256'],
                      'No-restore continuation does not match evaluated EMA')
        for key in ('last_epoch', 'ema_updates', 'ema_decay'):
            guard.require(after_control[key] == handle['control_before'][key], 'Unexpected no-restore state change: ' + key)
    else:
        guard.require(callable(continuation_writer), 'Restored state requires a separate checkpoint')
    stage = handle['stage']
    continuation_sha = write_checkpoint(stage / 'continuation.pt', continuation_writer) if restored else None
    guard.require(validate_rng(rng_probe()) == after_rng, 'RNG changed while writing continuation evidence')
    record = {'run_id': handle['run_id'], 'arm': handle['arm'], 'protocol_sha256': handle['protocol_sha256'],
              'epoch_index': epoch, 'optimizer_updates': (epoch + 1) * guard.BATCHES,
              'validation_passes': epoch + 1, 'restore_after_evaluation': restored,
              'evaluated_checkpoint_sha256': handle['evaluated_checkpoint_sha256'],
              'continuation_checkpoint_sha256': continuation_sha,
              'prediction_sha256': handle['prediction_sha256'], 'evaluation_arrays_sha256': handle['evaluation_arrays_sha256'],
              'evaluated_tensor_sha256': handle['evaluated_tensor_sha256'],
              'continuation_tensor_sha256': continuation_tensor_sha, 'identities': handle['identities'],
              'metrics': handle['metrics'], 'rng_before': handle['rng_before'], 'rng_after': after_rng,
              'control_before': handle['control_before'], 'control_after': after_control}
    data = guard.encode_json(record)
    guard.require(len(data) <= guard.MAX_MANIFEST_BYTES, 'Phase record is too large')
    with (stage / 'phase_record.json').open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    manifest = barrier.publish_manifest(stage, handle['run_id'], handle['arm'], handle['protocol_sha256'],
                                        epoch, restored, nonce=nonce)
    guard.require(validate_rng(rng_probe()) == after_rng, 'RNG changed during manifest publication')
    return manifest
