"""One isolated phase of the explicitly approved BASE diagnostic."""
from pathlib import Path
import argparse
import gzip
import json
import math
import os
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
import diagnostic_contract as c

def verify_prepared_inputs(validation_only):
    c.auth_check()
    c.package_check()
    ready = c.read(c.ROOT / 'setup_ready.json')
    c.require(ready['status'] == 'SETUP_READY_NO_MODEL_EXECUTION' and ready['run_id'] == c.RUN_ID,
              'Setup incomplete')
    c.require(ready['authorization_sha256'] == c.AUTH_SHA and
              ready['code_manifest_sha256'] == c.sha(c.ROOT / 'code_manifest.json'), 'Setup binding changed')
    c.require(c.sha(c.ROOT / 'original_protocol.json') == c.PROTOCOL_SHA, 'Original protocol changed')
    c.require(c.sha(c.ROOT / 'dataset_bundle_manifest.json') == c.DATA_MANIFEST_SHA, 'Dataset manifest changed')
    c.require(c.sha(c.ROOT / 'input/best.pt') == c.CHECKPOINT_SHA, 'Input checkpoint changed')
    c.require(c.sha(c.ROOT / 'execution_configs/BASE.yml') == ready['diagnostic_config_sha256'], 'Diagnostic config changed')
    protocol = c.read(c.ROOT / 'original_protocol.json')
    for item in protocol['frozen_files']:
        if item['path'].startswith('source_snapshot/') or item['path'] in ['fair_runner.py', 'source_loader.py']:
            c.require(c.sha(c.ROOT / item['path']) == item['sha256'], 'Copied source changed: ' + item['path'])
    manifest = c.read(c.ROOT / 'dataset_bundle_manifest.json')
    items = [x for x in manifest['files'] if not validation_only or x['path'].startswith('val/')]
    c.require(len(items) == (549 if validation_only else 7021), 'Unexpected data count')
    for item in items:
        p = c.CACHE / item['path']
        c.require(p.is_file() and not p.is_symlink() and p.stat().st_size == item['bytes'] and
                  c.sha(p) == item['sha256'], 'Cached data changed: ' + item['path'])
    return ready

def lock_phase(phase):
    path = c.ROOT / 'locks' / (phase + '.lock')
    with path.open('x', encoding='utf-8') as f:
        f.write(str(os.getpid()) + '\n')
        f.flush()
        os.fsync(f.fileno())
    c.fsync_dir(path.parent)

def progress(phase, **values):
    c.atomic_json(c.ROOT / 'worker_live.json', {'phase': phase, 'updated_utc': c.now(),
                                                'pid': os.getpid(), **values})

def start_runtime():
    # Imported only after explicit authorization and prepared-input checks.
    import fair_runner as fair
    fair.controls()
    return fair

def checkpoint_load(path, torch):
    return torch.load(path, map_location='cpu', weights_only=True)

def atomic_checkpoint(path, state, torch):
    temp = path.with_name(path.name + '.partial')
    with temp.open('xb') as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    c.require(not path.exists() and not path.is_symlink(), 'Checkpoint output already exists')
    os.replace(temp, path)
    c.fsync_dir(path.parent)

def eval_output(fair, model, post, val, evaluator, stage, phase, epoch, checkpoint_sha):
    metrics, pred_sha, arrays_sha, weights_sha = fair.evaluate(model, post, val, evaluator, stage, epoch, None)
    c.finite_metrics(metrics)
    os.replace(stage / 'latest.predictions.json', stage / 'predictions.json')
    os.replace(stage / 'latest.evaluation_arrays.npz', stage / 'evaluation_arrays.npz')
    c.require(c.sha(stage / 'predictions.json') == pred_sha and c.sha(stage / 'evaluation_arrays.npz') == arrays_sha,
              'Evaluation output changed')
    c.atomic_json(stage / 'metrics.json', {'phase': phase, 'epoch_index': epoch, 'metrics': metrics,
                  'prediction_sha256': pred_sha, 'evaluation_arrays_sha256': arrays_sha,
                  'evaluated_tensor_sha256': weights_sha, 'checkpoint_sha256': checkpoint_sha,
                  'images': 548, 'postprocessor_predictions_per_image': 300}, new=True)
    return metrics, weights_sha

def backup_stage(phase, epoch):
    manifest = c.make_backup_manifest(phase, epoch)
    progress(phase, status='WAITING_FOR_LOCAL_BACKUP', epoch_index=epoch,
             optimizer_updates=manifest['optimizer_updates'], validation_passes=manifest['validation_passes'])
    return c.await_backup(phase, manifest)

def validate_dispatch(phase):
    dispatch = c.read(c.ROOT / 'dispatch.json')
    c.require(dispatch['status'] == 'DISPATCHED' and dispatch['phase'] == phase,
              'Unexpected or missing controller dispatch')
    c.require(dispatch['run_id'] == c.RUN_ID and dispatch['authorization_sha256'] == c.AUTH_SHA,
              'Dispatch binding mismatch')
    return dispatch

def eval_phase(phase, dispatch):
    import torch
    fair = start_runtime()
    cfg, model, criterion, post, train, val, evaluator, lrfn = fair.build('BASE')
    c.require(train.persistent_workers is False and val.persistent_workers is False, 'Persistent workers not supported')
    if phase in ['initial_ema', 'initial_raw']:
        path = c.ROOT / 'input/best.pt'
        checkpoint_sha, epoch = c.CHECKPOINT_SHA, c.START_EPOCH
    else:
        result = c.read(c.ROOT / 'training_result.json')
        c.require(result['status'] in ['AP_DROP_STOP', 'MAX_EPOCHS_STOP'], 'Training did not finish at an approved boundary')
        epoch = result['epoch_index']
        path = c.ROOT / 'phases' / ('epoch_%04d' % epoch) / 'checkpoint.pt'
        checkpoint_sha = result['checkpoint_sha256']
    c.require(c.sha(path) == checkpoint_sha, 'Evaluation checkpoint hash mismatch')
    checkpoint = checkpoint_load(path, torch)
    c.require(checkpoint['arm'] == 'BASE' and checkpoint['actual_loop_epoch'] == epoch, 'Wrong checkpoint identity')
    state = checkpoint['ema']['module'] if phase == 'initial_ema' else checkpoint['model']
    model.load_state_dict(state, strict=True)
    model.cuda()
    post.cuda()
    fair.seeds(0)
    stage = c.ROOT / 'phases' / phase
    stage.mkdir(exist_ok=False)
    updates, validations = c.expected_counters(phase, epoch)
    progress(phase, status='VALIDATING', epoch_index=epoch, optimizer_updates=updates,
             validation_passes=validations)
    metrics, tensor_sha = eval_output(fair, model, post, val, evaluator, stage, phase, epoch, checkpoint_sha)
    if phase == 'initial_ema':
        saved = c.read(c.ROOT / 'input/best.metrics.json')
        c.require(tensor_sha == saved['ema_tensor_sha256'], 'Initial EMA tensor identity mismatch')
    c.atomic_json(stage / 'phase_record.json', {'phase': phase, 'epoch_index': epoch,
                  'optimizer_updates': updates, 'validation_passes': validations,
                  'evaluated_tensor_sha256': tensor_sha, 'completed_utc': c.now()}, new=True)
    ack = backup_stage(phase, epoch)
    progress(phase, status='BACKUP_CONFIRMED', epoch_index=epoch, **ack)
    return {'phase': phase, 'epoch_index': epoch, 'metrics': metrics, 'evaluated_tensor_sha256': tensor_sha, **ack}

def capture_rng(random, np, torch, train, val):
    name, values, pos, has_gauss, cached_gaussian = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': {'name': name, 'values': values.tolist(), 'pos': pos,
                      'has_gauss': has_gauss, 'cached_gaussian': cached_gaussian},
            'torch_cpu': torch.get_rng_state(), 'torch_cuda_all': torch.cuda.get_rng_state_all(),
            'train_loader_generator': train.generator.get_state(),
            'train_sampler_generator': train.sampler.generator.get_state(),
            'validation_loader_generator': val.generator.get_state()}

def train_phase(dispatch):
    import random
    import numpy as np
    import torch
    fair = start_runtime()
    cfg, model, criterion, post, train, val, evaluator, lrfn = fair.build('BASE')
    c.require(train.persistent_workers is False and val.persistent_workers is False, 'Persistent workers not supported')
    checkpoint = checkpoint_load(c.ROOT / 'input/best.pt', torch)
    c.require(checkpoint['arm'] == 'BASE' and checkpoint['actual_loop_epoch'] == c.START_EPOCH and
              checkpoint['optimizer_step_attempts'] == (c.START_EPOCH + 1) * c.BATCHES,
              'Unexpected initial checkpoint state')
    model.load_state_dict(checkpoint['model'], strict=True)
    model.cuda()
    criterion.cuda()
    post.cuda()
    optimizer = cfg.optimizer
    _ = cfg.lr_scheduler
    warmup = cfg.lr_warmup_scheduler
    ema = cfg.ema.to('cuda')
    optimizer.load_state_dict(checkpoint['optimizer'])
    ema.load_state_dict(checkpoint['ema'], strict=True)
    criterion.load_state_dict(checkpoint['criterion'], strict=True)
    post.load_state_dict(checkpoint['postprocessor'], strict=True)
    warmup.load_state_dict(checkpoint['lr_warmup_scheduler'])
    c.require(ema.updates == checkpoint['optimizer_step_attempts'], 'EMA update count mismatch')
    expected_initial = c.read(c.ROOT / 'input/best.metrics.json')
    c.require([g['lr'] for g in optimizer.param_groups] == expected_initial['post_final_update_lrs'],
              'Restored optimizer learning rates differ')
    base_lrs = [g['initial_lr'] for g in optimizer.param_groups]
    c.require(len(base_lrs) == 4, 'Unexpected optimizer group count')
    anchor = float(dispatch['initial_ema_AP'])
    c.require(math.isfinite(anchor), 'Invalid EMA anchor')
    fair.seeds(0)
    updates = 0
    last_epoch = checkpoint['last_epoch']
    result = None
    for epoch in range(c.START_EPOCH + 1, c.FINAL_EPOCH + 1):
        c.require(updates == (epoch - c.START_EPOCH - 1) * c.BATCHES, 'Unexpected update count')
        phase = 'epoch_%04d' % epoch
        stage = c.ROOT / 'phases' / phase
        stage.mkdir(exist_ok=False)
        fair.prepare_epoch(train, epoch)
        model.train()
        criterion.train()
        loss_total, grad_sum, grad_max, grad_count = 0.0, 0.0, 0.0, 0
        first_lrs = [g['lr'] for g in optimizer.param_groups]
        started = time.monotonic()
        with gzip.open(stage / 'learning_rates.csv.gz', 'wt', compresslevel=1, encoding='utf-8', newline='') as lrlog:
            lrlog.write('additional_update,absolute_attempt,actual_epoch,batch,' +
                        ','.join('group%d' % i for i in range(len(base_lrs))) + '\n')
            for step, (samples, targets) in enumerate(train):
                c.require(step < c.BATCHES and updates < c.MAX_UPDATES, 'Optimizer budget would be exceeded')
                absolute_attempt = (c.START_EPOCH + 1) * c.BATCHES + updates
                lrlog.write('%d,%d,%d,%d,' % (updates, absolute_attempt, epoch, step) +
                            ','.join(repr(g['lr']) for g in optimizer.param_groups) + '\n')
                samples = samples.cuda(non_blocking=True)
                targets = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in targets]
                outputs = model(samples, targets=targets)
                loss_dict = criterion(outputs, targets, epoch=epoch, step=step,
                                      global_step=epoch * c.BATCHES + step, epoch_step=c.BATCHES)
                loss = sum(loss_dict.values())
                c.require(bool(torch.isfinite(loss)), 'Nonfinite loss')
                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1, error_if_nonfinite=True)
                grad_value = float(grad_norm.detach())
                c.require(math.isfinite(grad_value), 'Nonfinite gradient norm')
                optimizer.step()
                updates += 1
                ema.update(model)
                for group, initial_lr in zip(optimizer.param_groups, base_lrs):
                    group['lr'] = lrfn(c.BATCHES * 320, 2000, c.BATCHES * 164, c.BATCHES * 24,
                                       epoch * c.BATCHES + step, initial_lr, initial_lr * cfg.lr_gamma)
                loss_total += float(loss.detach())
                grad_sum += grad_value
                grad_max = max(grad_max, grad_value)
                grad_count += 1
                if step % 100 == 0 or step == c.BATCHES - 1:
                    progress(phase, status='TRAINING', epoch_index=epoch, batch=step + 1,
                             optimizer_updates=updates, validation_passes=2 + epoch - c.START_EPOCH - 1)
            c.require(grad_count == c.BATCHES and updates == (epoch - c.START_EPOCH) * c.BATCHES,
                      'Incomplete training epoch')
        last_epoch += 1
        validations = 2 + epoch - c.START_EPOCH
        c.require(validations <= c.MAX_VALIDATIONS - 1, 'No budget left for final raw evaluation')
        progress(phase, status='VALIDATING_EMA', epoch_index=epoch,
                 optimizer_updates=updates, validation_passes=validations)
        # The routine EMA validation is in the training process, as in the frozen driver.
        metrics, tensor_sha = eval_output(fair, ema.module, post, val, evaluator, stage, phase, epoch, None)
        state = {'model': model.state_dict(), 'ema': ema.state_dict(), 'optimizer': optimizer.state_dict(),
                 'criterion': criterion.state_dict(), 'postprocessor': post.state_dict(),
                 'lr_warmup_scheduler': warmup.state_dict(), 'last_epoch': last_epoch,
                 'actual_loop_epoch': epoch, 'optimizer_step_attempts': (c.START_EPOCH + 1) * c.BATCHES + updates,
                 'additional_optimizer_updates': updates, 'validation_passes': validations,
                 'arm': 'BASE', 'weight_source': 'ema.module', 'original_protocol_sha256': c.PROTOCOL_SHA,
                 'original_checkpoint_sha256': c.CHECKPOINT_SHA, 'rng_state': capture_rng(random, np, torch, train, val)}
        atomic_checkpoint(stage / 'checkpoint.pt', state, torch)
        checkpoint_sha = c.sha(stage / 'checkpoint.pt')
        # Bind the saved checkpoint after the evaluation without changing metric values.
        metric_path = stage / 'metrics.json'
        saved_metrics = c.read(metric_path)
        c.require(saved_metrics['checkpoint_sha256'] is None, 'Unexpected existing checkpoint binding')
        saved_metrics['checkpoint_sha256'] = checkpoint_sha
        c.atomic_json(metric_path, saved_metrics)
        c.atomic_json(stage / 'gradient_summary.json', {'count': grad_count, 'nonfinite_count': 0,
                      'mean_preclip_norm': grad_sum / grad_count, 'maximum_preclip_norm': grad_max,
                      'clip_max_norm': 0.1}, new=True)
        c.atomic_json(stage / 'phase_record.json', {'phase': phase, 'epoch_index': epoch,
                      'optimizer_updates': updates, 'validation_passes': validations,
                      'train_loss_mean': loss_total / c.BATCHES, 'first_executed_lrs': first_lrs,
                      'post_final_update_lrs': [g['lr'] for g in optimizer.param_groups],
                      'ema_tensor_sha256': tensor_sha, 'checkpoint_sha256': checkpoint_sha,
                      'elapsed_seconds': time.monotonic() - started, 'completed_utc': c.now()}, new=True)
        ack = backup_stage(phase, epoch)
        drop = anchor - metrics['AP']
        stop = drop >= 0.05 or epoch == c.FINAL_EPOCH
        progress(phase, status='BACKUP_CONFIRMED', epoch_index=epoch, optimizer_updates=updates,
                 validation_passes=validations, AP=metrics['AP'], AP_drop_from_anchor=drop, **ack)
        if stop:
            result = {'status': 'AP_DROP_STOP' if drop >= 0.05 else 'MAX_EPOCHS_STOP',
                      'epoch_index': epoch, 'optimizer_updates': updates, 'validation_passes': validations,
                      'initial_ema_AP': anchor, 'final_ema_AP': metrics['AP'], 'AP_drop_from_anchor': drop,
                      'checkpoint_sha256': checkpoint_sha, 'backup_manifest_sha256': ack['manifest_sha256'],
                      'completed_utc': c.now()}
            break
    c.require(result is not None and updates <= c.MAX_UPDATES, 'Training finished without a valid stop')
    c.atomic_json(c.ROOT / 'training_result.json', result, new=True)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['initial_ema', 'initial_raw', 'train', 'terminal_raw'])
    args = parser.parse_args()
    c.auth_check()
    c.package_check()
    dispatch = validate_dispatch(args.phase)
    lock_phase(args.phase)
    try:
        progress(args.phase, status='VERIFYING_INPUTS')
        verify_prepared_inputs(validation_only=args.phase != 'train')
        result = train_phase(dispatch) if args.phase == 'train' else eval_phase(args.phase, dispatch)
        c.atomic_json(c.ROOT / ('worker_result_' + args.phase + '.json'), result, new=True)
        progress(args.phase, status='PHASE_COMPLETE')
        print(json.dumps({'status': 'PHASE_COMPLETE', 'phase': args.phase}), flush=True)
    except BaseException:
        c.atomic_json(c.ROOT / ('worker_failure_' + args.phase + '.json'),
                      {'status': 'PHASE_FAILED_NO_AUTOMATIC_RETRY', 'phase': args.phase,
                       'failed_utc': c.now(), 'traceback': traceback.format_exc()}, new=True)
        raise

if __name__ == '__main__':
    main()
