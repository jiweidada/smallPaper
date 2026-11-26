import argparse, gzip, json, math, os, shutil, subprocess, time, traceback
from pathlib import Path
EPOCHS = 320
BATCHES = 1617
STAGE_TWO = 296
ROLLING_FORMAT = 'fair_runner_post_policy_epoch_v1'
BEST_FILES = ('best.pt', 'best.metrics.json', 'best.predictions.json', 'best.evaluation_arrays.npz')

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def sync_dir(path):
    if os.name == 'posix':
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

def write_atomic(path, write):
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('xb') as stream:
        write(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_dir(path.parent)

def copy_atomic(source, target):
    def copy(stream):
        with source.open('rb') as input_stream:
            shutil.copyfileobj(input_stream, stream)

    write_atomic(target, copy)

def write_checkpoint(torch, path, state):
    write_atomic(path, lambda stream: torch.save(state, stream))

def write_json(path, value):
    payload = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
    write_atomic(path, lambda stream: stream.write(payload))

def package_path(output, ref):
    return output / 'selected' / f"epoch{ref['epoch']:04d}"

def capture_rng(fair, train, val):
    name, keys, position, has_gauss, cached = fair.np.random.get_state()
    return {
        'python': fair.random.getstate(),
        'numpy': (name, keys.tolist(), position, has_gauss, cached),
        'torch_cpu': fair.torch.get_rng_state(),
        'torch_cuda_all': fair.torch.cuda.get_rng_state_all(),
        'train_loader': train.generator.get_state(),
        'train_sampler': train.sampler.generator.get_state(),
        'validation_loader': val.generator.get_state(),
    }


def restore_rng(fair, saved, train, val):
    torch = fair.torch
    require(not train.persistent_workers and not val.persistent_workers
            and isinstance(val.sampler, torch.utils.data.SequentialSampler), 'Unexpected loader workers/sampler')
    rng = saved['rng_state']
    require(set(rng) == {'python', 'numpy', 'torch_cpu', 'torch_cuda_all', 'train_loader',
                        'train_sampler', 'validation_loader'}, 'Incomplete saved RNG state')
    fair.random.setstate(rng['python'])
    name, keys, position, has_gauss, cached = rng['numpy']
    fair.np.random.set_state((name, fair.np.asarray(keys, dtype=fair.np.uint32), position, has_gauss, cached))
    torch.set_rng_state(rng['torch_cpu'])
    require(len(rng['torch_cuda_all']) == torch.cuda.device_count() == 1, 'CUDA device count changed')
    torch.cuda.set_rng_state_all(rng['torch_cuda_all'])
    train.generator.set_state(rng['train_loader'])
    train.sampler.generator.set_state(rng['train_sampler'])
    val.generator.set_state(rng['validation_loader'])
    current = capture_rng(fair, train, val)
    require(current['python'] == rng['python'] and current['numpy'] == rng['numpy'], 'Python/NumPy RNG differs')
    for key in ['torch_cpu', 'train_loader', 'train_sampler', 'validation_loader']:
        require(torch.equal(current[key], rng[key]), 'Restored RNG differs: ' + key)
    require(all(torch.equal(a, b) for a, b in zip(current['torch_cuda_all'], rng['torch_cuda_all'])),
            'Restored CUDA RNG differs')

def train_run(fair, args, bindings, gate):
    torch = fair.torch
    out = args.output
    arm = args.arm
    resume_path = out.parent / 'recovery_anchor/next_epoch.pt'
    require(fair.sha(resume_path) == args.recovery['checkpoint_sha256'], 'Recovery anchor changed')
    initial = torch.load(resume_path, map_location='cpu', weights_only=True)
    first_epoch = initial['next_epoch']
    require(initial['resume_format'] == ROLLING_FORMAT and first_epoch == 29
            and initial['actual_loop_epoch'] == 28 and initial['last_epoch'] == 28
            and initial['optimizer_step_attempts'] == 46893, 'Unexpected recovery boundary')
    require(all(initial[k] == v for k, v in bindings.items()), 'Recovery bindings differ')
    require(out.is_dir(), 'Missing restored output directory')
    attempts = initial['optimizer_step_attempts']
    last_epoch = initial['last_epoch']
    best, stage_one = initial['best'], initial['stage_one']
    best_ap = fair.readj(package_path(out, best) / 'best.metrics.json')['metrics']['AP']
    origin = fair.readj(out.parent / 'inputs/preflight.json')['initialization']
    receipt = {
        **bindings, 'config_sha256': bindings['configuration_sha256'],
        'status': 'INITIALIZING', 'pid': os.getpid(), 'started_utc': fair.now(),
        'actual_loop_epoch': first_epoch - 1, 'completed_epochs': first_epoch, 'next_epoch': first_epoch,
        'first_epoch_in_this_segment': first_epoch, 'optimizer_step_attempts': attempts,
        'resumed_from_checkpoint_sha256': args.recovery['checkpoint_sha256'],
        'recovery_addendum_sha256': args.recovery_sha256,
        'recovery_scope': 'Explicitly authorized continuation from retained round29 on another matched RTX3090',
        'initialization_origin': origin, 'runner_sha256': fair.sha(Path(__file__)),
        'torch_version': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
    }
    fair.savej(out / 'receipt.json', receipt)
    try:
        require((out / 'selected').is_dir(), 'Missing restored selected packages')
        fair.controls()
        cfg, model, criterion, post, train, val, evaluator, lrfn = fair.build(arm)
        frozen_initial = torch.load(out.parent / 'inputs/initial.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(frozen_initial, strict=True)
        require(fair.tensor_hash(model.state_dict()) == fair.readj(out.parent / 'inputs/preflight.json')['initial_tensor_sha256'],
                'Training initial tensors differ')
        del frozen_initial
        model.cuda(); criterion.cuda(); post.cuda()
        optimizer = cfg.optimizer
        _ = cfg.lr_scheduler
        warmup = cfg.lr_warmup_scheduler
        ema = cfg.ema.to('cuda')
        base_lrs = [group['initial_lr'] for group in optimizer.param_groups]

        def state(loop_epoch):
            return {
                **bindings, 'model': model.state_dict(), 'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(), 'criterion': criterion.state_dict(),
                'postprocessor': post.state_dict(), 'lr_warmup_scheduler': warmup.state_dict(),
                'last_epoch': last_epoch, 'actual_loop_epoch': loop_epoch, 'optimizer_step_attempts': attempts,
            }

        def load_state(saved):
            model.load_state_dict(saved['model'], strict=True)
            ema.load_state_dict(saved['ema'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            criterion.load_state_dict(saved['criterion'], strict=True)
            post.load_state_dict(saved['postprocessor'], strict=True)
            warmup.load_state_dict(saved['lr_warmup_scheduler'])
            require([group['initial_lr'] for group in optimizer.param_groups] == base_lrs,
                    'Restored optimizer base learning rates differ from frozen config')

        def restore(loop_epoch, reason):
            nonlocal last_epoch
            path = package_path(out, stage_one) / 'best.pt'
            require(fair.sha(path) == stage_one['checkpoint_sha256'], 'Stage-one checkpoint SHA mismatch')
            previous_last_epoch = last_epoch
            saved = torch.load(path, map_location='cpu', weights_only=True)
            load_state(saved)
            last_epoch = saved['last_epoch']
            fair.appendj(out / 'restore_events.jsonl', {
                'actual_loop_epoch': loop_epoch, 'reason': reason, 'checkpoint_sha256': stage_one['checkpoint_sha256'],
                'previous_last_epoch': previous_last_epoch, 'restored_last_epoch': last_epoch,
                'monotonic_step_attempts': attempts, 'ema_updates': ema.updates,
            })

        def commit(loop_epoch, evaluated):
            require(attempts == (loop_epoch + 1) * BATCHES, 'Wrong attempt count at epoch boundary')
            rolling = out / 'next_epoch.pt'
            write_checkpoint(torch, rolling, {
                **state(loop_epoch), 'resume_format': ROLLING_FORMAT, 'next_epoch': loop_epoch + 1,
                'ema_decay_attribute': ema.decay, 'rng_state': capture_rng(fair, train, val),
                'best': best, 'stage_one': stage_one, 'initialization_origin': origin, 'latest_evaluated_row': evaluated,
            })
            rolling_sha = fair.sha(rolling)
            receipt.update(status='RUNNING', phase='awaiting_local_backup', actual_loop_epoch=loop_epoch,
                           backup_epoch=loop_epoch, next_epoch=loop_epoch+1, optimizer_step_attempts=attempts,
                           rolling_checkpoint_sha256=rolling_sha, updated_utc=fair.now())
            fair.savej(out / 'receipt.json', receipt)
            print(json.dumps({'arm': arm, 'status': 'AWAITING_LOCAL_BACKUP', 'backup_epoch': loop_epoch,
                              'rolling_checkpoint_sha256': rolling_sha}), flush=True)
            receipt['local_backup_verified'] = gate.protect(loop_epoch, rolling_sha, best, stage_one, evaluated)
            return rolling_sha

        load_state(initial)
        ema.decay = initial['ema_decay_attribute']
        require(fair.tensor_hash(model.state_dict()) == fair.tensor_hash(initial['model']),
                'Restored raw model differs')
        require(fair.tensor_hash(ema.module.state_dict()) == fair.tensor_hash(initial['ema']['module']),
                'Restored EMA differs')
        restore_rng(fair, initial, train, val)
        receipt.update(status='RUNNING', phase='epoch_complete',
                       rolling_checkpoint_sha256=args.recovery['checkpoint_sha256'],
                       best_epoch=best['epoch'], best_AP=best_ap, updated_utc=fair.now())
        fair.savej(out / 'receipt.json', receipt)
        del initial
        expected = fair.readj(out.parent / 'inputs/first100.json')

        lrpath = out / 'executed_learning_rates.csv.gz'
        require(lrpath.is_file(), 'Missing retained cumulative LR history')
        for epoch in range(first_epoch, EPOCHS):
            started = time.monotonic()
            fair.prepare_epoch(train, epoch)
            if epoch == STAGE_TWO:
                restore(epoch, 'stage_two_entry')
                ema.decay = train.collate_fn.ema_restart_decay
            model.train()
            criterion.train()
            total_loss = 0.0
            first_lr = [g['lr'] for g in optimizer.param_groups]
            with gzip.open(lrpath, 'at', compresslevel=1) as lrlog:
                for step, (samples, targets) in enumerate(train):
                    if epoch == 0 and step < 100:
                        observed = {'step': step, 'sha256': fair.batch_hash(samples, targets),
                                    'image_ids': [int(target['image_id']) for target in targets]}
                        require(observed == expected[step], 'Actual training inputs differ')
                        fair.appendj(out / 'actual_first100.jsonl', observed)
                    lrlog.write(f'{attempts},{epoch},{step},' + ','.join(repr(g['lr']) for g in optimizer.param_groups) + '\n')
                    samples = samples.cuda(non_blocking=True)
                    targets = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in targets]
                    outputs = model(samples, targets=targets)
                    loss_dict = criterion(outputs, targets, epoch=epoch, step=step, global_step=epoch*1617+step, epoch_step=1617)
                    loss = sum(loss_dict.values())
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError(f'Nonfinite loss epoch={epoch} batch={step}')
                    optimizer.zero_grad()
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1, error_if_nonfinite=True)
                    optimizer.step()
                    attempts += 1
                    ema.update(model)
                    # Preserve the LR applied after each optimizer and EMA update.
                    for group, init_lr in zip(optimizer.param_groups, base_lrs):
                        group['lr'] = lrfn(1617*320, 2000, 1617*164, 1617*24, epoch*1617+step, init_lr, init_lr*cfg.lr_gamma)
                    total_loss += float(loss.detach())
                    if step % 100 == 0 or step == 1616:
                        receipt.update(status='RUNNING', phase='training', actual_loop_epoch=epoch, batch=step+1,
                                       optimizer_step_attempts=attempts, last_loss=float(loss.detach()), updated_utc=fair.now())
                        fair.savej(out / 'receipt.json', receipt)
                        print(json.dumps({'arm': arm, 'epoch': epoch, 'batch': step+1,
                                          'loss': float(loss.detach()), 'attempts': attempts}), flush=True)
            last_epoch += 1
            receipt.update(phase='validation', actual_loop_epoch=epoch, updated_utc=fair.now())
            fair.savej(out / 'receipt.json', receipt)
            metrics, pred_sha, array_sha, weight_sha = fair.evaluate(ema.module, post, val, evaluator, out, epoch, None)
            row = {'epoch': epoch, 'last_epoch': last_epoch, 'optimizer_step_attempts': attempts, 'ema_updates': ema.updates,
                   'weight_source': 'ema.module', 'train_loss_mean': total_loss/1617, 'metrics': metrics,
                   'ema_tensor_sha256': weight_sha, 'prediction_sha256': pred_sha, 'evaluation_arrays_sha256': array_sha,
                   'first_executed_lrs': first_lr, 'post_final_update_lrs': [g['lr'] for g in optimizer.param_groups],
                   'seconds': time.monotonic()-started, 'created_utc': fair.now(),
                   'protocol_sha256': bindings['protocol_sha256'], 'config_sha256': bindings['configuration_sha256']}
            if epoch == EPOCHS - 1:
                terminal = out / 'terminal_epoch0319.pt'
                write_checkpoint(torch, terminal, state(epoch))
                copy_atomic(out / 'latest.predictions.json', out / 'terminal.predictions.json')
                copy_atomic(out / 'latest.evaluation_arrays.npz', out / 'terminal.evaluation_arrays.npz')
                write_json(out / 'terminal.metrics.json', {**row, 'checkpoint_sha256': fair.sha(terminal)})
            previous_best = best
            if metrics['AP'] > best_ap:
                best_ap = metrics['AP']
                folder = package_path(out, {'epoch': epoch})
                folder.mkdir()
                write_checkpoint(torch, folder / 'best.pt', state(epoch))
                copy_atomic(out / 'latest.predictions.json', folder / 'best.predictions.json')
                copy_atomic(out / 'latest.evaluation_arrays.npz', folder / 'best.evaluation_arrays.npz')
                checkpoint_sha = fair.sha(folder / 'best.pt')
                write_json(folder / 'best.metrics.json', {**row, 'checkpoint_sha256': checkpoint_sha})
                sync_dir(folder.parent)
                best = {'epoch': epoch, 'checkpoint_sha256': checkpoint_sha,
                        'metrics_sha256': fair.sha(folder / 'best.metrics.json')}
                if epoch < STAGE_TWO:
                    stage_one = best
            elif epoch >= STAGE_TWO:
                ema.decay -= 0.0001
                restore(epoch, 'late_epoch_without_new_global_best')
            fair.appendj(out / 'epochs.jsonl', row)
            rolling_sha = commit(epoch, row)
            if previous_best is not None and previous_best != best and previous_best != stage_one:
                obsolete = package_path(out, previous_best)
                for name in BEST_FILES:
                    (obsolete / name).unlink()
                obsolete.rmdir()
                sync_dir(obsolete.parent)
            receipt.update(phase='epoch_complete', completed_epochs=epoch+1, next_epoch=epoch+1,
                           best_epoch=best['epoch'], best_AP=best_ap, latest_metrics=metrics,
                           rolling_checkpoint_sha256=rolling_sha, updated_utc=fair.now())
            fair.savej(out / 'receipt.json', receipt)
            print(json.dumps({'arm': arm, 'epoch_complete': epoch, 'AP': metrics['AP'], 'APs': metrics['APs'],
                              'seconds': row['seconds'], 'rolling_checkpoint_sha256': rolling_sha}), flush=True)
        require(attempts == 517440, 'Wrong total optimizer attempt count')
        for name in BEST_FILES:
            copy_atomic(package_path(out, best) / name, out / name)
        copy_atomic(package_path(out, stage_one) / 'best.pt', out / 'best_stg1.pt')
        receipt.update(status='COMPLETED', completed_epochs=EPOCHS, optimizer_step_attempts=attempts, ended_utc=fair.now())
        write_json(out / 'receipt.json', receipt)
    except Exception:
        receipt.update(status='FAILED', ended_utc=fair.now(), traceback=traceback.format_exc())
        fair.savej(out / 'receipt.json', receipt)
        raise

def main():
    import fair_helpers as fair
    import backup_protocol as backup
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--recovery-sha256', required=True)
    args = parser.parse_args()
    plan = fair.readj(here / 'execution_protocol.json')
    recovery_path = here / 'recovery_addendum.json'
    require(fair.sha(recovery_path) == args.recovery_sha256, 'Recovery addendum differs')
    args.recovery = fair.readj(recovery_path)
    require(args.recovery['authorized'] is True and args.recovery['arm'] == 'NO_MSCA'
            and args.recovery['first_epoch_index'] == 29 and args.recovery['final_epoch_index'] == 319
            and args.recovery['new_segment_evaluation_limit'] == 291
            and args.recovery['new_segment_optimizer_update_limit'] == 470547
            and args.recovery['retained_optimizer_update_limit'] == 517440
            and args.recovery['original_protocol_sha256'] == args.protocol_sha256,
            'Unexpected explicitly authorized recovery scope')
    for item in args.recovery['recovery_files']:
        path = (here / item['path']).resolve(strict=True)
        require(path.is_relative_to(here) and fair.sha(path) == item['sha256'], 'Recovery input changed')
    require(fair.sha(Path(__file__)) == args.recovery['resume_runner_sha256'], 'Resume runner changed')
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == args.recovery['target_boot_id'],
            'Recovery target boot changed')
    require(fair.sha(here / 'execution_protocol.json') == args.protocol_sha256 == backup.PROTOCOL,
            'Execution protocol differs')
    require(plan['authorized'] is True and plan['epochs'] == 320 and plan['arm'] in ('NO_MPS', 'NO_MSCA'),
            'Unexpected authorization or experiment budget')
    args.arm = plan['arm']
    args.output = here / 'run'
    for item in plan['frozen_files']:
        path = (here / item['path']).resolve(strict=True)
        require(path.is_relative_to(here) and fair.sha(path) == item['sha256'], 'Frozen input changed')
    preflight = fair.readj(here / 'inputs/preflight.json')
    require(preflight['status'] == 'PREFLIGHT_PASSED_NO_OPTIMIZER_UPDATE' and preflight['optimizer_updates'] == 0,
            'Preflight did not pass')
    environment = fair.readj(here / 'environment.current.json')
    require(str(fair.torch.__version__) == environment['torch'] and fair.torch.version.cuda == environment['cuda']
            and fair.torch.backends.cudnn.version() == environment['cudnn'], 'Training runtime changed')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free',
                                   '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    require(gpu[0] == args.recovery['target_gpu_uuid'] and int(gpu[1]) >= 20000,
            'Recovery GPU differs or is occupied')
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                      '--format=csv,noheader,nounits'], text=True).strip()
    require(not active, 'GPU has another compute process')
    cache = fair.readj(here / 'cache_verification.json')
    cache_root = Path(cache['cache_root']).resolve(strict=True)
    for item in fair.readj(here / 'dataset_bundle_manifest.json')['files']:
        path = (cache_root / item['path']).resolve(strict=True)
        require(path.is_relative_to(cache_root) and fair.sha(path) == item['sha256'], 'Dataset changed')
    gate = backup.EpochGate(args.output)
    gate.require_probe()
    bindings = {'arm': args.arm, 'protocol_sha256': backup.PROTOCOL,
                'configuration_sha256': backup.CONFIG,
                'environment_sha256': fair.sha(here / 'environment.current.json'),
                'initialization_sha256': preflight['initial_file_sha256'],
                'parent_protocol_sha256': plan['parent_protocol_sha256'], 'weight_source': 'ema.module'}
    train_run(fair, args, bindings, gate)


if __name__ == '__main__':
    main()
