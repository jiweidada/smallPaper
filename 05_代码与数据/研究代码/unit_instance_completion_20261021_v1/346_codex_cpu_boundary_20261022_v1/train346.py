"""346 synthetic boundary acceptance only. No full-training launch path.

The loop is copied from sealed train345; the only in-loop edit renames epoch
timing metadata so a synthetic cursor cannot masquerade as executed history.
All loss/geometry/optimizer functions remain imported, never edited.
"""
import importlib,random,time
from pathlib import Path
C=importlib.import_module('346_context')
import numpy as np
import torch
torch.set_num_threads(2)
E=C.E;T=C.T
HARD_TOTAL_STEPS=E.HARD_TOTAL_STEPS
require=E.require;verify_file=E.verify_file;write_json=E.write_json
state_sha=E.state_sha;batch_sha=E.batch_sha
validate_bindings=E.validate_bindings;make_optimizer=E.make_optimizer;check_cursor=E.check_cursor

def execute_boundary(arm, seed, run_dir, device, max_steps, *, _mode='formal', _resume=None, _case=None):
    """CLI uses formal order; private acceptance worker also uses mini order/resume.

    max_steps is an absolute global endpoint. Resume(3 -> 6) executes three steps.
    Resume always writes to a fresh child run directory, keeping the parent sealed.
    """
    require(isinstance(max_steps, int) and 0 < max_steps <= HARD_TOTAL_STEPS, 'max_steps_budget')
    require(device == 'cpu', 'CPU_only_authorization')
    require(_mode == 'formal', '346_formal_only')
    require(_case in C.CASES, '346_unknown_case')
    specification=C.CASES[_case]
    require(max_steps == specification['end'], '346_endpoint_not_whitelisted')
    require(_resume is not None, '346_fixture_required')
    C.verify_sources()
    verify_file(_resume['path'], _resume['sha256'], '346_resume_receipt')
    preliminary=C.read(_resume['path'])
    require(preliminary.get('execution_kind') == C.KIND and preliminary.get('case') == _case,
            '346_fixture_kind_or_case')
    require(specification['start'] <= preliminary['step'] < max_steps,
            'resume_endpoint_not_after_checkpoint')
    require(max_steps - preliminary['step'] <= specification['maximum_actual_updates'],
            '346_actual_update_cap')
    run_dir = Path(run_dir).resolve()
    require(run_dir.is_relative_to(C.RUN_ROOT.resolve()), '346_run_directory_scope')
    require(not run_dir.exists(), 'run_directory_exists', run_dir)
    config, order, curve, binding = validate_bindings(arm, seed, _mode)
    binding=C.binding_for(binding,_case)
    require(preliminary['binding'] == binding, '346_preliminary_binding')
    run_dir.mkdir(parents=True, exist_ok=False)
    np.random.seed(config['head_seed'])
    random.seed(config['head_seed'])
    model = T.make_common_init(arm, seed).train()
    initial_sha = state_sha(model.state_dict())
    require(initial_sha == config['untrained_initial_state_sha256'], 'initial_state_SHA')
    require(sum(p.numel() for p in model.parameters()) == config['head_parameters'], 'parameter_count')
    require(all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters()), 'CPU_FP32')
    with open(run_dir / '346_initial_state.pt', 'xb') as f:
        torch.save({'model': model.state_dict(), 'state_sha256': initial_sha, 'binding': binding}, f)
    optimizer = make_optimizer(model, config)
    step = epoch = batch_index = 0
    step_logs, epoch_times = [], []
    epoch_partial_seconds = 0.0
    if _resume is not None:
        receipt_path = Path(_resume['path'])
        verify_file(receipt_path, _resume['sha256'], 'resume_receipt')
        receipt = C.read(receipt_path)
        require(receipt['binding'] == binding, 'resume_receipt_binding')
        verify_file(receipt['checkpoint_path'], receipt['checkpoint_sha256'], 'resume_checkpoint')
        checkpoint = torch.load(receipt['checkpoint_path'], map_location='cpu', weights_only=False)
        require(checkpoint['binding'] == binding and checkpoint['initial_state_sha256'] == initial_sha,
                'resume_checkpoint_binding')
        require(checkpoint['format'] == C.KIND and checkpoint['synthetic_origin_step'] == specification['start'],
                '346_checkpoint_kind')
        require(checkpoint['actual_optimizer_updates_since_fixture'] == checkpoint['step']-specification['start'],
                '346_actual_update_count')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        require([g['lr'] for g in optimizer.param_groups] == checkpoint['param_group_lr'], 'resume_group_lr')
        step, epoch, batch_index = checkpoint['step'], checkpoint['epoch'], checkpoint['batch_index']
        require(checkpoint['position_in_epoch'] == batch_index * config['batch'], 'resume_position')
        step_logs = checkpoint['step_logs']
        epoch_times = checkpoint['epoch_times']
        epoch_partial_seconds = checkpoint['epoch_partial_seconds']
        require(len(step_logs) == step-specification['start'], 'resume_log_length')
        require(checkpoint['model_state_sha256'] == state_sha(model.state_dict()), 'resume_model_SHA')
        T.restore_rng(checkpoint['rng'])
    check_cursor(step, epoch, batch_index, order, config)
    require(step < max_steps, 'resume_endpoint_not_after_checkpoint')
    write_json(run_dir / '346_run_binding.json', {'binding': binding, 'initial_state_sha256': initial_sha,
               'inherited_recipe': config, 'max_steps': max_steps, 'resume': _resume,
               'epoch_timing_policy': 'Synthetic acceptance: timing covers executed batches only. Cursor/order sizes are not counts of past training. Skipped prefix has no fabricated logs or elapsed time.'})
    local_steps = loaded_entries = 0
    checkpoint_receipts = []
    last_receipt = None

    def save_state():
        nonlocal last_receipt
        check_cursor(step, epoch, batch_index, order, config)
        path = run_dir / f'346_checkpoint_step{step:06d}.pt'
        payload = {'format': C.KIND, 'binding': binding,
                   'initial_state_sha256': initial_sha, 'model_state_sha256': state_sha(model.state_dict()),
                   'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'param_group_lr': [g['lr'] for g in optimizer.param_groups],
                   'step': step, 'epoch': epoch, 'batch_index': batch_index,
                   'position_in_epoch': batch_index * config['batch'], 'shuffle_sha256': binding['shuffle_sha256'],
                   'rng': T.capture_rng(), 'step_logs': step_logs, 'epoch_times': epoch_times,
                   'epoch_partial_seconds': epoch_partial_seconds,
                   'synthetic_origin_step': specification['start'],
                   'actual_optimizer_updates_since_fixture': step-specification['start'],
                   'real_training_steps':0}
        with open(path, 'xb') as f:
            torch.save(payload, f)
        receipt_path = run_dir / f'346_checkpoint_step{step:06d}_receipt.json'
        write_json(receipt_path, {'checkpoint_path': str(path), 'checkpoint_sha256': C.sha(path),
                   'binding': binding, 'step': step, 'epoch': epoch, 'batch_index': batch_index,
                   'position_in_epoch': batch_index * config['batch'],
                   'execution_kind':C.KIND,'case':_case,
                   'actual_optimizer_updates_since_fixture':step-specification['start']})
        last_receipt = {'path': str(receipt_path), 'sha256': C.sha(receipt_path)}
        checkpoint_receipts.append(last_receipt)

    while step < max_steps:
        require(step < config['total_steps'] == HARD_TOTAL_STEPS, 'hard_stop_1218')
        started = time.perf_counter()
        ids = order['epochs'][epoch][batch_index * config['batch']:(batch_index + 1) * config['batch']]
        require(0 < len(ids) <= config['batch'], 'empty_or_oversized_batch')
        entries = [T.load_entry(seed, iid) for iid in ids]
        loaded_entries += len(entries)
        batch = T.collate(entries)
        for i, group in enumerate(optimizer.param_groups):
            group['lr'] = curve['G_kappa'][step] if arm == 'G' and i == 2 else curve['non_kappa'][step]
        optimizer.zero_grad(set_to_none=True)
        losses, _ = T.compute_losses_340(arm, model, batch)
        require(all(bool(torch.isfinite(v).all()) for v in losses.values()), 'nonfinite_loss')
        losses['loss_total'].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
        require(bool(torch.isfinite(norm)), 'nonfinite_grad_norm')
        record = {'step': step, 'epoch': epoch, 'batch_index': batch_index,
                  'position_in_epoch': batch_index * config['batch'], 'image_ids': ids,
                  'batch_tensor_sha256': batch_sha(batch),
                  **{key: float(value.detach()) for key, value in losses.items()},
                  'param_group_lr': [group['lr'] for group in optimizer.param_groups],
                  'grad_norm_before_clip': float(norm), 'kappa_before': None, 'kappa_grad_after_clip': None}
        if arm == 'G':
            record['kappa_before'] = model.class_null.detach().tolist()
            record['kappa_grad_after_clip'] = (None if model.class_null.grad is None else model.class_null.grad.detach().tolist())
        optimizer.step()
        if arm == 'G':
            record['kappa_after'] = model.class_null.detach().tolist()
        step_logs.append(record)
        step += 1
        local_steps += 1
        batch_index += 1
        epoch_partial_seconds += time.perf_counter() - started
        epoch_finished = batch_index * config['batch'] >= len(order['epochs'][epoch])
        if epoch_finished:
            epoch_times.append({'epoch': epoch, 'seconds': epoch_partial_seconds,
                                'cursor_batches_at_epoch_end': batch_index,
                                'order_epoch_windows': len(order['epochs'][epoch]),
                                'actual_batches_since_fixture':sum(r['epoch']==epoch for r in step_logs),
                                'actual_windows_since_fixture':sum(len(r['image_ids']) for r in step_logs if r['epoch']==epoch)})
            epoch_partial_seconds = 0.0
            epoch += 1
            batch_index = 0
        # Every epoch and every requested mid-epoch endpoint gets a full snapshot.
        if epoch_finished or step == max_steps:
            save_state()
    write_json(run_dir / '346_steps.json', step_logs)
    write_json(run_dir / '346_epoch_times.json', {'completed_epochs': epoch_times,
               'partial_epoch': epoch, 'partial_seconds': epoch_partial_seconds})
    result = {'status': '346构造边界CPU短验收；真实训练未执行', 'arm': arm, 'seed': seed, 'mode': _mode,
              'binding': binding, 'initial_state_sha256': initial_sha, 'final_model_state_sha256': state_sha(model.state_dict()),
              'step': step, 'epoch': epoch, 'batch_index': batch_index,
              'position_in_epoch': batch_index * config['batch'], 'last_checkpoint': last_receipt,
              'checkpoints_this_process': checkpoint_receipts,
              'counts_this_process': {'head_forwards': local_steps, 'loss_adapter_calls': local_steps,
                  'native_loss_calls': 2 * loaded_entries, 'backward_calls': local_steps,
                  'optimizer_steps': local_steps, 'cache_loads': loaded_entries,
                  'global_forwards': 0, 'real_training_steps': 0},
              'input_binding_counts_this_process': dict(T.INPUT_COUNTS),
              'execution_kind':C.KIND,'case':_case,
              'synthetic_origin_step':specification['start'],
              'synthetic_unexecuted_prefix_steps':specification['start'],
              'actual_step_log_records_since_fixture':len(step_logs)}
    write_json(run_dir / '346_run_result.json', result)
    return result


if __name__=='__main__':
    raise SystemExit('346 boundary acceptance worker only; real training is not authorized.')
