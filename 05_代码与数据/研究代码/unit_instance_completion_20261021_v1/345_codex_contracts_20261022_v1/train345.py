"""345 entry: frozen train344 mathematics, persisted order and resumable cursor.

Current authorization permits CPU acceptance only (formal order <=2 steps;
private miniature acceptance <=6 steps). Real training: 需另行授权.
The 1218-step loop and epoch checkpoint machinery are implemented below;
the authorization cap is separate from that immutable scientific budget.
"""
import argparse
import hashlib
import importlib
import json
import math
import random
import time
from pathlib import Path

C = importlib.import_module('345_context')
import numpy as np
import torch
T = importlib.import_module('train344')
torch.set_num_threads(2)

HARD_TOTAL_STEPS = 1218
CURRENT_CPU_CAP = {'formal': 2, 'mini_acceptance': 6}


class GuardError(RuntimeError):
    def __init__(self, code, detail=''):
        self.code = code
        super().__init__(code + (': ' + str(detail) if detail else ''))


def require(condition, code, detail=''):
    if not condition:
        raise GuardError(code, detail)


def verify_file(path, expected, label):
    require(Path(path).is_file(), label + '_missing', path)
    actual = C.sha(path)
    require(actual == expected, label + '_SHA_mismatch', str(path))
    return actual


def write_json(path, document):
    # Encode before exclusive creation: invalid metadata cannot leave half a receipt.
    data = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)
    with open(path, 'x', encoding='utf-8', newline='\n') as f:
        f.write(data + '\n')


def state_sha(state):
    """Exactly the already sealed stage5 initial-state byte convention."""
    h = hashlib.sha256()
    for name, value in state.items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def batch_sha(batch):
    h = hashlib.sha256()
    for key in ('rgb', 'P_boxes', 'P_cls', 'P_scores', 'P_valid'):
        value = batch[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def validate_order(document, mode, config, development_ids, formal=None):
    expected_kind = 'development_6epoch' if mode == 'formal' else 'mini_acceptance_only'
    require(document['kind'] == expected_kind, 'order_kind_mismatch')
    require(document['global_seed'] == config['global_seed'], 'order_seed_mismatch')
    require(document['head_seed'] == config['head_seed'], 'order_head_seed_mismatch')
    epochs = document['epochs']
    require(len(epochs) == config['epochs'], 'order_epoch_count')
    if mode == 'formal':
        for epoch in epochs:
            require(len(epoch) == config['windows_per_epoch'] and
                    len(set(epoch)) == len(epoch) and set(epoch) == set(development_ids),
                    'order_not_development_permutation')
    else:
        require(formal is not None, 'mini_parent_order_missing')
        n = 2 * config['batch']
        require(all(row == parent[:n] for row, parent in zip(epochs, formal['epochs'])),
                'mini_not_parent_prefix')
        require(all(len(row) == n for row in epochs), 'mini_not_two_batches')
    return epochs


def validate_bindings(arm, seed, mode='formal', _test_paths=None):
    """Same SHA checks in acceptance and production. Test paths can only fail SHA.

    _test_paths is private negative-test plumbing, unavailable to the CLI. Expected
    hashes always come from the pinned bundle; test paths cannot change a recipe.
    """
    anchor = importlib.import_module('345_bindings_anchor')
    bundle_path = C.HERE / '345_B_冻结入口绑定.json'
    bundle_sha = verify_file(bundle_path, anchor.BUNDLE_SHA256, 'bundle')
    bundle = C.read(bundle_path)
    require(arm in 'NSGR' and len(arm) == 1 and seed in (1, 2), 'arm_or_seed')
    require(mode in CURRENT_CPU_CAP, 'mode')
    for path, expected in bundle['sources'].items():
        verify_file(path, expected, 'source')
    paths = {} if _test_paths is None else dict(_test_paths)
    group = bundle['config_group']
    verify_file(group['path'], group['sha256'], 'config_group')
    config_record = bundle['configs'][f'{arm}_seed{seed}']
    config_path = paths.get('config', config_record['path'])
    verify_file(config_path, config_record['sha256'], 'config')
    config = C.read(config_path)['config']
    frozen = next(r for r in C.read(group['path'])['runs'] if r['run_id'] == f'{arm}_seed{seed}')
    require(config == frozen, 'config_group_content_mismatch')
    require(config['arm'] == arm and config['global_seed'] == seed, 'config_identity')
    # Validate fixed contracts, then use the sealed configuration for actual values.
    require((config['epochs'], config['batch'], config['windows_per_epoch'],
             config['steps_per_epoch'], config['last_batch_size'], config['total_steps'])
            == (6, 16, 3235, 203, 3, HARD_TOTAL_STEPS), 'fixed_budget')
    require(not config['drop_last'] and config['fixed_last'] and not config['select_best'], 'fixed_last')
    require(config['optimizer'] == 'AdamW' and config['dtype'] == 'FP32', 'optimizer_or_dtype')
    require(config['loss_weights'] == {'L_det': 1.0, 'L_mass': 1.0, 'L_assoc': 1.0}, 'loss_contract')
    require(config['global_state_key'] == 'model' and not config['EMA'] and
            config['augmentation'] == 'none' and not config['extra_epochs'], 'frozen_recipe')
    verify_file(config['global_checkpoint'], config['global_checkpoint_sha256'], 'raw_checkpoint')
    manifest_path = paths.get('manifest', config['development_manifest'])
    verify_file(manifest_path, config['development_manifest_sha256'], 'manifest')
    require(config['development_manifest_sha256'] == T.MANIFEST_SHA256 and
            Path(config['development_manifest']).resolve() == T.MANIFEST_PATH.resolve(), 'manifest_binding')
    verify_file(config['val_manifest'], config['val_manifest_sha256'], 'val_manifest')
    order_index = bundle['shuffle'][str(seed)]
    formal_path = paths.get('shuffle', order_index['formal']['path'])
    verify_file(formal_path, order_index['formal']['sha256'], 'shuffle')
    formal = C.read(formal_path)
    development_ids = T.load_order()
    validate_order(formal, 'formal', config, development_ids)
    order_record = order_index['formal']
    order = formal
    if mode == 'mini_acceptance':
        order_record = order_index['mini']
        verify_file(order_record['path'], order_record['sha256'], 'mini_shuffle')
        order = C.read(order_record['path'])
        require(order['parent_formal_sha256'] == order_index['formal']['sha256'], 'mini_parent_SHA')
        validate_order(order, mode, config, development_ids, formal)
    curve_record = bundle['lr_curve']
    verify_file(curve_record['path'], curve_record['sha256'], 'lr_curve')
    curve = C.read(curve_record['path'])
    regular = config['lr_non_kappa']
    require(regular['post_warmup'] == 'cosine', 'lr_post_warmup')
    require(len(curve['non_kappa']) == HARD_TOTAL_STEPS and len(curve['G_kappa']) == HARD_TOTAL_STEPS,
            'lr_curve_length')
    for step in range(HARD_TOTAL_STEPS):
        value = T.lr_at(step, config['total_steps'], regular['warmup_steps'], regular['base'], regular['lo'])
        require(value == curve['non_kappa'][step], 'lr_curve_regular_mismatch', step)
        if arm == 'G':
            k = config['lr_class_null']
            require(T.lr_at_kappa(step, config['total_steps'], k['warmup_steps'], k['base'], k['lo'])
                    == curve['G_kappa'][step], 'lr_curve_kappa_mismatch', step)
    binding = {'bundle_sha256': bundle_sha, 'config_sha256': config_record['sha256'],
               'config_group_sha256': group['sha256'],
               'development_manifest_sha256': config['development_manifest_sha256'],
               'raw_checkpoint_sha256': config['global_checkpoint_sha256'],
               'shuffle_sha256': order_record['sha256'], 'shuffle_path': order_record['path'],
               'lr_curve_sha256': curve_record['sha256'], 'arm': arm, 'seed': seed, 'mode': mode}
    return config, order, curve, binding


def make_optimizer(model, config):
    k = config['lr_class_null']
    multiplier = k['base'] / config['lr_non_kappa']['base'] if k is not None else T.KAPPA_LR_MULT
    optimizer = T.build_optimizer(model, kappa_mult=multiplier)
    # Inherit the function itself, and reject any drift from configuration defaults.
    for i, group in enumerate(optimizer.param_groups):
        is_kappa = k is not None and i == 2
        require(group['betas'] == tuple(config['betas']) and group['eps'] == config['eps'], 'optimizer_defaults')
        expected_decay = (k['weight_decay'] if is_kappa else
                          config['weight_decay'] if i == 0 else config['bias_norm_decay'])
        require(group['weight_decay'] == expected_decay, 'optimizer_decay')
        require(group['lr'] == (k['base'] if is_kappa else config['lr_non_kappa']['base']), 'optimizer_base_lr')
    return optimizer


def check_cursor(step, epoch, batch_index, order, config):
    steps_per_epoch = math.ceil(len(order['epochs'][0]) / config['batch'])
    require(0 <= step <= HARD_TOTAL_STEPS, 'checkpoint_step_budget')
    require(0 <= epoch <= config['epochs'], 'checkpoint_epoch')
    require(step == epoch * steps_per_epoch + batch_index, 'checkpoint_cursor_step')
    require((epoch < config['epochs'] and 0 <= batch_index < steps_per_epoch) or
            (epoch == config['epochs'] and batch_index == 0), 'checkpoint_batch_cursor')


def execute(arm, seed, run_dir, device, max_steps, *, _mode='formal', _resume=None):
    """CLI uses formal order; private acceptance worker also uses mini order/resume.

    max_steps is an absolute global endpoint. Resume(3 -> 6) executes three steps.
    Resume always writes to a fresh child run directory, keeping the parent sealed.
    """
    require(isinstance(max_steps, int) and 0 < max_steps <= HARD_TOTAL_STEPS, 'max_steps_budget')
    require(device == 'cpu', 'CPU_only_authorization')
    require(_mode in CURRENT_CPU_CAP and max_steps <= CURRENT_CPU_CAP[_mode], 'acceptance_step_cap')
    run_dir = Path(run_dir).resolve()
    require(not run_dir.exists(), 'run_directory_exists', run_dir)
    config, order, curve, binding = validate_bindings(arm, seed, _mode)
    run_dir.mkdir(parents=True, exist_ok=False)
    np.random.seed(config['head_seed'])
    random.seed(config['head_seed'])
    model = T.make_common_init(arm, seed).train()
    initial_sha = state_sha(model.state_dict())
    require(initial_sha == config['untrained_initial_state_sha256'], 'initial_state_SHA')
    require(sum(p.numel() for p in model.parameters()) == config['head_parameters'], 'parameter_count')
    require(all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters()), 'CPU_FP32')
    with open(run_dir / '345_initial_state.pt', 'xb') as f:
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
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        require([g['lr'] for g in optimizer.param_groups] == checkpoint['param_group_lr'], 'resume_group_lr')
        step, epoch, batch_index = checkpoint['step'], checkpoint['epoch'], checkpoint['batch_index']
        require(checkpoint['position_in_epoch'] == batch_index * config['batch'], 'resume_position')
        step_logs = checkpoint['step_logs']
        epoch_times = checkpoint['epoch_times']
        epoch_partial_seconds = checkpoint['epoch_partial_seconds']
        require(len(step_logs) == step, 'resume_log_length')
        require(checkpoint['model_state_sha256'] == state_sha(model.state_dict()), 'resume_model_SHA')
        T.restore_rng(checkpoint['rng'])
    check_cursor(step, epoch, batch_index, order, config)
    require(step < max_steps, 'resume_endpoint_not_after_checkpoint')
    write_json(run_dir / '345_run_binding.json', {'binding': binding, 'initial_state_sha256': initial_sha,
               'inherited_recipe': config, 'max_steps': max_steps, 'resume': _resume,
               'epoch_timing_policy': 'Actual wall times are a separate ledger. Byte equality covers deterministic per-step numeric/batch logs; wall times are not deterministic.'})
    local_steps = loaded_entries = 0
    checkpoint_receipts = []
    last_receipt = None

    def save_state():
        nonlocal last_receipt
        check_cursor(step, epoch, batch_index, order, config)
        path = run_dir / f'345_checkpoint_step{step:06d}.pt'
        payload = {'format': '345_full_state_v1', 'binding': binding,
                   'initial_state_sha256': initial_sha, 'model_state_sha256': state_sha(model.state_dict()),
                   'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'param_group_lr': [g['lr'] for g in optimizer.param_groups],
                   'step': step, 'epoch': epoch, 'batch_index': batch_index,
                   'position_in_epoch': batch_index * config['batch'], 'shuffle_sha256': binding['shuffle_sha256'],
                   'rng': T.capture_rng(), 'step_logs': step_logs, 'epoch_times': epoch_times,
                   'epoch_partial_seconds': epoch_partial_seconds}
        with open(path, 'xb') as f:
            torch.save(payload, f)
        receipt_path = run_dir / f'345_checkpoint_step{step:06d}_receipt.json'
        write_json(receipt_path, {'checkpoint_path': str(path), 'checkpoint_sha256': C.sha(path),
                   'binding': binding, 'step': step, 'epoch': epoch, 'batch_index': batch_index,
                   'position_in_epoch': batch_index * config['batch']})
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
                                'batches': batch_index, 'windows': len(order['epochs'][epoch])})
            epoch_partial_seconds = 0.0
            epoch += 1
            batch_index = 0
        # Every epoch and every requested mid-epoch endpoint gets a full snapshot.
        if epoch_finished or step == max_steps:
            save_state()
    write_json(run_dir / '345_steps.json', step_logs)
    write_json(run_dir / '345_epoch_times.json', {'completed_epochs': epoch_times,
               'partial_epoch': epoch, 'partial_seconds': epoch_partial_seconds})
    result = {'status': 'CPU短验收已执行；真实训练未执行', 'arm': arm, 'seed': seed, 'mode': _mode,
              'binding': binding, 'initial_state_sha256': initial_sha, 'final_model_state_sha256': state_sha(model.state_dict()),
              'step': step, 'epoch': epoch, 'batch_index': batch_index,
              'position_in_epoch': batch_index * config['batch'], 'last_checkpoint': last_receipt,
              'checkpoints_this_process': checkpoint_receipts,
              'counts_this_process': {'head_forwards': local_steps, 'loss_adapter_calls': local_steps,
                  'native_loss_calls': 2 * loaded_entries, 'backward_calls': local_steps,
                  'optimizer_steps': local_steps, 'cache_loads': loaded_entries,
                  'global_forwards': 0, 'real_training_steps': 0},
              'input_binding_counts_this_process': dict(T.INPUT_COUNTS)}
    write_json(run_dir / '345_run_result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='345 CPU entry acceptance; 真实训练需另行授权', allow_abbrev=False)
    parser.add_argument('--arm', choices=list('NSGR'), required=True)
    parser.add_argument('--seed', type=int, choices=(1, 2), required=True)
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--device', choices=('cpu',), required=True)
    parser.add_argument('--max_steps', type=int, required=True)
    args = parser.parse_args(argv)
    result = execute(args.arm, args.seed, args.run_dir, args.device, args.max_steps)
    print(json.dumps({'status': result['status'], 'arm': args.arm, 'seed': args.seed,
                      'step': result['step'], 'last_checkpoint': result['last_checkpoint']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
