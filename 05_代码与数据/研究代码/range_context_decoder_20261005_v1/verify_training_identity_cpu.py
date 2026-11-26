"""Check zero-init identity with real DN randomness, BN updates and native loss."""
import argparse
import json
import torch
import bootstrap
from factory import build_model, build_criterion, load_independent_ema_for_cpu_probe
from dataset import PairedCocoImages, paired_collate
from dual_view import paired_loss


def detached(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: detached(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [detached(v) for v in value]
    return value


def exact(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def run(args):
    from pathlib import Path
    if Path(args.output).exists():
        raise RuntimeError('Preserving existing training identity receipt')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = build_model().cpu().train()
    init = load_independent_ema_for_cpu_probe(model, args.checkpoint)
    dataset = PairedCocoImages(args.images, args.annotations, augment=False)
    samples, targets = paired_collate([dataset[0]])
    original_bn = {k: v.clone() for k, v in model.named_buffers() if 'running_' in k or 'num_batches_tracked' in k}
    criterion = build_criterion()
    native_grads = {}
    def evaluate(variant):
        for name, value in model.named_buffers():
            if name in original_bn:
                value.copy_(original_bn[name])
        torch.manual_seed(20261005)
        model.zero_grad(set_to_none=True)
        model.range_interaction.variant = variant
        output = model(samples, targets)
        losses = paired_loss(criterion, output, targets)
        total = sum(losses.values())
        total.backward()
        result = {'outputs': detached(output), 'losses': detached(losses),
                  'bn': {k: v.clone() for k, v in model.named_buffers() if k in original_bn},
                  'rng_after': torch.random.get_rng_state().clone()}
        return result
    native = evaluate('independent')
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not name.startswith('range_interaction.'):
            native_grads[name] = parameter.grad.detach().clone()
    checks = {}
    for variant in ['fixed_range_global_context', 'adaptive_range_global_context', 'adaptive_range_local_context']:
        active = evaluate(variant)
        checks[variant + '_outputs_DN_aux_exact'] = exact(native['outputs'], active['outputs'])
        checks[variant + '_all_loss_terms_exact'] = exact(native['losses'], active['losses'])
        checks[variant + '_BN_updates_exact'] = exact(native['bn'], active['bn'])
        checks[variant + '_random_state_exact'] = torch.equal(native['rng_after'], active['rng_after'])
        parameters = dict(model.named_parameters())
        checks[variant + '_native_gradients_equal'] = all(parameters[k].grad is not None and
            torch.allclose(parameters[k].grad, v, atol=1e-6, rtol=1e-5) for k, v in native_grads.items())
        checks[variant + '_restore_gets_first_step_gradient'] = bool(model.range_interaction.restore.weight.grad.abs().sum() > 0)
    # Empty crop: the native loss legitimately has no local DN group. No fake
    # labels or fabricated targets may be added to make the new branch work.
    empty = {**targets, 'local': [{k: (v[:0] if k in ('boxes', 'labels', 'area', 'iscrowd', 'annotation_ids') else v)
                                  for k, v in targets['local'][0].items()}]}
    model.zero_grad(set_to_none=True)
    output = model(samples, empty)
    losses = paired_loss(criterion, output, empty)
    checks['empty_local_crop_loss_finite'] = bool(torch.isfinite(sum(losses.values())))
    checks['empty_local_crop_does_not_fabricate_DN'] = 'dn_outputs' not in output['local']
    checks['CUDA_never_initialized'] = not torch.cuda.is_initialized()
    for name, ok in checks.items():
        print(('PASS ' if ok else 'FAIL ') + name, flush=True)
    if not all(checks.values()):
        raise AssertionError('Training identity check failed')
    report = {'status': 'CPU_training_identity_pass_no_optimizer_step', 'checks': checks,
              'native_parameter_gradients_compared': len(native_grads), 'checkpoint': init,
              'image_id': int(targets['global'][0]['image_id']),
              'limitation': 'Single real image; CPU backward verification, not GPU capacity or trained AP'}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['checkpoint', 'images', 'annotations', 'output']:
        parser.add_argument('--' + name, required=True)
    run(parser.parse_args())
