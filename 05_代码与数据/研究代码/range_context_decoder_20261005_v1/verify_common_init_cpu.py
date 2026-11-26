"""Verify the frozen training initialization loader without starting training."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import bootstrap
from factory import build_model, load_common_init
from range_context import VARIANTS


def run(args):
    if Path(args.output).exists():
        raise RuntimeError('Preserving existing verification')
    torch.set_num_threads(4)
    expected = hashlib.sha256(Path(args.common_init).read_bytes()).hexdigest()
    states, receipts, parameters = [], [], {}
    for variant in VARIANTS:
        torch.manual_seed(20261005)
        model = build_model({'variant': variant}).cpu()
        receipts.append(load_common_init(model, args.common_init, expected))
        states.append({k: v.clone() for k, v in model.state_dict().items()})
        parameters[variant] = {'stored': sum(p.numel() for p in model.parameters()),
                               'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)}
    checks = {'all_states_exact_before_training': all(all(torch.equal(states[0][k], s[k]) for k in s) for s in states),
              'restore_zero_all_variants': all(torch.count_nonzero(s['range_interaction.restore.weight']) == 0 for s in states)}
    try:
        load_common_init(model, args.common_init, '0' * 64)
        checks['wrong_hash_rejected'] = False
    except ValueError:
        checks['wrong_hash_rejected'] = True
    try:
        probe_hash = hashlib.sha256(Path(args.probe_checkpoint).read_bytes()).hexdigest()
        load_common_init(model, args.probe_checkpoint, probe_hash)
        checks['trained_EMA_rejected_for_training_init'] = False
    except ValueError:
        checks['trained_EMA_rejected_for_training_init'] = True
    checks['CUDA_never_initialized'] = not torch.cuda.is_initialized()
    assert all(checks.values())
    report = {'status': 'common_initialization_CPU_verified_no_training_started', 'checks': checks,
              'initialization_receipt': receipts[0], 'parameters': parameters}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['common_init', 'probe_checkpoint', 'output']:
        parser.add_argument('--' + name, required=True)
    run(parser.parse_args())
