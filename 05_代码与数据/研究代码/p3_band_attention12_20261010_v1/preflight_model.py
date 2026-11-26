"""Zero-update native-model identity, RNG and learnability checks."""
import argparse
import gc
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    output = Path(a.output)
    if output.exists():
        raise FileExistsError(output)
    from experiment import verify_bundle
    from runtime import activate_training_snapshot, load_common_initialization
    receipt, plan, paths = verify_bundle(a.bundle)
    activate_training_snapshot(paths['native_root'])
    from evaluator_binding import activate
    activate(paths['evaluator_root'], receipt['identity']['evaluator_binding'])
    import torch
    from model import P3BandAttentionDEIM
    from engine.deim.deim_criterion import DEIMCriterion
    from engine.deim.matcher import HungarianMatcher
    from copy import deepcopy
    torch.set_num_threads(4)
    torch.manual_seed(plan['seed'])
    torch.cuda.manual_seed_all(plan['seed'])
    start = torch.get_rng_state().clone()
    cpu_after = {}
    models = {}
    init_receipts = {}
    trainability = {}
    for arm in ('baseline', 'wavelet', 'spatial'):
        torch.set_rng_state(start)
        model = P3BandAttentionDEIM(plan['model_recipe'], arm)
        cpu_after[arm] = torch.get_rng_state().clone()
        init_receipts[arm] = load_common_initialization(model, paths['common_init'])
        trainability[arm] = {name: param.requires_grad for name, param in model.named_parameters()}
        models[arm] = model
    assert all(torch.equal(cpu_after['baseline'], state) for state in cpu_after.values())
    native_state = models['baseline'].state_dict()
    for arm in ('wavelet', 'spatial'):
        state = models[arm].state_dict()
        assert all(torch.equal(value, state[key]) for key, value in native_state.items())
    # Constructing the probe tensor never touches the formal training process.
    x = torch.rand(1, 3, 640, 640, device='cuda')
    probes = []
    for amp in (False, True):
        base = None
        for arm in ('baseline', 'wavelet', 'spatial'):
            model = models[arm].cuda().eval().requires_grad_(False)
            with torch.no_grad(), torch.autocast('cuda', enabled=amp):
                out = model(x)
            values = {key: out[key].detach().float().cpu() for key in ('pred_logits', 'pred_boxes')}
            assert all(torch.isfinite(v).all() for v in values.values())
            if arm == 'baseline':
                base = values
            else:
                exact = {key: bool(torch.equal(base[key], value)) for key, value in values.items()}
                assert all(exact.values()), (arm, amp, exact)
                probes.append(dict(arm=arm, autocast=amp, exact_native_initial_outputs=exact))
            model.cpu()
            del out
    # A single FP32 native loss/backward on synthetic targets establishes that
    # the new gate is connected to real detection losses. No optimizer exists.
    train_model = models['wavelet'].cuda().train()
    # Restore the native flags, including intentionally frozen integer and
    # backbone parameters, after the EMA-like requires_grad_(False) checks.
    for name, param in train_model.named_parameters():
        param.requires_grad_(trainability['wavelet'][name])
    c = deepcopy(plan['criterion_recipe'])
    matcher = c.pop('matcher')
    matcher.pop('type', None)
    criterion = DEIMCriterion(matcher=HungarianMatcher(**matcher, use_focal_loss=True),
                             num_classes=10, **c).cuda()
    criterion.set_model(train_model)
    criterion.set_epoch(0)
    targets = [dict(labels=torch.tensor([0, 9], device='cuda'),
                    boxes=torch.tensor([[.31, .37, .035, .045], [.65, .62, .12, .09]], device='cuda'))]
    losses = criterion(train_model(x, targets), targets)
    total = sum(losses.values())
    assert torch.isfinite(total), 'Native synthetic loss is nonfinite'
    total.backward()
    gradients = {name: dict(connected=param.grad is not None,
                            finite=param.grad is not None and bool(torch.isfinite(param.grad).all()),
                            norm=None if param.grad is None else float(param.grad.norm()))
                 for name, param in train_model.band_attention.named_parameters()}
    assert all(item['finite'] for item in gradients.values()), gradients
    assert any(item['norm'] > 0 for item in gradients.values()), gradients
    report = dict(status='PASS_ZERO_UPDATE_NATIVE_MODEL', optimizer_constructions=0,
                  optimizer_steps=0, synthetic_training_batches=1,
                  native_state_exact=True, constructor_rng_exact=True,
                  initialization=init_receipts, initial_output_checks=probes,
                  native_loss=float(total.detach()), gradients=gradients,
                  identity=receipt['identity'],
                  limitation='Temporary models only. No training updates; not a data/RNG trajectory or AP test.')
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'optimizer_steps', 'constructor_rng_exact', 'native_loss')}))
    del models, train_model, criterion
    gc.collect()


if __name__ == '__main__':
    main()
