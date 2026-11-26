"""Mechanism tests plus a real validation image, not an AP experiment."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
import bootstrap
from distribution_context import DistributionContext, boundary_points, two_bin
from factory import build_model, build_criterion, verify_snapshot
from dataset import PairedCocoImages, paired_collate
from dual_view import paired_loss
from engine.deim.dfine_utils import weighting_function, distance2bbox


def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(17)
    checks = {}
    def check(name, ok):
        checks[name] = bool(ok)
        if not ok:
            raise AssertionError(name)
        print('PASS', name, flush=True)
    support = weighting_function(32, torch.tensor([.5]), torch.tensor([4.])).float()
    q = torch.randn(2, 17, 256, requires_grad=True)
    p = torch.randn(2, 17, 132, requires_grad=True)
    ref = torch.rand(2, 17, 4)*.3+.3
    crop = torch.tensor([[.1, .2, .6, .7], [.2, .1, 1.2, 1.1]])
    points, box = boundary_points(p, ref, support, 4., 'full')
    prob = p.detach().reshape(2, 17, 4, 33).softmax(-1)
    expected = distance2bbox(ref, (prob*support).sum(-1), torch.tensor([4.]))
    check('native_FDR_mean_box', torch.allclose(box, expected, atol=2e-7, rtol=0))
    e = two_bin(prob, support)
    check('two_bin_preserves_mean', torch.allclose((e*support).sum(-1), (prob*support).sum(-1), atol=2e-6, rtol=0))
    check('geometry_detached', not points.requires_grad and not box.requires_grad)
    check('twelve_local_points', points.shape == (2, 17, 12, 2))
    states = []
    for arm in 'ABCDE':
        torch.manual_seed(99)
        m = DistributionContext(arm)
        states.append(m.state_dict())
    check('identical_parameter_initialization_all_arms', all(all(torch.equal(states[0][k], s[k]) for k in s) for s in states))
    m = DistributionContext('D')
    g = torch.randn(2, 256, 10, 10, requires_grad=True)
    l = torch.randn(2, 256, 20, 20, requires_grad=True)
    def forward():
        return m(q, p, ref, support, 4., crop, m.prepare_context(g), m.prepare_local(l))
    out = forward()
    check('zero_init_exact_identity', torch.equal(out, q))
    out.square().mean().backward()
    check('restore_gradient_first_step', m.restore.weight.grad.abs().sum() > 0)
    check('geometry_logits_no_gradient', p.grad is None)
    with torch.no_grad():
        m.restore.weight.add_(-.01*m.restore.weight.grad)
    m.zero_grad(); q.grad = None
    out = forward()
    out.square().mean().backward()
    for name, param in m.named_parameters():
        check('gradient_'+name, param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0)
    check('global_feature_gradient', g.grad is not None and g.grad.abs().sum() > 0)
    check('local_feature_gradient', l.grad is not None and l.grad.abs().sum() > 0)
    with torch.no_grad():
        memg, meml = m.prepare_context(g), m.prepare_local(l)
        original = m(q, p, ref, support, 4., crop, memg, meml)
        changed = m(q, p, ref, support, 4., crop, m.prepare_context(g+torch.randn_like(g)), meml)
        check('global_context_changes_local_update', not torch.equal(original, changed))
        invalid = torch.ones_like(ref)*100
        empty = m(q, p, invalid, support, 4., crop, memg, meml)
        check('all_invalid_exact_identity_finite', torch.equal(empty, q) and torch.isfinite(empty).all())
        altered_q = q.clone(); altered_q[:, :3] += 3
        altered_p = p.clone(); altered_p[:, :3] += torch.randn_like(altered_p[:, :3])
        altered = m(altered_q, altered_p, ref, support, 4., crop, memg, meml)
        check('no_query_to_query_or_DN_leakage', torch.equal(original[:, 3:], altered[:, 3:]))
        bf = None
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            bf = m(q, p, ref, support, 4., crop, m.prepare_context(g), m.prepare_local(l))
        check('CPU_autocast_finite', torch.isfinite(bf).all())
    verify_snapshot()
    cfg = {'kind': 'dual_view', 'arm': 'D'}
    model = build_model(cfg)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(payload['ema']['module'], strict=False)
    check('real_checkpoint_only_new_module_missing', not unexpected and all(k.startswith('distribution_interaction.') for k in missing))
    data = PairedCocoImages(args.images, args.annotations, augment=False)
    samples, targets = paired_collate([data[0]])
    model.eval()
    with torch.inference_mode():
        model.distribution_interaction.arm = 'A'
        native = model(samples)
        model.distribution_interaction.arm = 'D'
        candidate = model(samples)
        check('real_zero_init_native_identical', all(torch.equal(native[v][k], candidate[v][k]) for v in native for k in native[v]))
        cache, global_out = model.cache_global(samples['global'])
        cached = model.predict_local(samples['local'], samples['crop_norm'], cache)
        check('real_cache_vs_paired_tolerance', all(torch.allclose(cached[k], candidate['local'][k], atol=2e-5, rtol=1e-4) for k in cached))
        torch.nn.init.normal_(model.distribution_interaction.restore.weight, std=.002)
        a = model(samples)
        changed_samples = dict(samples, **{'global': torch.rand_like(samples['global'])})
        b = model(changed_samples)
        check('real_global_to_local_effect', not torch.equal(a['local']['pred_logits'], b['local']['pred_logits']))
        check('global_output_native_unchanged', all(torch.equal(a['global'][k], native['global'][k]) for k in a['global']))
    model.train()
    criterion = build_criterion()
    output = model(samples, targets)
    loss_dict = paired_loss(criterion, output, targets)
    loss = sum(loss_dict.values())
    check('native_training_losses_finite', torch.isfinite(loss))
    loss.backward()
    check('native_training_new_module_gradient', model.distribution_interaction.context_kv.weight.grad is not None and model.distribution_interaction.context_kv.weight.grad.abs().sum() > 0)
    check('native_training_DN_present', 'dn_outputs' in output['local'])
    report = {'status': 'CPU_pass_not_GPU_capacity_or_effectiveness', 'checks': checks,
              'check_count': len(checks), 'real_image_id': int(targets['global'][0]['image_id']),
              'real_native_loss': float(loss.detach()), 'new_parameters': sum(x.numel() for x in m.parameters()),
              'total_parameters': sum(x.numel() for x in model.parameters()),
              'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              'torch': torch.__version__, 'time': time.time()}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k != 'checks'}), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['checkpoint', 'images', 'annotations', 'output']:
        parser.add_argument('--'+name, required=True)
    run(parser.parse_args())
