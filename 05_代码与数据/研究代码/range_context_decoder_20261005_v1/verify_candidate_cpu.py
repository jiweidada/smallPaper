"""CPU-only functional verification; no optimizer, AP evaluation or GPU run."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
import bootstrap
from bootstrap import ROOT
from range_context import RangeContext, VARIANTS, edge_geometry, original_coordinates
from factory import build_model, build_criterion, load_independent_ema_for_cpu_probe, verify_snapshot
from dataset import PairedCocoImages, paired_collate
from dual_view import paired_loss
from engine.deim.dfine_utils import weighting_function, distance2bbox


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    if Path(args.output).exists():
        raise RuntimeError('Preserving the previous verification receipt')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    checks = {}
    def check(name, ok):
        checks[name] = bool(ok)
        if not ok:
            raise AssertionError(name)
        print('PASS', name, flush=True)
    sealed = json.loads((ROOT / 'sealed_source_before.json').read_text(encoding='utf-8-sig'))
    def sealed_unchanged():
        return all(sha(Path(sealed['source']) / r['file']) == r['sha256'] for r in sealed['files'])
    check('sealed_code_and_protocol_before_unchanged', sealed_unchanged())
    verify_snapshot()
    check('isolated_native_snapshot_hashes_verified', True)
    check('CUDA_not_initialized_at_start', not torch.cuda.is_initialized())
    support = weighting_function(32, torch.tensor([.5]), torch.tensor([4.])).float()
    logits = torch.randn(2, 17, 132, requires_grad=True)
    ref = (torch.rand(2, 17, 4) * .04 + .04)
    ref[..., :2] = torch.rand(2, 17, 2) * .4 + .3
    ref.requires_grad_()
    adaptive, box, stat = edge_geometry(logits, ref, support, 4.)
    fixed, fixed_box, fixed_stat = edge_geometry(logits, ref, support, 4., 'fixed')
    probability = logits.detach().reshape(2, 17, 4, 33).softmax(-1)
    expected = distance2bbox(ref.detach(), (probability * support).sum(-1), torch.tensor([4.]))
    check('mean_box_matches_native_FDR', torch.allclose(box, expected, atol=2e-7, rtol=0))
    check('controls_have_exact_same_mean_box', torch.equal(box, fixed_box))
    check('controls_have_exact_same_central_points', torch.equal(adaptive[:, :, 1::3], fixed[:, :, 1::3]))
    check('adaptive_triplets_are_mean_centred', torch.allclose(stat['selected'].mean(-1), stat['mean'], atol=2e-7, rtol=0))
    check('physical_quantiles_sorted_all_four_edges', bool((stat['quantiles'].diff(dim=-1) >= 0).all()))
    check('geometry_no_logit_or_reference_gradient', not adaptive.requires_grad and not box.requires_grad)
    # Degenerate distributions must genuinely collapse, without a hidden cell
    # floor that would silently turn the proposed range into a fixed radius.
    sharp = torch.full((1, 1, 132), -1000.).reshape(1, 1, 4, 33)
    sharp[..., 16] = 0.
    sharp_points, _, sharp_stat = edge_geometry(sharp.flatten(-2), ref[:1, :1], support, 4.)
    check('sharp_distribution_exactly_collapses', bool((sharp_stat['radius'] == 0).all()))
    check('collapsed_triplets_identical', torch.equal(sharp_points.reshape(1, 1, 4, 3, 2)[..., 0, :],
                                                     sharp_points.reshape(1, 1, 4, 3, 2)[..., 2, :]))
    # A symmetric spread is useful to distinguish range from asymmetry.
    symmetric_support = torch.tensor([-1., 0., 1.])
    symmetric_p = torch.tensor([.25, .5, .25]).log().repeat(1, 1, 4).reshape(1, 1, 12)
    _, _, symmetric_stat = edge_geometry(symmetric_p, ref[:1, :1], symmetric_support, 4.)
    check('symmetric_distribution_still_has_nonzero_range', bool((symmetric_stat['radius'] > 0).all()))
    shifted_ref = ref.detach().clone()
    shifted_ref[..., :2] += shifted_ref.new_tensor([.07, -.03])
    shifted, _, shifted_stat = edge_geometry(logits, shifted_ref, support, 4.)
    check('translation_covariant_points', torch.allclose(shifted - adaptive, shifted.new_tensor([.07, -.03]), atol=2e-7, rtol=0))
    check('translation_invariant_range', torch.allclose(shifted_stat['radius'], stat['radius'], atol=2e-7, rtol=0))
    scaled_ref = ref.detach().clone()
    scaled_ref[..., 2:] *= 1.7
    _, _, scaled_stat = edge_geometry(logits, scaled_ref, support, 4.)
    check('range_scales_with_reference_sides', torch.allclose(scaled_stat['radius'], stat['radius'] * 1.7, atol=2e-7, rtol=0))
    states, counts = [], []
    for variant in VARIANTS:
        torch.manual_seed(99)
        module = RangeContext(variant)
        states.append(module.state_dict())
        counts.append(sum(p.numel() for p in module.parameters()))
    check('identical_initialization_all_variants', all(all(torch.equal(states[0][k], s[k]) for k in s) for s in states))
    check('same_stored_parameter_count_all_variants', len(set(counts)) == 1)
    module = RangeContext()
    q = torch.randn(2, 17, 256, requires_grad=True)
    g = torch.randn(2, 256, 10, 10, requires_grad=True)
    l = torch.randn(2, 256, 20, 20, requires_grad=True)
    crop = torch.tensor([[.1, .2, .6, .7], [.2, .1, 1.2, 1.1]])
    def forward():
        return module(q, logits, ref, support, 4., crop, module.prepare_context(g), module.prepare_local(l), (10, 10))
    output = forward()
    check('zero_restore_exact_identity', torch.equal(output, q))
    output.square().mean().backward()
    check('first_step_restore_has_gradient', bool(module.restore.weight.grad.abs().sum() > 0))
    check('geometry_sources_receive_no_extra_gradient', logits.grad is None and ref.grad is None)
    with torch.no_grad():
        module.restore.weight.add_(-.01 * module.restore.weight.grad)
    module.zero_grad(set_to_none=True)
    q.grad = g.grad = l.grad = None
    forward().square().mean().backward()
    check('all_active_parameters_have_finite_nonzero_gradient', all(p.grad is not None and
          torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in module.parameters()))
    check('both_feature_sources_receive_gradient', all(x.grad is not None and torch.isfinite(x.grad).all() and
                                                      x.grad.abs().sum() > 0 for x in (g, l)))
    with torch.no_grad():
        module.variant = 'adaptive_range_global_context'
        geom_g = module.sampling_geometry(logits, ref, support, 4., crop, (10, 10))
        module.variant = 'adaptive_range_local_context'
        geom_l = module.sampling_geometry(logits, ref, support, 4., crop, (10, 10))
        check('context_controls_have_identical_physical_points', torch.equal(geom_g['context_global'], geom_l['context_global']))
        check('context_controls_have_identical_local_points', torch.equal(geom_g['local_points'], geom_l['local_points']))
        module.variant = 'adaptive_range_global_context'
        learned = forward()
        changed = module(q, logits, ref, support, 4., crop, module.prepare_context(g + torch.randn_like(g)),
                         module.prepare_local(l), (10, 10))
        check('global_feature_changes_local_read', not torch.equal(learned, changed))
        q2, p2 = q.clone(), logits.clone()
        q2[:, :3] += 3
        p2[:, :3] += torch.randn_like(p2[:, :3])
        isolated = module(q2, p2, ref, support, 4., crop, module.prepare_context(g), module.prepare_local(l), (10, 10))
        check('no_new_cross_query_or_DN_leakage', torch.equal(learned[:, 3:], isolated[:, 3:]))
        invalid_ref = torch.ones_like(ref) * 100
        empty = module(q, logits, invalid_ref, support, 4., crop, module.prepare_context(g), module.prepare_local(l), (10, 10))
        check('all_invalid_local_points_exact_identity', torch.equal(empty, q) and torch.isfinite(empty).all())
        nan_points = torch.full((2, 17, 12, 2), float('nan'))
        empty_read = module.read(module.prepare_local(l), nan_points, torch.zeros(2, 17, 12, dtype=torch.bool),
                                 module.query(module.norm(q)), nan_points)
        check('invalid_nan_points_have_exact_zero_read', bool((empty_read == 0).all()))
        padding_ref = ref[:1, :1].detach().clone()
        padding_ref[..., :2] = .9
        padding_crop = torch.tensor([[0., 0., 1.3, 1.3]])
        padding = module.sampling_geometry(logits[:1, :1], padding_ref, support, 4., padding_crop, (40, 40))
        check('original_image_mask_excludes_padded_crop_points', not bool(padding['local_valid'].any()))
        edge_ref = ref[:1, :1].detach().clone()
        edge_ref[..., :2] = torch.tensor([.01, .5])
        edge_crop = torch.tensor([[.4, .2, .9, .7]])
        edge = module.sampling_geometry(logits[:1, :1], edge_ref, support, 4., edge_crop, (40, 40))
        outside = (edge['context_crop'][..., 0] < 0) & edge['context_valid']
        check('global_context_allowed_outside_crop_inside_original', bool(outside.any()))
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            amp = forward()
        check('CPU_autocast_finite', bool(torch.isfinite(amp).all()))
    model = build_model().cpu().eval()
    initialization = load_independent_ema_for_cpu_probe(model, args.checkpoint)
    check('pinned_independent_probe_checkpoint', initialization['sha256'] == 'cf8bcea8d38e9ee826929f1b6d161a9ceff0ac4851e3a4318e1f197b2312aa97')
    check('pinned_annotations', sha(args.annotations) == '0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd')
    dataset = PairedCocoImages(args.images, args.annotations, augment=False)
    samples, targets = paired_collate([dataset[0]])
    def equal_outputs(a, b):
        return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    actual_geometry = []
    original_sampling = model.range_interaction.sampling_geometry
    def capture(*a, **kw):
        result = original_sampling(*a, **kw)
        actual_geometry.append({k: v.detach().cpu().clone() for k, v in result.items()})
        return result
    model.range_interaction.sampling_geometry = capture
    with torch.inference_mode():
        model.range_interaction.variant = 'independent'
        native = model(samples)
        for variant in list(VARIANTS)[1:]:
            model.range_interaction.variant = variant
            candidate = model(samples)
            check('real_zero_init_native_exact_' + variant, all(equal_outputs(native[v], candidate[v]) for v in native))
        model.range_interaction.variant = 'adaptive_range_global_context'
        candidate = model(samples)
        cache, global_out = model.cache_global(samples['global'])
        cached = model.predict_local(samples['local'], samples['crop_norm'], cache)
        check('real_cached_vs_paired_outputs', all(torch.allclose(cached[k], candidate['local'][k], atol=2e-5, rtol=1e-4) for k in cached))
        check('real_global_output_native_exact', equal_outputs(global_out, native['global']))
        model.range_interaction.variant = 'adaptive_range_local_context'
        try:
            model.predict_local(samples['local'], samples['crop_norm'], cache)
            check('wrong_variant_cache_rejected', False)
        except ValueError:
            check('wrong_variant_cache_rejected', True)
        torch.nn.init.normal_(model.range_interaction.restore.weight, std=.002)
        model.range_interaction.variant = 'adaptive_range_global_context'
        a = model(samples)
        changed_samples = {**samples, 'global': torch.rand_like(samples['global'])}
        b = model(changed_samples)
        check('real_global_feature_to_local_prediction_path', not torch.equal(a['local']['pred_logits'], b['local']['pred_logits']))
        check('real_global_predictions_unchanged_by_reader', equal_outputs(a['global'], native['global']))
        model.range_interaction.variant = 'adaptive_range_local_context'
        a_local = model(samples)
        b_local = model(changed_samples)
        check('local_context_control_has_no_global_feature_path_in_eval', equal_outputs(a_local['local'], b_local['local']))
    # Restore the method before training: do not retain large DN tensors in the probe.
    model.range_interaction.sampling_geometry = original_sampling
    model.range_interaction.variant = 'adaptive_range_global_context'
    model.train()
    criterion = build_criterion()
    output = model(samples, targets)
    losses = paired_loss(criterion, output, targets)
    loss = sum(losses.values())
    check('native_paired_losses_finite', bool(torch.isfinite(loss)))
    loss.backward()
    check('native_DN_training_contract_retained', 'dn_outputs' in output['local'])
    check('native_training_gradient_reaches_context', model.range_interaction.context_kv.weight.grad is not None and
          bool(model.range_interaction.context_kv.weight.grad.abs().sum() > 0))
    check('native_training_all_reader_gradients_finite', all(p.grad is not None and torch.isfinite(p.grad).all()
                                                        for p in model.range_interaction.parameters()))
    check('CUDA_never_initialized', not torch.cuda.is_initialized())
    check('sealed_code_and_protocol_after_unchanged', sealed_unchanged())
    real = actual_geometry[1]  # adaptive_global, after fixed_global in insertion order
    fixed_real = actual_geometry[0]
    point_separation = (real['local_points'] - fixed_real['local_points']).norm(dim=-1) * 640
    report = {
        'status': 'CPU_functional_pass_not_AP_or_GPU_capacity', 'checks': checks, 'check_count': len(checks),
        'checkpoint': initialization, 'image_id': int(targets['global'][0]['image_id']),
        'annotation_sha256': sha(args.annotations),
        'new_reader_parameters': counts[0], 'total_stored_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters_candidate': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'reader_learned_offsets': 0, 'new_read_points': {'local_P3': 12, 'context_P4': 4},
        'real_first_image_geometry': {
            'all_queries_not_GT_selected': int(real['box'].shape[1]),
            'collapsed_edges_fraction': float((real['adaptive_radius'] == 0).float().mean()),
            'mean_adaptive_radius_native_px': float(real['adaptive_radius'].mean() * 640),
            'adaptive_vs_fixed_point_distance_mean_px': float(point_separation.mean()),
            'adaptive_vs_fixed_point_distance_median_px': float(point_separation.median()),
            'adaptive_vs_fixed_point_distance_mean_P3_cells': float(point_separation.mean() / 8),
            'local_valid_fraction': float(real['local_valid'].float().mean()),
        },
        'native_loss_before_backward': float(loss.detach()), 'loss_terms': sorted(losses),
        'torch': torch.__version__, 'finished_unix': time.time(),
        'limitations': ['No optimizer step or saved trained candidate', 'No AP effectiveness claim',
                        'No GPU capacity, latency or FLOPs verification',
                        'Evaluation local control is feature-independent; training BN still uses paired views'],
    }
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('checks', 'checkpoint')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['checkpoint', 'images', 'annotations', 'output']:
        parser.add_argument('--' + name, required=True)
    run(parser.parse_args())
