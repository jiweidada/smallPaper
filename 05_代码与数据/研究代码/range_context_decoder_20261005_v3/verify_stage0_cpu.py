"""Meaningful CPU checks of the revised pathway; no optimizer or GPU."""
import hashlib
import json
import time
from pathlib import Path
import torch
import bootstrap
from bootstrap import ROOT
from factory import build_model, load_independent_ema_for_cpu_probe, verify_snapshot
from range_context import RangeContext, VARIANTS, edge_geometry
from dataset import PairedCocoImages, paired_collate
from engine.deim.dfine_utils import weighting_function, distance2bbox

PROJECT = ROOT.parents[2]
CHECKPOINT = PROJECT / '英文小论文/data/04_冻结权重诊断_20261001/c_seed1_runtime_v1/readonly_20261004_1828/epoch_053.pth'
ANNOTATIONS = PROJECT / '数据/03_本轮实验记录_20260928/转换COCO标注原件_20260929/VisDrone2019-DET-val/annotations/val.json'
IMAGES = 'C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val/images'


def main():
    dest = ROOT / 'stage0_functional_cpu.json'
    if dest.exists():
        raise RuntimeError('Preserve previous receipt')
    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(20261005)
    checks = {}
    def check(name, ok):
        checks[name] = bool(ok)
        print(('PASS ' if ok else 'FAIL ') + name, flush=True)
        if not ok:
            raise AssertionError(name)
    verify_snapshot()
    check('native_snapshot_preserved', True)
    before = json.loads((ROOT / 'source_before_revision.json').read_text(encoding='utf8'))
    source = Path(before['source'])
    def source_exact():
        return all(hashlib.sha256((source / k).read_bytes()).hexdigest() == v
                   for k, v in before['original_source_hashes'].items())
    check('old_candidate_original_files_exact_before', source_exact())
    check('CPU_only_at_start', not torch.cuda.is_initialized())
    m = RangeContext()
    b, q, n = 1, 3, 12
    xy = torch.rand(b, q, n, 2) * .5 + .25
    relative = torch.randn_like(xy)
    valid = torch.ones(b, q, n, dtype=torch.bool)
    valid[:, 1, 1:] = False
    valid[:, 2] = False
    xy[:, 2] = float('nan')
    relative[:, 2] = float('inf')
    memory = torch.randn(b, 256, 8, 8)
    query = torch.randn(b, q, 128)
    a, weights, values = m.read(memory, xy, valid, query, relative, True)
    check('valid_weights_sum_one_each_head', torch.allclose(weights[:, :2].sum(2), torch.ones(1, 2, 4), atol=2e-7))
    check('invalid_weights_are_exact_zero', bool((weights[~valid] == 0).all()))
    check('all_invalid_NaN_Inf_geometry_gives_exact_zero', bool((a[:, 2] == 0).all()) and bool(torch.isfinite(a).all()))
    a2 = m.read(memory, xy, valid, query * 4 + 5, relative)
    check('one_valid_point_query_invariant', torch.equal(a[:, 1], a2[:, 1]))
    # Constant values, diverse keys and positions: the old always-null
    # implementation would change the total value mass in this test.
    memory[:, 128:] = torch.randn(1, 128, 1, 1)
    all_valid = torch.ones(1, 3, 12, dtype=torch.bool)
    clean_xy = torch.rand(1, 3, 12, 2) * .5 + .25
    const_a = m.read(memory, clean_xy, all_valid, query, torch.randn_like(clean_xy))
    const_b = m.read(memory, clean_xy, all_valid, query * 20, torch.randn_like(clean_xy))
    check('constant_values_remove_query_gain_with_diverse_keys', torch.allclose(const_a, const_b, atol=2e-6, rtol=1e-6))
    # Do not call a full-vector hull bound a multihead bound.
    v = torch.randn(19, 12, 4, 32)
    uniform = v.mean(1)
    bound = (v - uniform[:, None]).norm(dim=-1).amax(1).square().sum(-1).sqrt()
    w = torch.randn(19, 12, 4).softmax(1)
    movement = ((v * w[..., None]).sum(1) - uniform).flatten(1).norm(dim=-1)
    check('headwise_convex_bound_contains_random_mixtures', bool((movement <= bound + 1e-5).all()))
    states = []
    for variant in VARIANTS:
        torch.manual_seed(1)
        states.append(RangeContext(variant).state_dict())
    check('all_controls_have_exact_same_initial_tensors', all(all(torch.equal(states[0][k], s[k]) for k in s) for s in states))
    logits = torch.randn(2, 11, 132, requires_grad=True)
    ref = torch.rand(2, 11, 4) * .02 + .04
    ref[..., :2] = torch.rand(2, 11, 2) * .3 + .35
    ref.requires_grad_()
    support = weighting_function(32, torch.tensor([.5]), torch.tensor([4.]))
    geom = {tag: edge_geometry(logits, ref, support, 4., tag)
            for tag in ('adaptive', 'updated_size', 'axis_mean', 'opposite_swap')}
    p = logits.detach().reshape(2, 11, 4, 33).softmax(-1)
    expected = distance2bbox(ref.detach(), (p * support).sum(-1), torch.tensor([4.]))
    check('mean_box_agrees_with_native_FDR', torch.allclose(geom['adaptive'][1], expected, atol=2e-7, rtol=0))
    check('all_geometry_controls_have_exact_same_mean_box', all(torch.equal(geom['adaptive'][1], item[1]) for item in geom.values()))
    check('all_geometry_controls_have_exact_same_centre_points', all(torch.equal(geom['adaptive'][0][:, :, 1::3], item[0][:, :, 1::3]) for item in geom.values()))
    ra = geom['adaptive'][2]['radius']
    check('adaptive_multiplier_is_exactly_two', torch.equal(ra, 2 * geom['adaptive'][2]['adaptive_radius']))
    for name in ('axis_mean', 'opposite_swap'):
        radius = geom[name][2]['radius']
        check(name + '_preserves_each_axis_radius_sum', torch.allclose(radius + radius[..., [2, 3, 0, 1]], ra + ra[..., [2, 3, 0, 1]], atol=1e-7, rtol=0))
    check('swap_preserves_radius_multiset', torch.equal(geom['opposite_swap'][2]['radius'].sort(-1).values, ra.sort(-1).values))
    check('geometry_stops_logit_and_reference_gradient', all(not item[0].requires_grad for item in geom.values()))
    m = RangeContext()
    raw_query = torch.randn(2, 11, 256, requires_grad=True)
    global_feat = torch.randn(2, 256, 10, 10, requires_grad=True)
    local_feat = torch.randn(2, 256, 20, 20, requires_grad=True)
    crop = torch.tensor([[.1, .2, .6, .7], [.2, .1, .8, .8]])
    def forward():
        return m(raw_query, logits, ref, support, 4., crop, m.prepare_context(global_feat), m.prepare_local(local_feat), (10, 10))
    check('zero_restore_exact_query_identity', torch.equal(forward(), raw_query))
    forward().square().mean().backward()
    check('zero_init_restore_receives_first_gradient', bool(m.restore.weight.grad.abs().sum() > 0))
    with torch.no_grad():
        # Deliberate diagnostic intervention, not an optimizer/trained model.
        m.restore.weight[:128].copy_(torch.eye(128) * .05)
    m.zero_grad(set_to_none=True)
    global_feat.grad = local_feat.grad = None
    forward().square().mean().backward()
    check('nonzero_restore_has_finite_gradient_to_both_views', all(x.grad is not None and bool(torch.isfinite(x.grad).all()) and bool(x.grad.abs().sum() > 0) for x in (global_feat, local_feat)))
    check('no_added_geometry_gradients', logits.grad is None and ref.grad is None)
    torch.manual_seed(1)
    model = build_model().eval()
    init = load_independent_ema_for_cpu_probe(model, CHECKPOINT)
    check('frozen_probe_checkpoint_hash_exact', init['sha256'] == 'cf8bcea8d38e9ee826929f1b6d161a9ceff0ac4851e3a4318e1f197b2312aa97')
    check('annotations_hash_exact', hashlib.sha256(ANNOTATIONS.read_bytes()).hexdigest() == '0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd')
    dataset = PairedCocoImages(IMAGES, ANNOTATIONS, augment=False)
    samples, targets = paired_collate([dataset[0]])
    def equal_output(a, b):
        return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    with torch.inference_mode():
        model.range_interaction.variant = 'independent'
        native = model(samples)
        for variant in list(VARIANTS)[1:]:
            model.range_interaction.variant = variant
            active = model(samples)
            check('real_zero_init_predictions_exact_' + variant, all(equal_output(active[k], native[k]) for k in native))
        model.range_interaction.variant = 'adaptive_range_global_context'
        model.range_interaction.restore.weight[:128].copy_(torch.eye(128) * .05)
        changed = {**samples, 'global': torch.rand_like(samples['global'])}
        active, perturbed = model(samples), model(changed)
        delta = float((active['local']['pred_logits'] - perturbed['local']['pred_logits']).abs().max())
        check('real_global_input_can_change_local_prediction_after_restore_intervention', delta > 0)
        model.range_interaction.variant = 'adaptive_range_local_context'
        active, perturbed = model(samples), model(changed)
        check('real_local_context_control_has_no_global_input_path_in_eval', equal_output(active['local'], perturbed['local']))
    check('CPU_only_at_end', not torch.cuda.is_initialized())
    check('old_candidate_original_files_exact_after', source_exact())
    result = {'status': 'CPU_functional_pass_not_effectiveness_or_training_permission',
              'checks': checks, 'check_count': len(checks), 'seconds': time.time() - started,
              'checkpoint': init, 'image_id': int(targets['global'][0]['image_id']),
              'new_reader_parameters': sum(p.numel() for p in model.range_interaction.parameters()),
              'synthetic_restore_intervention': '0.05 identity in first 128 rows; no optimizer',
              'real_query_global_input_max_logit_change': delta,
              'CUDA_initialized': False,
              'limitations': ['Zero-init identity and reachable path are implementation properties, not useful context or AP.',
                              'Frozen old EMA plus untrained reader; no candidate training or GPU work.']}
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps({'checks': len(checks), 'seconds': result['seconds'], 'real_path_delta': delta}), flush=True)


if __name__ == '__main__':
    main()
