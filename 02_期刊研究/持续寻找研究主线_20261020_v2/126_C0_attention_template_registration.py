"""Fixed attention-template local registration. Parent reviews and executes.

Four eval forwards, no backward/VJP/update. Existing model outputs are observed
unchanged; only the copied final boxes are compared offline. See protocol 125.
"""
import os
import sys
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
import gc
import hashlib
import importlib.util
import json
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parent
PROTOCOL = ROOT / '125_事前协议.txt'
OUT = ROOT / '127_C0_attention_template_registration'
ARMS = ('original', 'GN', 'trace', 'barycenter')


def load_helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


h = load_helper('c0_build_113_for_126', ROOT / '113_C0_restore_route_frozen.py')
metric = load_helper('center_metrics_102_for_126', ROOT / '102_final_center_feasibility.py')
n, torch = h.n, h.torch
F = torch.nn.functional


def bn_hash(model):
    digest = hashlib.sha256()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            for key, value in module.state_dict().items():
                digest.update((name + '.' + key).encode('utf-8'))
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sample_feature(feature, points):
    """Native normalized grid convention, zero padding, in float64."""
    assert feature.ndim == 3 and points.shape[-1] == 2
    feature = feature.double()
    grid = points.double().reshape(1, -1, 1, 2) * 2 - 1
    values = F.grid_sample(feature.unsqueeze(0), grid, mode='bilinear',
                           padding_mode='zeros', align_corners=False)
    return values[0, :, :, 0].T.reshape(*points.shape[:-1], feature.shape[0])


def attention_template(feature, locations, weights):
    assert locations.shape[:-1] == weights.shape and locations.ndim == 4
    sampled = sample_feature(feature, locations)
    # Sum all native level/point weights within each head, then average heads.
    template = (sampled * weights.double().unsqueeze(-1)).sum(2).mean(1)
    barycenter = (locations.double() * weights.double().unsqueeze(-1)).sum(2).mean(1)
    return template, barycenter


def local_terms(feature, centers, image_wh):
    image_wh = torch.as_tensor(image_wh, dtype=torch.float64)
    offsets_pixels = torch.tensor([[0., 0.], [2., 0.], [-2., 0.],
                                   [0., 2.], [0., -2.]], dtype=torch.float64)
    points = centers.double()[:, None, :] + offsets_pixels[None, :, :] / image_wh
    values = sample_feature(feature, points)
    jacobian = torch.stack(((values[:, 1] - values[:, 2]) / 4,
                            (values[:, 3] - values[:, 4]) / 4), dim=-1)
    return values[:, 0], jacobian


def local_solve(template, center_features, jacobian):
    gram = jacobian.transpose(-1, -2) @ jacobian
    ridge = 1e-3 * gram.diagonal(dim1=-2, dim2=-1).sum(-1) / 2 + 1e-6
    matrix = gram + ridge[:, None, None] * torch.eye(2, dtype=torch.float64)
    rhs = (jacobian.transpose(-1, -2) @ (template - center_features).unsqueeze(-1)).squeeze(-1)
    gn = torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)
    trace = rhs / (matrix.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None] / 2)
    assert all(bool(torch.isfinite(v).all()) for v in (gram, matrix, rhs, gn, trace))
    return gn, trace, matrix, ridge


def synthetic_tests():
    assert metric.count(torch.tensor([[True, True]])) == 1
    assert metric.count(torch.eye(2, dtype=torch.bool)) == 2
    box = torch.tensor([[[.5, .5, .1, .1]]], dtype=torch.float64)
    assert abs(float(metric.pair_iou(box, box)) - 1) < 1e-12
    feature = torch.arange(16, dtype=torch.float64).reshape(1, 4, 4)
    points = torch.tensor([[.5 / 4, .5 / 4], [3.5 / 4, 2.5 / 4],
                           [-1., -1.], [2., 2.]], dtype=torch.float64)
    actual = sample_feature(feature, points)[:, 0]
    assert torch.allclose(actual, torch.tensor([0., 11., 0., 0.], dtype=torch.float64), atol=1e-12, rtol=0)
    yy, xx = torch.meshgrid((torch.arange(16, dtype=torch.float64) + .5) * 4,
                            (torch.arange(16, dtype=torch.float64) + .5) * 4, indexing='ij')
    affine = torch.stack((xx, 2 * yy, xx + yy))
    center = torch.tensor([[.5, .5]], dtype=torch.float64)
    expected_step = torch.tensor([[1., -1.]], dtype=torch.float64)
    template = sample_feature(affine, center + expected_step / 64)
    m, jac = local_terms(affine, center, (64, 64))
    expected_jac = torch.tensor([[[1., 0.], [0., 2.], [1., 1.]]], dtype=torch.float64)
    assert torch.allclose(jac, expected_jac, atol=1e-12, rtol=0)
    gn, scalar, matrix, ridge = local_solve(template, m, jac)
    assert torch.allclose(gn, expected_step, atol=.01, rtol=0)
    assert not torch.allclose(gn, scalar, atol=.01, rtol=0)
    assert torch.allclose(matrix @ gn.unsqueeze(-1),
                          jac.transpose(-1, -2) @ (template - m).unsqueeze(-1), atol=1e-12, rtol=0)
    cm, cj = local_terms(torch.ones_like(affine), center, (64, 64))
    cgn, cscalar, _, _ = local_solve(cm, cm, cj)
    assert torch.equal(cgn, torch.zeros_like(cgn)) and torch.equal(cscalar, torch.zeros_like(cscalar))
    locations = torch.tensor([[[[.25, .25], [.75, .75]],
                                [[.25, .75], [.75, .25]]]], dtype=torch.float64)
    weights = torch.tensor([[[.25, .75], [.6, .4]]], dtype=torch.float64)
    at, ab = attention_template(affine, locations, weights)
    expected_bary = (locations[0, 0, 0] * .25 + locations[0, 0, 1] * .75
                     + locations[0, 1, 0] * .6 + locations[0, 1, 1] * .4) / 2
    assert torch.allclose(ab[0], expected_bary, atol=1e-12, rtol=0)
    assert torch.allclose(at, sample_feature(affine, ab), atol=1e-12, rtol=0)
    assert torch.equal(torch.tensor([-3., 0., 3.]).clamp(-2, 2), torch.tensor([-2., 0., 2.]))
    return {'query_capacity_one': True, 'IoU_identity': True,
            'native_grid_pixel_center_and_zero_padding': True,
            'affine_feature_J_sign_and_input_pixel_units': True,
            'ridge_linear_system_and_trace_distinction': True,
            'constant_feature_zero_step': True, 'head_mean_and_alpha_weighting': True,
            'componentwise_step_bound': True, 'synthetic_GN_step': gn.tolist(),
            'synthetic_trace_step': scalar.tolist(), 'synthetic_ridge': ridge.tolist()}


def eval_with_records(model, image):
    """Observe original functions/hooks; never replace native tensor outputs."""
    assert not model.training and not model.decoder.training and not torch.is_grad_enabled()
    decoder = model.decoder.decoder
    last_attention = decoder.layers[decoder.eval_idx].cross_attn
    original_core = last_attention.ms_deformable_attn_core
    captures = {'reduce': [], 'attention': [], 'decoder': []}

    def core_observer(value, shapes, locations, weights, points_list):
        assert locations.shape == (1, 300, 8, 12, 2)
        assert weights.shape == (1, 300, 8, 12)
        captures['attention'].append({'sampling_locations': locations[0].detach().cpu().clone(),
                                      'attention_weights': weights[0].detach().cpu().clone(),
                                      'native_spatial_shapes': [[int(v) for v in shape] for shape in shapes],
                                      'native_points_list': [int(v) for v in points_list]})
        return original_core(value, shapes, locations, weights, points_list)

    def reduce_observer(module, args, output):
        assert output.shape == (1, 16, 160, 160)
        captures['reduce'].append(output[0].detach().cpu().clone())
        return None

    def decoder_observer(module, args, output):
        assert isinstance(output, tuple) and len(output) == 6
        assert output[2][-1].shape == (1, 300, 132)
        assert output[3][-1].shape == (1, 300, 4)
        captures['decoder'].append({'final_corners': output[2][-1][0].detach().cpu().clone(),
                                    'ref_points_initial': output[3][-1][0].detach().cpu().clone(),
                                    'boxes': output[0][-1][0].detach().cpu().clone(),
                                    'logits': output[1][-1][0].detach().cpu().clone()})
        return None

    handles = [model.backbone.p2_skip.reduce.register_forward_hook(reduce_observer),
               decoder.register_forward_hook(decoder_observer)]
    last_attention.ms_deformable_attn_core = core_observer
    try:
        h.memory_guard()
        assert h.COUNTS['model_forwards'] < 4
        output, indices = h.eval_with_selection_record(model, image)
        assert {key: len(value) for key, value in captures.items()} == {'reduce': 1, 'attention': 1, 'decoder': 1}
        final = captures['decoder'][0]
        assert torch.equal(output['pred_boxes'][0], final.pop('boxes'))
        assert torch.equal(output['pred_logits'][0], final.pop('logits'))
        return output, indices, {'reduced_p2': captures['reduce'][0],
                                 **captures['attention'][0], **final}
    finally:
        last_attention.ms_deformable_attn_core = original_core
        for handle in handles:
            handle.remove()


def natural_budget(logits):
    assert logits.shape == (300, 10)
    flat = logits.sigmoid().flatten().topk(300).indices
    allowed = torch.zeros((300, 10), dtype=torch.bool)
    allowed[flat // 10, flat % 10] = True
    assert int(allowed.sum()) == 300
    return flat, allowed


def existence(pair, allowed, labels):
    result = {}
    for kind, edges in [('geo', pair), ('class', pair * allowed[:, labels])]:
        for threshold, suffix in [(.5, '05'), (.75, '075')]:
            result[kind + suffix] = (edges >= threshold).any(0)
    return result


def compare_existence(current, original, small):
    report = {}
    for key, covered in current.items():
        base = original[key]
        report[key] = {'covered_GT_indices': torch.where(covered)[0].tolist(),
                       'covered_boolean_by_GT': covered.tolist()}
        for group, mask in [('all', torch.ones_like(small)), ('sub4', small)]:
            conditions = {'new': covered & ~base & mask, 'lost': ~covered & base & mask,
                          'retained': covered & base & mask, 'uncovered_both': ~covered & ~base & mask}
            report[key][group] = {name: {'count': int(value.sum()),
                                        'GT_indices': torch.where(value)[0].tolist()}
                                  for name, value in conditions.items()}
    return report


def arms_from_capture(boxes, indices, capture, image_wh):
    h.memory_guard()
    p3 = indices < 6400
    assert indices.min() >= 0 and indices.max() < 8400
    feature = capture['reduced_p2']
    locations, weights = capture['sampling_locations'], capture['attention_weights']
    assert all(bool(torch.isfinite(value).all()) for value in (boxes, feature, locations, weights))
    alpha_error = float((weights.double().sum(-1) - 1).abs().max())
    assert alpha_error <= 2e-6 and bool((weights >= 0).all())
    template, barycenter = attention_template(feature, locations, weights)
    m, jacobian = local_terms(feature, boxes[:, :2], image_wh)
    gn, scalar, matrix, ridge = local_solve(template, m, jacobian)
    wh = torch.as_tensor(image_wh, dtype=torch.float64)
    raw_steps = {'GN': gn, 'trace': scalar, 'barycenter': (barycenter - boxes[:, :2]) * wh}
    boxes_by_arm = {'original': boxes.clone()}
    deltas = {'original': torch.zeros((300, 2), dtype=torch.float64)}
    step_stats = {}
    for arm, raw_step in raw_steps.items():
        assert bool(torch.isfinite(raw_step).all())
        delta = raw_step.clamp(-2, 2)
        delta[~p3] = 0
        changed = boxes.clone()
        changed[p3, :2] = boxes[p3, :2] + delta[p3] / wh
        assert torch.equal(changed[:, 2:], boxes[:, 2:])
        assert torch.equal(changed[~p3], boxes[~p3])
        assert bool((delta.abs() <= 2).all()) and bool(torch.isfinite(changed).all())
        boxes_by_arm[arm], deltas[arm] = changed, delta
        step_stats[arm] = {'mean_P3_step_L2_pixels': float(torch.linalg.vector_norm(delta[p3], dim=-1).mean()),
                           'max_P3_abs_component_pixels': float(delta[p3].abs().max()),
                           'P3_queries_with_any_raw_component_outside_2px': int((raw_step[p3].abs() > 2).any(-1).sum()),
                           'P3_queries': int(p3.sum())}
    outside = ((locations < 0) | (locations > 1)).any(-1)
    outside_mass = (weights.double() * outside).sum(-1).mean(-1)
    diagnostics = {'alpha_head_sum_max_abs_error': alpha_error,
                   'P3_count': int(p3.sum()), 'P4_P5_count': int((~p3).sum()),
                   'P3_mean_weight_at_centers_outside_normalized_canvas': float(outside_mass[p3].mean()),
                   'P3_max_weight_at_centers_outside_normalized_canvas': float(outside_mass[p3].max()),
                   'P3_ridge_min': float(ridge[p3].min()), 'P3_ridge_max': float(ridge[p3].max()),
                   'P3_A_condition_max': float(torch.linalg.cond(matrix[p3]).max()),
                   'step_stats': step_stats,
                   'note': 'Out-of-canvas center mass is not exact bilinear zero-padding mass; no masking/renormalization is applied.'}
    return boxes_by_arm, deltas, diagnostics


def check_parity(output, indices, case, reference):
    assert case['event'] == reference['event'] and case['input_sha256'] == reference['input_sha256']
    assert torch.equal(case['target']['boxes'], reference['target']['boxes'])
    assert torch.equal(case['target']['labels'], reference['target']['labels'])
    assert torch.equal(case['original_sub4'].bool(), reference['original_sub4'].bool())
    assert torch.equal(indices, reference['selected_indices'])
    report = {}
    for key, maximum in [('pred_boxes', 1e-6), ('pred_logits', 1e-4)]:
        current, cached = output[key][0], reference[key]
        error = float((current.double() - cached.double()).abs().max())
        report[key] = {'max_absolute_error': error, 'exact_equal': torch.equal(current, cached),
                       'predeclared_maximum_absolute_error': maximum}
        assert error <= maximum, (key, error, maximum)
    flat, allowed = natural_budget(output['pred_logits'][0])
    old_flat, _ = natural_budget(reference['pred_logits'])
    assert torch.equal(flat, old_flat)
    report.update(selected_indices_exact=True, natural_top300_flat_ranking_exact=True,
                  native_selection_gather_exact=True)
    return report, flat, allowed


def main():
    started = time.monotonic()
    resources = h.resources()
    torch.set_grad_enabled(False)
    OUT.mkdir(exist_ok=False)
    model, meta = None, None
    before, before_bn = None, None
    try:
        n.save_json(OUT / '00_synthetic_tests.json', synthetic_tests())
        input_paths = [ROOT / '104_C0_本地资产严格加载核验.json',
                       ROOT / '77_all_count_frozen/00_all_count_inputs.pt',
                       ROOT / '85_source_self_attention/00_selected.json',
                       ROOT / '114_C0_restore_route_frozen/03_eval_predictions_and_queries.pt',
                       ROOT / '114_C0_restore_route_frozen/04_execution.json',
                       ROOT / '121_C0_center_feasibility/01_results.json',
                       ROOT / '113_C0_restore_route_frozen.py', ROOT / '102_final_center_feasibility.py',
                       PROTOCOL, pathlib.Path(__file__)]
        hashes = {str(path): n.sha(path) for path in input_paths}
        meta = json.loads(input_paths[0].read_text(encoding='utf-8'))
        for path_key, hash_key in [('checkpoint', 'checkpoint_sha256'), ('original_source', 'source_sha256'),
                                   ('recipe', 'recipe_sha256')]:
            assert n.sha(meta[path_key]) == meta[hash_key], path_key
            hashes[meta[path_key]] = meta[hash_key]
        for relative in ['common.py', 'native/engine/deim/deim.py', 'native/engine/deim/dfine_decoder.py',
                         'native/engine/deim/dfine_utils.py', 'native/engine/deim/utils.py',
                         'native/engine/deim/box_ops.py', 'native/engine/deim/hybrid_encoder.py']:
            hashes[str(h.COMMON / relative)] = n.sha(h.COMMON / relative)
        n.save_json(OUT / '00_inputs_and_protocol_sha256.json', hashes)
        raw = torch.load(meta['checkpoint'], map_location='cpu', weights_only=False)
        assert raw['last_epoch'] == meta['last_epoch'] == 94
        model, criterion, _, strict_load = h.build(meta, raw)
        del raw, criterion
        gc.collect()
        h.mode(model, False)
        assert all(not module.training for module in model.modules())
        assert not torch.backends.mha.get_fastpath_enabled()
        before, before_bn = n.model_hash(model), bn_hash(model)
        prior_run = json.loads(input_paths[4].read_text(encoding='utf-8'))
        assert prior_run['state'] == 'complete' and prior_run['checkpoint_unchanged']
        assert before == prior_run['state_hash_before'] == prior_run['state_hash_after']
        old = torch.load(input_paths[3], map_location='cpu', weights_only=False)
        assert old['mode'] == 'eval' and old['mha_fastpath_enabled'] is False and len(old['cases']) == 4
        assert old['checkpoint_sha256'] == meta['checkpoint_sha256'] and old['model_state_sha256'] == before
        selected = json.loads(input_paths[2].read_text(encoding='utf-8'))
        data = torch.load(input_paths[1], map_location='cpu', weights_only=False)
        cases = [next(case for case in data['cases'] if case['event'] == item['event']) for item in selected]
        assert len(cases) == 4 and sum(len(case['target']['labels']) for case in cases) == 902
        assert sum(int(case['original_sub4'].sum()) for case in cases) == 453
        images = [data['images'][case['input_sha256']].clone() for case in cases]
        for case, item, image in zip(cases, selected, images):
            assert case['input_sha256'] == item['input_sha256'] == h.tensor_sha(image)
        del data
        gc.collect()
        reg_max = int(model.decoder.reg_max)
        up, reg_scale = float(model.decoder.up.item()), float(model.decoder.reg_scale.item())
        assert reg_max == model.decoder.decoder.reg_max == 32
        assert up == float(model.decoder.decoder.up.item())
        assert reg_scale == float(model.decoder.decoder.reg_scale.item())
        rows, records = [], []
        totals = {arm: {} for arm in ARMS}
        existence_totals = {arm: {} for arm in ARMS}
        for number, (case, image) in enumerate(zip(cases, images)):
            assert image.shape == (3, 640, 640)
            reference = old['cases'][number]
            assert reference['image_index'] == number
            with torch.no_grad():
                output, indices, capture = eval_with_records(model, image.unsqueeze(0))
                parity, flat, allowed = check_parity(output, indices, case, reference)
                boxes = output['pred_boxes'][0].double()  # Actual final boxes, never encoder/pre reference.
                boxes_by_arm, deltas, diagnostics = arms_from_capture(boxes, indices, capture, (640, 640))
            target, small = case['target'], case['original_sub4'].bool()
            truth, labels = target['boxes'].double(), target['labels']
            original_pair = metric.pair_iou(boxes[:, None, :], truth[None, :, :])
            base_existence = existence(original_pair, allowed, labels)
            cached_pair = metric.pair_iou(reference['pred_boxes'].double()[:, None, :], truth[None, :, :])
            assert metric.metrics(original_pair, allowed, labels, small) == metric.metrics(cached_pair, allowed, labels, small)
            image_rows = []
            for arm in ARMS:
                pair = metric.pair_iou(boxes_by_arm[arm][:, None, :], truth[None, :, :])
                counts = metric.metrics(pair, allowed, labels, small)
                presence = compare_existence(existence(pair, allowed, labels), base_existence, small)
                row = {'image_index': number, 'event': case['event'], 'arm': arm,
                       'GT': len(labels), 'original_sub4': int(small.sum()),
                       'counts': counts, 'per_GT_existence_vs_original': presence}
                image_rows.append(row)
                rows.append(row)
                for key, value in counts.items():
                    totals[arm][key] = totals[arm].get(key, 0) + value
                for kind, detail in presence.items():
                    for group in ('all', 'sub4'):
                        for change in ('new', 'lost', 'retained', 'uncovered_both'):
                            key = f'{group}_{kind}_{change}'
                            existence_totals[arm][key] = existence_totals[arm].get(key, 0) + detail[group][change]['count']
            record = {'image_index': number, 'event': case['event'], 'input_sha256': case['input_sha256'],
                      'pred_boxes': output['pred_boxes'][0].detach().cpu().clone(),
                      'pred_logits': output['pred_logits'][0].detach().cpu().clone(),
                      'selected_indices': indices.clone(), 'target': n.cpu_tree(target),
                      'original_sub4': small.clone(), **capture, 'delta_pixels': deltas,
                      'boxes_by_arm': boxes_by_arm, 'natural_top300_flat_indices': flat,
                      'parity': parity}
            records.append(record)
            n.save_json(OUT / f'02_image_{number}.json', {'parity': parity, 'diagnostics': diagnostics, 'rows': image_rows})
            assert all(parameter.grad is None for parameter in model.parameters())
            del output
            gc.collect()
        expected = json.loads(input_paths[5].read_text(encoding='utf-8'))
        assert expected['state'] == 'complete' and expected['checkpoint_sha256'] == meta['checkpoint_sha256']
        assert totals['original'] == expected['totals']['original']
        checks = {'GN_sub4_class05_strictly_over_' + arm: totals['GN']['sub4_class05'] > totals[arm]['sub4_class05']
                  for arm in ('original', 'trace', 'barycenter')}
        checks['GN_all_class05_not_below_original'] = totals['GN']['all_class05'] >= totals['original']['all_class05']
        gate = {'passed': all(checks.values()), 'checks': checks, 'eligible_candidate': 'GN_only',
                'meaning': 'resource condition for continuing this structural variable; not significance, training approval, AP or novelty'}
        assert h.COUNTS == {'model_forwards': 4, 'model_vjps': 0, 'optimizer_steps': 0}
        after, after_bn = n.model_hash(model), bn_hash(model)
        assert before == after and before_bn == after_bn
        assert n.sha(meta['checkpoint']) == meta['checkpoint_sha256']
        assert all(parameter.grad is None for parameter in model.parameters())
        cache = {'records': records, 'reg_max': reg_max, 'up': up, 'reg_scale': reg_scale,
                 'checkpoint_sha256': meta['checkpoint_sha256'], 'weight_key': 'model_online', 'model_state_sha256': before,
                 'BN_state_sha256': before_bn, 'mode': 'eval', 'input_shape': [1, 3, 640, 640],
                 'coordinate_convention': 'native normalized grid; grid_sample align_corners=False, bilinear, zeros; boxes normalized cxcywh',
                 'protocol_sha256': hashes[str(PROTOCOL)], 'script_sha256': hashes[str(pathlib.Path(__file__))]}
        torch.save(cache, OUT / '01_cache.pt')
        n.save_json(OUT / '03_results.json', {'state': 'complete', 'GT': 902, 'original_sub4': 453,
                    'totals': totals, 'existence_totals_vs_original': existence_totals, 'gate': gate,
                    'rows_file_pattern': '02_image_*.json', 'weight_key': 'model_online', 'epoch': 94,
                    'scope': 'Four reused development images, one C0 seed; frozen final-center intervention only'})
        execution = {'state': 'complete', 'resources': resources, 'counts': h.COUNTS.copy(),
                     'strict_load_online': strict_load, 'state_hash_before': before, 'state_hash_after': after,
                     'BN_hash_before': before_bn, 'BN_hash_after': after_bn,
                     'checkpoint_unchanged': True, 'model_state_and_BN_unchanged': True,
                     'new_training': False, 'optimizer_called': False, 'criterion_called': False,
                     'MHA_fastpath_enabled': False, 'gate': gate, 'seconds': time.monotonic() - started,
                     'cache_path': str(OUT / '01_cache.pt')}
        n.save_json(OUT / '04_execution.json', execution)
        n.save_json(OUT / '05_output_sha256.json', {path.name: n.sha(path) for path in OUT.iterdir() if path.is_file()})
        print(json.dumps({'state': 'complete', 'totals': totals, 'gate': gate, 'counts': h.COUNTS,
                          'seconds': execution['seconds']}, ensure_ascii=False), flush=True)
    except Exception as exc:
        n.save_json(OUT / 'FAILED.json', {'state': 'failed', 'type': type(exc).__name__, 'message': str(exc),
                    'counts': h.COUNTS.copy(), 'model_state_unchanged': None if model is None or before is None else n.model_hash(model) == before,
                    'BN_state_unchanged': None if model is None or before_bn is None else bn_hash(model) == before_bn,
                    'checkpoint_unchanged': None if meta is None else n.sha(meta['checkpoint']) == meta['checkpoint_sha256'],
                    'new_training': False, 'seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    main()
