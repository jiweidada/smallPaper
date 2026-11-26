"""C0 restore-only gradient correspondence. No updates; parent executes after review."""
import os
import sys
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
import copy
import gc
import hashlib
import importlib.util
import json
import pathlib
import random
import time
import types

ROOT = pathlib.Path(__file__).resolve().parent
WORK = ROOT.parents[1]
COMMON = WORK / '英文小论文/code/dn_pre_frozen_diagnostic_20261020_v1'
sys.path.insert(0, str(COMMON))
import common as n
import yaml

torch, np = n.torch, n.np
F = torch.nn.functional
PROTOCOL = ROOT / '112_C0_restore参数路由_事前协议.txt'
OUT = ROOT / '114_C0_restore_route_frozen'
W_NAME = 'backbone.p2_skip.restore.weight'
DN_SEED = 20261021
COUNTS = {'model_forwards': 0, 'model_vjps': 0, 'optimizer_steps': 0}


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def resources():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    proc = n.psutil.Process()
    proc.nice(n.psutil.BELOW_NORMAL_PRIORITY_CLASS)
    allowed = proc.cpu_affinity()
    assert {13, 15}.issubset(allowed), allowed
    proc.cpu_affinity([13, 15])
    assert sorted(proc.cpu_affinity()) == [13, 15]
    assert torch.version.cuda is None and not torch.cuda.is_available()
    memory_guard()
    torch.backends.mha.set_fastpath_enabled(False)
    return {'threads': 2, 'interop_threads': 1, 'priority': 'below_normal',
            'affinity': proc.cpu_affinity(), 'torch': torch.__version__,
            'available_RAM_GiB': n.psutil.virtual_memory().available / 2**30,
            'mha_fastpath_enabled': False}


def memory_guard():
    assert n.psutil.virtual_memory().available / 2**30 > 8, 'RAM guard'


def project(g, p):
    assert g.ndim == 4 and g.shape[1] % 4 == 0 and g.shape[2:] == (1, 1)
    z = g.reshape(g.shape[0], g.shape[1] // 4, 4)
    return torch.einsum('ock,kl->ocl', z, p).reshape_as(g)


def bases_and_tests():
    h = torch.tensor([[1, 1, 1, 1], [-1, 1, -1, 1],
                      [-1, -1, 1, 1], [1, -1, -1, 1]], dtype=torch.float64) / 2
    eye = torch.eye(4, dtype=torch.float64)
    assert torch.allclose(h @ h.T, eye, atol=1e-12, rtol=0)
    p = [v[:, None] @ v[None, :] for v in h]
    pa = torch.outer((h[1] + h[2]) / 2**.5, (h[1] + h[2]) / 2**.5)
    pb = torch.outer((-h[1] + h[2]) / 2**.5, (-h[1] + h[2]) / 2**.5)
    ps = {'mean': p[0], 'x': p[1], 'y': p[2], 'd': p[3],
          'delta': p[1] + p[2] + p[3], 'xy': p[1] + p[2], 'a': pa, 'b': pb}
    for key, mat in ps.items():
        assert torch.allclose(mat, mat.T, atol=1e-12, rtol=0)
        assert torch.allclose(mat @ mat, mat, atol=1e-12, rtol=0)
        assert int(torch.linalg.matrix_rank(mat)) == (3 if key == 'delta' else 2 if key == 'xy' else 1)
    assert torch.allclose(sum(p), eye, atol=1e-12, rtol=0)
    assert torch.allclose(pa + pb, ps['xy'], atol=1e-12, rtol=0)
    source = torch.arange(32, dtype=torch.float64).reshape(1, 2, 4, 4)
    packed = F.pixel_unshuffle(source, 2)
    for c in range(2):
        for i in range(2):
            for j in range(2):
                assert torch.equal(packed[0, c * 4 + 2 * i + j], source[0, c, i::2, j::2])
    g = torch.arange(16, dtype=torch.float64).reshape(2, 8, 1, 1)
    blocks = g.reshape(2, 2, 4)
    for a in range(4):
        expected = (blocks @ h[a])[..., None] * h[a]
        assert torch.allclose(project(g, p[a]).reshape_as(blocks), expected, atol=1e-12, rtol=0)
    gx, gy = g, g.flip(0) * 0.37
    axis = project(gx, ps['x']) + project(gy, ps['y'])
    swap = project(gx, ps['y']) + project(gy, ps['x'])
    mixed = 0.5 * project(gx + gy, ps['xy'])
    assert torch.allclose(mixed, 0.5 * (axis + swap), atol=1e-12, rtol=0)
    jac = torch.tensor([[1., 1.], [0., 1.]], dtype=torch.float64)
    pin = torch.diag(torch.tensor([1., 0.], dtype=torch.float64))
    delta_z = jac @ jac.T @ pin @ torch.tensor([1., 0.], dtype=torch.float64)
    assert delta_z[1] != 0  # Input projection does not constrain a shared-parameter response.
    return ps, {'haar_orthogonal': True, 'projector_ranks': {k: int(torch.linalg.matrix_rank(v)) for k, v in ps.items()},
                'pixel_unshuffle_order_c4_2i_j': True, 'right_projection_matches_haar': True,
                'M_equals_half_axis_plus_swap': True,
                'input_projection_shared_J_counterexample': delta_z.tolist()}


class AuditCriterion(n.DEIMCriterion):
    def mark(self, out):
        self.final_boxes = out['pred_boxes']
        self.final_record = None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        result = super().get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)
        assert all(bool(torch.isfinite(v).all()) for v in result.values()), ('native nonfinite loss', loss)
        return result

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        result = super().loss_boxes(outputs, targets, indices, num_boxes, boxes_weight)
        if outputs['pred_boxes'] is self.final_boxes:
            assert self.final_record is None
            assert boxes_weight is None
            self.final_record = {'indices': n.cpu_tree(indices), 'denominator': float(num_boxes)}
        return result


def build(meta, raw):
    recipe = yaml.safe_load(pathlib.Path(meta['recipe']).read_text(encoding='utf-8-sig'))
    assert recipe['num_classes'] == 10 and recipe['DEIMCriterion']['losses'] == ['mal', 'boxes', 'local']
    mod_name = 'engine.backbone.c0_restore_route_source_113'
    spec = importlib.util.spec_from_file_location(mod_name, meta['original_source'])
    source = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = source  # register() inspects its actual module.
    spec.loader.exec_module(source)
    bc = copy.deepcopy(recipe['HGNetv2'])
    bc.update(pretrained=False, use_p2_skip=True, p2_skip_width=16,
              use_detail=False, use_context=False)
    model = n.DEIM(source.HGNetv2_Light(**bc),
                   n.HybridEncoder(**copy.deepcopy(recipe['HybridEncoder']), eval_spatial_size=[640, 640]),
                   n.DFINETransformer(**copy.deepcopy(recipe['DFINETransformer']), num_classes=10,
                                      eval_spatial_size=[640, 640]), num_classes=10)
    loaded = str(model.load_state_dict(raw['model'], strict=True))
    assert len(model.state_dict()) == 801
    assert model.backbone.p2_skip.reduce.conv.weight.shape == (16, 64, 1, 1)
    assert dict(model.named_parameters())[W_NAME].shape == (256, 64, 1, 1)
    cc = copy.deepcopy(recipe['DEIMCriterion'])
    mc = cc.pop('matcher')
    assert mc.pop('type') == 'HungarianMatcher'
    criterion = AuditCriterion(matcher=n.HungarianMatcher(**mc, use_focal_loss=True), num_classes=10, **cc)
    criterion.load_state_dict(raw['criterion'], strict=True)
    assert criterion.use_uni_set and criterion.boxes_weight_format is None and criterion.boxes_iou_type == 'giou'
    assert criterion.weight_dict['loss_bbox'] == 5 and criterion.weight_dict['loss_giou'] == 2
    assert not model.decoder.learn_query_content
    assert all(m.p == 0 for m in model.modules() if isinstance(m, torch.nn.Dropout))
    return model, criterion, recipe, loaded


def mode(model, decoder_training):
    model.eval()
    if decoder_training:
        model.decoder.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    assert not model.training and not model.backbone.training and not model.encoder.training
    assert model.decoder.training == decoder_training
    assert all(not m.training for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))


def forward(model, image, target=None):
    memory_guard()
    assert COUNTS['model_forwards'] < 8
    COUNTS['model_forwards'] += 1
    return model(image, target)


def eval_with_selection_record(model, image):
    """Observe native selection; return the original tensors without modification."""
    decoder = model.decoder
    assert decoder.query_select_method == 'default' and not decoder.training
    original = decoder._select_topk
    seen = []

    def observe(self, memory, logits, anchors, topk):
        result = original(memory, logits, anchors, topk)
        assert topk == 300 and memory.shape[:2] == (1, 8400)
        idx = torch.topk(logits.max(-1).values, topk, dim=-1).indices
        expected_memory = memory.gather(1, idx.unsqueeze(-1).expand(-1, -1, memory.shape[-1]))
        expected_anchors = anchors.gather(1, idx.unsqueeze(-1).expand(-1, -1, anchors.shape[-1]))
        assert torch.equal(result[0], expected_memory)
        assert torch.equal(result[2], expected_anchors)
        assert result[1] is None  # Native eval does not return selected logits.
        assert idx.unique().numel() == 300
        seen.append(idx.detach().cpu().clone()[0])
        return result

    decoder._select_topk = types.MethodType(observe, decoder)
    try:
        output = forward(model, image)
        assert len(seen) == 1
        return output, seen[0]
    finally:
        decoder._select_topk = original


def vjp(loss, w, retain=True):
    memory_guard()
    assert COUNTS['model_vjps'] < 21
    COUNTS['model_vjps'] += 1
    grad = torch.autograd.grad(loss, w, retain_graph=retain, create_graph=False, allow_unused=False)[0]
    assert bool(torch.isfinite(grad).all())
    assert w.grad is None
    return grad.detach().double().clone()


def dot(a, b):
    return float((a * b).sum())


def norm(a):
    return float(torch.linalg.vector_norm(a))


def tol(a, b=0.):
    return 1e-6 * max(abs(a), abs(b), 1e-12)


def gt(a, b=0.):
    return a > b + tol(a, b)


def ge(a, b):
    return a >= b - tol(a, b)


def directions(g, ps):
    x, y, total = g['cx'], g['cy'], g['total']
    c = x + y
    rest = total - c
    centers = {'O': c, 'Pdelta': project(c, ps['delta']), 'Pxy': project(c, ps['xy']),
               'M': 0.5 * project(c, ps['xy']),
               'axis': project(x, ps['x']) + project(y, ps['y']),
               'swap': project(x, ps['y']) + project(y, ps['x']),
               'rot45': project(x, ps['a']) + project(y, ps['b'])}
    cn = norm(c)
    if cn == 0:
        return None, {'nonzero_center': False}
    for arm, part in list(centers.items()):
        if arm != 'O':
            centers['S_' + arm] = c * (norm(part) / cn)
    raw = {arm: rest + part for arm, part in centers.items()}
    on = norm(raw['O'])
    balanced = {arm: value * (on / norm(value)) if norm(value) > 0 else value.clone()
                for arm, value in raw.items()}
    return {'raw': raw, 'total_direction_norm_matched': balanced}, {
        'nonzero_center': True, 'original_center_norm': cn, 'rest_norm': norm(rest),
        'center_norms': {k: norm(v) for k, v in centers.items()},
        'raw_direction_norms': {k: norm(v) for k, v in raw.items()},
        'paired_direction_norms': {k: norm(v) for k, v in balanced.items()}}


def analyze(grads, rows, ps):
    folds = []
    for j in range(4):
        train = {key: sum(grads[i][key] for i in range(4) if i != j) / 3
                 for key in ['cx', 'cy', 'total']}
        ds, norms = directions(train, ps)
        test = dict(grads[j])
        test['center'] = test['cx'] + test['cy']
        record = {'test_development_image': j, 'train_development_images': [i for i in range(4) if i != j],
                  'norms': norms, 'gain_sign': 'positive predicts loss decrease under -epsilon*D', 'gains': {}}
        if ds is not None:
            for pairing, arms in ds.items():
                record['gains'][pairing] = {arm: {key: dot(value, test[key]) for key in
                                                ['total', 'center', 'cx', 'cy', 'classification', 'hard_geometry']}
                                             for arm, value in arms.items()}
        folds.append(record)
    valid = all('total_direction_norm_matched' in f['gains'] for f in folds)
    if not valid:
        return {'folds': folds, 'gate': {'passed': False, 'reason': 'zero training-side center gradient'},
                'scope': 'restore W only; repeated development images; local SGD directional derivative'}
    means = {}
    for pairing in ['raw', 'total_direction_norm_matched']:
        means[pairing] = {arm: {key: sum(f['gains'][pairing][arm][key] for f in folds) / 4
                               for key in folds[0]['gains'][pairing][arm]}
                         for arm in folds[0]['gains'][pairing]}
    m = means['total_direction_norm_matched']
    center_wins = sum(gt(f['gains']['total_direction_norm_matched']['axis']['center'],
                         f['gains']['total_direction_norm_matched']['O']['center']) for f in folds)
    checks = {
        'hard_group_nonzero_at_least_3_images': sum(r['hard_union_pairs'] > 0 and norm(g['hard_geometry']) > 0
                                                  for r, g in zip(rows, grads)) >= 3,
        'O_and_axis_directions_nonzero_all_folds': all(f['norms']['raw_direction_norms']['O'] > 0 and
                                                      f['norms']['raw_direction_norms']['axis'] > 0 for f in folds),
        'axis_mean_center_gain_positive': gt(m['axis']['center']),
        'center_wins_over_O_at_least_3_of_4': center_wins >= 3,
        'axis_mean_hard_geometry_gain_positive': gt(m['axis']['hard_geometry'])}
    for control in ['O', 'M', 'S_axis', 'swap', 'rot45']:
        checks['axis_mean_center_strictly_over_' + control] = gt(m['axis']['center'], m[control]['center'])
    for key in ['total', 'classification', 'hard_geometry']:
        checks['axis_mean_' + key + '_not_below_O'] = ge(m['axis'][key], m['O'][key])
    return {'folds': folds, 'means': means, 'gate': {'passed': all(checks.values()), 'checks': checks,
             'center_wins_over_O': center_wins, 'eligible_candidate': 'axis_only',
             'meaning': 'continue structural/optimizer audit only; no training authorization',
             'three_of_four_is_resource_rule_not_statistical_significance': True},
            'scope': 'restore W only; repeated development images; local SGD directional derivative, not Adam/AP'}


def main():
    started = time.monotonic()
    status = resources()
    ps, tests = bases_and_tests()
    OUT.mkdir(exist_ok=False)
    n.save_json(OUT / '00_synthetic_tests.json', tests)
    input_paths = [ROOT / '104_C0_本地资产严格加载核验.json',
                   ROOT / '77_all_count_frozen/00_all_count_inputs.pt',
                   ROOT / '85_source_self_attention/00_selected.json']
    hashes = {str(p): n.sha(p) for p in input_paths}
    meta = json.loads(input_paths[0].read_text(encoding='utf-8'))
    for path_key, hash_key in [('checkpoint', 'checkpoint_sha256'), ('original_source', 'source_sha256'),
                               ('recipe', 'recipe_sha256')]:
        assert n.sha(meta[path_key]) == meta[hash_key], path_key
        hashes[meta[path_key]] = meta[hash_key]
    for relative in ['common.py', 'native/engine/deim/deim.py', 'native/engine/deim/dfine_decoder.py',
                     'native/engine/deim/deim_criterion.py', 'native/engine/deim/matcher.py',
                     'native/engine/deim/dfine_utils.py', 'native/engine/deim/box_ops.py',
                     'native/engine/deim/denoising.py', 'native/engine/deim/hybrid_encoder.py',
                     'native/engine/backbone/hgnetv2.py']:
        hashes[str(COMMON / relative)] = n.sha(COMMON / relative)
    hashes[str(PROTOCOL)] = n.sha(PROTOCOL)
    hashes[str(pathlib.Path(__file__))] = n.sha(__file__)
    n.save_json(OUT / '00_inputs_and_protocol_sha256.json', hashes)
    raw = torch.load(meta['checkpoint'], map_location='cpu', weights_only=False)
    assert raw['last_epoch'] == meta['last_epoch'] == 94
    assert len(raw['optimizer']['param_groups']) == 4
    model, criterion, recipe, strict_load = build(meta, raw)
    epoch = int(raw['last_epoch'])
    del raw
    gc.collect()
    before = n.model_hash(model)
    original_requires = {key: p.requires_grad for key, p in model.named_parameters()}
    for key, p in model.named_parameters():
        p.requires_grad_(key == W_NAME)
    w = dict(model.named_parameters())[W_NAME]
    selected = json.loads(input_paths[2].read_text(encoding='utf-8'))
    data = torch.load(input_paths[1], map_location='cpu', weights_only=False)
    cases = [next(c for c in data['cases'] if c['event'] == item['event']) for item in selected]
    assert len(cases) == 4 and sum(len(c['target']['labels']) for c in cases) == 902
    assert sum(int(c['original_sub4'].sum()) for c in cases) == 453
    images = [data['images'][c['input_sha256']].clone() for c in cases]
    for c, item, im in zip(cases, selected, images):
        assert c['input_sha256'] == item['input_sha256']
        assert tensor_sha(im) == c['input_sha256']
    del data
    gc.collect()
    rows, grads, predictions, eval_cache = [], [], [], []
    try:
        for number, (case, image) in enumerate(zip(cases, images)):
            target = n.cpu_tree(case['target'])
            targets = [target]
            image = image.unsqueeze(0)
            assert image.shape == (1, 3, 640, 640)
            mode(model, False)
            with torch.no_grad():
                evaluated, selected_indices = eval_with_selection_record(model, image)
                pred = {key: evaluated[key].detach().clone() for key in ['pred_boxes', 'pred_logits']}
                iou = n.box_iou(n.box_cxcywh_to_xyxy(pred['pred_boxes'][0]),
                                n.box_cxcywh_to_xyxy(target['boxes']))[0]
                miss = case['original_sub4'].bool() & ~(iou >= .5).any(0)
                eval_cache.append({'image_index': number, 'event': case['event'],
                                   'input_sha256': case['input_sha256'],
                                   'pred_boxes': pred['pred_boxes'][0].clone(),
                                   'pred_logits': pred['pred_logits'][0].clone(),
                                   'selected_indices': selected_indices,
                                   'target': n.cpu_tree(target),
                                   'original_sub4': case['original_sub4'].bool().clone()})
            del evaluated, iou
            mode(model, True)
            random.seed(DN_SEED)
            np.random.seed(DN_SEED)
            torch.manual_seed(DN_SEED)
            rng_before = tensor_sha(torch.get_rng_state())
            out = forward(model, image, targets)
            rng_after = tensor_sha(torch.get_rng_state())
            assert all(k in out for k in ['aux_outputs', 'enc_aux_outputs', 'pre_outputs', 'dn_outputs',
                                         'dn_pre_outputs', 'dn_meta', 'pred_corners', 'ref_points'])
            assert all(bool(torch.isfinite(out[k]).all()) for k in ['pred_boxes', 'pred_logits'])
            criterion.mark(out)
            losses = criterion(out, targets, epoch=epoch, step=0, epoch_step=1, global_step=None)
            rec = criterion.final_record
            assert rec is not None
            indices, denom = rec['indices'], rec['denominator']
            assert len(indices) == 1 and denom == max(1, sum(len(q) for q, _ in indices))
            q, gi = indices[0]
            src = out['pred_boxes'][0, q]
            truth = target['boxes'][gi]
            abs_error = F.l1_loss(src, truth, reduction='none')
            coefficient = criterion.weight_dict['loss_bbox']
            cx = coefficient * abs_error[:, 0].sum() / denom
            cy = coefficient * abs_error[:, 1].sum() / denom
            wh = coefficient * abs_error[:, 2:].sum() / denom
            full = sum(losses.values())
            rest = sum(value for key, value in losses.items() if key != 'loss_bbox') + wh
            assert torch.allclose(cx + cy + wh, losses['loss_bbox'], rtol=2e-5, atol=2e-6)
            assert torch.allclose(cx + cy + rest, full, rtol=2e-5, atol=2e-6)
            class_keys = [key for key in losses if key == 'loss_mal' or key.startswith('loss_mal_')]
            assert class_keys and 'loss_mal' in class_keys
            classification = sum(losses[key] for key in class_keys)
            hard_keep = miss[gi]
            subset = [(q[hard_keep], gi[hard_keep])]
            if bool(hard_keep.any()):
                hard_terms = n.DEIMCriterion.loss_boxes(criterion, {'pred_boxes': out['pred_boxes']},
                                                      targets, subset, denom, boxes_weight=None)
                hard = sum(criterion.weight_dict[key] * value for key, value in hard_terms.items())
            else:
                hard = out['pred_boxes'].sum() * 0
            terms = {'cx': cx, 'cy': cy, 'total': full, 'classification': classification, 'hard_geometry': hard}
            g = {}
            for term_idx, (key, value) in enumerate(terms.items()):
                g[key] = vjp(value, w, retain=(number == 0 or term_idx < 4))
            gradient_reconstruction = None
            if number == 0:
                g_rest = vjp(rest, w, retain=False)
                rebuilt = g['cx'] + g['cy'] + g_rest
                relative = norm(rebuilt - g['total']) / max(norm(g['total']), 1e-30)
                assert relative < 2e-4, ('gradient reconstruction', relative)
                gradient_reconstruction = {'relative_L2_error': relative,
                                           'max_absolute_error': float((rebuilt - g['total']).abs().max())}
            row = {'image_index': number, 'event': case['event'], 'input_sha256': case['input_sha256'],
                   'GT': len(target['labels']), 'original_sub4': int(case['original_sub4'].sum()),
                   'C0_eval_sub4_without_any_geo05_GT': int(miss.sum()),
                   'hard_union_pairs': int(hard_keep.sum()),
                   'hard_unique_GT_with_union_pair': int(torch.unique(gi[hard_keep]).numel()),
                   'hard_GT_indices': torch.where(miss)[0].tolist(),
                   'eval_selected_indices': selected_indices.tolist(),
                   'native_selection_gather_exact': True,
                   'final_union_query_indices': q.tolist(), 'final_union_GT_indices': gi.tolist(),
                   'final_native_denominator': denom, 'native_weight_dict': criterion.weight_dict,
                   'loss_values': {key: float(value.detach()) for key, value in losses.items()},
                   'measured_terms': {key: float(value.detach()) for key, value in terms.items()},
                   'independent_rest_value': float(rest.detach()),
                   'loss_reconstruction_abs_error': float((full - cx - cy - rest).detach().abs()),
                   'gradient_reconstruction_first_image': gradient_reconstruction,
                   'gradient_norms': {key: norm(value) for key, value in g.items()},
                   'classification_keys': class_keys, 'DN_seed': DN_SEED,
                   'DN_num_group': int(out['dn_meta']['dn_num_group']),
                   'DN_split': [int(v) for v in out['dn_meta']['dn_num_split']],
                   'rng_before_sha256': rng_before, 'rng_after_sha256': rng_after,
                   'eval_vs_train_ordinary_max_error': {
                       key: float((pred[key] - out[key].detach()).abs().max()) for key in pred}}
            rows.append(row)
            grads.append(g)
            predictions.append({'event': case['event'], 'eval': pred,
                                'train_ordinary': {key: out[key].detach().clone() for key in pred},
                                'DN_meta': n.cpu_tree(out['dn_meta'])})
            assert all(p.grad is None for p in model.parameters())
            n.save_json(OUT / ('01_image_%d.json' % number), row)
            # Free the graph and criterion's non-state target caches before the next image.
            criterion.final_boxes = None
            criterion.final_record = None
            criterion._clear_cache()
            del out, losses, terms, cx, cy, wh, full, rest, classification, hard, abs_error, src, truth
            if 'hard_terms' in locals():
                del hard_terms
            gc.collect()
        analysis = analyze(grads, rows, ps)
        n.save_json(OUT / '02_direction_results.json', analysis)
        torch.save({'gradients_W_only': grads, 'predictions': predictions, 'rows': rows}, OUT / '03_gradients_and_predictions.pt')
        torch.save({'checkpoint_sha256': meta['checkpoint_sha256'], 'model_state_sha256': before,
                    'mode': 'eval', 'mha_fastpath_enabled': False, 'cases': eval_cache,
                    'source_input_sha256': hashes[str(input_paths[1])]}, OUT / '03_eval_predictions_and_queries.pt')
        assert COUNTS == {'model_forwards': 8, 'model_vjps': 21, 'optimizer_steps': 0}
        state_unchanged = n.model_hash(model) == before
        checkpoint_unchanged = n.sha(meta['checkpoint']) == meta['checkpoint_sha256']
        assert state_unchanged and checkpoint_unchanged
        result = {'state': 'complete', 'resources': status, 'counts': COUNTS.copy(), 'strict_load_online': strict_load,
                  'epoch': epoch, 'state_hash_before': before, 'state_hash_after': n.model_hash(model),
                  'checkpoint_unchanged': checkpoint_unchanged, 'model_state_and_BN_unchanged': state_unchanged,
                  'new_training': False, 'optimizer_called': False, 'Adam_state_used': False,
                  'heldout_reads': False, 'parameter_scope': W_NAME, 'gate': analysis['gate'],
                  'model_seconds': time.monotonic() - started,
                  'scope': 'C0 single-seed restore-W correspondence on four reused development images; not full-network or Adam/AP evidence'}
        n.save_json(OUT / '04_execution.json', result)
        n.save_json(OUT / '05_output_sha256.json', {p.name: n.sha(p) for p in OUT.iterdir() if p.is_file()})
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as exc:
        n.save_json(OUT / 'FAILED.json', {'state': 'failed', 'type': type(exc).__name__, 'message': str(exc),
                    'counts': COUNTS.copy(), 'model_state_unchanged': n.model_hash(model) == before,
                    'checkpoint_unchanged': n.sha(meta['checkpoint']) == meta['checkpoint_sha256'],
                    'new_training': False, 'elapsed_seconds': time.monotonic() - started})
        raise
    finally:
        for key, p in model.named_parameters():
            p.requires_grad_(original_requires[key])
        assert all(p.grad is None for p in model.parameters())


if __name__ == '__main__':
    main()
