"""Cache-only coupled marginal FGL audit; root reviews before execution.

No model/checkpoint/optimizer construction, no model forward or model VJP.
Autograd acts only on cached logits and a zero 132-D symbolic bias increment.
"""
import os
import sys
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
import gc
import json
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parent
WORK = ROOT.parents[1]
COMMON = WORK / '英文小论文/code/dn_pre_frozen_diagnostic_20261020_v1'
sys.path.insert(0, str(COMMON))
import common as n
import yaml
from engine.deim.dfine_utils import weighting_function, bbox2distance, distance2bbox
from engine.deim.box_ops import generalized_box_iou

torch = n.torch
F = torch.nn.functional
PROTOCOL = ROOT / '148_配对边际FGL共享bias_事前协议.txt'
OUT = ROOT / '150_C0_coupled_marginal_bias'
CACHE = ROOT / '127_C0_attention_template_registration/01_cache.pt'
BIAS_NAME = 'decoder.dec_bbox_head.2.layers.2.bias'
ARMS = ('O', 'M', 'C')
GROUPS = ('all_ordinary', 'all_sub4', 'hard_sub4', 'other_ordinary')
COUNTS = {'model_forwards': 0, 'model_vjps': 0, 'checkpoint_loads': 0,
          'model_instances': 0, 'optimizer_instances': 0, 'optimizer_steps': 0,
          'cache_loads': 0, 'cache_autograd_calls': 0, 'cache_autograd_targets': 0,
          'synthetic_autograd_calls': 0, 'synthetic_autograd_targets': 0}


def memory_guard():
    assert n.psutil.virtual_memory().available / 2**30 > 8, 'RAM guard'


def resources():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    proc = n.psutil.Process()
    proc.nice(n.psutil.BELOW_NORMAL_PRIORITY_CLASS)
    assert {13, 15}.issubset(proc.cpu_affinity())
    proc.cpu_affinity([13, 15])
    assert sorted(proc.cpu_affinity()) == [13, 15]
    assert torch.version.cuda is None and not torch.cuda.is_available()
    memory_guard()
    return {'threads': 2, 'interop_threads': 1, 'priority': 'below_normal',
            'affinity': proc.cpu_affinity(), 'torch': torch.__version__,
            'available_RAM_GiB': n.psutil.virtual_memory().available / 2**30}


def grad(loss, inputs, domain, retain=False):
    memory_guard()
    assert domain in ('cache', 'synthetic')
    sequence = isinstance(inputs, (tuple, list))
    requested = tuple(inputs) if sequence else (inputs,)
    key = domain + '_autograd_calls'
    assert COUNTS[key] < (32 if domain == 'cache' else 16)
    COUNTS[key] += 1
    COUNTS[domain + '_autograd_targets'] += len(requested)
    values = torch.autograd.grad(loss, requested, retain_graph=retain, create_graph=False,
                                 allow_unused=False)
    assert all(bool(torch.isfinite(value).all()) for value in values)
    values = tuple(value.detach().double().clone() for value in values)
    return values if sequence else values[0]


def norm(value):
    return float(torch.linalg.vector_norm(value))


def dot(a, b):
    return float((a * b).sum())


def tolerance(a, b=0.):
    return 1e-6 * max(abs(a), abs(b), 1e-12)


def strictly_greater(a, b=0.):
    return a > b + tolerance(a, b)


def not_below(a, b):
    return a >= b - tolerance(a, b)


def projection(nodes, locations):
    """Barycentric map to the original nodes; no support expansion."""
    nodes, locations = nodes.double(), locations.double()
    assert bool((nodes[1:] > nodes[:-1]).all())
    assert float(locations.min()) >= float(nodes[0]) - 1e-12
    assert float(locations.max()) <= float(nodes[-1]) + 1e-12
    left = (torch.searchsorted(nodes, locations.contiguous(), right=True) - 1).clamp(0, len(nodes) - 2)
    right = left + 1
    wr = (locations - nodes[left]) / (nodes[right] - nodes[left])
    assert bool((wr >= -1e-12).all()) and bool((wr <= 1 + 1e-12).all())
    matrix = torch.zeros((*locations.shape, len(nodes)), dtype=torch.float64)
    matrix.scatter_add_(-1, left.unsqueeze(-1), (1 - wr).unsqueeze(-1))
    matrix.scatter_add_(-1, right.unsqueeze(-1), wr.unsqueeze(-1))
    assert torch.allclose(matrix.sum(-1), torch.ones_like(locations), atol=1e-12, rtol=0)
    assert torch.allclose(matrix @ nodes, locations, atol=1e-12, rtol=0)
    return matrix


def transformations(nodes):
    left, right = nodes[:, None], nodes[None, :]
    t = (right - left) / 2
    s = (right + left) / 2
    return projection(nodes, t), projection(nodes, s)


def push_pair(left, right, transform):
    assert left.shape == right.shape and left.ndim == 2
    bins = left.shape[-1]
    joint = (left.unsqueeze(-1) * right.unsqueeze(-2)).reshape(-1, bins * bins)
    return joint @ transform.reshape(bins * bins, bins)


def push_four(probability, bt, bs):
    # Native edge order l,t,r,b. New auxiliary order tx,sx,ty,sy.
    return torch.stack((push_pair(probability[:, 0], probability[:, 2], bt),
                        push_pair(probability[:, 0], probability[:, 2], bs),
                        push_pair(probability[:, 1], probability[:, 3], bt),
                        push_pair(probability[:, 1], probability[:, 3], bs)), dim=1)


def safe_cross_entropy(target, probability):
    # Fail closed instead of introducing result-dependent numerical smoothing.
    assert bool(torch.isfinite(probability).all()) and bool((probability > 0).all())
    return -(target * probability.log()).sum(-1)


def synthetic_tests():
    nodes = torch.tensor([-2., -.5, 0., .5, 2.], dtype=torch.float64)
    bt, bs = transformations(nodes)
    left = torch.softmax(torch.linspace(-1., 1., len(nodes), dtype=torch.float64), -1)[None]
    right = torch.softmax(torch.tensor([[.7, -.4, .2, 1.1, -.1]], dtype=torch.float64), -1)
    pt, ps = push_pair(left, right, bt), push_pair(left, right, bs)
    assert torch.allclose(pt.sum(-1), torch.ones(1, dtype=torch.float64), atol=1e-12, rtol=0)
    assert torch.allclose(ps.sum(-1), torch.ones(1, dtype=torch.float64), atol=1e-12, rtol=0)
    assert torch.allclose(pt @ nodes, (right @ nodes - left @ nodes) / 2, atol=1e-12, rtol=0)
    assert torch.allclose(ps @ nodes, (right @ nodes + left @ nodes) / 2, atol=1e-12, rtol=0)
    # Fixed-target counterexample: the right PMF changes M's left gradient.
    v = torch.tensor([-1., 0., 1.], dtype=torch.float64)
    tmap, smap = transformations(v)
    yl = torch.tensor([[1., 0., 0.]], dtype=torch.float64)
    yr = torch.tensor([[0., 0., 1.]], dtype=torch.float64)
    yt, ys = push_pair(yl, yr, tmap), push_pair(yl, yr, smap)

    def losses(z):
        p = z.softmax(-1)
        lm = safe_cross_entropy(yt, push_pair(p[0:1], p[1:2], tmap)).sum()
        lm = lm + safe_cross_entropy(ys, push_pair(p[0:1], p[1:2], smap)).sum()
        lo = -(yl[0] * z[0].log_softmax(-1) + yr[0] * z[1].log_softmax(-1)).sum()
        return lo, lm

    x = torch.tensor([[-.7, .3, 1.1], [.2, -.4, .9]], dtype=torch.float64, requires_grad=True)
    lo, lm = losses(x)
    go = grad(lo, x, 'synthetic', retain=True)
    gm = grad(lm, x, 'synthetic')
    alpha = dot(gm, go) / dot(go, go)
    nonproportional = norm(gm - alpha * go) / norm(gm)
    assert nonproportional > 1e-4
    assert torch.allclose(gm.sum(-1), torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0)
    finite_difference = {}
    for side in range(2):
        direction = torch.zeros_like(x)
        direction[side] = torch.tensor([1., -.3, -.7], dtype=torch.float64)
        eps = 1e-6
        observed = float((losses(x.detach() + eps * direction)[1] - losses(x.detach() - eps * direction)[1]) / (2 * eps))
        expected = dot(gm, direction)
        assert abs(observed - expected) < 1e-6 * max(abs(observed), abs(expected), 1.)
        finite_difference[str(side)] = {'central_difference': observed, 'chain_gradient': expected}
    other = x.detach().clone()
    other[1] += torch.tensor([.8, -.1, -.5], dtype=torch.float64)
    other.requires_grad_(True)
    oo, om = losses(other)
    g_other_o = grad(oo, other, 'synthetic', retain=True)
    g_other_m = grad(om, other, 'synthetic')
    assert torch.allclose(go[0], g_other_o[0], atol=1e-12, rtol=0)
    assert not torch.allclose(gm[0], g_other_m[0], atol=1e-6, rtol=0)
    # One shared bias has the sum of per-query gradients, not their mean.
    bias = torch.zeros_like(x, requires_grad=True)
    logits = torch.stack([x.detach(), other.detach(), x.detach() * .7]) + bias[None]
    shared_loss = sum(losses(logits[i])[1] for i in range(3))
    gb, gz = grad(shared_loss, (bias, logits), 'synthetic')
    assert torch.allclose(gb, gz.sum(0), atol=1e-12, rtol=0)
    return {'pushforward_probability_mass': True, 'pushforward_t_s_mean': True,
            'target_probability_mass': bool(torch.allclose(yt.sum(-1), yl.sum(-1)) and torch.allclose(ys.sum(-1), yl.sum(-1))),
            'left_and_right_chain_finite_differences': finite_difference,
            'constant_logit_shift_gradient_sum_zero': True,
            'changing_right_changes_M_left_but_not_O_left': True,
            'M_not_scalar_rescaling_of_O_relative_residual': nonproportional,
            'shared_bias_gradient_equals_sum_query_gradients': True}


def source_scope_audit(meta, prior_hashes):
    # No model instantiation. Hashes lock the specific statically reviewed graph.
    fixed = {'native/engine/deim/dfine_decoder.py': 'b31408e49d4961e1e82692703b8f9fc3f07f1e280fd13ee12f896cb337f9bae5',
             'native/engine/deim/deim_criterion.py': '16cdd80b2197b8def8ee3f5f19f6311f5221c3da08abacb99f50d346610a2c30',
             'native/engine/deim/dfine_utils.py': 'ad87ec5a50b1970e76b06f7242298ffa332beaf78c828ac3e8d41ac656ec5f6c'}
    for relative, expected in fixed.items():
        path = COMMON / relative
        assert n.sha(path) == prior_hashes[str(path)] == expected, str(path)
    helper = ROOT / '113_C0_restore_route_frozen.py'
    assert n.sha(helper) == prior_hashes[str(helper)] == '14c3b180ee394cd91840d5369cc44715ad3918d2a38fd3a495913a67f63dd2d3'
    assert n.sha(meta['recipe']) == meta['recipe_sha256'] == prior_hashes[meta['recipe']]
    recipe = yaml.safe_load(pathlib.Path(meta['recipe']).read_text(encoding='utf-8-sig'))
    config = recipe['DFINETransformer']
    assert config['num_layers'] == 3 and config['eval_idx'] == -1 and config['reg_max'] == 32
    assert config['reg_scale'] == 4 and config['layer_scale'] == 1
    weights = recipe['DEIMCriterion']['weight_dict']
    assert weights['loss_bbox'] == 5 and weights['loss_giou'] == 2 and weights['loss_fgl'] == .15
    return {'parameter': BIAS_NAME, 'dimensions': 132, 'decoder_layers': 3, 'eval_layer': 2,
            'verified_by': 'statically reviewed source construction, forward and strict-copy load, locked by hashes',
            'runtime_storage_alias_audit_performed': False,
            'each_decoder_layer_constructs_fresh_MLP': True, 'each_MLP_constructs_fresh_Linear': True,
            'final_MLP_layer_is_linear_without_activation': True,
            'final_corners_add_earlier_residual_independent_of_this_bias': True,
            'ordinary_queries_share_bias': True,
            'LQE_also_depends_on_corners_classification_not_measured': True}


def make_target_distribution(labels, wr, wl, pairs, bins):
    left = labels.long()
    assert left.shape == (pairs * 4,)
    assert int(left.min()) >= 0 and int(left.max()) < bins - 1
    y = torch.zeros((pairs * 4, bins), dtype=torch.float64)
    y.scatter_add_(1, left[:, None], wl.double().reshape(-1, 1))
    y.scatter_add_(1, (left + 1)[:, None], wr.double().reshape(-1, 1))
    assert torch.allclose(y.sum(-1), torch.ones(pairs * 4, dtype=torch.float64), atol=1e-7, rtol=0)
    return y.reshape(pairs, 4, bins)


def prepare(record, prior, nodes, up, reg_scale, bt, bs):
    assert record['image_index'] == prior['image_index']
    assert record['event'] == prior['event'] and record['input_sha256'] == prior['input_sha256']
    assert torch.equal(record['selected_indices'], torch.tensor(prior['eval_selected_indices'], dtype=torch.long))
    target = record['target']
    assert len(target['labels']) == prior['GT'] and int(record['original_sub4'].sum()) == prior['original_sub4']
    q = torch.tensor(prior['final_union_query_indices'], dtype=torch.long)
    gi = torch.tensor(prior['final_union_GT_indices'], dtype=torch.long)
    denom = float(prior['final_native_denominator'])
    assert len(q) == len(gi) == int(denom) and denom > 0
    assert q.unique().numel() == len(q) and int(q.min()) >= 0 and int(q.max()) < 300
    assert int(gi.min()) >= 0 and int(gi.max()) < len(target['labels'])
    assert record['final_corners'].shape == (300, 132)
    assert record['ref_points_initial'].shape == (300, 4)
    assert record['final_corners'].dtype == record['ref_points_initial'].dtype == torch.float32
    assert target['boxes'].dtype == record['pred_boxes'].dtype == torch.float32
    hard_mask = torch.zeros(len(target['labels']), dtype=torch.bool)
    hard_mask[torch.tensor(prior['hard_GT_indices'], dtype=torch.long)] = True
    all_iou = n.box_iou(n.box_cxcywh_to_xyxy(record['pred_boxes']), n.box_cxcywh_to_xyxy(target['boxes']))[0]
    rechecked_hard = record['original_sub4'].bool() & ~(all_iou >= .5).any(0)
    assert torch.equal(hard_mask, rechecked_hard)
    hard_keep = hard_mask[gi]
    assert int(hard_keep.sum()) == prior['hard_union_pairs']
    assert int(gi[hard_keep].unique().numel()) == prior['hard_unique_GT_with_union_pair']
    alpha = all_iou[q, gi].detach().float()
    assert bool(torch.isfinite(alpha).all()) and bool((alpha >= 0).all()) and bool((alpha <= 1 + 1e-6).all())
    ref32, truth32 = record['ref_points_initial'][q], target['boxes'][gi]
    assert bool((ref32[:, 2:] > 0).all()) and bool((truth32[:, 2:] > 0).all())
    # Native target routine requires FP32 boxes and tensor[1] parameters.
    labels, wr, wl = bbox2distance(ref32, n.box_cxcywh_to_xyxy(truth32), 32, reg_scale, up)
    y = make_target_distribution(labels, wr, wl, len(q), len(nodes))
    ym = push_four(y, bt, bs).detach()
    assert torch.allclose(ym.sum(-1), torch.ones_like(ym[..., 0]), atol=2e-7, rtol=0)
    ref, truth = ref32.double(), truth32.double()
    R = float(reg_scale.abs().item())
    continuous_target = torch.stack((R * (truth[:, 0] - ref[:, 0]) / ref[:, 2],
                                      R / 2 * (truth[:, 2] / ref[:, 2] - 1),
                                      R * (truth[:, 1] - ref[:, 1]) / ref[:, 3],
                                      R / 2 * (truth[:, 3] / ref[:, 3] - 1)), dim=-1)
    all_decoded = distance2bbox(record['ref_points_initial'].double(),
                                (record['final_corners'].double().reshape(300, 4, 33).softmax(-1) * nodes).sum(-1),
                                reg_scale.double())
    error = float((all_decoded - record['pred_boxes'].double()).abs().max())
    assert error <= 1e-6, ('decode parity', error)
    groups = {'all_ordinary': torch.ones(len(q), dtype=torch.bool),
              'all_sub4': record['original_sub4'].bool()[gi], 'hard_sub4': hard_keep,
              'other_ordinary': ~hard_keep}
    group_info = {name: {'pairs': int(mask.sum()), 'unique_GT': int(gi[mask].unique().numel()),
                         'positive_IoU_weight_pairs': int(((alpha > 0) & mask).sum()),
                         'IoU_weight_sum': float(alpha[mask].double().sum())}
                  for name, mask in groups.items()}
    row = {'image_index': record['image_index'], 'event': record['event'], 'input_sha256': record['input_sha256'],
           'GT': prior['GT'], 'original_sub4': prior['original_sub4'], 'original_union_denominator': denom,
           'original_query_indices': q.tolist(), 'original_GT_indices': gi.tolist(),
           'fixed_hard_GT_indices': prior['hard_GT_indices'], 'group_info': group_info,
           'FP64_decode_vs_127_FP32_max_absolute_error': error,
           'historical_114_train_eval_difference': prior['eval_vs_train_ordinary_max_error'],
           'native_FGL_target_probability_mass_max_error': float((y.sum(-1) - 1).abs().max()),
           'derived_target_probability_mass_max_error': float((ym.sum(-1) - 1).abs().max())}
    return {'logits': record['final_corners'].double(), 'q': q, 'gi': gi, 'denom': denom,
            'ref': ref, 'truth': truth, 'alpha': alpha, 'y': y, 'ym': ym,
            'continuous_target': continuous_target, 'labels': labels, 'wr': wr, 'wl': wl,
            'groups': groups, 'row': row}


def losses_at_bias(case, bias, nodes, bt, bs, reg_scale):
    z = (case['logits'] + bias[None])[case['q']].reshape(-1, 4, 33)
    p = z.softmax(-1)
    mean = (p * nodes).sum(-1)
    derived = push_four(p, bt, bs)
    continuous = torch.stack(((mean[:, 2] - mean[:, 0]) / 2, (mean[:, 2] + mean[:, 0]) / 2,
                               (mean[:, 3] - mean[:, 1]) / 2, (mean[:, 3] + mean[:, 1]) / 2), dim=-1)
    per_aux = {'O': -(case['y'] * z.log_softmax(-1)).sum((-1, -2)),
               'M': safe_cross_entropy(case['ym'], derived).sum(-1),
               'C': (continuous - case['continuous_target']).abs().sum(-1)}
    aux = {arm: .15 * (value * case['alpha'].double()).sum() / case['denom'] for arm, value in per_aux.items()}
    boxes = distance2bbox(case['ref'], mean, reg_scale.double())
    l1 = (boxes - case['truth']).abs().sum(-1)
    giou_loss = 1 - torch.diag(generalized_box_iou(n.box_cxcywh_to_xyxy(boxes), n.box_cxcywh_to_xyxy(case['truth'])))
    per_geometry = 5 * l1 + 2 * giou_loss
    evaluation = {name: per_geometry[mask].sum() / case['denom'] for name, mask in case['groups'].items()}
    assert all(bool(torch.isfinite(value).all()) for value in (*aux.values(), *evaluation.values()))
    values = {name: {'G_5L1_plus_2GIoU': float(evaluation[name].detach()),
                     'weighted_5L1': float((5 * l1[mask].sum() / case['denom']).detach()),
                     'weighted_2GIoU': float((2 * giou_loss[mask].sum() / case['denom']).detach()),
                     'auxiliary_values': {arm: float((.15 * (value[mask] * case['alpha'][mask].double()).sum() / case['denom']).detach())
                                          for arm, value in per_aux.items()}}
              for name, mask in case['groups'].items()}
    return z, aux, evaluation, values


def analyze(gradients, rows, active_images):
    folds = []
    for heldout in range(4):
        train = [index for index in range(4) if index != heldout]
        directions = {arm: sum(gradients[index]['auxiliary'][arm] for index in train) / 3 for arm in ARMS}
        norms = {arm: norm(value) for arm, value in directions.items()}
        complete = all(value > 0 for value in norms.values())
        factors = {arm: (norms['O'] / norms[arm] if complete else None) for arm in ARMS}
        scores = {}
        for group in GROUPS:
            g = gradients[heldout]['evaluation'][group]
            scores[group] = {'raw': {arm: dot(g, directions[arm]) for arm in ARMS},
                             'normmatched': {arm: (dot(g, directions[arm]) * factors[arm] if complete else None) for arm in ARMS}}
        folds.append({'heldout_development_image': heldout, 'training_image_indices': train,
                      'direction_norms': norms, 'normmatch_factors': factors,
                      'directions_nonzero': complete, 'gain': scores})
    all_nonzero = all(fold['directions_nonzero'] for fold in folds)
    means = {group: {style: {arm: (sum(fold['gain'][group][style][arm] for fold in folds) / 4
                                        if style == 'raw' or all_nonzero else None) for arm in ARMS}
                     for style in ('raw', 'normmatched')} for group in GROUPS}
    checks = {'positive_weight_main_group_at_least_3_images': active_images >= 3,
              'all_folds_O_M_C_directions_nonzero': all_nonzero}
    wins = 0
    if all_nonzero:
        main, full = means['hard_sub4']['normmatched'], means['all_ordinary']['normmatched']
        wins = sum(strictly_greater(fold['gain']['hard_sub4']['normmatched']['M'],
                                   fold['gain']['hard_sub4']['normmatched']['O']) for fold in folds)
        checks.update(M_main_mean_positive=strictly_greater(main['M']),
                      M_main_mean_strictly_over_O=strictly_greater(main['M'], main['O']),
                      M_main_mean_strictly_over_continuous=strictly_greater(main['M'], main['C']),
                      M_main_beats_O_at_least_3_images=wins >= 3,
                      M_all_ordinary_mean_not_below_O=not_below(full['M'], full['O']))
    else:
        checks.update({key: False for key in ('M_main_mean_positive', 'M_main_mean_strictly_over_O',
                       'M_main_mean_strictly_over_continuous', 'M_main_beats_O_at_least_3_images',
                       'M_all_ordinary_mean_not_below_O')})
    return {'folds': folds, 'mean_gain': means,
            'gate': {'passed': all(checks.values()), 'checks': checks, 'M_main_wins_over_O': wins,
                     'eligible_candidate': 'M_only', 'three_of_four_is_resource_rule_not_significance': True,
                     'meaning': 'continue this exact supervision-variable audit only; not training authorization or novelty'},
            'positive_weight_main_images': active_images,
            'scope': 'final-head shared bias partial on cached eval readouts with fixed recorded train union pairs; geometry only'}


def main():
    started = time.monotonic()
    status = resources()
    OUT.mkdir(exist_ok=False)
    hashes = {}
    try:
        paths = [CACHE, ROOT / '104_C0_本地资产严格加载核验.json',
                 ROOT / '114_C0_restore_route_frozen/00_inputs_and_protocol_sha256.json',
                 ROOT / '114_C0_restore_route_frozen/04_execution.json',
                 ROOT / '127_C0_attention_template_registration/04_execution.json',
                 ROOT / '127_C0_attention_template_registration/05_output_sha256.json',
                 ROOT / '113_C0_restore_route_frozen.py', PROTOCOL, pathlib.Path(__file__)]
        paths += [ROOT / f'114_C0_restore_route_frozen/01_image_{index}.json' for index in range(4)]
        meta = json.loads(paths[1].read_text(encoding='utf-8'))
        paths.append(pathlib.Path(meta['recipe']))
        paths += [COMMON / relative for relative in ('common.py', 'native/engine/deim/dfine_decoder.py',
                  'native/engine/deim/deim_criterion.py', 'native/engine/deim/dfine_utils.py', 'native/engine/deim/box_ops.py')]
        hashes = {str(path): n.sha(path) for path in paths}
        prior_hashes = json.loads(paths[2].read_text(encoding='utf-8'))
        scope = source_scope_audit(meta, prior_hashes)
        n.save_json(OUT / '00_inputs_and_protocol_sha256.json', hashes)
        n.save_json(OUT / '00_parameter_scope.json', scope)
        tests = synthetic_tests()
        n.save_json(OUT / '00_synthetic_tests.json', tests)
        prior_execution = json.loads(paths[3].read_text(encoding='utf-8'))
        cache_execution = json.loads(paths[4].read_text(encoding='utf-8'))
        cache_manifest = json.loads(paths[5].read_text(encoding='utf-8'))
        assert prior_execution['state'] == cache_execution['state'] == 'complete'
        assert cache_execution['checkpoint_unchanged'] and cache_execution['model_state_and_BN_unchanged']
        assert hashes[str(CACHE)] == cache_manifest['01_cache.pt']
        memory_guard()
        # This is the sole torch.load call. It reads predictions, never checkpoint weights.
        assert CACHE.name == '01_cache.pt' and CACHE.parent.name == '127_C0_attention_template_registration'
        data = torch.load(CACHE, map_location='cpu', weights_only=False)
        COUNTS['cache_loads'] += 1
        assert data['checkpoint_sha256'] == meta['checkpoint_sha256']
        assert data['weight_key'] == 'model_online' and data['mode'] == 'eval'
        assert data['model_state_sha256'] == cache_execution['state_hash_before'] == prior_execution['state_hash_before']
        assert len(data['records']) == 4 and data['reg_max'] == 32
        up = torch.tensor([data['up']], dtype=torch.float32)
        reg_scale = torch.tensor([data['reg_scale']], dtype=torch.float32)
        assert up.item() == .5 and reg_scale.item() == 4
        nodes = weighting_function(32, up, reg_scale).double()
        assert nodes.shape == (33,) and nodes[0] == -4 and nodes[-1] == 4
        bt, bs = transformations(nodes)
        prepared = []
        for index, record in enumerate(data['records']):
            assert record['image_index'] == index
            prior = json.loads((ROOT / f'114_C0_restore_route_frozen/01_image_{index}.json').read_text(encoding='utf-8'))
            assert prior['final_native_denominator'] == [172., 290., 300., 70.][index]
            assert prior['hard_union_pairs'] == [41, 105, 124, 13][index]
            assert len(prior['hard_GT_indices']) == [36, 104, 237, 9][index]
            assert prior['native_weight_dict']['loss_fgl'] == .15
            prepared.append(prepare(record, prior, nodes, up, reg_scale, bt, bs))
        assert sum(case['row']['GT'] for case in prepared) == 902
        assert sum(case['row']['original_sub4'] for case in prepared) == 453
        active_images = sum(case['row']['group_info']['hard_sub4']['positive_IoU_weight_pairs'] > 0 for case in prepared)
        gradients, rows = [], []
        for index, case in enumerate(prepared):
            memory_guard()
            bias = torch.zeros(132, dtype=torch.float64, requires_grad=active_images >= 3)
            z, aux, evaluation, values = losses_at_bias(case, bias, nodes, bt, bs, reg_scale)
            row = {**case['row'], 'group_original_values': values,
                   'all_ordinary_auxiliary_values': {arm: float(value.detach()) for arm, value in aux.items()}}
            if active_images >= 3:
                ga = {arm: grad(value, bias, 'cache', retain=True) for arm, value in aux.items()}
                if index == 0:
                    native = .15 * n.DEIMCriterion.unimodal_distribution_focal_loss(
                        None, z.reshape(-1, 33), case['labels'], case['wr'], case['wl'],
                        weight=case['alpha'].repeat_interleave(4), avg_factor=case['denom'])
                    native_g = grad(native, bias, 'cache', retain=True)
                    native_value_error = float((native - aux['O']).detach().abs())
                    native_grad_error = norm(native_g - ga['O']) / max(norm(ga['O']), 1e-30)
                    assert native_value_error <= 1e-10 * max(1., abs(float(native.detach())))
                    assert native_grad_error <= 1e-10
                    row['native_O_same_point_value_gradient_check'] = {'value_abs_error': native_value_error,
                                                                      'gradient_relative_L2_error': native_grad_error}
                ge = {name: grad(value, bias, 'cache', retain=number < len(evaluation) - 1)
                      for number, (name, value) in enumerate(evaluation.items())}
                gradients.append({'auxiliary': ga, 'evaluation': ge})
                row['auxiliary_bias_gradient_norms'] = {arm: norm(value) for arm, value in ga.items()}
                row['geometry_bias_gradient_norms'] = {name: norm(value) for name, value in ge.items()}
                assert bias.grad is None
            rows.append(row)
            n.save_json(OUT / f'01_image_{index}.json', row)
            del bias, z, aux, evaluation
            gc.collect()
        if active_images >= 3:
            assert COUNTS['cache_autograd_calls'] == 29
            analysis = analyze(gradients, rows, active_images)
        else:
            assert COUNTS['cache_autograd_calls'] == 0
            analysis = {'folds': [], 'mean_gain': None, 'positive_weight_main_images': active_images,
                        'gate': {'passed': False, 'checks': {'positive_weight_main_group_at_least_3_images': False},
                                 'eligible_candidate': 'M_only', 'meaning': 'insufficient active main-group images; exact version unsupported'}}
        n.save_json(OUT / '02_LOO_results.json', analysis)
        torch.save({'parameter_scope': BIAS_NAME, 'rows': rows, 'gradients': gradients,
                    'native_nodes_from_FP32': nodes, 'checkpoint_sha256_metadata_only': data['checkpoint_sha256'],
                    'model_state_sha256_metadata_only': data['model_state_sha256']}, OUT / '03_cached_bias_gradients.pt')
        unchanged = {path: n.sha(path) == expected for path, expected in hashes.items()}
        assert all(unchanged.values())
        assert all(COUNTS[key] == 0 for key in ('model_forwards', 'model_vjps', 'checkpoint_loads',
                   'model_instances', 'optimizer_instances', 'optimizer_steps'))
        result = {'state': 'complete', 'resources': status, 'counts': COUNTS.copy(),
                  'gate': analysis['gate'], 'positive_weight_main_images': active_images,
                  'input_and_source_hashes_unchanged': unchanged,
                  'parameter_scope': BIAS_NAME, 'new_training': False,
                  'classifier_or_LQE_gradient_measured': False, 'complete_native_criterion_reproduced': False,
                  'learned_edge_covariance_or_full_TS_joint': False,
                  'model_state_hash_recomputed': False, 'checkpoint_loaded_or_hashed_by_this_script': False,
                  'seconds': time.monotonic() - started}
        n.save_json(OUT / '04_execution.json', result)
        n.save_json(OUT / '05_output_sha256.json', {path.name: n.sha(path) for path in OUT.iterdir() if path.is_file()})
        print(json.dumps({'state': 'complete', 'counts': COUNTS, 'gate': analysis['gate'],
                          'mean_gain': analysis['mean_gain'], 'seconds': result['seconds']}, ensure_ascii=False), flush=True)
    except Exception as exc:
        n.save_json(OUT / 'FAILED.json', {'state': 'failed', 'type': type(exc).__name__, 'message': str(exc),
                    'counts': COUNTS.copy(), 'input_and_source_hashes_unchanged': {path: n.sha(path) == expected for path, expected in hashes.items()},
                    'new_training': False, 'seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    main()
