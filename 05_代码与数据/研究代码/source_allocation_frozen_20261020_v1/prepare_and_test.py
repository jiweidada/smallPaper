"""Synthetic checks and read-only asset binding, before diagnostic outcomes."""
import datetime, hashlib, json
from pathlib import Path
import diagnostic as d


def tests():
    t = d.torch
    t.manual_seed(6102026)
    pair = lambda q, g: (t.tensor(q, dtype=t.long), t.tensor(g, dtype=t.long))
    union = [pair([0, 1, 2, 3, 4], [0, 0, 1, 2, 1])]
    matches = [{'pre': pair([0, 1, 2], [0, 1, 2]), 'enc_0': pair([4], [1]),
                'aux_0': pair([1], [0]), 'aux_1': pair([1, 2], [0, 1]),
                'final': pair([0, 1, 2, 3], [0, 0, 1, 2])}]
    targets = [{'labels': t.tensor([0, 0, 1]),
                'boxes': t.tensor([[.2, .2, .1, .1], [.5, .5, .2, .2], [.7, .7, .15, .15]], dtype=t.float64)}]
    selected, source, fixed, pop = d.coefficient_tables(union, matches, targets)
    assert selected.tolist() == [False, True, True, True, False]
    expected = t.tensor([[0, 0, 0], [1/3, 1/3, 1/3], [0, .5, .5], [0, 0, 1], [0, 0, 0]], dtype=t.float64)
    assert t.equal(source, expected)
    assert pop[0]['pre_unmatched'] == pop[0]['same_class_other_GT'] == pop[0]['different_class_other_GT'] == 1
    assert pop[0]['encoder_only_unchanged'] == 1
    assert t.allclose(fixed.sum(0), source.sum(0), atol=1e-14, rtol=0)
    assert not t.equal(source, fixed)
    # Empty transfer keeps NA proportions and zero budgets, never divides by zero.
    empty, es, ef, ep = d.coefficient_tables([pair([0], [0])], matches, targets)
    assert not empty.any() and es.sum() == ef.sum() == 0 and ep[0]['fixed_proportions'] is None
    recipes = json.loads((d.OLD/'baseline_recipes.json').read_text(encoding='utf-8-sig'))
    criterion = d.native.build_criterion(recipes['1'])
    boxes = [(t.rand(1, 5, 4, dtype=t.float64)*.4+.2).requires_grad_() for k in range(4)]
    output = {'pred_boxes': boxes[3], 'pre_outputs': {'pred_boxes': boxes[0]},
              'aux_outputs': [{'pred_boxes': boxes[1]}, {'pred_boxes': boxes[2]}]}
    delta, _ = d.correction(output, targets, union, selected, source, 5)
    independent = delta*0
    for j in [1, 2, 3]:
        q, g = union[0][0][j:j+1], union[0][1][j:j+1]
        for stage, coefficient in enumerate([-1., *source[j].tolist()]):
            one = d.native.DEIMCriterion.loss_boxes(criterion, {'pred_boxes': boxes[stage]}, targets, [(q, g)], 5)
            independent = independent+coefficient*(5*one['loss_bbox']+2*one['loss_giou'])
    assert abs(float(delta-independent)) < 1e-14
    gd = t.autograd.grad(delta, boxes, retain_graph=True)
    gi = t.autograd.grad(independent, boxes)
    gradient_error = max(float((a-b).abs().max()) for a, b in zip(gd, gi))
    assert gradient_error < 1e-14
    # Virtual Adam uses copied moments and preserves None-gradient parameters.
    toy = t.nn.Linear(2, 1, bias=True).double()
    named = list(toy.named_parameters())
    state = {'step': t.tensor(17.), 'exp_avg': t.full_like(toy.weight, .03),
             'exp_avg_sq': t.full_like(toy.weight, .04)}
    group = {'amsgrad': False, 'betas': [.9, .999], 'lr': .0004,
             'weight_decay': .0001, 'eps': 1e-8, 'maximize': False}
    binding = {'weight': {'group': group, 'state': state}, 'bias': {'group': group, 'state': {}}}
    grad = t.full_like(toy.weight, .4)
    before = {n: p.detach().clone() for n, p in named}
    virtual, info = d.virtual_endpoint(toy, named, binding, [grad, None])
    g = grad*info['global_clip_factor']
    m = .9*state['exp_avg']+.1*g
    v = .999*state['exp_avg_sq']+.001*g.square()
    expected_update = before['weight']*(1-.0004*.0001)-.0004*(m/(1-.9**18))/(t.sqrt(v/(1-.999**18))+1e-8)
    assert float((virtual['weight']-expected_update).abs().max()) < 1e-14
    assert t.equal(virtual['bias'], before['bias'])
    assert all(t.equal(p, before[n]) for n, p in named)
    assert state['step'].item() == 17 and t.equal(state['exp_avg'], t.full_like(toy.weight, .03))
    clone = t.nn.Parameter(t.zeros_like(grad)); clone.grad = grad.clone()
    actual_norm = t.nn.utils.clip_grad_norm_([clone], .1)
    assert abs(float(actual_norm)-info['global_gradient_norm_native_FP32']) < 1e-14
    assert t.allclose(clone.grad, g, atol=1e-14, rtol=0)
    # Pure functional call must isolate BatchNorm buffers and parameters.
    bn = t.nn.Sequential(t.nn.BatchNorm1d(2), t.nn.Linear(2, 1)).train()
    inp = t.randn(4, 2)
    original = d.native.model_hash(bn)
    initial_buffers = {n: b.clone() for n, b in bn.named_buffers()}
    t.func.functional_call(bn, ({n: p.detach().clone() for n, p in bn.named_parameters()},
                              d.native.cpu_tree(initial_buffers)), (inp,), strict=True)
    assert d.native.model_hash(bn) == original
    return {'coefficient_selection_cases_and_encoder_only': True,
            'per_pair_per_GT_and_per_image_receiver_budgets_conserved': True,
            'empty_transfer_NA_not_zero_proportions': True,
            'native_geometric_correction_loss_and_gradient_max_abs': gradient_error,
            'cloned_historical_AdamW_formula': True, 'None_gradient_skips_weight_decay': True,
            'native_global_clip_formula': True, 'functional_BN_and_parameters_isolated': True}


def main():
    limits = d.native.resource_limits()
    test_results = tests()
    old_protocol = json.loads((d.WORK/'data/09_DN初始化冻结诊断_20261020_v1/01_锁定资产与诊断口径.json').read_text(encoding='utf8'))
    manifest = json.loads((d.OLD/'shared_inputs/inputs/manifest.json').read_text(encoding='utf-8-sig'))
    batches = sorted([x for x in manifest['batches'] if x['group'] == 'development'], key=lambda x: x['batch'])
    assert [x['batch'] for x in batches] == list(range(8))
    sources = []
    for row in batches:
        p = d.OLD/'shared_inputs/inputs'/row['file']
        assert d.native.sha(p) == row['sha256']
        sources.append({'path': str(p), 'sha256': row['sha256']})
    for item in old_protocol['matching_and_native_source_identity']:
        p = d.PREVIOUS/item['relative']
        assert d.native.sha(p) == item['sha256']
        sources.append({'path': str(p), 'sha256': item['sha256']})
    for p in [d.PREVIOUS/'common.py', d.OLD/'baseline_recipes.json']:
        sources.append({'path': str(p), 'sha256': d.native.sha(p)})
    recipes = json.loads((d.OLD/'baseline_recipes.json').read_text(encoding='utf-8-sig'))
    domains = []
    for seed in [1, 2]:
        cp = old_protocol['checkpoints'][str(seed)]
        assert d.native.sha(cp['path']) == cp['sha256']
        sources.append({'path': cp['path'], 'sha256': cp['sha256']})
        raw = d.torch.load(cp['path'], map_location='cpu', weights_only=False)
        model = d.native.build_model(recipes[str(seed)])
        model.load_state_dict(raw['model'], strict=True)
        assert raw['last_epoch'] == 23 and recipes[str(seed)]['clip_max_norm'] == .1
        named, binding = d.full_optimizer_binding(model, raw, recipes[str(seed)])
        domains.append({'seed': seed, 'trainable_tensors': len(named),
                        'trainable_scalars': sum(p.numel() for n, p in named),
                        'parameter_blocks': sorted(set(d.parameter_block(n) for n, p in named)),
                        'all_historical_states_present': all(binding[n]['state'] for n, p in named)})
        del raw, model, named, binding
        d.gc.collect()
    protocol = {'created_utc': d.now(), 'user_authorization': 'Two seeds; original, source allocation and same-budget fixed proportion; decoder/shared parameters; native matching/loss; no training; no impact to other tasks.',
        'batches': batches, 'seeds': [1, 2], 'checkpoints': old_protocol['checkpoints'],
        'weight_state': 'raw online BASE24, last_epoch=23, not EMA',
        'cross_batch_pairs': [[i, i^1] for i in range(8)],
        'sampling': 'All 8 previously used development batches per seed, same 32 augmented inputs across seeds, 6629 GT appearances per seed. Exploratory, not heldout or 64 independent inputs.',
        'source_arm': 'Remove pre L1/GIoU coefficient for union pairs absent from pre own match but present in decoder own matches; equally add to all those decoder source layers. Encoder-only unchanged.',
        'fixed_proportion_control': 'Identical per-pair deletion and transfer total; each image receiver totals exactly match source arm, divided among all selected pairs by the same per-image proportions, independent of per-pair provenance.',
        'unmodified': 'native union, num_boxes_go, MAL, DN, FGL, DDF, encoder losses, original weights and all other losses',
        'update_domain': 'Every originally trainable parameter, including backbone, hybrid encoder, shared decoder attention/FFN and all prediction heads. Frozen native parameters remain fixed.',
        'optimizer': 'Independent cloned functional AdamW endpoints, historical moments/steps and native parameter groups/LR/decay; full original gradient norm and max_norm .1; no optimizer object or .step and no writeback.',
        'precision': 'Native CPU FP32 forward/loss/gradients/virtual parameters; FP64 original geometric formula on FP32 endpoint predictions. No AMP or historical CUDA-bitstream equivalence claim.',
        'matching': 'Native unmodified CPU matcher and union from each base forward, reused for all arms and endpoint loss; no rematching of endpoints. Native tie handling retained; not substituted with old GPU indices.',
        'primary_endpoint': 'On the other adjacent batch, final ordinary boxes 5L1+2GIoU using that batch base union and original num_boxes_go. Negative source-minus-control is better. Not AP or percentage points.',
        'query_selection': 'Primary endpoint freezes base native encoder top-k indices, preserving spatial query identity as upstream parameters change. Anchor values/features still recomputed. Unmodified native top-k endpoint separately measured for all arms as sensitivity; not mixed with primary.',
        'stochastic_control': 'Same stored per-batch forward_DN_seed on CPU for all arms; clone initial buffers per functional call, restore model buffers/RNG after each seed. Not historical CUDA RNG replay.',
        'aggregation': 'Primary seed summary sum(loss*original_num_boxes_go)/sum(original_num_boxes_go); also disclose each of 8 paired directions, mean batch values, classification MAL and own-match secondary geometry.',
        'decision_before_results': 'No training launched by this diagnostic. Only retain for consideration of a later short test if BOTH seeds have pooled source-minus-original and source-minus-fixed-proportion below the measured numerical repeatability floor, and native-query sensitivity does not contradict. Otherwise do not claim extra source benefit in these frozen conditions. Disclose mixed individual directions. This is an exploratory screening disposition, not significance/equivalence or proof about all training stages.',
        'numerical_floor': 'Two times largest zero-step functional geometric-loss error within that seed; absolute zero check bounds: boxes<3e-7, loss<3e-5. Effects within floor are unresolved, not positive or equivalent.',
        'resources': 'local CPU only, <=2 threads, below-normal priority, two-CPU affinity; sequential seeds/pairs; no remote or GPU/queue/monitor operations; 1200s diagnostic timeout; require >12GiB available at start and >8GiB between pairs.',
        'sources': sources, 'runner_sha256': d.native.sha(d.CODE/'diagnostic.py'),
        'prepare_sha256': d.native.sha(__file__), 'synthetic_tests': test_results,
        'read_only_preflight_parameter_domains': domains, 'resources_at_preflight': limits}
    d.OUT.mkdir(exist_ok=False)
    d.native.save_json(d.OUT/'01_冻结协议与资产.json', protocol)
    print(json.dumps({'protocol_sha256': d.native.sha(d.OUT/'01_冻结协议与资产.json'),
                      'synthetic_tests': test_results, 'domains': domains,
                      'native_source_identity_files': len(old_protocol['matching_and_native_source_identity']),
                      'real_diagnostic_forward_started': False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
