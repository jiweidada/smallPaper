"""Isolated full-parameter frozen diagnostic, no optimizer or checkpoint writes.

Imports the SHA-verified previous diagnostic's native code/runtime unchanged.
All three virtual AdamW endpoints start from the same original model and states.
"""
import argparse, copy, datetime, gc, hashlib, json, math, os
from pathlib import Path
import random, re, sys, time, traceback

CODE = Path(__file__).resolve().parent
WORK = CODE.parents[1]
PREVIOUS = WORK/'code/dn_pre_frozen_diagnostic_20261020_v1'
sys.path.insert(0, str(PREVIOUS))
import common as native
torch, np, psutil = native.torch, native.np, native.psutil
from engine.deim.box_ops import generalized_box_iou, box_cxcywh_to_xyxy, box_iou
from torch.optim import _functional as optim_functional

OLD = WORK/'data/08_匹配聚合覆盖探针_20261019_v1'
OUT = WORK/'data/10_来源分配冻结诊断_20261020_v1'
ARMS = ['original', 'source', 'fixed_proportion']
STAGES = ['aux_0', 'aux_1', 'final']


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def progress(**kw):
    row = {'utc': now(), 'pid': os.getpid(),
           'rss_GiB': psutil.Process().memory_info().rss/2**30, **kw}
    print(json.dumps(row, ensure_ascii=False), flush=True)
    with (OUT/'progress.jsonl').open('a', encoding='utf8') as f:
        f.write(json.dumps(row, ensure_ascii=False)+'\n')


def indices_json(indices):
    return [[q.tolist(), g.tolist()] for q, g in indices]


def geometric_terms(boxes, targets, indices):
    """Unreduced terms, exactly the native default GIoU path, no proxy metric."""
    src = torch.cat([boxes[b, q] for b, (q, g) in enumerate(indices)])
    tgt = torch.cat([t['boxes'][g].to(boxes.dtype) for t, (q, g) in zip(targets, indices)])
    l1 = (src-tgt).abs().sum(-1)
    giou = 1-generalized_box_iou(box_cxcywh_to_xyxy(src), box_cxcywh_to_xyxy(tgt)).diagonal()
    return l1, giou, src, tgt


def stage_outputs(outputs):
    assert len(outputs['aux_outputs']) == 2
    return [outputs['pre_outputs'], outputs['aux_outputs'][0],
            outputs['aux_outputs'][1], outputs]


def coefficient_tables(union, matches, targets):
    """Build source and fixed-control coefficients; exact double budget audit."""
    per_image, selected_all, source_all, fixed_all = [], [], [], []
    for b, (uq, ug) in enumerate(union):
        own = {stage: dict(zip(pair[0].tolist(), pair[1].tolist()))
               for stage, pair in matches[b].items()}
        n = len(uq)
        supports = torch.zeros(n, 3, dtype=torch.float64)
        selected = torch.zeros(n, dtype=torch.bool)
        counts = {'pre_unmatched': 0, 'same_class_other_GT': 0,
                  'different_class_other_GT': 0, 'encoder_only_unchanged': 0}
        set_sizes = {str(k): 0 for k in [1, 2, 3]}
        for j, (q, g) in enumerate(zip(uq.tolist(), ug.tolist())):
            if own['pre'].get(q) == g:
                continue
            d = [k for k, name in enumerate(STAGES) if own[name].get(q) == g]
            if not d:
                assert own['enc_0'].get(q) == g
                counts['encoder_only_unchanged'] += 1
                continue
            selected[j] = True
            supports[j, d] = 1/len(d)
            set_sizes[str(len(d))] += 1
            if q not in own['pre']:
                counts['pre_unmatched'] += 1
            elif targets[b]['labels'][own['pre'][q]] == targets[b]['labels'][g]:
                counts['same_class_other_GT'] += 1
            else:
                counts['different_class_other_GT'] += 1
        m = int(selected.sum())
        budget = supports.sum(0)
        fixed = selected.double().unsqueeze(1) * (budget/m if m else budget)
        assert torch.allclose(supports.sum(1), selected.double(), atol=1e-12, rtol=0)
        assert torch.allclose(fixed.sum(1), selected.double(), atol=1e-12, rtol=0)
        assert torch.allclose(fixed.sum(0), budget, atol=1e-12, rtol=0)
        # Per-GT and per-pair effective L1/GIoU coefficient conservation.
        for g in ug.unique():
            ix = ug == g
            assert abs(float(supports[ix].sum()-selected[ix].sum())) < 1e-10
            assert abs(float(fixed[ix].sum()-selected[ix].sum())) < 1e-10
        per_image.append({'image': b, 'GT': len(targets[b]['labels']),
                          'union_pairs': n, 'transfer_pairs': m, **counts,
                          'source_set_sizes': set_sizes,
                          'receiver_budget_before_original_denominator': budget.tolist(),
                          'fixed_proportions': (budget/m).tolist() if m else None,
                          'source_fixed_coefficient_L1_distance': float((supports-fixed).abs().sum())})
        selected_all.append(selected)
        source_all.append(supports)
        fixed_all.append(fixed)
    return torch.cat(selected_all), torch.cat(source_all), torch.cat(fixed_all), per_image


def correction(outputs, targets, union, selected, receivers, denominator):
    terms = [geometric_terms(o['pred_boxes'], targets, union)[:2] for o in stage_outputs(outputs)]
    w = selected.to(terms[0][0].dtype)
    l1 = -(terms[0][0]*w).sum()
    giou = -(terms[0][1]*w).sum()
    for k in range(3):
        c = receivers[:, k].to(terms[k+1][0].dtype)
        l1 = l1+(terms[k+1][0]*c).sum()
        giou = giou+(terms[k+1][1]*c).sum()
    return (5*l1+2*giou)/denominator, {'delta_weighted_L1': float(5*l1.detach()/denominator),
                                     'delta_weighted_GIoU': float(2*giou.detach()/denominator)}


def full_optimizer_binding(model, raw, recipe):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    groups, visited = [], []
    for config in recipe['optimizer']['params']:
        names = [n for n, p in named if re.findall(config['params'], n)]
        groups.append(names)
        visited.extend(names)
    assert len(visited) == len(set(visited))
    groups.append([n for n, p in named if n not in set(visited)])
    original = raw['optimizer']
    assert len(groups) == len(original['param_groups'])
    binding = {}
    for names, group in zip(groups, original['param_groups']):
        assert len(names) == len(group['params'])
        for name, ident in zip(names, group['params']):
            state = original['state'].get(ident, {})
            if state:
                assert state['exp_avg'].shape == dict(named)[name].shape
                assert state['exp_avg_sq'].shape == dict(named)[name].shape
            binding[name] = {'id': ident, 'group': {k: v for k, v in group.items() if k != 'params'},
                             'state': native.cpu_tree(state)}
    assert set(binding) == {n for n, p in named}
    return named, binding


def virtual_endpoint(model, named, binding, gradients):
    active = [g for g in gradients if g is not None]
    total_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g, 2) for g in active]), 2)
    factor = float(torch.clamp(.1/(total_norm+1e-6), max=1.))
    result = {n: p.detach() for n, p in model.named_parameters()}
    blocks, squared_delta = {}, 0.
    for (name, p), g in zip(named, gradients):
        if g is None:
            continue  # Native AdamW also skips parameters with grad=None, including weight decay.
        assert torch.isfinite(g).all()
        item = binding[name]
        group, state = item['group'], native.cpu_tree(item['state'])
        if not state:
            state = {'step': torch.tensor(0.), 'exp_avg': torch.zeros_like(p),
                     'exp_avg_sq': torch.zeros_like(p)}
            if group['amsgrad']:
                state['max_exp_avg_sq'] = torch.zeros_like(p)
        q = p.detach().clone()
        with torch.no_grad():
            optim_functional.adamw([q], [g.detach()*factor], [state['exp_avg']],
                [state['exp_avg_sq']], [state['max_exp_avg_sq']] if group['amsgrad'] else [],
                [state['step']], foreach=False, capturable=False, differentiable=False, fused=False,
                amsgrad=group['amsgrad'], beta1=group['betas'][0], beta2=group['betas'][1],
                lr=group['lr'], weight_decay=group['weight_decay'], eps=group['eps'], maximize=group['maximize'])
        assert torch.isfinite(q).all()
        result[name] = q
        block = parameter_block(name)
        z = blocks.setdefault(block, {'active_tensors': 0, 'gradient_squared_norm': 0., 'update_squared_norm': 0.})
        z['active_tensors'] += 1
        z['gradient_squared_norm'] += float(g.double().square().sum())
        delta = float((q.double()-p.detach().double()).square().sum())
        z['update_squared_norm'] += delta
        squared_delta += delta
    return result, {'global_gradient_norm_native_FP32': float(total_norm),
                    'global_clip_factor': factor, 'parameter_update_L2_norm': squared_delta**.5,
                    'active_tensors': len(active), 'blocks': blocks}


def parameter_block(name):
    if name.startswith('backbone.'):
        return 'backbone_shared_upstream'
    if name.startswith('encoder.'):
        return 'hybrid_encoder_shared_upstream'
    if name.startswith('decoder.decoder.'):
        return 'decoder_attention_FFN_shared'
    if name.startswith('decoder.pre_bbox_head.'):
        return 'pre_bbox_head'
    if name.startswith('decoder.dec_bbox_head.'):
        return 'decoder_bbox_heads'
    if name.startswith('decoder.dec_score_head.'):
        return 'decoder_classification_heads'
    return 'other_decoder_and_encoder_proposal_parameters'


class QuerySelection:
    """Records native discrete selection; fixes ONLY its indices for primary endpoints.

    At the base point the replacement gathers are checked exactly against native.
    A second endpoint uses the unmodified native selector as a sensitivity check.
    """
    def __init__(self, decoder):
        self.decoder = decoder
        self.native = decoder._select_topk
        self.mode, self.fixed, self.observed = 'native', None, None
        decoder._select_topk = self.call

    def call(self, memory, logits, anchors, k):
        method = self.decoder.query_select_method
        if method == 'default':
            idx = torch.topk(logits.max(-1).values, k, dim=-1).indices
        elif method == 'one2many':
            idx = torch.topk(logits.flatten(1), k, dim=-1).indices // logits.shape[-1]
        elif method == 'agnostic':
            idx = torch.topk(logits.squeeze(-1), k, dim=-1).indices
        else:
            raise ValueError(method)
        self.observed = idx.detach().clone()
        if self.mode == 'native':
            result = self.native(memory, logits, anchors, k)
            expected = memory.gather(1, idx.unsqueeze(-1).expand(-1, -1, memory.shape[-1]))
            assert torch.equal(result[0], expected)
            return result
        assert self.mode == 'fixed' and self.fixed is not None
        idx = self.fixed
        return (memory.gather(1, idx.unsqueeze(-1).expand(-1, -1, memory.shape[-1])),
                logits.gather(1, idx.unsqueeze(-1).expand(-1, -1, logits.shape[-1])),
                anchors.gather(1, idx.unsqueeze(-1).expand(-1, -1, anchors.shape[-1])))


def reset_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def restore_buffers(model, initial):
    with torch.no_grad():
        for n, b in model.named_buffers():
            b.copy_(initial[n])


def metric(outputs, targets, indices, denominator, criterion, own_indices):
    with torch.no_grad():
        boxes = outputs['pred_boxes'].double()
        l1, giou, src, gt = geometric_terms(boxes, targets, indices)
        iou = box_iou(box_cxcywh_to_xyxy(src), box_cxcywh_to_xyxy(gt))[0].diagonal()
        native_terms = native.DEIMCriterion.loss_boxes(criterion, outputs, targets, indices, denominator)
        mal = native.DEIMCriterion.loss_labels_mal(criterion, outputs, targets, own_indices,
                                                  max(1, sum(len(t['labels']) for t in targets)))['loss_mal']
        own_l1, own_giou, own_src, own_gt = geometric_terms(boxes, targets, own_indices)
        return {'pairs': len(l1), 'denominator': denominator,
                'weighted_geometry': float((5*l1.sum()+2*giou.sum())/denominator),
                'L1_original_normalization': float(l1.sum()/denominator),
                'GIoU_loss_original_normalization': float(giou.sum()/denominator),
                'weighted_geometry_native_FP32': float(5*native_terms['loss_bbox']+2*native_terms['loss_giou']),
                'mean_IoU': float(iou.mean()) if len(iou) else None,
                'IoU_ge_0_5_pairs': int((iou >= .5).sum()), 'IoU_ge_0_75_pairs': int((iou >= .75).sum()),
                'MAL_native_own_matching': float(mal),
                'own_matching_pairs_secondary': len(own_l1),
                'own_matching_mean_L1_secondary': float(own_l1.mean()) if len(own_l1) else None,
                'own_matching_mean_GIoU_loss_secondary': float(own_giou.mean()) if len(own_l1) else None}


def payload_for(row):
    path = OLD/'shared_inputs/inputs'/row['file']
    assert native.sha(path) == row['sha256']
    v = torch.load(path, map_location='cpu', weights_only=False)
    assert hashlib.sha256(v['images'].numpy().tobytes()).hexdigest() == row['float_input_sha256']
    return v


def capture(model, criterion, query_selector, named, binding, buffers, row, seed, seed_folder):
    started = time.perf_counter()
    payload = payload_for(row)
    rng = payload['spec']['forward_DN_seed']
    restore_buffers(model, buffers)
    reset_rng(rng)
    query_selector.mode = 'native'
    progress(seed=seed, batch=row['batch'], stage='native_forward')
    outputs = model(payload['images'], payload['targets'])
    topk = query_selector.observed.clone()
    criterion.mark(outputs)
    trace, union_holder = [], []
    old_match, old_union = criterion.matcher.forward, criterion._get_go_indices
    def track_match(*args, **kwargs):
        result = old_match(*args, **kwargs)
        trace.append(native.cpu_tree(result['indices']))
        return result
    def track_union(*args, **kwargs):
        result = old_union(*args, **kwargs)
        union_holder.append(native.cpu_tree(result))
        return result
    criterion.matcher.forward, criterion._get_go_indices = track_match, track_union
    try:
        losses = criterion(outputs, payload['targets'], epoch=24)
    finally:
        criterion.matcher.forward, criterion._get_go_indices = old_match, old_union
    assert len(trace) == 5 and len(union_holder) == 1
    assert all(torch.isfinite(v).all() for v in losses.values())
    union = union_holder[0]
    denominator = criterion.record['ordinary']['denominator']
    assert denominator == sum(len(q) for q, g in union)
    order = ['final', 'aux_0', 'aux_1', 'pre', 'enc_0']
    matches = [{stage: trace[k][b] for k, stage in enumerate(order)} for b in range(len(payload['targets']))]
    selected, source_coeffs, fixed_coeffs, population = coefficient_tables(union, matches, payload['targets'])
    delta_source, ds_terms = correction(outputs, payload['targets'], union, selected, source_coeffs, denominator)
    delta_fixed, df_terms = correction(outputs, payload['targets'], union, selected, fixed_coeffs, denominator)
    total = sum(losses.values())
    # Independent native loss comparison for the exact per-pair decomposition.
    pair_l1, pair_giou, _, _ = geometric_terms(outputs['pre_outputs']['pred_boxes'], payload['targets'], union)
    pre_error = abs(float((5*pair_l1.sum()+2*pair_giou.sum())/denominator-
                          losses['loss_bbox_pre']-losses['loss_giou_pre']))
    assert pre_error < 3e-5, pre_error
    endpoints, updates = {}, {}
    parameters = [p for n, p in named]
    deltas = {'original': total*0, 'source': delta_source, 'fixed_proportion': delta_fixed}
    for k, arm in enumerate(ARMS):
        progress(seed=seed, batch=row['batch'], stage='full_parameter_gradient', arm=arm)
        grads = torch.autograd.grad(total+deltas[arm], parameters, allow_unused=True, retain_graph=k < 2)
        endpoints[arm], updates[arm] = virtual_endpoint(model, named, binding, grads)
        del grads
    base_metric = metric(outputs, payload['targets'], union, denominator, criterion, trace[0])
    # A zero-step functional call checks buffer isolation, RNG, and native selection equivalence.
    base_params = {n: p.detach() for n, p in model.named_parameters()}
    query_selector.mode, query_selector.fixed = 'fixed', topk
    reset_rng(rng)
    with torch.no_grad():
        zero = torch.func.functional_call(model, (base_params, native.cpu_tree(buffers)),
                                          (payload['images'], payload['targets']), strict=True)
    zero_error = float((zero['pred_boxes']-outputs['pred_boxes'].detach()).abs().max())
    zero_metric = metric(zero, payload['targets'], union, denominator, criterion, trace[0])
    zero_loss_error = abs(zero_metric['weighted_geometry']-base_metric['weighted_geometry'])
    assert zero_error < 3e-7 and zero_loss_error < 3e-5, (zero_error, zero_loss_error)
    previous_check = None
    previous_cache = WORK/f'data/09_DN初始化冻结诊断_20261020_v1/seed{seed}/batch{row["batch"]:02d}_frozen_head.pt'
    if row['batch'] < 2:
        old = torch.load(previous_cache, map_location='cpu', weights_only=False)
        previous_check = float((old['native_pre_boxes']-outputs['pre_outputs']['pred_boxes'].detach()).abs().max())
        assert previous_check < 3e-7, previous_check
        del old
    compact = {'seed': seed, 'batch': row['batch'], 'num_boxes_go': denominator,
               'final_native_indices': indices_json(trace[0]), 'union_indices': indices_json(union),
               'branch_matches': [{s: [pair[0].tolist(), pair[1].tolist()] for s, pair in m.items()} for m in matches],
               'topk_indices': topk.tolist(), 'population': population,
               'native_loss_components': {k: float(v.detach()) for k, v in losses.items()},
               'corrections': {'source': ds_terms, 'fixed_proportion': df_terms},
               'updates': updates, 'base_metric': base_metric,
               'checks': {'per_pair_pre_native_loss_abs_error': pre_error,
                          'zero_functional_final_box_max_abs_error': zero_error,
                          'zero_functional_loss_abs_error': zero_loss_error,
                          'previous_diagnostic_pre_box_max_abs_error': previous_check},
               'seconds': time.perf_counter()-started}
    native.save_json(seed_folder/f'batch{row["batch"]:02d}_capture.json', compact)
    restore_buffers(model, buffers)
    del outputs, losses, total, deltas, delta_source, delta_fixed, zero
    gc.collect()
    assert not any(p.grad is not None for p in model.parameters())
    progress(seed=seed, batch=row['batch'], stage='captured', seconds=compact['seconds'],
             transfer_pairs=sum(r['transfer_pairs'] for r in population))
    return {'payload': payload, 'rng': rng, 'union': union, 'own': trace[0], 'denominator': denominator,
            'topk': topk, 'endpoints': endpoints, 'base_metric': base_metric, 'checks': compact['checks']}


def evaluate(model, criterion, selector, buffers, source, target, seed, generator, evaluation):
    row = {'seed': seed, 'generator_batch': generator, 'evaluation_batch': evaluation,
           'cross_batch': generator != evaluation, 'before': target['base_metric'], 'arms': {}}
    for arm in ARMS:
        item = {}
        for mode in ['fixed', 'native']:
            selector.mode, selector.fixed = mode, target['topk']
            reset_rng(target['rng'])
            with torch.no_grad():
                outputs = torch.func.functional_call(model,
                    (source['endpoints'][arm], native.cpu_tree(buffers)),
                    (target['payload']['images'], target['payload']['targets']), strict=True)
            metric_value = metric(outputs, target['payload']['targets'], target['union'],
                                   target['denominator'], criterion, target['own'])
            selection_changed = int((selector.observed != target['topk']).sum())
            item[mode] = {'metrics': metric_value,
                          'native_topk_ordered_slots_changed_vs_base': selection_changed,
                          'native_topk_total_ordered_slots': target['topk'].numel()}
            del outputs
        row['arms'][arm] = item
    for mode in ['fixed', 'native']:
        values = {arm: row['arms'][arm][mode]['metrics']['weighted_geometry'] for arm in ARMS}
        row.setdefault('contrasts', {})[mode] = {
            'source_minus_original': values['source']-values['original'],
            'source_minus_fixed_proportion': values['source']-values['fixed_proportion'],
            'fixed_proportion_minus_original': values['fixed_proportion']-values['original']}
    progress(seed=seed, generator=generator, evaluation=evaluation, stage='cross_batch_evaluated',
             primary_fixed_query_contrasts=row['contrasts']['fixed'])
    return row


def run():
    protocol_path = OUT/'01_冻结协议与资产.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf8'))
    resources = native.resource_limits()
    for item in protocol['sources']:
        assert native.sha(item['path']) == item['sha256'], item['path']
    assert native.sha(__file__) == protocol['runner_sha256']
    assert resources['available_RAM_GiB'] > 12
    started = time.monotonic()
    native.save_json(OUT/'02_启动核验.json', {'utc': now(), 'pid': os.getpid(), 'resources': resources,
                      'protocol_sha256': native.sha(protocol_path), 'virtual_only': True,
                      'hard_timeout_seconds': 1200, 'model_source_imported_unchanged': str(PREVIOUS)})
    progress(stage='preflight_passed', resources=resources)
    all_rows = []
    recipes = json.loads((OLD/'baseline_recipes.json').read_text(encoding='utf-8-sig'))
    for seed in [1, 2]:
        rng_before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        seed_folder = OUT/f'seed{seed}'
        seed_folder.mkdir(exist_ok=False)
        recipe = recipes[str(seed)]
        raw = torch.load(protocol['checkpoints'][str(seed)]['path'], map_location='cpu', weights_only=False)
        assert raw['last_epoch'] == 23 and recipe['clip_max_norm'] == .1
        model = native.build_model(recipe)
        model.load_state_dict(raw['model'], strict=True)
        model.train()
        criterion = native.build_criterion(recipe)
        criterion.load_state_dict(raw['criterion'], strict=True)
        named, binding = full_optimizer_binding(model, raw, recipe)
        original_state = native.model_hash(model)
        buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
        optimizer_state_hash = hashlib.sha256()
        for name, item in binding.items():
            optimizer_state_hash.update(name.encode())
            for key, value in item['state'].items():
                optimizer_state_hash.update(key.encode())
                optimizer_state_hash.update(value.numpy().tobytes() if torch.is_tensor(value) else str(value).encode())
        native.save_json(seed_folder/'parameter_domain.json', {
            'trainable_tensors': len(named), 'trainable_scalars': sum(p.numel() for n, p in named),
            'parameters': [{'name': n, 'shape': list(p.shape), 'block': parameter_block(n),
                            'optimizer_id': binding[n]['id'], 'lr': binding[n]['group']['lr'],
                            'historical_step': float(binding[n]['state']['step']) if binding[n]['state'] else None}
                           for n, p in named], 'original_optimizer_state_digest': optimizer_state_hash.hexdigest()})
        del raw
        gc.collect()
        selector = QuerySelection(model.decoder)
        rows = []
        for lo in range(0, 8, 2):
            assert time.monotonic()-started < 1200, 'Hard diagnostic timeout'
            if psutil.virtual_memory().available < 8*2**30:
                raise MemoryError('Less than 8GiB free: do not compete with other tasks')
            cases = {b: capture(model, criterion, selector, named, binding, buffers,
                                protocol['batches'][b], seed, seed_folder) for b in [lo, lo+1]}
            for source, target in [(lo, lo+1), (lo+1, lo)]:
                row = evaluate(model, criterion, selector, buffers, cases[source], cases[target], seed, source, target)
                rows.append(row)
                native.save_json(seed_folder/f'cross_{source:02d}_to_{target:02d}.json', row)
            del cases
            restore_buffers(model, buffers)
            assert native.model_hash(model) == original_state
            assert not any(p.grad is not None for p in model.parameters())
            gc.collect()
        # Verify the source Adam states as well as the model and the checkpoint.
        current_optimizer_hash = hashlib.sha256()
        for name, item in binding.items():
            current_optimizer_hash.update(name.encode())
            for key, value in item['state'].items():
                current_optimizer_hash.update(key.encode())
                current_optimizer_hash.update(value.numpy().tobytes() if torch.is_tensor(value) else str(value).encode())
        assert current_optimizer_hash.hexdigest() == optimizer_state_hash.hexdigest()
        selector.decoder._select_topk = selector.native
        restore_buffers(model, buffers)
        assert native.model_hash(model) == original_state
        assert native.sha(protocol['checkpoints'][str(seed)]['path']) == protocol['checkpoints'][str(seed)]['sha256']
        random.setstate(rng_before[0]); np.random.set_state(rng_before[1]); torch.set_rng_state(rng_before[2])
        native.save_json(seed_folder/'complete.json', {'state': 'complete', 'seed': seed, 'cross_batch_rows': len(rows),
            'model_state_initial_final_sha256': original_state, 'model_unchanged': True,
            'Adam_source_states_unchanged': True, 'BN_buffers_restored': True,
            'RNG_restored': True, 'checkpoint_sha256': protocol['checkpoints'][str(seed)]['sha256'],
            'optimizer_step_calls': 0, 'checkpoint_writes': 0})
        all_rows.extend(rows)
        progress(seed=seed, stage='seed_complete', cross_batch_rows=len(rows))
        del model, criterion, selector, binding, named, buffers
        gc.collect()
    native.save_json(OUT/'03_全部跨批原始结果.json', {'state': 'complete', 'completed_utc': now(),
        'rows': all_rows, 'protocol_sha256': native.sha(protocol_path), 'resources': resources,
        'elapsed_seconds': time.monotonic()-started, 'training_steps': 0, 'optimizer_step_calls': 0,
        'remote_operations': False, 'heldout_reads': False,
        'scope': 'Exploratory BASE24 local FP32 full-parameter native-gradient / cloned AdamW endpoints. '
                 'Primary freezes base native matching and encoder query selection; native query reselection sensitivity is separate. '
                 'Original L1/GIoU with frozen union/num_boxes_go; FP64 metric arithmetic on FP32 predictions. Not AP or training benefit.'})
    progress(stage='complete', seconds=time.monotonic()-started, cross_batch_rows=len(all_rows), training_steps=0)


if __name__ == '__main__':
    try:
        run()
    except BaseException as e:
        if OUT.exists():
            native.save_json(OUT/('failure_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),
                {'utc': now(), 'error': type(e).__name__+': '+str(e), 'traceback': traceback.format_exc(),
                 'training_steps': 0, 'automatic_identical_retry': False})
        raise
