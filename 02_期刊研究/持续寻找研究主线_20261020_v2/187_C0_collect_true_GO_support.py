"""186: four true train-output C0 forwards, native GO/loss observation only.

Write-only handoff: root reviews and schedules execution. No eval forward,
autograd, model update, optimizer, method effect gate, or remote operation.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import collections
import gc
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
import random
import sys
import time
import traceback
import types
sys.dont_write_bytecode = True

ROOT = pathlib.Path(__file__).resolve().parent
HELPER = ROOT / '113_C0_restore_route_frozen.py'
COMMON_FILE = ROOT.parent / 'code/dn_pre_frozen_diagnostic_20261020_v1/common.py'
PROTOCOL = ROOT / '186_C0真实GO支持域_采集事前协议.txt'
OLD = ROOT / '114_C0_restore_route_frozen'
DEFAULT_OUT = ROOT / '188_C0_true_GO_support_cache'
CP_SHA = '9d6c0dd7a9cd6cb56465c4eab9416c0e89547c7e45df936e559fa12b921f275e'
STATE_SHA = 'c10a2b8b8f7244c6bdb4d1a3261a85af829d303b18469ceefb25acc5fdd5e60b'
BN_SHA = '9c0e960e1a69e3a97aadece89680498c62b9ea9058863f66e5ee75adbf51cb17'
DN_SEED = 20261021
COUNTS = {'model_forwards': 0, 'criterion_calls': 0, 'model_backwards': 0,
          'model_vjps': 0, 'cache_autograd': 0, 'optimizer_steps': 0, 'eval_forwards': 0}


def require(value, message):
    if not bool(value):
        raise RuntimeError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


require(file_sha(HELPER) == '14c3b180ee394cd91840d5369cc44715ad3918d2a38fd3a495913a67f63dd2d3', '113 changed')
require(file_sha(COMMON_FILE) == '5117dd90c631797f2b30d5c48cef50fc9f7a4902143e8ea0086bd6645577ca58', 'common changed')
_spec = importlib.util.spec_from_file_location('_true_go187_build113', HELPER)
h = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = h
_spec.loader.exec_module(h)
n, torch, np = h.n, h.torch, h.np


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, float):
        require(math.isfinite(value), 'nonfinite report scalar')
    return value


def save(path, value):
    n.save_json(path, plain(value))


def bn_hash(model):
    digest = hashlib.sha256()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            for key, value in module.state_dict().items():
                digest.update((name + '.' + key).encode('utf-8'))
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def no_graph(value):
    if isinstance(value, torch.Tensor):
        require(not value.requires_grad and value.grad_fn is None, 'autograd graph unexpectedly retained')
    elif isinstance(value, dict):
        for item in value.values():
            no_graph(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            no_graph(item)


def equal_tree(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            return False
        return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(a.cpu(), b.cpu()))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def stage_map(out):
    stages = {'final': out}
    for key in ('aux_outputs', 'enc_aux_outputs', 'dn_outputs'):
        for index, value in enumerate(out.get(key, [])):
            stages[f'{key}.{index}'] = value
    for key in ('pre_outputs', 'dn_pre_outputs'):
        if key in out:
            stages[key] = out[key]
    pointers = {}
    for name, value in stages.items():
        require('pred_boxes' in value and 'pred_logits' in value, 'missing stage predictions')
        ident = id(value['pred_boxes'])
        require(ident not in pointers, 'ambiguous shared pred_boxes object across stages')
        pointers[ident] = name
    return stages, pointers


def inspect_reference_sharing(out):
    """Observe native refs before cpu_tree cloning discards alias information."""
    stages, _ = stage_map(out)
    groups = {'ordinary_FDR': ['final'] + [key for key in stages if key.startswith('aux_outputs.')],
              'DN_FDR': [key for key in stages if key.startswith('dn_outputs.')]}
    report = {'stages_without_ref_points': [key for key, value in stages.items() if 'ref_points' not in value],
              'groups': {}, 'comparison_scope': 'within identical ordinary or DN query slots; not ordinary versus DN'}
    for group, names in groups.items():
        require(names and all('ref_points' in stages[key] for key in names), 'missing group refs')
        base = stages[names[0]]['ref_points']
        comparisons = {}
        for name in names:
            ref = stages[name]['ref_points']
            same_shape = ref.shape == base.shape and ref.dtype == base.dtype
            comparisons[name] = {'shape': list(ref.shape), 'same_shape_and_dtype': same_shape,
                'values_exact_to_group_first': bool(same_shape and torch.equal(ref, base)),
                'max_abs_to_group_first': float((ref.double() - base.double()).abs().max()) if same_shape else None,
                'same_tensor_object': ref is base,
                'same_underlying_storage': ref.untyped_storage().data_ptr() == base.untyped_storage().data_ptr(),
                'same_first_element_address': ref.data_ptr() == base.data_ptr(),
                'tensor_sha256': h.tensor_sha(ref)}
        pre_name = 'pre_outputs' if group == 'ordinary_FDR' else 'dn_pre_outputs'
        pre = stages[pre_name]['pred_boxes']
        pre_same_shape = pre.shape == base.shape and pre.dtype == base.dtype
        report['groups'][group] = {'group_first': names[0], 'stage_count': len(names),
            'all_stage_values_exact': all(value['values_exact_to_group_first'] for value in comparisons.values()),
            'per_stage': comparisons, 'pre_boxes_comparator': pre_name,
            'group_first_equals_pre_boxes_exact': bool(pre_same_shape and torch.equal(base, pre)),
            'group_first_pre_boxes_max_abs': float((base.double() - pre.double()).abs().max()) if pre_same_shape else None}
    return report


def summarize_indices(indices, targets, query_count=300):
    require(len(indices) == len(targets) == 1, 'one-image index contract')
    q, g = [value.detach().cpu().long() for value in indices[0]]
    require(q.ndim == g.ndim == 1 and q.shape == g.shape, 'index shape')
    labels = targets[0]['labels'].detach().cpu()
    require(bool((q >= 0).all() and (q < query_count).all()), 'query out of range')
    require(bool((g >= 0).all() and (g < len(labels)).all()), 'GT out of range')
    unique_q = q.unique().numel() == q.numel()
    per_query_gt = torch.full((query_count,), -1, dtype=torch.long)
    per_query_class = torch.full((query_count,), -1, dtype=torch.long)
    if unique_q:
        per_query_gt[q], per_query_class[q] = g, labels[g]
    return {'query_indices': q, 'GT_indices': g, 'GT_multiset': g.clone(),
            'GT_multiplicity': torch.bincount(g, minlength=len(labels)),
            'pair_classes': labels[g].clone(), 'query_unique': bool(unique_q),
            'unique_GT_count': int(g.unique().numel()), 'pairs': int(q.numel()),
            'gt_per_query': per_query_gt, 'class_per_query': per_query_class}


class SelectionObserver:
    """Observe native training top-k, returning the untouched result tuple."""
    def __init__(self, decoder):
        self.decoder, self.original, self.seen = decoder, decoder._select_topk, []

    def __enter__(self):
        observer = self
        def observe(this, memory, logits, anchors, topk):
            require(this.training and this.query_select_method == 'default', 'training default selection required')
            result = observer.original(memory, logits, anchors, topk)
            require(topk == 300 and tuple(memory.shape[:2]) == (1, 8400), 'native selection geometry')
            indices = torch.topk(logits.max(-1).values, topk, dim=-1).indices
            for actual, source in zip(result, (memory, logits, anchors)):
                expected = source.gather(1, indices.unsqueeze(-1).expand(-1, -1, source.shape[-1]))
                require(actual is not None and torch.equal(actual, expected), 'native selection gather differs')
            require(indices.unique().numel() == 300, 'duplicate selected source')
            observer.seen.append(indices[0].detach().cpu().clone())
            return result
        self.decoder._select_topk = types.MethodType(observe, self.decoder)
        return self

    def __exit__(self, *_):
        self.decoder._select_topk = self.original


class CriterionObserver:
    """Capture original matcher, actual GO and every get_loss input/output."""
    def __init__(self, criterion, out, targets):
        self.criterion, self.targets = criterion, targets
        self.stages, self.pointers = stage_map(out)
        self.matcher_calls, self.loss_calls, self.go_calls = [], [], []
        self.original_get_loss = criterion.get_loss
        self.original_go = criterion._get_go_indices

    def stage(self, outputs):
        require(id(outputs['pred_boxes']) in self.pointers, 'unrecognized loss/matcher stage')
        return self.pointers[id(outputs['pred_boxes'])]

    def __enter__(self):
        observer = self
        def matcher_hook(module, inputs, output):
            require(len(inputs) >= 2 and 'indices' in output, 'native matcher return contract')
            observer.matcher_calls.append({'stage': observer.stage(inputs[0]),
                'indices': n.cpu_tree(output['indices']),
                'GT_count': [len(t['labels']) for t in inputs[1]]})
        self.matcher_handle = self.criterion.matcher.register_forward_hook(matcher_hook)
        def observe_go(this, indices, indices_aux_list):
            inputs_copy = {'final_own': n.cpu_tree(indices), 'auxiliary_own': n.cpu_tree(indices_aux_list)}
            result = observer.original_go(indices, indices_aux_list)
            observer.go_calls.append({**inputs_copy, 'indices_go': n.cpu_tree(result)})
            return result
        def observe_loss(this, loss, outputs, targets, indices, num_boxes, **kwargs):
            stage = observer.stage(outputs)
            result = observer.original_get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)
            observer.loss_calls.append({'stage': stage, 'loss': loss,
                'indices': n.cpu_tree(indices), 'denominator': float(num_boxes),
                'paired_GT_labels': [t['labels'][g].detach().cpu().clone() for t, (_, g) in zip(targets, indices)],
                'extra_loss_arguments': n.cpu_tree(kwargs),
                'raw_loss_values_before_weight_dict': {key: float(value.detach()) for key, value in result.items()}})
            return result
        self.criterion._get_go_indices = types.MethodType(observe_go, self.criterion)
        self.criterion.get_loss = types.MethodType(observe_loss, self.criterion)
        return self

    def __exit__(self, *_):
        self.matcher_handle.remove()
        self.criterion._get_go_indices = self.original_go
        self.criterion.get_loss = self.original_get_loss

    def export(self):
        expected_stages = ['final'] + [key for key in self.stages if key.startswith('aux_outputs.')]
        expected_stages += ['pre_outputs'] + [key for key in self.stages if key.startswith('enc_aux_outputs.')]
        require([call['stage'] for call in self.matcher_calls] == expected_stages, 'native own matcher stage order')
        require(len(self.go_calls) == 1, 'exactly one ordinary GO call expected')
        go = self.go_calls[0]
        require(equal_tree(go['final_own'], self.matcher_calls[0]['indices']), 'GO final input differs from observed own')
        require(equal_tree(go['auxiliary_own'], [v['indices'] for v in self.matcher_calls[1:]]), 'GO auxiliary input order differs')
        final_boxes = [x for x in self.loss_calls if x['stage'] == 'final' and x['loss'] == 'boxes']
        final_mal = [x for x in self.loss_calls if x['stage'] == 'final' and x['loss'] == 'mal']
        require(len(final_boxes) == len(final_mal) == 1, 'one ordinary final boxes/MAL loss expected')
        require(equal_tree(final_boxes[0]['indices'], go['indices_go']), 'final boxes did not use native GO')
        require(equal_tree(final_mal[0]['indices'], self.matcher_calls[0]['indices']), 'final MAL did not use own')
        summary = summarize_indices(go['indices_go'], self.targets)
        require(summary['query_unique'], 'native GO must retain at most one GT per query')
        require(final_boxes[0]['denominator'] == max(1, summary['pairs']), 'actual GO denominator')
        require(final_mal[0]['denominator'] == len(self.targets[0]['labels']), 'ordinary GT denominator')
        combined_q = torch.cat([call['indices'][0][0] for call in self.matcher_calls])
        combined_g = torch.cat([call['indices'][0][1] for call in self.matcher_calls])
        return {'own_by_stage': {call['stage']: call['indices'] for call in self.matcher_calls},
                'matcher_calls': self.matcher_calls, 'GO_native_call': go,
                'indices_own_final': self.matcher_calls[0]['indices'], 'indices_go': go['indices_go'],
                'num_boxes_ordinary': final_mal[0]['denominator'], 'num_boxes_go': final_boxes[0]['denominator'],
                'GO_summary': summary,
                'pre_GO_stage_pair_multiset': {'query_indices': combined_q, 'GT_indices': combined_g,
                    'GT_multiplicity': torch.bincount(combined_g, minlength=len(self.targets[0]['labels']))}}


def compare_tensor(actual, prior):
    require(actual.shape == prior.shape and actual.dtype == prior.dtype, '114 prediction interface changed')
    delta = (actual.detach().double().cpu() - prior.detach().double().cpu()).abs()
    return {'exact': bool(torch.equal(actual.cpu(), prior.cpu())), 'max_abs_error': float(delta.max()),
            'mean_abs_error': float(delta.mean()), 'actual_sha256': h.tensor_sha(actual), 'prior_sha256': h.tensor_sha(prior)}


def compare_114(record, row, prediction):
    require(record['image_index'] == row['image_index'] and record['event'] == row['event'] == prediction['event'], '114 event mismatch')
    require(record['input_sha256'] == row['input_sha256'], '114 image tensor identity mismatch')
    require(len(record['target']['labels']) == row['GT'] and int(record['original_sub4'].sum()) == row['original_sub4'], '114 GT group mismatch')
    q, g = record['matching']['indices_go'][0]
    actual_pairs = list(zip(q.tolist(), g.tolist()))
    old_pairs = list(zip(row['final_union_query_indices'], row['final_union_GT_indices']))
    loss_values = record['loss_values']
    prior_losses = row['loss_values']
    common = sorted(set(loss_values) & set(prior_losses))
    differences = {key: {'actual': loss_values[key], 'prior': prior_losses[key],
        'signed_difference': loss_values[key] - prior_losses[key],
        'absolute_difference': abs(loss_values[key] - prior_losses[key]),
        'relative_difference': abs(loss_values[key] - prior_losses[key]) / max(abs(prior_losses[key]), 1e-30)} for key in common}
    predictions = {key: compare_tensor(record['native_outputs'][key], prediction['train_ordinary'][key])
                   for key in ('pred_boxes', 'pred_logits')}
    report = {'metadata_exact': True, 'GO_ordered_exact': actual_pairs == old_pairs,
        'GO_pair_multiset_exact': collections.Counter(actual_pairs) == collections.Counter(old_pairs),
        'GO_actual_only': sorted((collections.Counter(actual_pairs) - collections.Counter(old_pairs)).elements()),
        'GO_prior_only': sorted((collections.Counter(old_pairs) - collections.Counter(actual_pairs)).elements()),
        'GO_changed_order_positions': [{'position': i, 'actual': a, 'prior': b}
            for i, (a, b) in enumerate(itertools.zip_longest(actual_pairs, old_pairs)) if a != b],
        'GO_denominator_actual': record['matching']['num_boxes_go'], 'GO_denominator_114': row['final_native_denominator'],
        'GO_denominator_exact': record['matching']['num_boxes_go'] == row['final_native_denominator'],
        'loss_keys_exact': set(loss_values) == set(prior_losses),
        'loss_actual_only_keys': sorted(set(loss_values) - set(prior_losses)),
        'loss_prior_only_keys': sorted(set(prior_losses) - set(loss_values)), 'loss_comparison': differences,
        'loss_total_actual': record['loss_total'], 'loss_total_114': row['measured_terms']['total'],
        'train_ordinary_predictions': predictions,
        'selected_vs_114_eval_exact': record['selected_indices'].tolist() == row['eval_selected_indices'],
        'selected_comparator_scope': '114 saved eval source indices, not a second train-index cache',
        'DN_meta_exact': equal_tree(record['native_outputs']['dn_meta'], prediction['DN_meta']),
        'DN_seed_exact': record['DN_seed'] == row['DN_seed'],
        'DN_num_group_exact': int(record['native_outputs']['dn_meta']['dn_num_group']) == row['DN_num_group'],
        'DN_split_exact': list(record['native_outputs']['dn_meta']['dn_num_split']) == row['DN_split'],
        'rng_before_exact': record['rng_before_sha256'] == row['rng_before_sha256'],
        'rng_after_forward_exact': record['rng_after_forward_sha256'] == row['rng_after_sha256'],
        'difference_interpretation': 'No method intervention. Any discrepancy is recorded, never repaired into expected parity.'}
    report['exact_train_reproduction_of_saved_fields'] = bool(
        report['GO_ordered_exact'] and report['GO_denominator_exact'] and report['loss_keys_exact']
        and all(d['absolute_difference'] == 0 for d in differences.values())
        and all(v['exact'] for v in predictions.values()) and report['DN_meta_exact']
        and report['rng_before_exact'] and report['rng_after_forward_exact'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    outdir = args.out.resolve()
    require(outdir.parent == ROOT and outdir.name.startswith('188_') and not outdir.exists(), 'new local 188_* output required')
    started = time.monotonic()
    resources = h.resources()
    require(torch.get_num_threads() == 2 and torch.get_num_interop_threads() == 1, 'thread contract')
    require(h.DN_SEED == DN_SEED and not torch.backends.mha.get_fastpath_enabled(), '113 seed/fastpath contract')
    outdir.mkdir()
    model = criterion = None
    forward_handle = None
    before_state = before_bn = before_criterion = None
    hashes, reports, records = {}, [], []
    stage = 'source_and_identity_checks'
    try:
        def lock(path, expected=None):
            path = pathlib.Path(path).resolve()
            current = hashes.get(str(path)) or file_sha(path)
            require(expected is None or current == expected, 'source hash mismatch: ' + str(path))
            hashes[str(path)] = current
            return path
        manifest_path = lock(OLD / '00_inputs_and_protocol_sha256.json', 'edeaf742a55655bdc1ee5c5c940f8b4c97a90a314663e28ddc50ff5276f3f328')
        old_inputs = json.loads(manifest_path.read_text(encoding='utf-8'))
        for path, digest in old_inputs.items():
            lock(path, digest)
        output_manifest_path = lock(OLD / '05_output_sha256.json', '85c6592ddc67060394db29ff036b2575719387fb608ee7609dc428294dd3c38d')
        old_hashes = json.loads(output_manifest_path.read_text(encoding='utf-8'))
        required_old = ['03_gradients_and_predictions.pt', '04_execution.json'] + [f'01_image_{i}.json' for i in range(4)]
        for name in required_old:
            lock(OLD / name, old_hashes[name])
        lock(PROTOCOL)
        lock(pathlib.Path(__file__))
        save(outdir / '00_inputs_sha256.json', hashes)
        meta = json.loads((ROOT / '104_C0_本地资产严格加载核验.json').read_text(encoding='utf-8'))
        old_execution = json.loads((OLD / '04_execution.json').read_text(encoding='utf-8'))
        require(meta['checkpoint_sha256'] == CP_SHA and old_execution['state_hash_before'] == STATE_SHA, 'C0 identity changed')
        require(old_execution['state'] == 'complete' and old_execution['resources']['mha_fastpath_enabled'] is False, '114 execution identity')
        selected = json.loads((ROOT / '85_source_self_attention/00_selected.json').read_text(encoding='utf-8'))
        data = torch.load(ROOT / '77_all_count_frozen/00_all_count_inputs.pt', map_location='cpu', weights_only=False)
        cases, images = [], []
        for item in selected:
            found = [case for case in data['cases'] if case['event'] == item['event']]
            require(len(found) == 1, 'event must identify one original input')
            case = found[0]
            image = data['images'][case['input_sha256']].clone()
            require(case['input_sha256'] == item['input_sha256'] == h.tensor_sha(image), 'input tensor SHA mismatch')
            require(image.dtype == torch.float32 and tuple(image.shape) == (3, 640, 640), 'input tensor interface')
            cases.append(n.cpu_tree(case))
            images.append(image)
        require(len(cases) == 4, 'four development inputs required')
        require([len(c['target']['labels']) for c in cases] == [146, 280, 418, 58], 'GT cohort identity')
        require([int(c['original_sub4'].sum()) for c in cases] == [46, 121, 271, 15], 'sub4 cohort identity')
        del data
        old_cache = torch.load(OLD / '03_gradients_and_predictions.pt', map_location='cpu', weights_only=False)
        old_predictions = n.cpu_tree(old_cache['predictions'])
        require(len(old_predictions) == 4, '114 prediction count')
        del old_cache
        old_rows = [json.loads((OLD / f'01_image_{i}.json').read_text(encoding='utf-8')) for i in range(4)]
        gc.collect()
        h.memory_guard()
        stage = 'strict_online_model_build'
        raw = torch.load(meta['checkpoint'], map_location='cpu', weights_only=False)
        require(raw['last_epoch'] == meta['last_epoch'] == 94, 'epoch identity')
        model, criterion, recipe, strict_load = h.build(meta, raw)
        del raw
        gc.collect()
        require(len(model.state_dict()) == 801 and strict_load == '<All keys matched successfully>', 'strict 801 tensor load')
        require(model.decoder.query_select_method == 'default' and not model.decoder.mgsa_groups
                and not model.decoder.o2m_t, 'ordinary decoder path required')
        before_state, before_bn, before_criterion = n.model_hash(model), bn_hash(model), n.model_hash(criterion)
        require(before_state == STATE_SHA and before_bn == BN_SHA, 'loaded model or BN state differs')
        def count_forward(module, inputs):
            require(COUNTS['model_forwards'] < 4, 'fifth model forward forbidden')
            require(not torch.is_grad_enabled(), 'forward must be no_grad')
            require(not model.training and model.decoder.training, 'forward mode differs')
            COUNTS['model_forwards'] += 1
        forward_handle = model.register_forward_pre_hook(count_forward)
        with torch.no_grad():
            for number, (case, image) in enumerate(zip(cases, images)):
                stage = f'collect_image_{number}'
                h.memory_guard()
                h.mode(model, True)
                require(all(not m.training for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)), 'BN not frozen')
                target, small = n.cpu_tree(case['target']), case['original_sub4'].bool().clone()
                targets = [target]
                random.seed(DN_SEED)
                np.random.seed(DN_SEED)
                torch.manual_seed(DN_SEED)
                rng_before = h.tensor_sha(torch.get_rng_state())
                with SelectionObserver(model.decoder) as selection:
                    output = model(image.unsqueeze(0), targets)
                rng_after = h.tensor_sha(torch.get_rng_state())
                require(len(selection.seen) == 1, 'one native top-k selection per image')
                require(all(key in output for key in ('pred_boxes', 'pred_logits', 'pred_corners', 'ref_points',
                    'aux_outputs', 'pre_outputs', 'enc_aux_outputs', 'dn_outputs', 'dn_pre_outputs', 'dn_meta')), 'missing train output')
                require(tuple(output['pred_boxes'].shape) == (1, 300, 4)
                        and tuple(output['pred_logits'].shape) == (1, 300, 10)
                        and tuple(output['pred_corners'].shape) == (1, 300, 132)
                        and tuple(output['ref_points'].shape) == (1, 300, 4), 'ordinary train output shape')
                for key in ('pred_boxes', 'pred_logits', 'pred_corners', 'ref_points'):
                    require(bool(torch.isfinite(output[key]).all()), 'nonfinite ordinary train prediction')
                no_graph(output)
                ref_sharing = inspect_reference_sharing(output)
                h.memory_guard()
                criterion.mark(output)
                with CriterionObserver(criterion, output, targets) as observation:
                    COUNTS['criterion_calls'] += 1
                    losses = criterion(output, targets, epoch=94, step=0, epoch_step=1, global_step=None)
                no_graph(losses)
                matching = observation.export()
                require(equal_tree(criterion.final_record['indices'], matching['indices_go'])
                        and criterion.final_record['denominator'] == matching['num_boxes_go'], '113 final observation mismatch')
                snapshot = n.cpu_tree(output)
                stages, _ = stage_map(snapshot)
                catalog = {name: {key: list(value[key].shape) for key in ('pred_boxes', 'pred_logits', 'pred_corners', 'ref_points') if key in value}
                           for name, value in stages.items()}
                record = {'image_index': number, 'event': case['event'], 'input_sha256': case['input_sha256'],
                    'target': target, 'original_sub4': small, 'pred_boxes': snapshot['pred_boxes'][0],
                    'pred_logits': snapshot['pred_logits'][0], 'pred_corners': snapshot['pred_corners'][0],
                    'final_corners': snapshot['pred_corners'][0], 'ref_points_initial': snapshot['ref_points'][0],
                    'reference_origin': "this true train forward output['ref_points'][0]; never borrowed from eval",
                    'selected_indices': selection.seen[0], 'native_selection_gather_exact': True,
                    'native_outputs': snapshot, 'stage_catalog': catalog, 'stage_ref_sharing': ref_sharing, 'matching': matching,
                    'loss_calls': observation.loss_calls,
                    'loss_values': {key: float(value.detach()) for key, value in losses.items()},
                    'loss_total': float(sum(losses.values())), 'native_weight_dict': dict(criterion.weight_dict),
                    'DN_seed': DN_SEED, 'rng_before_sha256': rng_before,
                    'rng_after_forward_sha256': rng_after, 'rng_after_criterion_sha256': h.tensor_sha(torch.get_rng_state())}
                comparison = compare_114(record, old_rows[number], old_predictions[number])
                record['comparison_114'] = comparison
                records.append(record)
                report = {key: record[key] for key in ('image_index', 'event', 'input_sha256', 'stage_catalog', 'stage_ref_sharing',
                    'matching', 'loss_calls', 'loss_values', 'loss_total', 'native_weight_dict', 'DN_seed',
                    'rng_before_sha256', 'rng_after_forward_sha256', 'rng_after_criterion_sha256', 'comparison_114')}
                report.update(GT=len(target['labels']), original_sub4=int(small.sum()),
                    selected_indices=record['selected_indices'], native_selection_gather_exact=True,
                    BN_state_unchanged=bn_hash(model) == before_bn)
                save(outdir / f'01_image_{number}.json', report)
                reports.append(report)
                require(all(p.grad is None for p in model.parameters()), 'unexpected parameter gradient')
                criterion.final_boxes = criterion.final_record = None
                criterion._clear_cache()
                del output, losses, observation, snapshot, stages, selection
                gc.collect()
                print('collected image', number, 'GO pairs', matching['GO_summary']['pairs'],
                      '114 ordered GO exact', comparison['GO_ordered_exact'], flush=True)
        require(COUNTS['model_forwards'] == COUNTS['criterion_calls'] == 4, 'exact four-forward collection budget')
        require(h.COUNTS == {'model_forwards': 0, 'model_vjps': 0, 'optimizer_steps': 0}, '113 execution helpers must not run')
        after_state, after_bn, after_criterion = n.model_hash(model), bn_hash(model), n.model_hash(criterion)
        require(before_state == after_state and before_bn == after_bn and before_criterion == after_criterion, 'model/BN/criterion state changed')
        require(all(p.grad is None for p in model.parameters()), 'unexpected gradients at completion')
        unchanged = {path: file_sha(path) == digest for path, digest in hashes.items()}
        require(all(unchanged.values()), 'source/checkpoint/input hash changed')
        mode = {'model': 'eval', 'backbone': 'eval', 'encoder': 'eval', 'decoder': 'train',
                'all_BN': 'eval', 'torch_no_grad': True, 'mha_fastpath_enabled': False}
        top = {'records': records, 'checkpoint_sha256': CP_SHA, 'weight_key': 'model_online', 'epoch': 94,
               'reg_max': int(model.decoder.reg_max), 'up': float(model.decoder.up.detach().item()),
               'reg_scale': float(model.decoder.reg_scale.detach().item()),
               'model_state_sha256': before_state, 'BN_state_sha256': before_bn, 'criterion_state_sha256': before_criterion,
               'mode': mode, 'DN_seed': DN_SEED, 'source_hashes': hashes, 'counts': COUNTS.copy(),
               'GT': 902, 'original_sub4': 453, 'schema': 'true-GO-train-ref-v1', 'heldout_reads': False}
        torch.save(top, outdir / '01_cache.pt')
        summary = {'state': 'complete', 'GT': 902, 'original_sub4': 453,
            'per_image': [{'image_index': r['image_index'], 'event': r['event'], 'GT': len(r['target']['labels']),
                'original_sub4': int(r['original_sub4'].sum()), 'GO_pairs': r['matching']['GO_summary']['pairs'],
                'GO_unique_GT': r['matching']['GO_summary']['unique_GT_count'],
                'stage_ref_sharing': r['stage_ref_sharing'], 'comparison_114': r['comparison_114']} for r in records],
            'parity_with_114_all_saved_train_fields_exact': all(r['comparison_114']['exact_train_reproduction_of_saved_fields'] for r in records),
            'method_effect_evaluated': False, 'heldout_reads': False}
        save(outdir / '02_summary.json', summary)
        execution = {'state': 'complete', 'resources': resources, 'counts': COUNTS.copy(), 'strict_load_online': strict_load,
            'num_state_tensors': len(model.state_dict()), 'model_state_before': before_state, 'model_state_after': after_state,
            'BN_state_before': before_bn, 'BN_state_after': after_bn, 'criterion_state_before': before_criterion,
            'criterion_state_after': after_criterion, 'source_hashes': hashes, 'source_unchanged': unchanged,
            'checkpoint_unchanged': file_sha(meta['checkpoint']) == CP_SHA, 'new_training': False,
            'optimizer_constructed': False, 'heldout_reads': False, 'mode': mode,
            'elapsed_seconds': time.monotonic() - started, 'method_gate': None}
        require(execution['checkpoint_unchanged'], 'checkpoint changed after cache write')
        save(outdir / '04_execution.json', execution)
        save(outdir / '05_output_sha256.json', {path.name: file_sha(path) for path in outdir.iterdir() if path.is_file()})
        print(json.dumps({'state': 'complete', 'counts': COUNTS, 'seconds': execution['elapsed_seconds'],
                          'parity_exact': summary['parity_with_114_all_saved_train_fields_exact']}, ensure_ascii=False), flush=True)
    except BaseException:
        save(outdir / 'FAILED.json', {'state': 'failed', 'stage': stage, 'counts': COUNTS.copy(),
            'traceback': traceback.format_exc(), 'elapsed_seconds': time.monotonic() - started,
            'model_state_unchanged': None if model is None or before_state is None else n.model_hash(model) == before_state,
            'BN_state_unchanged': None if model is None or before_bn is None else bn_hash(model) == before_bn,
            'new_training': False, 'heldout_reads': False})
        raise
    finally:
        if forward_handle is not None:
            forward_handle.remove()
        if criterion is not None:
            criterion.final_boxes = criterion.final_record = None
            criterion._clear_cache()


if __name__ == '__main__':
    main()
