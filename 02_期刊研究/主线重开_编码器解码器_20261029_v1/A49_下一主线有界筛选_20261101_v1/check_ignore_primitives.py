"""Analytic and native-function invariants only; no detector/GPU/optimizer."""
from pathlib import Path
import ast
import functools
import hashlib
import json
import sys
from typing import List

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / 'code/dn_pre_frozen_diagnostic_20261020_v1/runtime'))
import torch
import torch.nn.functional as F
from ignore_query_supervision import disjoint_union, fractional_field, box_ignore_fraction, read_ignore_mass, IgnorePolicy

SOURCE = ROOT.parent / 'A43_ECDet三组蒸馏最小验证_20261031_v1/source/EdgeCrafter-d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/ecdetseg/engine/edgecrafter'


def load_function(path, name, class_name=None):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    container = tree if class_name is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in container.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {'torch': torch, 'F': F, 'List': List}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class ToyCriterion:
    num_classes = 2
    gamma = 2.0
    mal_alpha = .75

    @staticmethod
    def _get_src_permutation_idx(indices):
        return (torch.cat([torch.full_like(i, b) for b, (i, _) in enumerate(indices)]),
                torch.cat([i for i, _ in indices]))


if __name__ == '__main__':
    target = ROOT / 'math_and_loss_check_v2.json'
    assert not target.exists(), 'Keep completed evidence; do not overwrite.'
    torch.set_num_threads(2)
    torch.manual_seed(20261101)
    tiles = torch.tensor(disjoint_union([[0, 0, .5, 1], [.25, 0, .75, 1]]), dtype=torch.float32)
    assert abs(float(((tiles[:, 2:] - tiles[:, :2]).prod(-1)).sum()) - .75) < 1e-7
    field = fractional_field(tiles, 4, 4)
    assert torch.equal(field, torch.tensor([[1., 1., 1., 0.]]).expand(4, 4))
    whole = torch.tensor([[.5, .5, 1., 1.]])
    assert torch.equal(box_ignore_fraction(whole, tiles), torch.tensor([.75]))
    mirrored = tiles.clone()
    mirrored[:, [0, 2]] = 1 - tiles[:, [2, 0]]
    assert torch.equal(fractional_field(mirrored, 4, 4), field.flip(-1))
    native_core = load_function(SOURCE / 'utils.py', 'deformable_attention_core_func_v2')
    counts = {'native_core_calls': 0, 'mask_core_reads': 0}
    native_checks = []
    shapes = [(4, 4), (2, 2)]
    locations = torch.tensor([[[[[.125, .125], [.25, .25]], [[.125, .125], [.25, .25]]],
                                [[[.875, .875], [.75, .75]], [[.875, .875], [.75, .75]]],
                                [[[.125, .125], [.75, .75]], [[.125, .125], [.75, .75]]]]])
    weights = torch.full((1, 3, 2, 2), .5)
    half_tiles = [torch.tensor([[0., 0., .5, 1.]])]
    for method, expected in [('default', [1., 0., .5]), ('discrete', [.5, 0., .5])]:
        def counted(*args, **kwargs):
            counts['native_core_calls'] += 1
            return native_core(*args, method=method, **kwargs)
        value = [torch.randn(1, 2, 3, h * w, requires_grad=True) for h, w in shapes]
        before_rng = torch.get_rng_state().clone()
        original = counted(value, shapes, locations, weights, [1, 1])
        gradient0 = torch.autograd.grad(original.sum(), value)
        mass = read_ignore_mass(counted, half_tiles, shapes, locations, weights, [1, 1])
        counts['mask_core_reads'] += 1
        again = counted(value, shapes, locations, weights, [1, 1])
        gradient1 = torch.autograd.grad(again.sum(), value)
        assert torch.equal(original, again) and all(torch.equal(a, b) for a, b in zip(gradient0, gradient1))
        assert torch.equal(before_rng, torch.get_rng_state())
        assert not mass.requires_grad and torch.equal(mass, torch.tensor([expected]))
        native_checks.append({'method': method, 'mass': mass.tolist(), 'original_output_gradient_rng_unchanged': True})
    # Use the exact pinned MAL function. Supplied detached IoUs isolate the new
    # classification-weight invariant; full native matching/IoU is left to capacity.
    criterion_file = SOURCE / 'criterion.py'
    classes = [n.name for n in ast.parse(criterion_file.read_text(encoding='utf-8-sig')).body if isinstance(n, ast.ClassDef)]
    mal = next(load_function(criterion_file, 'loss_labels_mal', c) for c in classes
               if any(isinstance(n, ast.FunctionDef) and n.name == 'loss_labels_mal'
                      for n in next(v for v in ast.parse(criterion_file.read_text(encoding='utf-8-sig')).body
                                    if isinstance(v, ast.ClassDef) and v.name == c).body))
    ToyCriterion.loss_labels_mal = mal
    logits = torch.randn(1, 3, 2)
    boxes = torch.tensor([[[.25, .5, .2, .2], [.25, .5, .2, .2], [.8, .5, .1, .1]]])
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    targets = [{'labels': torch.tensor([1]), 'boxes': boxes[0, :1], 'ignore_tiles': half_tiles[0]}]
    ious = torch.tensor([.7])

    def run(arm, empty=False, ordinary=True):
        criterion = ToyCriterion()
        policy = IgnorePolicy(arm)
        policy.install_loss(criterion)
        x = logits.clone().requires_grad_()
        out = {'pred_logits': x, 'pred_boxes': boxes, '_a49_ordinary': ordinary,
               '_a49_read_mass': torch.tensor([[1., 1., 0.]])}
        ts = [{**t, 'ignore_tiles': t['ignore_tiles'].new_empty((0, 4))} for t in targets] if empty else targets
        loss = criterion.loss_labels_mal(out, ts, indices, 1., values=ious)['loss_mal']
        grad = torch.autograd.grad(loss, x)[0]
        return loss.detach(), grad

    base_loss, base_grad = run('B')
    loss_checks = []
    for arm in ['M', 'Q']:
        empty_loss, empty_grad = run(arm, empty=True)
        dn_loss, dn_grad = run(arm, ordinary=False)
        loss, grad = run(arm)
        assert torch.equal(base_loss, empty_loss) and torch.equal(base_grad, empty_grad)
        assert torch.equal(base_loss, dn_loss) and torch.equal(base_grad, dn_grad)
        assert torch.equal(grad[:, 0], base_grad[:, 0])  # Matched row inside ignore is protected.
        assert torch.equal(grad[:, 2], base_grad[:, 2])  # Ordinary background outside ignore retained.
        assert torch.equal(grad[:, 1], torch.zeros_like(grad[:, 1]))
        loss_checks.append({'arm': arm, 'empty_mask_native_bitwise': True, 'DN_native_bitwise': True,
                            'matched_row_all_class_gradients_protected': True,
                            'unmasked_background_gradient_unchanged': True, 'omitted_row_gradient_zero': True})
    result = {'status': 'PASS', 'scope': 'CPU analytic geometry and native function toys with supplied IoUs, not a detector run, capacity check, effect or novelty proof.',
              'native_sampling_checks': native_checks, 'classification_gradient_checks': loss_checks,
              'component_counts': counts, 'model_forwards': 0, 'optimizer_updates': 0, 'GPU_jobs': 0,
              'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         [ROOT / 'ignore_query_supervision.py', ROOT / 'check_ignore_primitives.py', SOURCE / 'utils.py', criterion_file]}}
    with target.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'sha256'}))
