"""Bounded native-attention integration and algebra checks, no detector/image run."""
import ast
import copy
import functools
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import List

ROOT = Path(__file__).resolve().parent
ENG = ROOT.parents[1]
sys.path.insert(0, str(ENG / 'code/dn_pre_frozen_diagnostic_20261020_v1/runtime'))
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from contrastive_read import install

SOURCE = ROOT.parent / ('A43_ECDet三组蒸馏最小验证_20261031_v1/source/'
                       'EdgeCrafter-d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/'
                       'ecdetseg/engine/edgecrafter')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def extract(path, names, namespace):
    tree = ast.parse(path.read_text('utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def main():
    out = ROOT / '02_CPU算子核验_v1.json'
    assert not out.exists()
    torch.set_num_threads(2)
    torch.manual_seed(20261112)
    namespace = dict(torch=torch, nn=nn, F=F, init=init, math=math,
                     functools=functools, List=List)
    extract(SOURCE / 'utils.py', ['deformable_attention_core_func_v2'], namespace)
    extract(SOURCE / 'decoder.py', ['MSDeformableAttention'], namespace)
    cls = namespace['MSDeformableAttention']
    base = nn.Module()
    base.decoder = nn.Module()
    base.decoder.decoder = nn.Module()
    base.decoder.decoder.layers = nn.ModuleList([])
    for _ in range(4):
        layer = nn.Module()
        layer.cross_attn = cls(192, 8, 3, [3, 6, 3])
        base.decoder.decoder.layers.append(layer)
    models = {a: copy.deepcopy(base) for a in ['B', 'M', 'Q']}
    original = {k: v.clone() for k, v in base.state_dict().items()}
    branches = {}
    rng = torch.random.get_rng_state().clone()
    for arm in ['M', 'Q']:
        branches[arm] = install(models[arm], arm)
        assert torch.equal(torch.random.get_rng_state(), rng)
        assert all(torch.equal(v, models[arm].state_dict()[k]) for k, v in original.items())
    mstate = {k: v for k, v in models['M'].state_dict().items() if 'a56_contrast' in k}
    qstate = {k: v for k, v in models['Q'].state_dict().items() if 'a56_contrast' in k}
    assert mstate.keys() == qstate.keys()
    assert all(torch.equal(v, qstate[k]) for k, v in mstate.items())
    extra = sum(p.numel() for p in models['M'].parameters()) - sum(p.numel() for p in base.parameters())
    assert extra == 74144
    bs, nq, heads, channels = 2, 7, 8, 24
    shapes = [(11, 13), (6, 7), (3, 4)]
    values = [torch.randn(bs, heads, channels, h * w) for h, w in shapes]
    query = torch.randn(bs, nq, 192)
    refs = torch.rand(bs, nq, 1, 4)
    refs[..., :2] = refs[..., :2] * 1.4 - .2
    refs[..., 2:] = .8
    outputs = {a: models[a].decoder.decoder.layers[0].cross_attn(query, refs, values, shapes)
               for a in ['B', 'M', 'Q']}
    assert torch.equal(outputs['B'], outputs['M']) and torch.equal(outputs['B'], outputs['Q'])
    native_calls = 3
    positive = torch.randn(bs, nq, heads, 12)
    negative = torch.randn_like(positive)
    weights = {a: branches[a][0].signed_weights(positive, negative) for a in ['M', 'Q']}
    per_level = {a: [float(x.sum(-1).abs().max()) for x in w.split([3, 6, 3], -1)]
                 for a, w in weights.items()}
    assert max(per_level['Q']) < 2e-7
    assert max(per_level['M']) > .05
    shifted = [v + float(i + 1) * 3.0 for i, v in enumerate(values)]
    shifts = {}
    for arm in ['M', 'Q']:
        attn = models[arm].decoder.decoder.layers[0].cross_attn
        branch = branches[arm][0]
        with torch.no_grad():
            branch.gain.fill_(.3)
            before = branch.residual(query, attn, refs, values, shapes)
            after = branch.residual(query, attn, refs, shifted, shapes)
            shifts[arm] = float((after - before).abs().max())
            branch.gain.zero_()
    assert shifts['Q'] < 3e-6 and shifts['M'] > .01
    gradient_rows = []
    component_steps = 0
    target = torch.randn_like(outputs['B'])
    for arm in ['M', 'Q']:
        attn = models[arm].decoder.decoder.layers[0].cross_attn
        branch = branches[arm][0]
        optimizer = torch.optim.SGD(branch.parameters(), lr=.05)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = attn(query, refs, values, shapes)
            native_calls += 1
            loss = (output - target).square().mean()
            loss.backward()
            gate_grad = float(branch.gain.grad.norm())
            beta_grad = float(branch.negative_logits.weight.grad.norm())
            assert math.isfinite(gate_grad) and math.isfinite(beta_grad)
            assert gate_grad > 0
            if step == 0:
                assert beta_grad == 0
            else:
                assert beta_grad > 0
            optimizer.step()
            component_steps += 1
            gradient_rows.append({'arm': arm, 'step': step + 1,
                                  'loss': float(loss.detach()), 'gate_grad_norm': gate_grad,
                                  'negative_weight_grad_norm': beta_grad})
    result = {'status': 'PASS', 'scope': 'synthetic CPU attention components only',
              'extra_parameters_each_arm': extra, 'native_attention_calls': native_calls,
              'synthetic_component_optimizer_steps': component_steps,
              'whole_detector_forwards': 0, 'detector_optimizer_updates': 0,
              'real_images': 0, 'initial_output_bitwise_equal': True,
              'native_tensors_unchanged_at_install': True, 'native_rng_unchanged_at_install': True,
              'M_Q_new_initial_tensors_equal': True, 'max_level_signed_mass': per_level,
              'fixed_query_level_offset_residual_max_change': shifts,
              'boundary_test': 'non-square maps; many samples outside the feature extent; border residual in both arms',
              'gradient_checks': gradient_rows,
              'no_task_effect_or_novelty_demonstrated': True,
              'source_sha256': {p.name: sha(p) for p in [SOURCE / 'decoder.py', SOURCE / 'utils.py']},
              'implementation_sha256': sha(ROOT / 'contrastive_read.py'),
              'check_script_sha256': sha(Path(__file__))}
    assert native_calls == 7 and component_steps == 4
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
