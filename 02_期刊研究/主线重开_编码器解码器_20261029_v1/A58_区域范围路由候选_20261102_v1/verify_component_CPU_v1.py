"""Bound, padding, all-region equivalence and gradient checks; zero detector calls."""
from pathlib import Path
import sys
import json
import math
import hashlib
import copy

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parent.parent / 'code/dn_pre_frozen_diagnostic_20261020_v1/runtime'))
import torch
import regional_range_attention_v1 as r


def save(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def main():
    scope = json.loads((P / 'component_scope_FROZEN_v1.json').read_text('utf-8'))
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    torch.random.default_generator.manual_seed(20261102)
    results = {}
    # Compare a regional bound against the actual maximum of ALL fine dot products.
    q = torch.randn(2, 2, 5, 3, 7, dtype=torch.float64)
    k = torch.randn(2, 2, 6, 4, 7, dtype=torch.float64)
    qvalid = torch.ones(5, 3, dtype=torch.bool)
    kvalid = torch.ones(6, 4, dtype=torch.bool)
    qm, ql, qh = r.regional_stats(q, qvalid)
    _, kl, kh = r.regional_stats(k, kvalid)
    bound = r.two_sided_interval(ql, qh, kl, kh, chunk=2)
    actual = torch.einsum('bhird,bhjsd->bhijrs', q, k).flatten(-2).amax(-1) / math.sqrt(7)
    assert bool((bound + 1e-12 >= actual).all())
    results['bound_covers_all_fine_pairs'] = True
    point_bound = r.mean_query_key_interval(qm, kl, kh)
    actual_point = torch.einsum('bhid,bhjsd->bhijs', qm, k).amax(-1) / math.sqrt(7)
    assert bool((point_bound + 1e-12 >= actual_point).all())
    assert bool((bound + 1e-12 >= point_bound).all())
    results['M_only_bounds_mean_query'] = True
    fixed_q = qm.unsqueeze(-2).expand(-1, -1, -1, 3, -1)
    _, ql0, qh0 = r.regional_stats(fixed_q, qvalid)
    zero_query_range = r.two_sided_interval(ql0, qh0, kl, kh, chunk=3)
    torch.testing.assert_close(zero_query_range, point_bound, rtol=1e-12, atol=1e-12)
    results['Q_reduces_to_M_for_constant_query_regions'] = True
    # A constructed change of rankings is not UAV evidence.
    qr = torch.tensor([[[[[10., 1.], [-10., 1.]]]]], dtype=torch.float64)
    kr = torch.tensor([[[[[1., 0.]], [[0., 2.]]]]], dtype=torch.float64)
    qmean, qmin, qmax = r.regional_stats(qr, torch.ones(1, 2, dtype=torch.bool))
    _, kmin, kmax = r.regional_stats(kr, torch.ones(2, 1, dtype=torch.bool))
    sm = r.mean_query_key_interval(qmean, kmin, kmax)
    sq = r.two_sided_interval(qmin, qmax, kmin, kmax)
    assert int(sm.argmax(-1)) == 1 and int(sq.argmax(-1)) == 0
    results['constructed_minority_query_ranking_difference'] = {'M': sm.tolist(), 'Q': sq.tolist(), 'scope': 'synthetic only, not task harm or AP'}
    x = torch.randn(2, 6, 5, 7, dtype=torch.float64)
    packed, valid, layout = r.pack_regions(x, 2, 2)
    torch.testing.assert_close(r.unpack_regions(packed, layout, 2), x, rtol=0, atol=0)
    assert int(valid.sum()) == 35 and packed.shape[2:4] == (12, 4)
    # Padding values are altered by a huge finite number and must not affect stats.
    contaminated = torch.where(valid[None, None, :, :, None], packed, packed.new_tensor(1e6))
    s1 = r.regional_stats(packed, valid)
    s2 = r.regional_stats(contaminated, valid)
    assert all(torch.equal(a, b) for a, b in zip(s1, s2))
    results['packing_inverse_and_padding_exclusion'] = True
    b = r.RegionalRangeContext(6, 2, 2, 2, 'B').double()
    m = r.RegionalRangeContext(6, 2, 2, 2, 'M').double()
    qmod = r.RegionalRangeContext(6, 2, 2, 2, 'Q').double()
    qmod.load_state_dict(m.state_dict(), strict=True)
    assert torch.equal(b(x), x)
    assert torch.equal(m(x), x) and torch.equal(qmod(x), x)
    results['zero_gate_preserves_native_tensor_values_B_M_Q'] = True
    # With every region retained, both routes must be ordinary dense attention.
    m.topk = qmod.topk = 100
    with torch.no_grad():
        m.gate.fill_(.3)
        qmod.gate.fill_(.3)
    y_m, y_q = m(x), qmod(x)
    z = m.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
    q0, k0, v0 = m.qkv(z).chunk(3, 1)
    batch, _, h, w = q0.shape
    qd, kd, vd = [v.reshape(batch, 2, 3, h * w).transpose(-2, -1) for v in [q0, k0, v0]]
    dense = (qd @ kd.transpose(-2, -1) / math.sqrt(3)).softmax(-1) @ vd
    dense = dense.transpose(-2, -1).reshape(batch, 6, h, w)
    expected = x + .3 * m.proj(dense)
    torch.testing.assert_close(y_m, expected, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(y_q, expected, rtol=1e-11, atol=1e-11)
    results['all_regions_equal_dense_attention_with_odd_shape'] = True
    gradients = {}
    for label, module in [('M', m), ('Q', qmod)]:
        module.topk = 2
        module.zero_grad(set_to_none=True)
        xi = x.detach().clone().requires_grad_(True)
        module(xi).square().mean().backward()
        assert bool(torch.isfinite(xi.grad).all())
        values = {n: float(p.grad.abs().max()) for n, p in module.named_parameters() if p.grad is not None}
        assert len(values) == len(list(module.parameters()))
        assert all(math.isfinite(v) and v > 0 for v in values.values())
        gradients[label] = values
    results['nonzero_gate_gradients_all_parameters'] = gradients
    # Verify installation adds identical tensors for all modes without RNG drift.
    class StubEncoder(torch.nn.Module):
        hidden_dim = 192
        def forward(self, features):
            return features
    states = []
    for mode in ['B', 'M', 'Q']:
        holder = torch.nn.Module()
        holder.encoder = StubEncoder()
        before = torch.get_rng_state().clone()
        site = r.install(holder, mode)
        assert torch.equal(before, torch.get_rng_state())
        assert site['extra_parameters'] == 148609
        states.append(copy.deepcopy(holder.state_dict()))
    assert all(states[0].keys() == s.keys() and all(torch.equal(v, s[k]) for k, v in states[0].items()) for s in states[1:])
    results['identical_B_M_Q_new_parameters_and_CPU_RNG_preserved'] = True
    assert r.COUNTS['module_forwards'] == scope['module_forward_cap_CPU'] == 7
    assert sum(r.COUNTS[k] for k in ['M_route_calls', 'Q_route_calls']) <= scope['standalone_route_score_call_cap_CPU']
    assert not torch.cuda.is_initialized()
    out = {'status': 'PASS_SYNTHETIC_OPERATOR_AND_GRADIENT_CHECKS', 'checks': results,
           'counters': r.COUNTS, 'synthetic_backward_calls': 2,
           'full_detector_forwards': 0, 'optimizer_updates': 0, 'AP_evaluations': 0,
           'CUDA_initialized': False, 'new_full_detector_instances': 0,
           'new_component_parameters_at192channels': 148609,
           'source_sha256': hashlib.sha256((P / 'regional_range_attention_v1.py').read_bytes()).hexdigest(),
           'scope_sha256': hashlib.sha256((P / 'component_scope_FROZEN_v1.json').read_bytes()).hexdigest(),
           'claims_excluded': ['task benefit', 'novelty clearance', '24GB detector capacity', 'formal training success']}
    save(P / 'CPU_component_result_v1.json', out)
    print(json.dumps({'status': out['status'], 'counters': out['counters'], 'synthetic_backward_calls': 2,
                      'new_parameters': 148609, 'full_detector_forwards': 0, 'optimizer_updates': 0}))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        save(P / 'CPU_component_failure_v1.json', {'traceback': traceback.format_exc(), 'counters': r.COUNTS,
             'full_detector_forwards': 0, 'optimizer_updates': 0, 'AP_evaluations': 0})
        raise
