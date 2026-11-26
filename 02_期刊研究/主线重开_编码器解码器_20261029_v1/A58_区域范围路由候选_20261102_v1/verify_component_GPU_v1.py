"""Exactly two synthetic CUDA module forwards/backwards. No detector or optimizer."""
from pathlib import Path
import json
import hashlib
import os
import time
import traceback
import numpy as np
import torch
import regional_range_attention_v1 as r

P = Path(__file__).resolve().parent
COMPLETIONS = 0
BACKWARDS = 0


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def main():
    global COMPLETIONS, BACKWARDS
    binding = json.loads((P / 'GPU_component_inputs_FROZEN_v1.json').read_text())
    for name, expected in binding['files_sha256'].items():
        assert sha(P / name) == expected, name
    with (P / 'GPU_component_once.lock').open('x') as f:
        f.write(str(os.getpid()))
    raw = Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()
    save(P / 'GPU_component_launch.json', {'pid': os.getpid(), 'start_ticks': int(raw[19]),
         'monotonic': time.monotonic(), 'scope_sha256': sha(P / 'GPU_component_inputs_FROZEN_v1.json')})
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fixture = np.load(P / 'synthetic_GPU_fixture_v1.npz', allow_pickle=False)
    q = fixture['route_q']
    k = fixture['route_k']
    qlow, qhigh, qmean = q.min(-2), q.max(-2), q.mean(-2)
    klow, khigh = k.min(-2), k.max(-2)
    a = np.stack([qlow[..., :, None, :] * klow[..., None, :, :],
                  qlow[..., :, None, :] * khigh[..., None, :, :],
                  qhigh[..., :, None, :] * klow[..., None, :, :],
                  qhigh[..., :, None, :] * khigh[..., None, :, :]], axis=0)
    expected_Q = a.max(0).sum(-1) / np.sqrt(q.shape[-1])
    expected_M = np.maximum(qmean[..., :, None, :] * klow[..., None, :, :],
                            qmean[..., :, None, :] * khigh[..., None, :, :]).sum(-1) / np.sqrt(q.shape[-1])
    tensors = [torch.from_numpy(x.copy()).float().cuda() for x in [qlow, qhigh, qmean, klow, khigh]]
    ql, qh, qm, kl, kh = tensors
    out_Q = r.two_sided_interval(ql, qh, kl, kh, chunk=2)
    out_M = r.mean_query_key_interval(qm, kl, kh)
    torch.testing.assert_close(out_Q.cpu(), torch.from_numpy(expected_Q).float(), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(out_M.cpu(), torch.from_numpy(expected_M).float(), rtol=2e-5, atol=2e-6)
    cpu = json.loads((P / 'CPU_component_result_v1.json').read_text())
    state = {key[7:]: torch.from_numpy(fixture[key].copy()) for key in fixture.files if key.startswith('state::')}
    rows = {}
    for mode in ['M', 'Q']:
        module = r.RegionalRangeContext(6, 2, 2, 2, mode).float()
        module.load_state_dict(state, strict=True)
        module = module.cuda()
        x = torch.from_numpy(fixture['x'].copy()).float().cuda().requires_grad_(True)
        output = module(x)
        COMPLETIONS += 1
        assert bool(torch.isfinite(output).all())
        output.square().mean().backward()
        BACKWARDS += 1
        assert bool(torch.isfinite(x.grad).all())
        grads = {key: float(p.grad.abs().max()) for key, p in module.named_parameters() if p.grad is not None}
        expected = cpu['checks']['nonzero_gate_gradients_all_parameters'][mode]
        assert set(grads) == set(expected)
        for key in grads:
            assert np.isfinite(grads[key]) and grads[key] > 0
            assert abs(grads[key] - expected[key]) <= 2e-6 + 2e-4 * abs(expected[key]), (mode, key)
        rows[mode] = {'all_outputs_and_gradients_finite': True,
                      'gradient_maxima_match_CPU_fixture': True, 'parameter_gradient_maxima': grads}
        del output, x, module
    assert r.COUNTS['module_forwards'] == COMPLETIONS == BACKWARDS == 2
    assert r.COUNTS['M_route_calls'] == r.COUNTS['Q_route_calls'] == 2
    save(P / 'GPU_component_result_v1.json', {'status': 'PASS_SYNTHETIC_CUDA_OPERATOR_AND_GRADIENT_CHECKS',
         'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__),
         'route_scores_match_numpy_reference': True, 'rows': rows, 'counters': r.COUNTS,
         'module_completions': COMPLETIONS, 'synthetic_backward_completions': BACKWARDS,
         'full_detector_forwards': 0, 'optimizer_updates': 0, 'AP_evaluations': 0,
         'TF32': False, 'full_model_numeric_equivalence_claimed': False,
         'fixture_sha256': sha(P / 'synthetic_GPU_fixture_v1.npz')})
    save(P / 'GPU_component_exit_v1.json', {'status': 'COMPLETED', 'exit_code': 0,
         'pid': os.getpid(), 'counters': r.COUNTS, 'module_completions': COMPLETIONS,
         'synthetic_backward_completions': BACKWARDS, 'full_detector_forwards': 0, 'optimizer_updates': 0})
    print(json.dumps({'status': 'PASS_SYNTHETIC_CUDA_OPERATOR_AND_GRADIENT_CHECKS',
                      'counters': r.COUNTS, 'completions': COMPLETIONS, 'backwards': BACKWARDS,
                      'full_detector_forwards': 0, 'optimizer_updates': 0}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        save(P / 'GPU_component_failure_v1.json', {'traceback': traceback.format_exc(),
             'counters': r.COUNTS, 'module_completions': COMPLETIONS, 'synthetic_backward_completions': BACKWARDS,
             'full_detector_forwards': 0, 'optimizer_updates': 0})
        raise
