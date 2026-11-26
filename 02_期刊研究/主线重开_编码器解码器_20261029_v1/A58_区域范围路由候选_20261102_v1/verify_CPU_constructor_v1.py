"""One real CPU ECDet-S constructor; guarded against any forward."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import hashlib
import importlib.util
import json
import sys
import time
import traceback

P = Path(__file__).resolve().parent
ASSETS = Path('/root/uav_a47_20261101_v1/assets_A43_baseline')
SOURCE = ASSETS / 'source/EdgeCrafter-d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/ecdetseg'
COUNTS = dict(constructors_attempted=0, constructors_completed=0, whole_detector_forwards=0,
              backwards=0, optimizer_steps=0, AP_evaluations=0)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def save_new(name, obj):
    with (P / name).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)

def run():
    scope = json.loads((P / 'CPU_constructor_scope_FROZEN_v1.json').read_text('utf-8'))
    expected = json.loads((P / 'asset_expectations.json').read_text('utf-8'))
    assert sha(P / 'regional_range_attention_v1.py') == scope['module_sha256']
    assert sha(ASSETS / 'runtime_common.py') == scope['native_runtime_sha256']
    assert sha(ASSETS / 'common_init_10class.pth') == scope['base_initialization_sha256']
    for relative, digest in expected['native_source_sha256'].items():
        assert sha(SOURCE / relative) == digest, relative
    import torch
    assert not torch.cuda.is_initialized()
    sys.path.insert(0, '/root/uav_a47_20261101_v1/pydeps')
    sys.path.insert(0, str(ASSETS))
    sys.path.insert(0, str(P))
    import runtime_common as native
    from regional_range_attention_v1 import install, COUNTS as module_counts
    assert not torch.cuda.is_initialized()
    COUNTS['constructors_attempted'] += 1
    model, criterion, postprocessor = native.create_components('B', role='cpu_check')
    COUNTS['constructors_completed'] += 1
    def no_forward(module, args):
        COUNTS['whole_detector_forwards'] += 1
        raise RuntimeError('No whole-detector forward is allocated in this CPU scope')
    model.register_forward_pre_hook(no_forward)
    base_parameters = sum(p.numel() for p in model.parameters())
    assert base_parameters == scope['expected_base_parameters']
    original = torch.load(ASSETS / 'common_init_10class.pth', map_location='cpu', weights_only=True)['model']
    assert set(model.state_dict()) == set(original)
    for k, v in model.state_dict().items():
        assert torch.equal(v, original[k]), k
    rng_before = torch.random.get_rng_state().clone()
    installed = install(model, 'B')
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert installed['extra_parameters'] == scope['expected_extra_parameters']
    count = sum(p.numel() for p in model.parameters())
    assert count == scope['expected_augmented_parameters']
    state = model.state_dict()
    extra_keys = sorted(set(state) - set(original))
    assert extra_keys and all(k.startswith('encoder.a58_context.') for k in extra_keys)
    for k, v in original.items():
        assert torch.equal(state[k], v), k
    for k, v in model.named_parameters():
        assert torch.isfinite(v).all(), k
    allowed_inf = []
    for k, v in state.items():
        if v.is_floating_point():
            assert not torch.isnan(v).any(), k
            if torch.isinf(v).any():
                assert k == 'decoder.anchors' and k in original and torch.equal(v, original[k]), k
                valid = state['decoder.valid_mask'].bool().expand_as(v)
                assert torch.equal(torch.isposinf(v), ~valid), k
                allowed_inf.append(k)
    # Changing the arm does not create/reinitialize tensors or execute a forward.
    for mode in ['B', 'M', 'Q']:
        model.encoder.a58_context.mode = mode
        other = model.state_dict()
        assert set(other) == set(state)
        assert all(torch.equal(state[k], other[k]) for k in state)
    model.encoder.a58_context.mode = 'B'
    target = P / 'common_init_augmented_v1.pth'
    assert not target.exists()
    torch.save(dict(model=state, meta=dict(phase='A58', seed=20261101,
                    native_init_sha256=scope['base_initialization_sha256'],
                    module_sha256=scope['module_sha256'], mode_is_not_state=True)), target)
    loaded = torch.load(target, map_location='cpu', weights_only=True)['model']
    assert set(loaded) == set(state)
    assert all(torch.equal(loaded[k], state[k]) for k in state)
    assert not torch.cuda.is_initialized()
    assert native.MODEL_COUNTER == {'forwards': 0, 'teacher_forwards': 0}
    assert module_counts == {'module_forwards': 0, 'M_route_calls': 0, 'Q_route_calls': 0}
    return dict(status='PASS_CPU_CONSTRUCTOR_NO_FORWARD', counts=COUNTS,
                torch=torch.__version__, python=sys.executable, CUDA_initialized=False,
                base_parameters=base_parameters, augmented_parameters=count,
                extra_parameters=installed['extra_parameters'], extra_state_keys=extra_keys,
                native_state_tensors_preserved=len(original), all_learned_parameters_finite=True,
                documented_inf_buffers=allowed_inf, B_M_Q_state_identical=True,
                native_CPU_RNG_preserved_by_install=True,
                common_init_augmented=dict(path=str(target), bytes=target.stat().st_size, sha256=sha(target)),
                bindings={n: sha(P / n) for n in ['CPU_constructor_scope_FROZEN_v1.json',
                          'regional_range_attention_v1.py', 'asset_expectations.json',
                          'verify_CPU_constructor_v1.py']},
                limits='Constructor and serialization only; no detection or AP evidence')

if __name__ == '__main__':
    with (P / 'CPU_constructor_start_v1.json').open('x', encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(), counts_at_start=COUNTS), f)
    start = time.monotonic()
    try:
        result = run()
        result['monotonic_seconds'] = time.monotonic() - start
        save_new('CPU_constructor_result_v1.json', result)
        save_new('CPU_constructor_exit_v1.json', dict(exit=0, counts=COUNTS))
        print(json.dumps(result), flush=True)
    except BaseException:
        save_new('CPU_constructor_failure_v1.json', dict(status='FAILED_NO_RETRY',
                    counts=COUNTS, traceback=traceback.format_exc(), monotonic_seconds=time.monotonic()-start))
        save_new('CPU_constructor_exit_v1.json', dict(exit=1, counts=COUNTS))
        raise
