"""Synthetic module-only checks; no native model, optimizer or training step.

Run manually with --device cpu or --device cuda. CUDA also checks AMP. Emits a
JSON receipt to stdout, and an explicit FAILED receipt on any assertion error.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
import torch
from band_attention import P3BandAttention


def require(ok, text):
    if not ok:
        raise AssertionError(text)


def state_hash(module):
    result = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def main(device):
    torch.set_num_threads(4)
    if device == 'cuda':
        require(torch.cuda.is_available(), 'CUDA requested but unavailable; no fallback')
    cpu_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
    wavelet, spatial = P3BandAttention('wavelet'), P3BandAttention('spatial')
    require(torch.equal(cpu_before, torch.get_rng_state()), 'Constructors changed CPU RNG')
    if cuda_before is not None:
        require(torch.equal(cuda_before, torch.cuda.get_rng_state()), 'Constructors changed CUDA RNG')
    require(set(wavelet.state_dict()) == set(spatial.state_dict()), 'Different parameter keys')
    require(all(torch.equal(v, spatial.state_dict()[k]) for k,v in wavelet.state_dict().items()), 'Different initialization')
    for model in (wavelet, spatial):
        require(sum(p.numel() for p in model.parameters()) == 19232, 'Wrong parameter count')
        require(not list(model.buffers()), 'Unexpected persistent transform buffers')
        for key in model.identity_zero_parameters:
            require(torch.count_nonzero(model.state_dict()[key]).item() == 0, 'Nonzero identity parameter ' + key)
    initial_hash = state_hash(wavelet)
    models = {'wavelet':wavelet.to(device), 'spatial':spatial.to(device)}
    generator = torch.Generator(device='cpu').manual_seed(271828)
    identity, reconstruction = [], []
    for shape in ((2, 256, 8, 10), (2, 256, 7, 9), (1, 256, 1, 1), (1, 256, 2, 3)):
        original = torch.randn(shape, generator=generator).to(device)
        for arm, model in models.items():
            z, hw = model.decompose(original)
            recovered = model.reconstruct(z, hw)
            error = float((recovered-original).abs().max())
            require(torch.allclose(recovered, original, atol=1e-6, rtol=1e-6), 'Inverse reconstruction ' + arm)
            reconstruction.append(dict(arm=arm, shape=list(shape), max_abs_error=error))
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                x = original.to(dtype).detach().requires_grad_(True)
                y = model(x)
                require(y.dtype == x.dtype and y.shape == x.shape and torch.equal(y, x), 'Identity ' + arm + str(dtype))
                gradient, = torch.autograd.grad(y.float().sum(), x)
                require(torch.equal(gradient, torch.ones_like(x)), 'Identity input gradient')
                identity.append(dict(arm=arm, shape=list(shape), dtype=str(dtype), exact=True, input_gradient_exact=True))
    # Fixed known phases catch a mutually inverse but incorrectly labelled Haar
    # transform. Values are exactly representable, so this assertion is exact.
    phase_image = torch.tensor([[1., 2.], [3., 4.]], device=device).reshape(1, 1, 2, 2).expand(1, 256, 2, 2)
    hz, _ = models['wavelet'].decompose(phase_image)
    pz, _ = models['spatial'].decompose(phase_image)
    require(torch.equal(hz[0, :, 0, 0, 0], torch.tensor([5., 1., 2., 0.], device=device)), 'Haar basis/order')
    require(torch.equal(pz[0, :, 0, 0, 0], torch.tensor([1., 2., 3., 4.], device=device)), 'Spatial phase/order')
    for model in models.values():
        require(state_hash(model) == initial_hash, 'Identity checks mutated parameters')
    # A deterministic, explicitly synthetic manual opening (NOT an optimizer
    # step) verifies that the initially zero adapter can affect X and all gate
    # parameter groups receive finite, nonzero gradients afterwards.
    gradients = []
    for arm, model in models.items():
        with torch.no_grad():
            model.local.weight[:, :, 1, 1].fill_(.03)
            model.global_expand.weight.fill_(.02)
        opened_hash = state_hash(model)
        x = torch.randn((2, 256, 7, 9), generator=generator).to(device).requires_grad_(True)
        target = torch.randn(x.shape, generator=generator).to(device)
        y = model(x)
        require(torch.isfinite(y).all().item() and not torch.equal(y, x), 'Opened adapter ineffective/nonfinite')
        params = list(model.named_parameters())
        gs = torch.autograd.grad((y*target).mean(), [x] + [p for _,p in params])
        norms = {name:float(g.float().norm()) for name,g in zip(['input'] + [n for n,_ in params], gs)}
        require(all(torch.isfinite(g).all().item() for g in gs), 'Nonfinite opened gradient')
        require(all(value > 0 for value in norms.values()), 'Zero opened parameter-group gradient ' + str(norms))
        require(state_hash(model) == opened_hash, 'Backward mutated synthetic parameters')
        gradients.append(dict(arm=arm, gradient_norms=norms, parameter_updates=0,
                              manual_fixture_parameter_change=True))
    amp = {'status':'NOT_REQUESTED_CPU_RUN'}
    if device == 'cuda':
        amp_results = []
        for arm, model in models.items():
            fresh = P3BandAttention(arm).to(device)
            x = torch.randn((2,256,7,9), generator=generator).to(device=device, dtype=torch.float16).requires_grad_(True)
            with torch.autocast('cuda', dtype=torch.float16):
                initial_y = fresh(x)
            require(torch.equal(initial_y, x) and initial_y.dtype == x.dtype, 'AMP initial identity')
            with torch.no_grad():
                fresh.local.weight[:, :, 1, 1].fill_(.03)
                fresh.global_expand.weight.fill_(.02)
            before = state_hash(fresh)
            with torch.autocast('cuda', dtype=torch.float16):
                y = fresh(x)
                loss = y.float().square().mean()
            # Native default scale, backward only. No optimizer or scaler step.
            scaler = torch.amp.GradScaler('cuda')
            scaler.scale(loss).backward()
            require(torch.isfinite(y).all().item(), 'AMP nonfinite output')
            require(x.grad is not None and torch.isfinite(x.grad).all().item(), 'AMP nonfinite input gradient')
            require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in fresh.parameters()), 'AMP nonfinite gate gradient')
            require(state_hash(fresh) == before, 'AMP backward changed parameters')
            amp_results.append(dict(arm=arm, scaler_scale=float(scaler.get_scale()),
                                    output_dtype=str(y.dtype), finite_scaled_gradients=True,
                                    optimizer_constructed=False, optimizer_steps=0))
        amp = dict(status='PASS_SYNTHETIC_AMP_BACKWARD_NO_OPTIMIZER', results=amp_results)
    return dict(schema='p3_band_attention_module_preflight_v1', status='PASS_SYNTHETIC_MODULE_ZERO_OPTIMIZER_STEPS',
        device=device, torch_version=str(torch.__version__), trainable_parameters=19232,
        initial_state_sha256=initial_hash, arms_initial_parameters_exact=True,
        constructor_CPU_RNG_unchanged=True, constructor_CUDA_RNG_unchanged=True if device=='cuda' else None,
        reconstruction=reconstruction, zero_identity=identity, opened_fixture_gradients=gradients, amp=amp,
        optimizer_constructed=False, optimizer_steps=0, native_model_forward_calls=0,
        scope='Synthetic module checks only; manual nonzero fixture is not training or evidence of AP improvement')


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    try:
        result = main(args.device)
    except Exception as exc:
        print(json.dumps(dict(status='FAILED', device=args.device, error=str(exc),
                              traceback=traceback.format_exc(), optimizer_steps=0)))
        raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
