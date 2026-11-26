"""Three freshly initialized native CPU forwards. No checkpoint or optimizer."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import gc
import hashlib
import json
from pathlib import Path
import sys
from experiment import HERE, verify_bundle
from runtime import activate_training_snapshot, register_components, implementation_hashes
from evaluator_binding import activate, loaded_fingerprint

# Only this engineering process uses these supplementary import dependencies.
sys.path.insert(0, str(HERE/'native_import_deps'))
import torch

attempts = []
def forbidden(name):
    def block(*args, **kwargs):
        attempts.append(name)
        raise RuntimeError('Fresh CPU probe forbids: ' + name)
    return block
torch.load = forbidden('checkpoint_load')
torch.hub.load_state_dict_from_url = forbidden('weight_download')
torch.optim.Optimizer.__init__ = forbidden('optimizer_construction')
torch.cuda._lazy_init = forbidden('CUDA_initialization')
torch.set_num_threads(4)


def state_hashes(model):
    return {name: hashlib.sha256(str((tuple(value.shape), value.dtype)).encode() +
                   value.detach().cpu().numpy().tobytes()).hexdigest()
            for name, value in model.state_dict().items()}


def main():
    out = HERE/'engineering_native_models'
    out.mkdir(exist_ok=False)
    receipt, _, paths = verify_bundle(HERE/'prepared_local/baseline_seed1')
    activate_training_snapshot(paths['native_root'])
    activate(paths['evaluator_root'], receipt['identity']['evaluator_binding'])
    from engine.core import YAMLConfig
    register_components()
    sample = torch.linspace(0, 1, 3*640*640).reshape(1, 3, 640, 640)
    checks = []
    reports = []
    baseline_state = baseline_output = None
    for arm in ('baseline', 'fused_residual', 'fine_residual'):
        bundle = HERE/'prepared_local'/f'{arm}_seed1'
        _, plan, _ = verify_bundle(bundle)
        assert plan['model_recipe']['HGNetv2']['pretrained'] is False
        cfg = YAMLConfig(str(bundle/'resolved.yml'))
        torch.manual_seed(810218)
        model = cfg.model.cpu().eval()
        assert model.geometry_arm == arm
        assert cfg._optimizer is None and cfg._ema is None and cfg._scaler is None
        initial = state_hashes(model)
        rng = torch.get_rng_state().clone()
        with torch.inference_mode():
            output = model(sample)
        assert set(output) == {'pred_boxes', 'pred_logits'}
        assert output['pred_boxes'].shape == (1, 300, 4)
        assert output['pred_logits'].shape == (1, 300, 10)
        assert all(torch.isfinite(v).all() for v in output.values())
        checks.append(arm + ': native YAML construction and 640 CPU output shapes valid')
        assert initial == state_hashes(model)
        checks.append(arm + ': parameters and persistent buffers unchanged after forward')
        assert torch.equal(rng, torch.get_rng_state())
        checks.append(arm + ': eval forward consumes no RNG')
        if arm == 'baseline':
            baseline_state, baseline_output = initial, output
        else:
            assert all(initial.get(k) == h for k, h in baseline_state.items())
            expected = {'decoder.geometry_residual.' + name for name in
                        ('kernel', 'mix_weight', 'mix_bias', 'output_weight', 'output_bias')}
            assert set(initial) - set(baseline_state) == expected
            checks.append(arm + ': native initial state identical except five new tensors')
            assert all(torch.equal(output[k], baseline_output[k]) for k in output)
            checks.append(arm + ': initial outputs bitwise identical to native baseline')
        reports.append({'arm':arm, 'state_sha256':hashlib.sha256(
            json.dumps(initial,sort_keys=True).encode()).hexdigest(),
            'parameter_count':sum(p.numel() for p in model.parameters())})
        del model, cfg, output
        gc.collect()
    assert not attempts and not torch.cuda.is_initialized()
    loaded_fingerprint(receipt['identity']['evaluator_binding'], paths['evaluator_root'])
    report = {'status':'PASS_FRESH_NATIVE_CPU_MODEL_INTERFACE_ONLY',
        'checks': checks, 'check_count':len(checks), 'arms':reports,
        'full_fresh_detector_forwards':3, 'trained_checkpoint_forwards':0,
        'gradient_calls':0, 'optimizer_constructions':0, 'checkpoint_loads':0,
        'parameter_updates_after_initialization':0, 'CUDA_initialized':False,
        'guard_attempts':attempts, 'training_started':False,
        'engine_root':str(paths['native_root']), 'implementation':implementation_hashes(),
        'test_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'historical_baseline_reproduction_verified':False,
        'actual_optimizer_EMA_AMP_training_verified':False}
    with (out/'report.json').open('x',encoding='utf-8') as handle:
        json.dump(report,handle,ensure_ascii=False,indent=2)
    print(json.dumps({'status':report['status'],'checks':len(checks),'fresh_forwards':3}))


if __name__ == '__main__':
    main()
