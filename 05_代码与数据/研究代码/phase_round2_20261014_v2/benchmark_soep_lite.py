"""Same-GPU FP32 cost audit, common-init equivalence, and optimizer checks."""
import gc
import json
import pathlib
import sys
import torch
import numpy as np

torch.set_num_threads(4)
sys.path.insert(0, str(pathlib.Path.cwd()))
import engine.deim.soep_lite_round2  # noqa: F401
from engine.core import YAMLConfig
from calflops import calculate_flops

phase = pathlib.Path.cwd().parent
addons = pathlib.Path(__file__).resolve().parent
destination = addons/'soep_cost_audit.json'
assert not destination.exists(), 'Refuse to replace an existing cost audit'
common = torch.load(phase/'artifacts/common_init_seed0.pth', map_location='cpu', weights_only=False)
common = common['model'] if 'model' in common else common
sample = torch.rand(1, 3, 640, 640, device='cuda')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
results = []
baseline_outputs = None
for arm in ['baseline_full96', 'c_full96', 'soep_lite_full96']:
    cfg = YAMLConfig('configs/experiments/phase/'+arm+'.yml')
    model = cfg.model.eval().cuda()
    loading = model.load_state_dict(common, strict=False)
    allowed = 'encoder.p2_fusion.' if arm == 'soep_lite_full96' else 'backbone.p2_skip.'
    assert not loading.unexpected_keys
    assert all(key.startswith(allowed) for key in loading.missing_keys)
    with torch.inference_mode():
        output = model(sample)
        if baseline_outputs is None:
            baseline_outputs = {key: value.detach().clone() for key, value in output.items()}
        else:
            for key, value in output.items():
                torch.testing.assert_close(value, baseline_outputs[key], rtol=1e-5, atol=1e-6)
        for _ in range(30):
            model(sample)
        torch.cuda.synchronize()
        timings = []
        for _ in range(100):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); model(sample); end.record(); end.synchronize()
            timings.append(start.elapsed_time(end))
        flops, macs, _ = calculate_flops(model=model, input_shape=None, args=[sample], kwargs={},
                                        output_as_string=False, print_results=False, print_detailed=False)
    row = {'arm': arm, 'parameters': sum(p.numel() for p in model.parameters()),
           'calflops_FLOPs': int(flops), 'calflops_MACs': int(macs), 'latency_ms_mean': float(np.mean(timings)),
           'latency_ms_median': float(np.median(timings)), 'latency_ms_p95': float(np.quantile(timings, .95)),
           'initial_output_equals_baseline': True, 'missing_common_init_keys': loading.missing_keys}
    if arm == 'soep_lite_full96':
        branch = model.encoder.p2_fusion
        row['branch_parameters'] = sum(p.numel() for p in branch.parameters())
        assert row['branch_parameters'] == 15424
        groups = cfg.get_optim_params(cfg.yaml_cfg['optimizer'], model)
        groups = [{**group, 'params': list(group['params'])} for group in groups]
        row['branch_optimizer_groups'] = []
        for name, parameter in model.named_parameters():
            if name.startswith('encoder.p2_fusion.'):
                matches = [group for group in groups if any(parameter is item for item in group['params'])]
                assert len(matches) == 1, ('ambiguous optimizer group', name)
                group = matches[0]
                assert group.get('lr', cfg.yaml_cfg['optimizer']['lr']) == .0001
                row['branch_optimizer_groups'].append({'name': name, 'lr': group['lr'],
                                                       'weight_decay': group.get('weight_decay', cfg.yaml_cfg['optimizer']['weight_decay'])})
    results.append(row)
    print(json.dumps(row), flush=True)
    del model, cfg, output
    if arm == 'soep_lite_full96':
        del branch, groups
    gc.collect(); torch.cuda.empty_cache()
report = {'device': torch.cuda.get_device_name(), 'batch': 1, 'input': [1,3,640,640], 'dtype': 'FP32',
          'tf32': False, 'warmup': 30, 'iterations': 100, 'deploy_fusion': False,
          'profiler': 'calflops; same operator coverage for all arms', 'results': results}
with destination.open('x') as output:
    json.dump(report, output, indent=2)
print('SOEP_LITE_COST_AUDIT_PASS', flush=True)
