"""Real registry/model forward check; run only when the GPU is idle."""
import gc
import json
import pathlib
import sys
import torch

torch.set_num_threads(4)
sys.path.insert(0, str(pathlib.Path.cwd()))
from engine.core import YAMLConfig

results = []
for name, expected in [('c_full96', 17440), ('c_mean_full96', 17440),
                       ('rank16_full96', 6176), ('rank16_mean_full96', 6176),
                       ('lowpass_full96', 17376)]:
    torch.manual_seed(37)
    cfg = YAMLConfig('configs/experiments/phase/' + name + '.yml')
    model = cfg.model.eval().cuda()
    branch = model.backbone.p2_skip
    count = sum(p.numel() for p in branch.parameters())
    assert count == expected, (name, count, expected)
    with torch.inference_mode():
        outputs = model(torch.rand(1, 3, 640, 640, device='cuda'))
    assert all(torch.isfinite(v).all() for v in outputs.values() if isinstance(v, torch.Tensor))
    results.append({'arm': name, 'branch': type(branch).__name__, 'branch_parameters': count,
                    'total_parameters': sum(p.numel() for p in model.parameters()),
                    'outputs': {k: list(v.shape) for k, v in outputs.items() if isinstance(v, torch.Tensor)}})
    print(json.dumps(results[-1]), flush=True)
    del model, cfg, outputs, branch
    gc.collect()
    torch.cuda.empty_cache()
print('ROUND2_GPU_PREFLIGHT_PASS', flush=True)
