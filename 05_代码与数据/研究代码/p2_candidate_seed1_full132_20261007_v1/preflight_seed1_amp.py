"""One bounded seed1 AMP fixture; its weights are never used for training."""
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
from torch.nn.functional import interpolate
import p2_candidate
from engine.core import YAMLConfig
from screen_runtime import strict_common_init


def main():
    root = Path(sys.argv[1])
    auth = json.loads((root / 'authorization.json').read_text())
    assert auth['human_authorization'] == '两张显卡，你都用起来跑实验'
    assert auth['seed'] == 1 and auth['epoches'] == 132
    torch.set_num_threads(6)
    random.seed(1); np.random.seed(1); torch.manual_seed(1); torch.cuda.manual_seed_all(1)
    cfg = YAMLConfig(str(root / 'p2_full132.yml'))
    assert cfg.yaml_cfg['seed'] == 1
    model = cfg.model
    init = strict_common_init(model, auth['common_init'])
    model.to('cuda:0').train()
    criterion = cfg.criterion.to('cuda:0'); criterion.set_model(model)
    optimizer, scaler = cfg.optimizer, cfg.scaler
    ema = cfg.ema.to('cuda:0')
    loader = cfg.train_dataloader; loader.set_epoch(0)
    images, targets = next(iter(loader))
    images = interpolate(images.to('cuda:0'), size=(800, 800), mode='bilinear', align_corners=False)
    targets = [{k: v.to('cuda:0') if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
    counts = {'updates': 0}
    optimizer.register_step_post_hook(lambda *_: counts.__setitem__('updates', counts['updates'] + 1))
    torch.cuda.reset_peak_memory_stats()
    steps = []
    for i in range(10):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            outputs = model(images, targets=targets)
        with torch.autocast('cuda', enabled=False):
            loss = sum(criterion(outputs, targets, epoch=0, step=i, global_step=i, epoch_step=len(loader)).values())
        assert torch.isfinite(loss)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        g = model.decoder.query_recruitment.project.weight.grad
        grad = float(g.float().norm())
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)
        inf = sum(float(v.item()) for v in scaler._found_inf_per_device(optimizer).values())
        before = counts['updates']; scale = scaler.get_scale()
        scaler.step(optimizer); scaler.update(); ema.update(model)
        updated = counts['updates'] > before
        if updated:
            assert np.isfinite(grad) and grad > 0
        info = dict(model.decoder._p2_last_info)
        assert info['old_count'] == 240 and info['p2_count'] == 60
        assert len(info['spatial_shapes']) == 3
        steps.append({'step': i, 'loss': float(loss), 'scale': scale, 'found_inf': inf,
                      'updated': updated, 'P2_grad_norm': grad if np.isfinite(grad) else None})
        if i >= 2 and counts['updates'] >= 2:
            break
    assert counts['updates'] >= 2 and model.decoder.query_recruitment.project.weight.norm() > 0
    receipt = {'passed': True, 'seed': 1, 'init': init, 'effective_updates': counts['updates'],
               'steps': steps, 'batch_size': len(images), 'image_size': list(images.shape[-2:]),
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
               'P2_info': info, 'GPU_UUID': auth['GPU_UUID'],
               'scope': 'development fixture only; formal trainer restarts from common initialization'}
    (root / 'full132_amp_preflight.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
