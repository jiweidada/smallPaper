"""Compare saved pre-intervention checkpoints on CPU; never constructs a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def compare(base, candidate, output):
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    base, candidate, output = map(Path, (base, candidate, output))
    assert base.name == candidate.name == 'checkpoint0003.pth'
    assert not output.exists()
    before = [(p.stat().st_size, p.stat().st_mtime_ns, sha(p)) for p in (base, candidate)]
    started = time.monotonic()
    a = torch.load(base, map_location='cpu', weights_only=False)
    b = torch.load(candidate, map_location='cpu', weights_only=False)
    assert a['last_epoch'] == b['last_epoch'] == 3
    assert b['scale_protection12_identity']['arm'] == 'target_protect'
    def states(left, right):
        assert left.keys() == right.keys() and len(left) == 794
        rows, families = {}, {}
        for name, x in left.items():
            y = right[name]
            assert x.device.type == y.device.type == 'cpu'
            assert x.shape == y.shape and x.dtype == y.dtype
            exact = torch.equal(x, y)
            d, v = y.double()-x.double(), x.double()
            delta_sq, base_sq = float(d.square().sum()), float(v.square().sum())
            row = dict(exact=exact, dtype=str(x.dtype), elements=x.numel(),
                       max_abs=float(d.abs().max()), delta_squared=delta_sq, base_squared=base_sq)
            rows[name] = row
            group = name.split('.', 1)[0]
            z = families.setdefault(group, dict(tensors=0, unequal_tensors=0, delta_squared=0., base_squared=0.))
            z['tensors'] += 1; z['unequal_tensors'] += int(not exact)
            z['delta_squared'] += delta_sq; z['base_squared'] += base_sq
        for z in families.values():
            z['relative_L2'] = (z['delta_squared']/max(z['base_squared'], 1e-30))**.5
        return dict(tensors=len(rows), unequal_tensors=sum(not v['exact'] for v in rows.values()),
                    families=families, tensor_differences=rows)
    model, ema = states(a['model'], b['model']), states(a['ema']['module'], b['ema']['module'])
    def steps(payload):
        opt = payload['optimizer']
        return dict(groups=opt['param_groups'],
                    state_steps={str(k):float(v['step']) for k,v in opt['state'].items() if 'step' in v})
    oa, ob = steps(a), steps(b)
    result = dict(schema='scale_protection12_prefix_cpu_v1', epoch_one_based=4,
        seed=b['scale_protection12_identity']['seed'], scale_protection_has_not_started=True,
        baseline_checkpoint_sha256=before[0][2], candidate_checkpoint_sha256=before[1][2],
        model=model, EMA=ema, optimizer_groups_exact=oa['groups']==ob['groups'],
        optimizer_state_steps_exact=oa['state_steps']==ob['state_steps'],
        baseline_optimizer_steps=oa['state_steps'], candidate_optimizer_steps=ob['state_steps'],
        baseline_EMA_updates=a['ema'].get('updates'), candidate_EMA_updates=b['ema'].get('updates'),
        baseline_scaler=a.get('scaler'), candidate_scaler=b.get('scaler'),
        torch=str(torch.__version__), elapsed_seconds=time.monotonic()-started,
        script_sha256=sha(__file__), new_model_forwards=0, new_optimizer_updates=0,
        new_epochs=0, CUDA_initialized=torch.cuda.is_initialized(),
        scope='Saved state divergence before intervention. Does not identify its cause or reject the fixed12 gate.')
    assert not result['CUDA_initialized']
    assert before == [(p.stat().st_size, p.stat().st_mtime_ns, sha(p)) for p in (base, candidate)]
    with output.open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    return dict(seed=result['seed'], model_unequal_tensors=model['unequal_tensors'],
                EMA_unequal_tensors=ema['unequal_tensors'], optimizer_groups_exact=result['optimizer_groups_exact'],
                optimizer_state_steps_exact=result['optimizer_state_steps_exact'], output=str(output))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('baseline','candidate','output'): p.add_argument('--'+name, required=True)
    args = p.parse_args()
    print(json.dumps(compare(args.baseline,args.candidate,args.output)),flush=True)
