"""Describe raw and EMA tensors in the saved baseline checkpoint without model execution."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import json
import math
import torch

PAPER = Path(__file__).resolve().parent.parent
REMOTE = PAPER / '远程公平对照_20261103'
CHECKPOINT = REMOTE / 'remote_evidence/interrupted_BASE_20261105/runs/BASE/best.pt'
AUDIT = REMOTE / 'interrupted_base_saved_file_audit_20261105.json'
REPORT = REMOTE / 'interrupted_base_best_raw_ema_state_20261105.json'

def sha(path):
    h = sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

assert not REPORT.exists()
audit = json.loads(AUDIT.read_text(encoding='utf-8'))
assert audit['status'] == 'BEST_CHECKPOINT_BOUND_LAST_ARRAYS_ONLY'
assert sha(CHECKPOINT) == audit['best_checkpoint_sha256']
torch.set_num_threads(1)
checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
raw, ema = checkpoint['model'], checkpoint['ema']['module']
assert checkpoint['actual_loop_epoch'] == 213 and set(raw) == set(ema)
out = []
nonfinite = {'raw': 0, 'ema': 0}
variance = {'raw_negative': 0, 'ema_negative': 0, 'raw_nonfinite': 0, 'ema_nonfinite': 0,
            'tensor_count': 0}
integer_differences = 0
for key in sorted(raw):
    a, b = raw[key].detach().cpu(), ema[key].detach().cpu()
    assert a.shape == b.shape and a.dtype == b.dtype
    if not a.dtype.is_floating_point:
        integer_differences += not torch.equal(a, b)
        continue
    fa, fb = torch.isfinite(a), torch.isfinite(b)
    nonfinite['raw'] += int((~fa).sum().item())
    nonfinite['ema'] += int((~fb).sum().item())
    if key.endswith('running_var'):
        variance['tensor_count'] += 1
        variance['raw_negative'] += int((a < 0).sum().item())
        variance['ema_negative'] += int((b < 0).sum().item())
        variance['raw_nonfinite'] += int((~fa).sum().item())
        variance['ema_nonfinite'] += int((~fb).sum().item())
    if bool(fa.all()) and bool(fb.all()):
        x, y = a.to(torch.float64), b.to(torch.float64)
        a_norm, b_norm = torch.linalg.vector_norm(x).item(), torch.linalg.vector_norm(y).item()
        difference = torch.linalg.vector_norm(x-y).item()
        relative = difference / max(a_norm, b_norm, 1e-12)
        assert math.isfinite(relative)
        out.append({'key': key, 'elements': a.numel(), 'raw_norm': a_norm,
                    'ema_norm': b_norm, 'relative_difference': relative,
                    'kind': 'running_mean' if key.endswith('running_mean') else
                            'running_var' if key.endswith('running_var') else 'other'})
by_kind = {}
for kind in ['running_mean', 'running_var', 'other']:
    records = [v for v in out if v['kind'] == kind]
    ordered = sorted((v['relative_difference'] for v in records))
    by_kind[kind] = {'tensors': len(records),
                     'median_relative_difference': ordered[len(ordered)//2] if ordered else None,
                     'maximum_relative_difference': max(ordered) if ordered else None,
                     'top_five': sorted(records, key=lambda x: x['relative_difference'], reverse=True)[:5]}
report = {'status': 'BEST_CHECKPOINT_RAW_AND_EMA_TENSORS_DESCRIBED',
          'checked_local_utc': datetime.now(timezone.utc).isoformat(),
          'checkpoint_sha256': sha(CHECKPOINT), 'prior_saved_file_audit_sha256': sha(AUDIT),
          'actual_epoch_zero_based': checkpoint['actual_loop_epoch'],
          'relative_difference_definition': 'L2(raw-EMA)/max(L2(raw),L2(EMA),1e-12), per floating tensor',
          'nonfinite_element_counts': nonfinite, 'running_variance': variance,
          'integer_tensors_with_different_values': integer_differences, 'per_kind': by_kind,
          'limits': ['Raw and EMA tensors are expected to differ; their size alone does not establish an error.',
                     'Only epoch 213 is available; no corresponding epoch-224 model or EMA tensors were saved.',
                     'No model prediction, model execution, evaluator execution or training is performed.'],
          'model_executions': 0, 'evaluator_executions': 0, 'optimizer_updates': 0}
with REPORT.open('x', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({'status': report['status'], 'nonfinite': nonfinite, 'running_variance': variance,
                  'median_relative_differences': {k: v['median_relative_difference'] for k, v in by_kind.items()}}))
