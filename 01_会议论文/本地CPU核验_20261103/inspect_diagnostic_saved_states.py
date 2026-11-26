"""Inspect saved state tensors on CPU; no model construction or evaluation."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import math
import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import torch

PAPER = Path(__file__).resolve().parent.parent
REMOTE = PAPER / '远程公平对照_20261103'
sys.path.insert(0, str(REMOTE / '诊断准备_20261105'))
import backup_guard as guard

ROOT = REMOTE / 'diagnostic_backups_20261105' / guard.RUN_ID
ORIGINAL = REMOTE / 'remote_evidence/interrupted_BASE_20261105/runs/BASE/best.pt'

def all_tensors(value, prefix=''):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from all_tensors(item, prefix + '/' + str(key))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from all_tensors(item, prefix + '/' + str(index))

def inspect_group(tensors):
    count = elements = nonfinite = negative_variances = 0
    maximum = 0.0
    variance_names = 0
    for name, tensor in tensors:
        if not (tensor.is_floating_point() or tensor.is_complex()):
            continue
        count += 1
        elements += tensor.numel()
        nonfinite += int((~torch.isfinite(tensor)).sum().item())
        if tensor.numel():
            finite = tensor[torch.isfinite(tensor)]
            if finite.numel():
                maximum = max(maximum, float(finite.abs().max().item()))
        if name.endswith('running_var'):
            variance_names += 1
            negative_variances += int((tensor < 0).sum().item())
    return {'floating_tensors': count, 'floating_elements': elements,
            'nonfinite_elements': nonfinite, 'max_abs_finite_value': maximum,
            'running_variance_tensors': variance_names, 'negative_running_variance_elements': negative_variances}

def compare_raw_ema(raw, ema):
    assert set(raw) == set(ema)
    diff_sq = ema_sq = 0.0
    for key in raw:
        a, b = raw[key], ema[key]
        assert a.shape == b.shape and a.dtype == b.dtype
        if a.is_floating_point() and a.numel():
            a64 = a.to(torch.float64)
            b64 = b.to(torch.float64)
            diff_sq += float(torch.sum((a64 - b64) ** 2).item())
            ema_sq += float(torch.sum(b64 ** 2).item())
    value = math.sqrt(diff_sq / ema_sq) if ema_sq else None
    return {'combined_raw_minus_ema_l2_over_ema_l2': value,
            'note': 'Includes floating parameters and buffers. A nonzero value is expected and does not by itself identify a fault.'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['early', 'final'], required=True)
    args = parser.parse_args()
    report_path = REMOTE / ('diagnostic_saved_state_inspection_' + args.mode + '_20261105.json')
    assert not report_path.exists() and not report_path.is_symlink()
    guard.check_plans()
    assert guard.digest_file(ORIGINAL) == guard.CHECKPOINT_SHA
    torch.set_num_threads(1)
    entries = [(213, ORIGINAL)]
    for epoch in range(214, 225):
        name = 'epoch_%04d' % epoch
        if not (ROOT / name / 'ack.json').is_file():
            break
        guard.read_ack_for_previous(ROOT, name)
        entries.append((epoch, ROOT / name / 'checkpoint.pt'))
    assert len(entries) > 1
    results = []
    for epoch, path in entries:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        assert checkpoint['arm'] == 'BASE' and checkpoint['actual_loop_epoch'] == epoch
        raw, ema = checkpoint['model'], checkpoint['ema']['module']
        results.append({'epoch_index': epoch, 'checkpoint_sha256': guard.digest_file(path),
                        'raw_model': inspect_group(all_tensors(raw)),
                        'ema_model': inspect_group(all_tensors(ema)),
                        'optimizer': inspect_group(all_tensors(checkpoint['optimizer'])),
                        'raw_ema_difference': compare_raw_ema(raw, ema)})
    report = {'status': 'SAVED_STATES_INSPECTED', 'created_local_utc': datetime.now(timezone.utc).isoformat(),
              'mode': args.mode, 'epochs': results, 'model_inference_passes': 0,
              'evaluator_reruns': 0, 'optimizer_updates': 0,
              'limit': 'Finite saved tensors and raw/EMA distance do not establish the cause of an earlier anomaly.'}
    with report_path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({'status': report['status'], 'epochs': len(entries), 'report': str(report_path)}))

if __name__ == '__main__':
    main()
