"""Audit saved diagnostic update logs and counters; no model/evaluator execution."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import gzip
import json
import math
import os
import backup_guard as guard

ROOT = guard.REMOTE_WORK / 'diagnostic_backups_20261105' / guard.RUN_ID
OUTPUT = guard.REMOTE_WORK / 'diagnostic_saved_steps_audit_20261105.json'
N = 1617
START = 214 * N
EXPECTED = [0.0004, 0.0004, 0.0008, 0.0008]

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def main():
    guard.check_plans()
    assert not OUTPUT.exists() and not OUTPUT.is_symlink()
    summary = []
    for epoch in range(214, 225):
        phase = 'epoch_%04d' % epoch
        guard.read_ack_for_previous(ROOT, phase)
        stage = ROOT / phase
        manifest = guard.parse_manifest((stage / 'manifest.json').read_bytes())
        record, gradients, metrics = (read(stage / name) for name in
                                      ['phase_record.json', 'gradient_summary.json', 'metrics.json'])
        assert manifest['optimizer_updates'] == (epoch - 213) * N
        assert record['optimizer_updates'] == manifest['optimizer_updates']
        assert record['validation_passes'] == manifest['validation_passes']
        assert record['checkpoint_sha256'] == metrics['checkpoint_sha256']
        assert record['ema_tensor_sha256'] == metrics['evaluated_tensor_sha256']
        assert math.isfinite(record['train_loss_mean'])
        assert gradients['count'] == N and gradients['nonfinite_count'] == 0
        assert all(math.isfinite(gradients[k]) for k in ['mean_preclip_norm', 'maximum_preclip_norm'])
        assert gradients['clip_max_norm'] == 0.1
        count = 0
        with gzip.open(stage / 'learning_rates.csv.gz', 'rt', encoding='utf-8', newline='') as f:
            for step, row in enumerate(csv.DictReader(f)):
                offset = (epoch - 214) * N + step
                assert int(row['additional_update']) == offset
                assert int(row['absolute_attempt']) == START + offset
                assert int(row['actual_epoch']) == epoch and int(row['batch']) == step
                assert [float(row['group%d' % k]) for k in range(4)] == EXPECTED
                count += 1
        assert count == N
        summary.append({'epoch_index': epoch, 'optimizer_updates': manifest['optimizer_updates'],
                        'validation_passes': manifest['validation_passes'], 'learning_rate_rows': count,
                        'train_loss_mean': record['train_loss_mean'], 'EMA_AP': metrics['metrics']['AP'],
                        'gradient_nonfinite_count': gradients['nonfinite_count']})
    report = {'status': 'DIAGNOSTIC_SAVED_STEPS_VERIFIED',
              'created_local_utc': datetime.now(timezone.utc).isoformat(), 'epochs': summary,
              'total_learning_rate_rows': sum(x['learning_rate_rows'] for x in summary),
              'total_optimizer_updates_from_record': summary[-1]['optimizer_updates'],
              'expected_lr_groups': EXPECTED, 'model_inference_passes': 0,
              'evaluator_reruns': 0, 'optimizer_updates_performed_by_audit': 0,
              'limit': 'This checks saved update records, not the unrecorded trajectory of the interrupted run.'}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({'status': report['status'], 'rows': report['total_learning_rate_rows'],
                      'report': str(OUTPUT)}))

if __name__ == '__main__':
    main()
