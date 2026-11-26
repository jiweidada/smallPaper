"""One-time export of the original completed B12; never start or load a model.

Run only after the original supervisor has produced its B learning-gate record.
Partial outputs are intentionally retained for inspection, never auto-retried.
"""
from pathlib import Path
import hashlib
import json
import os
import zipfile

ROOT = Path('/root/uav_a60_20261102_v1/pilot_v1')
EXPORT = Path('/root/uav_a54_20261101_v1/A60_exports')
SCOPE = EXPORT / 'B_fixed12_archive_scope_FROZEN_v1.json'
ARCHIVE = EXPORT / 'B_fixed12_artifacts_bundle_v1.zip'
CHECKPOINT = EXPORT / 'B_fixed12_raw_checkpoint_v1.pth'
READY = EXPORT / 'B_fixed12_export_ready_v1.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    scope = read(SCOPE)
    assert sha(Path(__file__)) == scope['remote_export_script_sha256']
    assert not any(path.exists() for path in [ARCHIVE, CHECKPOINT, READY])
    assert sha(ROOT / 'protocol_FROZEN_v1.json') == scope['protocol_sha256']
    protocol = read(ROOT / 'protocol_FROZEN_v1.json')
    assert protocol['initialization_sha256'] == scope['initialization_sha256']
    assert sha(ROOT / 'common_init_A60_v1.pth') == scope['initialization_sha256']
    for name, digest in protocol['source_sha256'].items():
        assert sha(ROOT / 'code' / name) == digest

    result = read(ROOT / 'B/result.json')
    exit_record = read(ROOT / 'B/exit.json')
    reaped = read(ROOT / 'B_worker_reaped_v1.json')
    gate = read(ROOT / 'B_learning_gate_v1.json')
    row = read(ROOT / 'B/epoch12.json')
    metrics = read(ROOT / 'B/epoch12_raw_metrics.json')
    expected_counts = dict(started=78514, completed=78514, updates=9708, backwards=77652,
                           epochs=12, train_forwards=77652, eval_forwards=822, latency_forwards=40)
    assert result['status'] == 'COMPLETED_FIXED12_RAW' and result['arm'] == 'B'
    assert result['counts'] == exit_record['counts'] == expected_counts
    assert exit_record['exit'] == reaped['exit'] == 0 and reaped['wait_completed']
    assert result['protocol_sha256'] == scope['protocol_sha256']
    assert result['initialization_sha256'] == scope['initialization_sha256']
    assert result['source_sha256'] == protocol['source_sha256']
    assert result['no_best_epoch_selection'] and result['training_epochs'] == 12
    assert row['epoch'] == 12 and row['metrics'] == metrics == result['metrics']
    assert gate['AP'] == metrics['AP'] and gate['APs'] == metrics['APs']
    assert gate['minimum_AP'] == 8. and gate['minimum_APs'] == 3.
    assert gate['passed'] == (metrics['AP'] >= 8. and metrics['APs'] >= 3.)
    assert metrics['eval_prior_stream_sha256'] == scope['eval_prior_stream_sha256']
    assert sha(ROOT / 'B/epoch12_raw_GT.json') == scope['B8_GT_sha256']
    raw = ROOT / 'B/epoch12_raw.pth'
    assert raw.is_file() and raw.stat().st_size > 0 and raw.stat().st_dev == EXPORT.stat().st_dev
    raw_sha = sha(raw)
    assert raw_sha == result['fixed12_checkpoint_sha256'] == row['checkpoint_sha256']

    paths = ['protocol_FROZEN_v1.json', 'supervisor_start_v1.json', 'B_launch_v1.json',
             'B_worker_reaped_v1.json', 'B_learning_gate_v1.json']
    paths += [f'B/epoch{i:02d}{suffix}.json' for i in range(1, 13) for suffix in ['', '_flow']]
    paths += ['B/initialization.json', 'B/run_once.json', 'B/first8_input_flow.jsonl',
              'B/actual_events.jsonl', 'B/result.json', 'B/exit.json', 'B/fixed12_latency.json',
              'B/epoch12_raw_GT.json', 'B/epoch12_raw_predictions.json',
              'B/epoch12_raw_metrics.json', 'B/epoch12_raw_precision_recall.npz']
    assert len(paths) == len(set(paths)) and all((ROOT / p).is_file() for p in paths)
    manifest = {}
    with zipfile.ZipFile(ARCHIVE, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for name in paths:
            payload = (ROOT / name).read_bytes()
            manifest[name] = {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
            archive.writestr(name, payload)
        archive.writestr('manifest.json', json.dumps(manifest, indent=2, sort_keys=True))
    os.link(raw, CHECKPOINT)
    assert raw.stat().st_ino == CHECKPOINT.stat().st_ino and raw.stat().st_dev == CHECKPOINT.stat().st_dev
    ready = {'status': 'B_FIXED12_EXISTING_ARTIFACTS_AND_CHECKPOINT_READY',
             'scope_sha256': sha(SCOPE), 'archive_bytes': ARCHIVE.stat().st_size,
             'archive_sha256': sha(ARCHIVE), 'archive_payloads': len(paths),
             'checkpoint_bytes': raw.stat().st_size, 'checkpoint_sha256': raw_sha,
             'checkpoint_original': str(raw), 'checkpoint_hardlink': str(CHECKPOINT),
             'same_inode_verified': True, 'gate': gate, 'metrics': metrics,
             'new_forwards': 0, 'new_updates': 0, 'new_COCO_evaluations': 0, 'checkpoint_loads': 0}
    with READY.open('x', encoding='utf-8') as out:
        json.dump(ready, out, indent=2, allow_nan=False)
    print(json.dumps(ready, allow_nan=False))


if __name__ == '__main__':
    main()
