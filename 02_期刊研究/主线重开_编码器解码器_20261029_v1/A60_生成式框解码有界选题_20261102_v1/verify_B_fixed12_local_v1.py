"""Verify one received B12 bundle/checkpoint; no model replay or COCO rematch."""
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath
import hashlib
import json
import math
import zipfile
import numpy as np

P = Path(__file__).resolve().parent
D = P.parent / 'A54_场景尺度约束候选_20261101_v1'
SCOPE = P / 'B_fixed12_archive_scope_FROZEN_v1.json'
DOWNLOAD = D / 'A60_downloads'
ARCHIVE = DOWNLOAD / 'B_fixed12_artifacts_bundle_v1.zip'
CHECKPOINT = DOWNLOAD / 'B_fixed12_raw_checkpoint_v1.pth'
READY = DOWNLOAD / 'B_fixed12_export_ready_v1.json'
TRANSFER = D / 'A60_B_fixed12_download_v1.json'
OUT = P / 'fixed12_B_remote_v1'
RESULT = P / 'B_fixed12_local_verification_v1.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    assert not OUT.exists() and not RESULT.exists()
    scope = read(SCOPE)
    assert sha(Path(__file__)) == scope['local_verifier_script_sha256']
    ready = read(READY)
    assert ready['status'] == 'B_FIXED12_EXISTING_ARTIFACTS_AND_CHECKPOINT_READY'
    assert ready['scope_sha256'] == sha(SCOPE)
    transfer = read(TRANSFER)
    assert transfer['status'] == 'TRANSFERRED' and transfer['operation'] == 'get'
    received = {Path(x['local']).name: x for x in transfer['files']}
    assert set(received) == {ARCHIVE.name, CHECKPOINT.name, READY.name}
    for path in [ARCHIVE, CHECKPOINT, READY]:
        record = received[path.name]
        assert Path(record['local']).resolve() == path.resolve()
        assert record['remote'] == scope['remote_exports'][path.name]
        assert path.stat().st_size == record['bytes'] and sha(path) == record['local_sha256']
    assert ARCHIVE.stat().st_size == ready['archive_bytes'] and sha(ARCHIVE) == ready['archive_sha256']
    assert CHECKPOINT.stat().st_size == ready['checkpoint_bytes'] and sha(CHECKPOINT) == ready['checkpoint_sha256']
    assert ready['same_inode_verified'] and ready['checkpoint_original'] == scope['checkpoint_original']
    assert ready['checkpoint_hardlink'] == scope['remote_exports'][CHECKPOINT.name]
    assert ready['archive_payloads'] == scope['expected_payloads']

    with zipfile.ZipFile(ARCHIVE) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)) and 'manifest.json' in names
        manifest = json.loads(archive.read('manifest.json'))
        assert len(manifest) == scope['expected_payloads'] and set(names) == set(manifest) | {'manifest.json'}
        payloads = {}
        for name, metadata in manifest.items():
            path = PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts and not name.endswith('/')
            data = archive.read(name)
            assert len(data) == metadata['bytes'] and hashlib.sha256(data).hexdigest() == metadata['sha256']
            payloads[name] = data
    def j(name):
        return json.loads(payloads[name])

    assert hashlib.sha256(payloads['protocol_FROZEN_v1.json']).hexdigest() == scope['protocol_sha256']
    protocol = j('protocol_FROZEN_v1.json')
    result, exit_record = j('B/result.json'), j('B/exit.json')
    metrics, row, gate = j('B/epoch12_raw_metrics.json'), j('B/epoch12.json'), j('B_learning_gate_v1.json')
    counts = dict(started=78514, completed=78514, updates=9708, backwards=77652,
                  epochs=12, train_forwards=77652, eval_forwards=822, latency_forwards=40)
    assert result['status'] == 'COMPLETED_FIXED12_RAW' and result['arm'] == 'B'
    assert result['counts'] == exit_record['counts'] == counts and exit_record['exit'] == 0
    assert result['protocol_sha256'] == scope['protocol_sha256']
    assert result['initialization_sha256'] == j('B/initialization.json')['initialization_sha256'] == scope['initialization_sha256']
    assert result['source_sha256'] == protocol['source_sha256']
    assert result['fixed12_checkpoint_sha256'] == row['checkpoint_sha256'] == ready['checkpoint_sha256']
    assert result['no_best_epoch_selection'] and result['training_epochs'] == 12
    assert row['metrics'] == result['metrics'] == metrics == ready['metrics']
    assert gate == ready['gate'] and gate['AP'] == metrics['AP'] and gate['APs'] == metrics['APs']
    assert gate['minimum_AP'] == 8. and gate['minimum_APs'] == 3.
    assert gate['passed'] == (metrics['AP'] >= 8. and metrics['APs'] >= 3.)
    assert j('B_worker_reaped_v1.json')['exit'] == 0
    assert metrics['eval_prior_stream_sha256'] == scope['eval_prior_stream_sha256']
    assert hashlib.sha256(payloads['B/epoch12_raw_GT.json']).hexdigest() == scope['B8_GT_sha256']
    for epoch in range(1, 13):
        r = j(f'B/epoch{epoch:02d}.json'); flow = j(f'B/epoch{epoch:02d}_flow.json')
        assert r['epoch'] == flow['epoch'] == epoch and r['updates'] == epoch * 809
        assert r['flow'] == flow and flow['images'] == 6471

    gt = j('B/epoch12_raw_GT.json'); predictions = j('B/epoch12_raw_predictions.json')
    assert len(gt['images']) == 548 and len(gt['annotations']) == metrics['evaluated_GT'] == 38602
    assert gt['ignored_valid_gt'] == metrics['ignored_valid_GT'] == 157
    assert sum(item['area'] <= 1024 for item in gt['annotations']) == 26444
    assert len(predictions) == metrics['predictions_after_ignore']
    sizes = {row['id']: (row['width'], row['height']) for row in gt['images']}
    per_image = Counter()
    for item in predictions:
        assert item['image_id'] in sizes and 1 <= item['category_id'] <= 10
        x, y, w, h = item['bbox']; width, height = sizes[item['image_id']]
        assert all(math.isfinite(v) for v in [x, y, w, h, item['score']])
        assert 0 <= item['score'] <= 1 and w > 0 and h > 0
        assert x >= -1e-5 and y >= -1e-5 and x + w <= width + 1e-3 and y + h <= height + 1e-3
        per_image[item['image_id']] += 1
    assert max(per_image.values(), default=0) <= 300

    with np.load(BytesIO(payloads['B/epoch12_raw_precision_recall.npz'])) as arrays:
        precision, recall, thresholds = arrays['precision'], arrays['recall'], arrays['iou_thresholds']
        assert precision.shape == (10, 101, 10, 4, 3) and recall.shape == (10, 10, 4, 3)
        def ap(area=0, iou=None):
            values = precision[:, :, :, area, -1]
            if iou is not None:
                values = values[np.isclose(thresholds, iou)]
            values = values[values >= 0]
            return float(values.mean() * 100)
        rs = recall[:, :, 1, -1]; rs = rs[rs >= 0]
        derived = dict(AP=ap(), AP50=ap(iou=.5), AP75=ap(iou=.75), APs=ap(1),
                       APs75=ap(1, .75), APm=ap(2), APl=ap(3), ARs=float(rs.mean() * 100))
    differences = {key: derived[key] - metrics[key] for key in derived}
    assert all(abs(value) < 1e-10 for value in differences.values())

    events = [json.loads(line) for line in payloads['B/actual_events.jsonl'].splitlines()]
    starts = [event for event in events if event['event'] == 'forward_start']
    completes = [event for event in events if event['event'] == 'forward_complete']
    updates = [event for event in events if event['event'] == 'optimizer_returned']
    assert len(events) == 166736 and len(starts) == len(completes) == 78514 and len(updates) == 9708
    assert [event['count'] for event in starts] == [event['count'] for event in completes] == list(range(1, 78515))
    assert Counter(event['phase'] for event in starts) == {'train': 77652, 'eval': 822, 'latency': 40}
    for epoch in range(1, 13):
        assert [x['position'] for x in starts if x['phase'] == 'train' and x['epoch'] == epoch] == list(range(6471))
        assert [x['group_images'] for x in updates if x['epoch'] == epoch] == [8] * 808 + [7]
    assert all(math.isfinite(x[k]) for x in updates for k in ['mean_loss', 'gradient_norm', 'time_MLP_gradient_L1'])
    baseline8 = read(P / 'epoch8_B_remote_v1/B/epoch08_raw_metrics.json')

    OUT.mkdir()
    for name, data in payloads.items():
        destination = OUT.joinpath(*PurePosixPath(name).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as target:
            target.write(data)
    with (OUT / 'manifest.json').open('x', encoding='utf-8') as target:
        json.dump(manifest, target, indent=2, sort_keys=True)
    report = {'status': 'PASS_ORIGINAL_B_FIXED12_ARTIFACTS_AND_LOCAL_MODEL_BACKUP',
              'recorded_client_utc': datetime.now(timezone.utc).isoformat(),
              'archive_sha256': ready['archive_sha256'], 'verified_payloads': len(payloads),
              'checkpoint_path': str(CHECKPOINT), 'checkpoint_sha256': ready['checkpoint_sha256'],
              'metrics': metrics, 'gate': gate, 'B12_minus_B8_pp': {k: metrics[k] - baseline8[k] for k in derived},
              'saved_precision_recall_metric_differences': differences,
              'fixed_GT_sha_matches_B8': True, 'fixed_eval_priors_match_B8': True,
              'original_events': len(events), 'original_counts': counts,
              'new_model_forwards': 0, 'new_optimizer_updates': 0, 'new_COCO_evaluations': 0,
              'checkpoint_loads': 0,
              'scope': 'Saved files, checkpoint bytes, counts and aggregation verified; no independent COCO rematch, CNN replay, or candidate efficacy result.'}
    with RESULT.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(report, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write('\n')
    print(json.dumps({'status': report['status'], 'AP': metrics['AP'], 'APs': metrics['APs'],
                      'gate_passed': gate['passed'], 'checkpoint_sha256': ready['checkpoint_sha256']}))


if __name__ == '__main__':
    main()
