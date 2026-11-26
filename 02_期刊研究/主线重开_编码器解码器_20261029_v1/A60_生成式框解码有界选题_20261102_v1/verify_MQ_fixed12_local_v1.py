"""One-time verification of a received M/Q fixed12 bundle and model bytes."""
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import math
import zipfile
import numpy as np

P = Path(__file__).resolve().parent
D = P.parent / 'A54_场景尺度约束候选_20261101_v1'
SCOPE = P / 'MQ_fixed12_archive_scope_FROZEN_v1.json'
DOWNLOAD = D / 'A60_downloads'
RAW_FIELDS = ['epoch', 'position', 'name', 'flip', 'queries', 'image', 'GT', 'labels', 'prior', 'endpoint', 'u']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def first8(payload):
    rows = [json.loads(line) for line in payload.splitlines()]
    assert len(rows) == 8 and [x['position'] for x in rows] == list(range(8))
    return rows


def main(arm):
    assert arm in ['M', 'Q']
    archive = DOWNLOAD / (arm + '_fixed12_artifacts_bundle_v1.zip')
    checkpoint = DOWNLOAD / (arm + '_fixed12_raw_checkpoint_v1.pth')
    ready_file = DOWNLOAD / (arm + '_fixed12_export_ready_v1.json')
    transfer_file = D / ('A60_' + arm + '_fixed12_download_v1.json')
    out = P / ('fixed12_' + arm + '_remote_v1')
    result_file = P / (arm + '_fixed12_local_verification_v1.json')
    assert not out.exists() and not result_file.exists()
    scope = read(SCOPE)
    assert sha(Path(__file__)) == scope['local_verifier_script_sha256']
    ready, transfer = read(ready_file), read(transfer_file)
    assert ready['status'] == arm + '_FIXED12_EXISTING_ARTIFACTS_AND_CHECKPOINT_READY' and ready['arm'] == arm
    assert ready['scope_sha256'] == sha(SCOPE)
    assert transfer['status'] == 'TRANSFERRED' and transfer['operation'] == 'get'
    received = {Path(item['local']).name: item for item in transfer['files']}
    assert set(received) == {archive.name, checkpoint.name, ready_file.name}
    for path in [archive, checkpoint, ready_file]:
        item = received[path.name]
        assert Path(item['local']).resolve() == path.resolve() and item['remote'] == scope['remote_exports'][arm][path.name]
        assert path.stat().st_size == item['bytes'] and sha(path) == item['local_sha256']
    assert archive.stat().st_size == ready['archive_bytes'] and sha(archive) == ready['archive_sha256']
    assert checkpoint.stat().st_size == ready['checkpoint_bytes'] and sha(checkpoint) == ready['checkpoint_sha256']
    assert ready['same_inode_verified'] and ready['checkpoint_original'] == scope['checkpoint_original'][arm]
    assert ready['checkpoint_hardlink'] == scope['remote_exports'][arm][checkpoint.name]
    assert ready['archive_payloads'] == scope['expected_payloads'][arm]
    assert ready['full_raw_stream_matches_B'] is True

    with zipfile.ZipFile(archive) as src:
        names = src.namelist()
        assert len(names) == len(set(names)) and 'manifest.json' in names
        manifest = json.loads(src.read('manifest.json'))
        assert len(manifest) == scope['expected_payloads'][arm] and set(names) == set(manifest) | {'manifest.json'}
        payloads = {}
        for name, expected in manifest.items():
            relative = PurePosixPath(name)
            assert not relative.is_absolute() and '..' not in relative.parts and not name.endswith('/')
            data = src.read(name)
            assert len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256']
            payloads[name] = data
    def j(name):
        return json.loads(payloads[name])

    assert hashlib.sha256(payloads['protocol_FROZEN_v1.json']).hexdigest() == scope['protocol_sha256']
    protocol = j('protocol_FROZEN_v1.json')
    result, exit_record = j(f'{arm}/result.json'), j(f'{arm}/exit.json')
    metrics, row = j(f'{arm}/epoch12_raw_metrics.json'), j(f'{arm}/epoch12.json')
    counts = dict(started=78514, completed=78514, updates=9708, backwards=77652,
                  epochs=12, train_forwards=77652, eval_forwards=822, latency_forwards=40)
    assert result['status'] == 'COMPLETED_FIXED12_RAW' and result['arm'] == arm
    assert result['counts'] == exit_record['counts'] == counts and exit_record['exit'] == 0
    assert result['protocol_sha256'] == scope['protocol_sha256']
    assert result['initialization_sha256'] == j(f'{arm}/initialization.json')['initialization_sha256'] == scope['initialization_sha256']
    assert result['source_sha256'] == protocol['source_sha256']
    assert result['fixed12_checkpoint_sha256'] == row['checkpoint_sha256'] == ready['checkpoint_sha256']
    assert result['no_best_epoch_selection'] and result['training_epochs'] == 12
    assert row['metrics'] == result['metrics'] == metrics == ready['metrics']
    assert j(arm + '_worker_reaped_v1.json')['exit'] == 0
    assert metrics['eval_prior_stream_sha256'] == scope['eval_prior_stream_sha256']
    assert hashlib.sha256(payloads[f'{arm}/epoch12_raw_GT.json']).hexdigest() == scope['B12_GT_sha256']
    baseline = j('B/result.json'); baseline_local = read(P / 'fixed12_B_remote_v1/B/result.json')
    assert baseline == baseline_local and baseline['fixed12_checkpoint_sha256'] == scope['B12_checkpoint_sha256']
    assert j('B_learning_gate_v1.json')['passed']
    b8, t8 = first8(payloads['B/first8_input_flow.jsonl']), first8(payloads[f'{arm}/first8_input_flow.jsonl'])
    assert all(x[k] == y[k] for x, y in zip(b8, t8) for k in RAW_FIELDS)
    assert payloads['B/first8_input_flow.jsonl'] == (P / 'fixed12_B_remote_v1/B/first8_input_flow.jsonl').read_bytes()
    tflow, bflow = [], []
    for epoch in range(1, 13):
        row_i = j(f'{arm}/epoch{epoch:02d}.json'); flow = j(f'{arm}/epoch{epoch:02d}_flow.json')
        base = j(f'B/epoch{epoch:02d}_flow.json')
        assert row_i['epoch'] == flow['epoch'] == base['epoch'] == epoch
        assert row_i['updates'] == epoch * 809 and row_i['flow'] == flow and flow['images'] == 6471
        assert base == read(P / f'fixed12_B_remote_v1/B/epoch{epoch:02d}_flow.json')
        assert flow['raw_stream_sha256'] == base['raw_stream_sha256']
        tflow.append(flow); bflow.append(base)
    stream_note = {'raw_matches_B_all_12_epochs': True, 'Q_noisy_matches_M_all_12_epochs': None,
                   'Q_conditions_differ_from_M_all_12_epochs': None}
    if arm == 'M':
        assert any(x['noisy'] != y['noisy'] for x, y in zip(b8, t8))
        assert any(x['time'] != y['time'] for x, y in zip(b8, t8))
        assert ready['judgment'] is None and ready['Q_noisy_stream_matches_M'] is None
    else:
        assert ready['Q_noisy_stream_matches_M'] is True
        m_verified = read(P / 'M_fixed12_local_verification_v1.json')
        assert m_verified['status'] == 'PASS_ORIGINAL_M_FIXED12_ARTIFACTS_AND_LOCAL_MODEL_BACKUP'
        middle = j('M/result.json'); assert middle == read(P / 'fixed12_M_remote_v1/M/result.json')
        m8 = first8(payloads['M/first8_input_flow.jsonl'])
        assert payloads['M/first8_input_flow.jsonl'] == (P / 'fixed12_M_remote_v1/M/first8_input_flow.jsonl').read_bytes()
        assert all(x[k] == y[k] for x, y in zip(m8, t8) for k in RAW_FIELDS + ['noisy'])
        assert any(x['time'] != y['time'] for x, y in zip(m8, t8))
        for epoch, qflow in enumerate(tflow, 1):
            mflow = j(f'M/epoch{epoch:02d}_flow.json')
            assert mflow == read(P / f'fixed12_M_remote_v1/M/epoch{epoch:02d}_flow.json')
            assert qflow['noisy_stream_sha256'] == mflow['noisy_stream_sha256']
            assert qflow['condition_stream_sha256'] != mflow['condition_stream_sha256']
        stream_note.update(Q_noisy_matches_M_all_12_epochs=True, Q_conditions_differ_from_M_all_12_epochs=True)
        judgment, supervisor = j('fixed12_judgment_v1.json'), j('supervisor_exit_v1.json')
        assert judgment == ready['judgment'] and supervisor['status'] == 'ALL_THREE_FIXED12_COMPLETED' and supervisor['exit'] == 0
        delta = {'Q_minus_B_APs': metrics['APs'] - baseline['metrics']['APs'],
                 'Q_minus_M_APs': metrics['APs'] - middle['metrics']['APs'],
                 'Q_minus_B_AP': metrics['AP'] - baseline['metrics']['AP'],
                 'Q_over_M_median_latency': result['latency']['median_ms'] / middle['latency']['median_ms']}
        assert all(math.isclose(delta[k], judgment['delta'][k], rel_tol=0, abs_tol=1e-12) for k in delta)
        gates = {'APs_vs_B': delta['Q_minus_B_APs'] >= .5, 'APs_vs_M': delta['Q_minus_M_APs'] >= .3,
                 'AP_preservation': delta['Q_minus_B_AP'] >= -.2, 'latency': delta['Q_over_M_median_latency'] <= 1.10}
        assert judgment['gates'] == gates and judgment['candidate_has_pilot_support'] == all(gates.values())
        assert judgment['metrics'] == {x: y['metrics'] for x, y in [('B', baseline), ('M', middle), ('Q', result)]}
        assert judgment['full_input_stream_audit_passed'] and not judgment['mainline_validated']

    gt, predictions = j(f'{arm}/epoch12_raw_GT.json'), j(f'{arm}/epoch12_raw_predictions.json')
    assert len(gt['images']) == 548 and len(gt['annotations']) == metrics['evaluated_GT'] == 38602
    assert gt['ignored_valid_gt'] == metrics['ignored_valid_GT'] == 157
    assert sum(item['area'] <= 1024 for item in gt['annotations']) == 26444
    assert len(predictions) == metrics['predictions_after_ignore']
    sizes = {row['id']: (row['width'], row['height']) for row in gt['images']}; per_image = Counter()
    for item in predictions:
        assert item['image_id'] in sizes and 1 <= item['category_id'] <= 10
        x, y, w, h = item['bbox']; width, height = sizes[item['image_id']]
        assert all(math.isfinite(v) for v in [x, y, w, h, item['score']]) and 0 <= item['score'] <= 1
        assert w > 0 and h > 0 and x >= -1e-5 and y >= -1e-5 and x + w <= width + 1e-3 and y + h <= height + 1e-3
        per_image[item['image_id']] += 1
    assert max(per_image.values(), default=0) <= 300
    with np.load(BytesIO(payloads[f'{arm}/epoch12_raw_precision_recall.npz'])) as arrays:
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
    events = [json.loads(line) for line in payloads[f'{arm}/actual_events.jsonl'].splitlines()]
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

    out.mkdir()
    for name, data in payloads.items():
        destination = out.joinpath(*PurePosixPath(name).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as target:
            target.write(data)
    with (out / 'manifest.json').open('x', encoding='utf-8') as target:
        json.dump(manifest, target, indent=2, sort_keys=True)
    report = {'status': 'PASS_ORIGINAL_' + arm + '_FIXED12_ARTIFACTS_AND_LOCAL_MODEL_BACKUP', 'arm': arm,
              'recorded_client_utc': datetime.now(timezone.utc).isoformat(),
              'archive_sha256': ready['archive_sha256'], 'verified_payloads': len(payloads),
              'checkpoint_path': str(checkpoint), 'checkpoint_sha256': ready['checkpoint_sha256'],
              'metrics': metrics, 'latency': result['latency'], 'stream_alignment': stream_note,
              'judgment': ready['judgment'], 'saved_precision_recall_metric_differences': differences,
              'fixed_GT_sha_matches_B12': True, 'fixed_eval_priors_match_B12': True,
              'original_events': len(events), 'original_counts': counts,
              'new_model_forwards': 0, 'new_optimizer_updates': 0, 'new_COCO_evaluations': 0,
              'checkpoint_loads': 0,
              'scope': 'Saved files, checkpoint bytes, journal, stream hashes and aggregation; no independent COCO rematch, CNN replay, convergence or global novelty proof.'}
    with result_file.open('x', encoding='utf-8', newline='\n') as target:
        json.dump(report, target, indent=2, ensure_ascii=False, allow_nan=False); target.write('\n')
    print(json.dumps({'status': report['status'], 'AP': metrics['AP'], 'APs': metrics['APs'],
                      'checkpoint_sha256': ready['checkpoint_sha256'],
                      'candidate_has_pilot_support': ready['judgment']['candidate_has_pilot_support'] if ready['judgment'] else None}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['M', 'Q'], required=True)
    main(parser.parse_args().arm)
