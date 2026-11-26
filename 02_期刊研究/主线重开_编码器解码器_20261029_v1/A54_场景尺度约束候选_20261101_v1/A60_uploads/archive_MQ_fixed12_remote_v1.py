"""One-time export of an original completed M or Q; never start/load a model."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import zipfile

ROOT = Path('/root/uav_a60_20261102_v1/pilot_v1')
EXPORT = Path('/root/uav_a54_20261101_v1/A60_exports')
SCOPE = EXPORT / 'MQ_fixed12_archive_scope_FROZEN_v1.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def first8(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 8 and [x['position'] for x in rows] == list(range(8))
    return rows


def result_for(arm, scope, protocol):
    out = ROOT / arm
    result, exit_record = read(out / 'result.json'), read(out / 'exit.json')
    launch, reaped = read(ROOT / (arm + '_launch_v1.json')), read(ROOT / (arm + '_worker_reaped_v1.json'))
    expected = dict(started=78514, completed=78514, updates=9708, backwards=77652,
                    epochs=12, train_forwards=77652, eval_forwards=822, latency_forwards=40)
    assert result['status'] == 'COMPLETED_FIXED12_RAW' and result['arm'] == arm
    assert result['counts'] == exit_record['counts'] == expected
    assert exit_record['exit'] == reaped['exit'] == 0 and reaped['wait_completed']
    assert reaped['original_pid'] == launch['pid'] and launch['arm'] == arm
    assert result['protocol_sha256'] == launch['protocol_sha256'] == scope['protocol_sha256']
    assert result['initialization_sha256'] == scope['initialization_sha256']
    assert result['source_sha256'] == protocol['source_sha256']
    assert result['no_best_epoch_selection'] and result['training_epochs'] == 12
    row, metrics = read(out / 'epoch12.json'), read(out / 'epoch12_raw_metrics.json')
    assert row['epoch'] == 12 and row['metrics'] == metrics == result['metrics']
    assert metrics['eval_prior_stream_sha256'] == scope['eval_prior_stream_sha256']
    assert sha(out / 'epoch12_raw_GT.json') == scope['B12_GT_sha256']
    assert row['checkpoint_sha256'] == result['fixed12_checkpoint_sha256']
    return result


def main(arm):
    assert arm in ['M', 'Q']
    scope = read(SCOPE)
    assert sha(Path(__file__)) == scope['remote_export_script_sha256']
    archive = EXPORT / (arm + '_fixed12_artifacts_bundle_v1.zip')
    checkpoint = EXPORT / (arm + '_fixed12_raw_checkpoint_v1.pth')
    ready_path = EXPORT / (arm + '_fixed12_export_ready_v1.json')
    assert not any(path.exists() for path in [archive, checkpoint, ready_path])
    assert sha(ROOT / 'protocol_FROZEN_v1.json') == scope['protocol_sha256']
    protocol = read(ROOT / 'protocol_FROZEN_v1.json')
    assert protocol['initialization_sha256'] == scope['initialization_sha256']
    for name, digest in protocol['source_sha256'].items():
        assert sha(ROOT / 'code' / name) == digest
    baseline = result_for('B', scope, protocol)
    assert baseline['fixed12_checkpoint_sha256'] == scope['B12_checkpoint_sha256']
    assert read(ROOT / 'B_learning_gate_v1.json')['passed']
    target = result_for(arm, scope, protocol)
    b8, t8 = first8(ROOT / 'B/first8_input_flow.jsonl'), first8(ROOT / arm / 'first8_input_flow.jsonl')
    raw_fields = ['epoch', 'position', 'name', 'flip', 'queries', 'image', 'GT', 'labels', 'prior', 'endpoint', 'u']
    assert all(x[k] == y[k] for x, y in zip(b8, t8) for k in raw_fields)
    bflow = [read(ROOT / ('B/epoch%02d_flow.json' % epoch)) for epoch in range(1, 13)]
    tflow = [read(ROOT / (arm + '/epoch%02d_flow.json' % epoch)) for epoch in range(1, 13)]
    assert all(x['raw_stream_sha256'] == y['raw_stream_sha256'] for x, y in zip(bflow, tflow))
    paths = ['protocol_FROZEN_v1.json', 'supervisor_start_v1.json', 'B_learning_gate_v1.json',
             arm + '_launch_v1.json', arm + '_worker_reaped_v1.json']
    paths += [f'{arm}/epoch{i:02d}{suffix}.json' for i in range(1, 13) for suffix in ['', '_flow']]
    paths += [f'{arm}/{name}' for name in ['initialization.json', 'run_once.json', 'first8_input_flow.jsonl',
              'actual_events.jsonl', 'result.json', 'exit.json', 'fixed12_latency.json',
              'epoch12_raw_GT.json', 'epoch12_raw_predictions.json', 'epoch12_raw_metrics.json',
              'epoch12_raw_precision_recall.npz']]
    paths += ['B/result.json', 'B/first8_input_flow.jsonl'] + [f'B/epoch{i:02d}_flow.json' for i in range(1, 13)]
    judgment = None
    if arm == 'M':
        assert any(x['noisy'] != y['noisy'] for x, y in zip(b8, t8))
        assert any(x['time'] != y['time'] for x, y in zip(b8, t8))
    else:
        middle = result_for('M', scope, protocol)
        m8 = first8(ROOT / 'M/first8_input_flow.jsonl')
        mflow = [read(ROOT / ('M/epoch%02d_flow.json' % epoch)) for epoch in range(1, 13)]
        assert all(x[k] == y[k] for x, y in zip(m8, t8) for k in raw_fields + ['noisy'])
        assert any(x['time'] != y['time'] for x, y in zip(m8, t8))
        assert all(x['noisy_stream_sha256'] == y['noisy_stream_sha256'] and
                   x['condition_stream_sha256'] != y['condition_stream_sha256'] for x, y in zip(mflow, tflow))
        judgment, supervisor_exit = read(ROOT / 'fixed12_judgment_v1.json'), read(ROOT / 'supervisor_exit_v1.json')
        assert supervisor_exit['exit'] == 0 and supervisor_exit['status'] == 'ALL_THREE_FIXED12_COMPLETED'
        assert supervisor_exit['completed_arms'] == ['B', 'M', 'Q'] and supervisor_exit['remaining_queue'] == []
        delta = {'Q_minus_B_APs': target['metrics']['APs'] - baseline['metrics']['APs'],
                 'Q_minus_M_APs': target['metrics']['APs'] - middle['metrics']['APs'],
                 'Q_minus_B_AP': target['metrics']['AP'] - baseline['metrics']['AP'],
                 'Q_over_M_median_latency': target['latency']['median_ms'] / middle['latency']['median_ms']}
        assert all(math.isclose(delta[k], judgment['delta'][k], rel_tol=0, abs_tol=1e-12) for k in delta)
        gates = {'APs_vs_B': delta['Q_minus_B_APs'] >= .5, 'APs_vs_M': delta['Q_minus_M_APs'] >= .3,
                 'AP_preservation': delta['Q_minus_B_AP'] >= -.2, 'latency': delta['Q_over_M_median_latency'] <= 1.10}
        assert judgment['gates'] == gates and judgment['candidate_has_pilot_support'] == all(gates.values())
        assert judgment['metrics'] == {x: y['metrics'] for x, y in [('B', baseline), ('M', middle), ('Q', target)]}
        assert judgment['full_input_stream_audit_passed'] and not judgment['mainline_validated']
        paths += ['M/result.json', 'M/first8_input_flow.jsonl'] + [f'M/epoch{i:02d}_flow.json' for i in range(1, 13)]
        paths += ['fixed12_judgment_v1.json', 'supervisor_exit_v1.json']
    assert len(paths) == len(set(paths)) == scope['expected_payloads'][arm]
    assert all((ROOT / name).is_file() for name in paths)
    raw = ROOT / (arm + '/epoch12_raw.pth')
    assert raw.is_file() and raw.stat().st_size > 0 and raw.stat().st_dev == EXPORT.stat().st_dev
    raw_sha = sha(raw); assert raw_sha == target['fixed12_checkpoint_sha256']
    manifest = {}
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as dest:
        for name in paths:
            payload = (ROOT / name).read_bytes()
            manifest[name] = {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
            dest.writestr(name, payload)
        dest.writestr('manifest.json', json.dumps(manifest, indent=2, sort_keys=True))
    os.link(raw, checkpoint)
    assert raw.stat().st_ino == checkpoint.stat().st_ino and raw.stat().st_dev == checkpoint.stat().st_dev
    ready = {'status': arm + '_FIXED12_EXISTING_ARTIFACTS_AND_CHECKPOINT_READY', 'arm': arm,
             'scope_sha256': sha(SCOPE), 'archive_bytes': archive.stat().st_size,
             'archive_sha256': sha(archive), 'archive_payloads': len(paths),
             'checkpoint_bytes': raw.stat().st_size, 'checkpoint_sha256': raw_sha,
             'checkpoint_original': str(raw), 'checkpoint_hardlink': str(checkpoint),
             'same_inode_verified': True, 'metrics': target['metrics'], 'judgment': judgment,
             'full_raw_stream_matches_B': True, 'Q_noisy_stream_matches_M': True if arm == 'Q' else None,
             'new_forwards': 0, 'new_updates': 0, 'new_COCO_evaluations': 0, 'checkpoint_loads': 0}
    with ready_path.open('x', encoding='utf-8') as out:
        json.dump(ready, out, indent=2, allow_nan=False)
    print(json.dumps(ready, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['M', 'Q'], required=True)
    main(parser.parse_args().arm)
