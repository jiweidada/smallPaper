"""Inspect saved M4 evidence; no detector loading, inference, or training."""
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath
import hashlib
import json
import zipfile
import numpy as np

P = Path(__file__).resolve().parent
D = P.parent / 'A54_场景尺度约束候选_20261101_v1'
DEST = P / 'epoch4_M_saved_metrics_remote_v1'
RESULT = P / 'M4_saved_metrics_local_verification_v1.json'
FIELDS = ['AP', 'AP50', 'AP75', 'APs', 'APs75', 'APm', 'APl', 'ARs']
RAW = ['epoch', 'position', 'name', 'flip', 'queries', 'image', 'GT', 'labels', 'prior', 'endpoint', 'u']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def main():
    assert not DEST.exists() and not RESULT.exists()
    archive = D / 'A60_downloads/M_epoch4_saved_metrics_bundle_v1.zip'
    ready_path = D / 'A60_downloads/M_epoch4_saved_metrics_ready_v1.json'
    ready = read(ready_path)
    transfer = read(D / 'A60_M_epoch4_saved_metrics_download_v1.json')
    assert transfer['status'] == 'TRANSFERRED' and transfer['operation'] == 'get'
    assert ready['status'] == 'ORIGINAL_M_EPOCH4_SAVED_METRICS_READY'
    assert archive.stat().st_size == ready['archive_bytes'] and sha(archive) == ready['archive_sha256']
    for item in transfer['files']:
        path = Path(item['local'])
        assert path in [archive, ready_path]
        assert path.stat().st_size == item['bytes'] and sha(path) == item['local_sha256']
    assert len(transfer['files']) == 2
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('manifest.json'))
        assert len(manifest) == ready['payloads'] == 14
        assert len(z.namelist()) == len(set(z.namelist())) == 15
        assert set(z.namelist()) == set(manifest) | {'manifest.json'}
        payload = {}
        for name, expected in manifest.items():
            rel = PurePosixPath(name)
            assert not rel.is_absolute() and '..' not in rel.parts
            data = z.read(name)
            assert len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256']
            payload[name] = data
    def j(name):
        return json.loads(payload[name])
    assert payload['protocol_FROZEN_v1.json'] == (P / 'protocol_FROZEN_v1.json').read_bytes()
    assert j('M/initialization.json')['initialization_sha256'] == '95da80506b4482a315294bfdcf47fa8bd46a1bbaa412bf0156f08105e9c33535'
    row, metrics = j('M/epoch04.json'), j('M/epoch04_raw_metrics.json')
    assert row['epoch'] == 4 and row['updates'] == 3236 and row['metrics'] == metrics == ready['metrics']
    assert row['checkpoint_sha256'] == ready['checkpoint_sha256']
    baseline = read(P / 'epoch4_B_remote_v1/B/epoch04_raw_metrics.json')
    base_gt = P / 'epoch4_B_remote_v1/B/epoch04_raw_GT.json'
    assert ready['GT_sha256'] == sha(base_gt)
    for key in ['eval_prior_stream_sha256', 'evaluated_GT', 'ignored_valid_GT', 'max_detections', 'area_convention', 'model_state']:
        assert metrics[key] == baseline[key]
    for epoch in range(1, 5):
        m = j(f'M/epoch{epoch:02d}_flow.json')
        b = j(f'B/epoch{epoch:02d}_flow.json')
        assert b == read(P / f'epoch4_B_remote_v1/B/epoch{epoch:02d}_flow.json')
        assert m['epoch'] == b['epoch'] == epoch and m['images'] == b['images'] == 6471
        assert m['raw_stream_sha256'] == b['raw_stream_sha256']
    b8 = [json.loads(x) for x in (P / 'epoch4_B_remote_v1/B/first8_input_flow.jsonl').read_text().splitlines()]
    m8 = [json.loads(x) for x in payload['M/first8_input_flow.jsonl'].splitlines()]
    assert len(b8) == len(m8) == 8
    assert all(b[k] == m[k] for b, m in zip(b8, m8) for k in RAW)
    assert any(b['noisy'] != m['noisy'] for b, m in zip(b8, m8))
    assert any(b['time'] != m['time'] for b, m in zip(b8, m8))
    with np.load(BytesIO(payload['M/epoch04_raw_precision_recall.npz'])) as arrays:
        precision, recall, thresholds = arrays['precision'], arrays['recall'], arrays['iou_thresholds']
        assert precision.shape == (10, 101, 10, 4, 3) and recall.shape == (10, 10, 4, 3)
        def ap(area=0, iou=None):
            values = precision[:, :, :, area, -1]
            if iou is not None:
                values = values[np.isclose(thresholds, iou)]
            return float(values[values >= 0].mean() * 100)
        rs = recall[:, :, 1, -1]
        derived = dict(AP=ap(), AP50=ap(iou=.5), AP75=ap(iou=.75), APs=ap(1),
                       APs75=ap(1, .75), APm=ap(2), APl=ap(3), ARs=float(rs[rs >= 0].mean() * 100))
    errors = {k: derived[k] - metrics[k] for k in FIELDS}
    assert all(abs(x) < 1e-10 for x in errors.values())
    obs_receipt = read(D / 'A60_formal_observe_296.json')
    assert obs_receipt['status'] == 'EXECUTED' and obs_receipt['exit'] == 0
    obs = json.loads(obs_receipt['stdout'])
    assert obs['source_binding'] and obs['supervisor']['identity_matched'] and obs['supervisor']['live']
    assert obs['arms']['M']['process']['identity_matched'] and obs['arms']['M']['process']['live']
    assert obs['arms']['M']['launch']['pid'] == 35385 and obs['arms']['M']['launch']['start_ticks'] == '32308346'
    assert obs['arms']['M']['epochs'][3] == row and not obs['arms']['Q']['exists']
    result = {
        'status': 'PASS_ORIGINAL_M4_SAVED_METRICS_AND_FIRST4_RAW_STREAMS',
        'recorded_client_utc': datetime.now(timezone.utc).isoformat(),
        'recorded_remote_utc': obs['recorded_utc'],
        'observation': str(D / 'A60_formal_observe_296.json'),
        'archive_sha256': ready['archive_sha256'], 'verified_payloads': 14,
        'B4_metrics': baseline, 'M4_metrics': metrics,
        'M_minus_B_pp': {k: metrics[k] - baseline[k] for k in FIELDS},
        'saved_precision_recall_metric_errors': errors,
        'raw_stream_matches_B_first4_epochs': True,
        'first8_raw_fields_match': RAW,
        'noisy_and_time_streams_differ_as_designed': True,
        'eval_GT_and_prior_hashes_match_B': True,
        'M4_checkpoint_remote_sha256': ready['checkpoint_sha256'],
        'M4_checkpoint_downloaded': False,
        'candidate_Q_tested': False, 'fixed12_decision_available': False,
        'new_model_forwards': 0, 'new_optimizer_updates': 0, 'new_COCO_evaluations': 0,
        'checkpoint_loads': 0,
        'scope': 'Saved aggregation and trainer-produced stream summaries, no independent image rehash, COCO rematch, detector replay or terminal efficacy claim.'
    }
    DEST.mkdir()
    for name, data in payload.items():
        dest = DEST.joinpath(*PurePosixPath(name).parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as f:
            f.write(data)
    with RESULT.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'M4_AP': metrics['AP'], 'M4_APs': metrics['APs'],
                      'M_minus_B_pp': result['M_minus_B_pp']}))

if __name__ == '__main__':
    main()
