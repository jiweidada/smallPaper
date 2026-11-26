"""Verify B8 downloaded bytes and cached arrays, without detector/evaluator calls."""
from pathlib import Path
from collections import Counter
import json
import hashlib
import os
import shutil
import sys
import zipfile

P = Path(__file__).resolve().parent
R54 = P.parent / 'A54_场景尺度约束候选_20261101_v1'
sys.path.insert(0, str(P.parent.parent / 'code/dn_pre_frozen_diagnostic_20261020_v1/runtime'))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
torch.set_num_threads(2)


def load(p):
    return json.loads(p.read_text('utf-8'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda: f.read(4 << 20), b''):
            h.update(x)
    return h.hexdigest()


def save(p, value):
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    rpc = load(R54 / 'A57_B_epoch08_small_results_export_rpc.json')
    assert rpc['exit'] == 0
    export = json.loads(rpc['stdout'])
    download = load(R54 / 'A57_B_epoch08_results_download_rpc.json')
    assert download['status'] == 'TRANSFERRED'
    archive = R54 / 'A57_downloads/B_epoch08_small_results_v1.zip'
    assert archive.stat().st_size == export['bytes'] and sha(archive) == export['sha256']
    target = P / 'B_epoch08_remote_v1'
    target.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == set(export['files'])
        for name, row in export['files'].items():
            raw = z.read(name)
            assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
            dest = (target / name).resolve()
            assert dest.is_relative_to(target.resolve())
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as f:
                f.write(raw)
    shutil.copy2(archive, P / archive.name)
    cp = export['checkpoint']
    source = R54 / 'A57_downloads/B_epoch08_RAW.pth'
    assert source.stat().st_size == cp['bytes'] and sha(source) == cp['sha256']
    os.link(source, target / 'arm_B/epoch08_RAW.pth')
    protocol_hash = 'd233e4a817a997a7a0c4ce203a098e3e0794d0622921fc6333659e374fc13a37'
    assert sha(target / 'protocol_FROZEN_v1.json') == sha(P / 'protocol_FROZEN_v1.json') == protocol_hash
    gt = load(target / 'arm_B/epoch08_RAW_GT.json')
    assert gt == load(P / 'validation_GT_FROZEN_v1.json')
    assert len(gt['images']) == 548 and len(gt['annotations']) == 38602
    metrics = load(target / 'arm_B/epoch08_RAW_metrics.json')
    previous_metrics = load(P / 'B_epoch04_remote_v1/arm_B/epoch04_RAW_metrics.json')
    pred = load(target / 'arm_B/epoch08_RAW_predictions.json')
    assert len(pred) == metrics['predictions_after_ignore'] == 156199
    ids = {v['id'] for v in gt['images']}
    assert all(v['image_id'] in ids and 1 <= v['category_id'] <= 10 for v in pred)
    boxes = np.asarray([v['bbox'] for v in pred], dtype=np.float64)
    scores = np.asarray([v['score'] for v in pred], dtype=np.float64)
    assert np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all()
    assert np.isfinite(scores).all() and (scores >= 0).all() and (scores <= 1).all()
    saved = np.load(target / 'arm_B/epoch08_RAW_precision_recall.npz', allow_pickle=False)
    precision, recall, ths = [saved[n] for n in ['precision', 'recall', 'iou_thresholds']]
    assert precision.shape == (10, 101, 10, 4, 3) and recall.shape == (10, 10, 4, 3)

    def ap(area=0, iou=None):
        values = precision[:, :, :, area, -1]
        if iou is not None:
            values = values[np.isclose(ths, iou)]
        values = values[values >= 0]
        return float(values.mean() * 100)

    small_recall = recall[:, :, 1, -1]
    small_recall = small_recall[small_recall >= 0]
    recomputed = {'AP': ap(), 'AP50': ap(iou=.5), 'AP75': ap(iou=.75),
                  'APs': ap(1), 'APs75': ap(1, .75), 'APm': ap(2), 'APl': ap(3),
                  'ARs': float(small_recall.mean() * 100)}
    assert all(abs(metrics[k] - v) < 1e-12 for k, v in recomputed.items())
    prefix = (target / 'B_epoch08_evaluation_events_prefix.jsonl').read_bytes()
    assert prefix.startswith((P / 'B_epoch04_remote_v1/B_epoch04_evaluation_events_prefix.jsonl').read_bytes())
    rows = [json.loads(line) for line in prefix.splitlines()]
    counts = dict(Counter(row['event'] for row in rows))
    assert counts == {'model_forward_entered': 13218, 'model_forward_completed': 13218,
                      'optimizer_step_attempt': 6472, 'optimizer_step_returned': 6472,
                      'training_epoch_completed': 8}
    assert [r['epoch'] for r in rows if r['event'] == 'training_epoch_completed'] == list(range(1, 9))
    assert rows[-1]['model_counter']['decoder_entries_including_recomputation'] == 156972
    ck = torch.load(source, map_location='cpu', weights_only=True)
    assert ck['epoch'] == 8 and ck['optimizer_updates'] == 6472 and ck['arm'] == 'B'
    assert ck['protocol_sha256'] == protocol_hash
    assert ck['initialization_sha256'] == '984c6a2413f08caf8db96c869f43a402bfd0449cb3a3dca6b439b73cfe508040'
    state = ck['model']
    assert len(state) == 2011 and all(bool(torch.isfinite(t).all()) for t in state.values() if t.is_floating_point())
    assert list(state['encoder.pos_encoding.pe'].shape) == [160, 160, 256]
    assert dict(Counter(int(t) for k, t in state.items() if k.endswith('num_batches_tracked'))) == {12944: 33}
    assert not torch.cuda.is_initialized()
    remote_audit = load(target / 'B_epoch08_CPU_checkpoint_audit_v1.json')
    assert remote_audit['checkpoint'] == cp and remote_audit['positional_cache_exactly_equal_to_B4']
    assert all(v['changed_from_initial'] for v in remote_audit['selected_tensors'].values())
    save(P / 'B_epoch08_transfer_verification.json', {'status': 'PASS_EXACT_BYTES_AND_SAFE_EXTRACTION',
         'archive_bytes': export['bytes'], 'archive_sha256': export['sha256'],
         'files_verified': len(export['files']), 'checkpoint': cp})
    audit = {'status': 'PASS_B8_SAVED_METRICS_GT_EVENTS_CHECKPOINT', 'metrics': metrics,
             'metrics_recomputed_from_saved_precision': recomputed,
             'B8_minus_B4_percentage_points': {k: metrics[k] - previous_metrics[k] for k in recomputed},
             'GT_semantically_identical': True, 'finite_positive_boxes_and_scores': True,
             'event_prefix_is_continuation_of_B4': True, 'event_prefix_counts': counts,
             'training_model_forwards': 12944, 'validation_model_forwards': 274,
             'checkpoint': cp, 'state_tensors': 2011, 'all_floating_tensors_finite': True,
             'positional_encoding_cache_shape': [160, 160, 256],
             'cold_restore': 'After common initialization, call encoder.pos_encoding._build_pe(160,160) before strict raw-state loading; no forward needed. Do not resume training.',
             'B8_observation_only': True, 'B12_learning_gate_assessed': False,
             'M_formal_training_started': False, 'Q_formal_training_started': False,
             'low_AP_cause_established': False, 'training_or_gate_changed': False,
             'additional_detector_instances': 0, 'additional_model_forwards': 0,
             'additional_optimizer_updates': 0, 'additional_COCO_evaluate_calls': 0, 'CUDA_initialized': False}
    save(P / 'B_epoch08_cached_verification_v1.json', audit)
    print(json.dumps({'status': audit['status'], 'B8_AP': metrics['AP'], 'B8_APs': metrics['APs'],
                      'B8_minus_B4_pp': audit['B8_minus_B4_percentage_points'],
                      'checkpoint': cp, 'gate_changed': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
