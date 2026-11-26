"""Local data-only terminal verification. No model construction or evaluator rerun."""
from pathlib import Path
from collections import Counter
import hashlib
import json
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
        for v in iter(lambda: f.read(4 << 20), b''):
            h.update(v)
    return h.hexdigest()


def save(p, obj):
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    export_rpc = load(R54 / 'A57_B12_terminal_export_rpc.json')
    assert export_rpc['exit'] == 0
    export = json.loads(export_rpc['stdout'])
    download = load(R54 / 'A57_B12_terminal_download_rpc.json')
    assert download['status'] == 'TRANSFERRED' and len(download['files']) == 3
    archive = R54 / 'A57_downloads/B12_terminal_small_v1.zip'
    assert sha(archive) == export['sha256'] and archive.stat().st_size == export['bytes']
    dest = P / 'B12_terminal_remote_v1'
    dest.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist()) == set(export['files'])
        for name, row in export['files'].items():
            data = z.read(name)
            assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
            path = (dest / name).resolve()
            assert path.is_relative_to(dest.resolve())
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as f:
                f.write(data)
    shutil.copy2(archive, P / archive.name)
    for field in ['raw_checkpoint', 'optimizer_checkpoint']:
        row = export[field]
        f = R54 / 'A57_downloads' / ('B12_' + row['file'])
        assert f.stat().st_size == row['bytes'] and sha(f) == row['sha256']
        os.link(f, dest / 'arm_B' / row['file'])
    for old_stage in ['B_epoch04_remote_v1', 'B_epoch08_remote_v1']:
        old = P / old_stage / 'arm_B'
        for f in old.iterdir():
            if f.is_file() and f.name not in ['run_manifest.json', 'epoch04_RAW.pth', 'epoch08_RAW.pth']:
                if (dest / 'arm_B' / f.name).exists():
                    assert sha(f) == sha(dest / 'arm_B' / f.name), f.name
    protocol = load(P / 'protocol_FROZEN_v1.json')
    protocol_sha = sha(P / 'protocol_FROZEN_v1.json')
    assert protocol_sha == sha(dest / 'protocol_FROZEN_v1.json') == 'd233e4a817a997a7a0c4ce203a098e3e0794d0622921fc6333659e374fc13a37'
    assert all(sha(P / n) == h for n, h in protocol['files_sha256'].items())
    assert sha(P / 'common_initialization.pth') == protocol['initialization_sha256']
    gt = load(dest / 'arm_B/epoch12_RAW_GT.json')
    assert gt == load(P / 'validation_GT_FROZEN_v1.json')
    assert len(gt['images']) == 548 and len(gt['annotations']) == 38602
    metrics = load(dest / 'arm_B/epoch12_RAW_metrics.json')
    predictions = load(dest / 'arm_B/epoch12_RAW_predictions.json')
    assert len(predictions) == metrics['predictions_after_ignore'] == 155183
    ids = {i['id'] for i in gt['images']}
    assert all(v['image_id'] in ids and 1 <= v['category_id'] <= 10 for v in predictions)
    boxes = np.array([v['bbox'] for v in predictions], dtype=np.float64)
    scores = np.array([v['score'] for v in predictions], dtype=np.float64)
    assert np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all()
    assert np.isfinite(scores).all() and (scores >= 0).all() and (scores <= 1).all()
    arrays = np.load(dest / 'arm_B/epoch12_RAW_precision_recall.npz', allow_pickle=False)
    precision, recall, ths = [arrays[k] for k in ['precision', 'recall', 'iou_thresholds']]
    assert precision.shape == (10, 101, 10, 4, 3) and recall.shape == (10, 10, 4, 3)

    def ap(area=0, iou=None):
        v = precision[:, :, :, area, -1]
        if iou is not None:
            v = v[np.isclose(ths, iou)]
        return float(v[v >= 0].mean() * 100)

    rec = recall[:, :, 1, -1]
    recomputed = {'AP': ap(), 'AP50': ap(iou=.5), 'AP75': ap(iou=.75),
                  'APs': ap(1), 'APs75': ap(1, .75), 'APm': ap(2), 'APl': ap(3),
                  'ARs': float(rec[rec >= 0].mean() * 100)}
    assert all(abs(metrics[k] - v) < 1e-12 for k, v in recomputed.items())
    events_raw = (dest / 'arm_B/events.jsonl').read_bytes()
    assert events_raw.startswith((P / 'B_epoch08_remote_v1/B_epoch08_evaluation_events_prefix.jsonl').read_bytes())
    events = [json.loads(line) for line in events_raw.splitlines()]
    counts = dict(Counter(row['event'] for row in events))
    assert len(events) == 59162 and counts == {
        'model_forward_entered': 19867, 'model_forward_completed': 19867,
        'optimizer_step_attempt': 9708, 'optimizer_step_returned': 9708, 'training_epoch_completed': 12}
    assert [e['epoch'] for e in events if e['event'] == 'training_epoch_completed'] == list(range(1, 13))
    counter = {'entries': 19867, 'completions': 19867, 'decoder_entries_including_recomputation': 235698}
    assert events[-1]['model_counter'] == counter
    supervisor = load(dest / 'supervisor_exit.json')
    worker = load(dest / 'arm_B/exit.json')
    snapshot = load(dest / 'B12_terminal_process_snapshot_v1.json')
    assert supervisor == load(dest / 'supervisor_state.json')
    assert supervisor['status'] == 'COMPLETED_BOUNDED_SCREEN' and supervisor['exit_code'] == 0
    assert supervisor['queue'] == [] and supervisor['cancelled'] == ['M', 'Q'] and supervisor['active_arm'] is None
    assert supervisor['post_training_queue'] is None
    assert supervisor['verdict']['status'] == 'BASELINE_INADEQUATE_SHORT_HORIZON_STOPPED'
    assert supervisor['verdict']['candidate_trained'] is False
    assert all(j['current_identity'] is None and not j['owned_live'] for j in snapshot['jobs'])
    assert snapshot['GPU_compute_processes'] == '' and snapshot['only_B_formal_directory_exists']
    assert worker['status'] == 'COMPLETED_FIXED12_RAW' and worker['exit_code'] == 0
    assert worker['model_counter'] == counter and worker['metrics_fixed12'] == metrics
    assert worker['counts']['epochs_completed'] == 12 and worker['counts']['optimizer_updates'] == 9708
    ck = torch.load(dest / 'arm_B/epoch12_RAW.pth', map_location='cpu', weights_only=True)
    assert ck['epoch'] == 12 and ck['optimizer_updates'] == 9708 and ck['protocol_sha256'] == protocol_sha
    assert ck['initialization_sha256'] == protocol['initialization_sha256']
    model = ck['model']
    assert len(model) == 2011 and all(bool(torch.isfinite(t).all()) for t in model.values() if t.is_floating_point())
    assert dict(Counter(int(t) for k, t in model.items() if k.endswith('num_batches_tracked'))) == {19416: 33}
    assert list(model['encoder.pos_encoding.pe'].shape) == [160, 160, 256]
    del ck, model
    opt = torch.load(dest / 'arm_B/epoch12_optimizer.pth', map_location='cpu', weights_only=True)
    assert opt['epoch'] == 12 and opt['optimizer_updates'] == 9708 and opt['protocol_sha256'] == protocol_sha
    states = opt['optimizer']['state']
    assert dict(Counter(int(s['step']) for s in states.values())) == {9708: 1898}
    assert all(bool(torch.isfinite(t).all()) for row in states.values()
               for t in row.values() if isinstance(t, torch.Tensor) and t.is_floating_point())
    assert not torch.cuda.is_initialized()
    gates = {'B_AP_ge_8': metrics['AP'] >= 8, 'B_APs_ge_3': metrics['APs'] >= 3}
    assert not any(gates.values())
    audit = {'status': 'PASS_TERMINAL_B12_BYTES_CACHED_AP_GT_EVENTS_WEIGHTS_OPTIMIZER',
             'metrics': metrics, 'metrics_recomputed_from_saved_arrays': recomputed,
             'gates': gates, 'decision': 'CANCEL_UNSTARTED_M_Q_EXIT_HOST_SHORT_RECIPE_Q_UNTESTED',
             'training_process_success': True, 'formal_B_epochs': 12, 'formal_B_updates': 9708,
             'formal_B_model_calls': 19867, 'train_model_calls': 19416, 'validation_model_calls': 411,
             'latency_model_calls': 40, 'event_counts': counts,
             'new_arms_started': ['B'], 'cancelled_arms': ['M', 'Q'],
             'owned_processes_absent': True, 'GPU_compute_processes_empty': True,
             'source_bound_protocol_unchanged': True, 'GT_semantically_identical': True,
             'all_model_and_optimizer_floating_tensors_finite': True,
             'optimizer_active_parameter_states': 1898, 'optimizer_steps_each': 9708,
             'raw_checkpoint': export['raw_checkpoint'], 'optimizer_checkpoint': export['optimizer_checkpoint'],
             'archive_bytes': export['bytes'], 'archive_sha256': export['sha256'],
             'small_files_verified': len(export['files']),
             'cold_restore': 'After common initialization, build encoder.pos_encoding cache160x160 before strict loading raw state; no forward required. No training resume authorized by this audit.',
             'low_AP_cause_established': False, 'candidate_efficacy_assessed': False,
             'recipe_or_gate_changed': False, 'additional_detector_instances': 0,
             'additional_model_forwards': 0, 'additional_optimizer_updates': 0,
             'additional_COCO_evaluate_calls': 0, 'CUDA_initialized': False}
    save(P / 'B12_terminal_local_verification_v1.json', audit)
    print(json.dumps({k: audit[k] for k in ['status', 'metrics', 'gates', 'decision',
                                           'formal_B_model_calls', 'formal_B_updates', 'small_files_verified']}))


if __name__ == '__main__':
    main()
