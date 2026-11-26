"""Audit saved diagnostic files only; no model inference, evaluator or optimizer."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import argparse
import json
import math
import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import torch

PAPER = Path(__file__).resolve().parent.parent
WORK = PAPER.parent
REMOTE = PAPER / '远程公平对照_20261103'
sys.path.insert(0, str(REMOTE / '诊断准备_20261105'))
import backup_guard as guard

ROOT = REMOTE / 'diagnostic_backups_20261105' / guard.RUN_ID
ORIGINAL = REMOTE / 'remote_evidence/interrupted_BASE_20261105/runs/BASE/best.pt'
ANNOTATIONS = WORK / '05_代码与数据/原始论文数据/03_本轮实验记录_20260928/转换COCO标注原件_20260929/VisDrone2019-DET-val/annotations/val.json'
ANNOTATION_SHA = '0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd'
NAMES = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def tensor_hash(state):
    h = sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        h.update(json.dumps([key, str(tensor.dtype), list(tensor.shape)]).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()

def summaries(path):
    with np.load(path, allow_pickle=False) as arrays:
        p, r = arrays['precision'], arrays['recall']
        assert p.shape == (10, 101, 10, 4, 3) and r.shape == (10, 10, 4, 3)
        selected = [p[:, :, :, 0, 2], p[0, :, :, 0, 2], p[5, :, :, 0, 2],
                    p[:, :, :, 1, 2], p[:, :, :, 2, 2], p[:, :, :, 3, 2],
                    r[:, :, 0, 0], r[:, :, 0, 1], r[:, :, 0, 2],
                    r[:, :, 1, 2], r[:, :, 2, 2], r[:, :, 3, 2]]
        def mean_valid(a):
            values = a[a > -1]
            return float(np.mean(values)) if values.size else -1.0
        return dict(zip(NAMES, map(mean_valid, selected)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['initial', 'available', 'complete'], required=True)
    args = parser.parse_args()
    output = REMOTE / ('diagnostic_saved_file_audit_' + args.mode + '_20261105.json')
    assert not output.exists() and not output.is_symlink()
    guard.check_plans()
    assert guard.digest_file(ORIGINAL) == guard.CHECKPOINT_SHA
    assert guard.digest_file(ANNOTATIONS) == ANNOTATION_SHA
    annotation = read(ANNOTATIONS)
    ids = {x['id'] for x in annotation['images']}
    categories = {x['id'] for x in annotation['categories']}
    assert len(ids) == 548 and len(categories) == 10
    initial = ['initial_ema', 'initial_raw']
    epoch_names = ['epoch_%04d' % n for n in range(214, 225) if (ROOT / ('epoch_%04d' % n) / 'ack.json').is_file()]
    phases = initial if args.mode == 'initial' else initial + epoch_names
    if args.mode != 'initial' and (ROOT / 'terminal_raw/ack.json').is_file():
        phases += ['terminal_raw']
    if args.mode == 'complete':
        assert epoch_names and phases[-1] == 'terminal_raw'
    assert epoch_names == ['epoch_%04d' % n for n in range(214, 214 + len(epoch_names))]
    torch.set_num_threads(1)
    original = torch.load(ORIGINAL, map_location='cpu', weights_only=True)
    results = []
    for phase in phases:
        guard.read_ack_for_previous(ROOT, phase)
        stage = ROOT / phase
        manifest = guard.parse_manifest((stage / 'manifest.json').read_bytes())
        metric = read(stage / 'metrics.json')
        assert metric['phase'] == phase and metric['epoch_index'] == manifest['epoch_index']
        assert metric['images'] == 548 and metric['postprocessor_predictions_per_image'] == 300
        assert guard.digest_file(stage / 'predictions.json') == metric['prediction_sha256']
        assert guard.digest_file(stage / 'evaluation_arrays.npz') == metric['evaluation_arrays_sha256']
        if phase in initial:
            checkpoint, checkpoint_path = original, ORIGINAL
        else:
            checkpoint_path = ROOT / ('epoch_%04d' % manifest['epoch_index']) / 'checkpoint.pt'
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            assert checkpoint['arm'] == 'BASE' and checkpoint['actual_loop_epoch'] == manifest['epoch_index']
            assert checkpoint['additional_optimizer_updates'] == manifest['optimizer_updates']
            assert checkpoint['optimizer_step_attempts'] == 214 * 1617 + manifest['optimizer_updates']
            assert checkpoint['original_checkpoint_sha256'] == guard.CHECKPOINT_SHA
            assert {'python', 'numpy', 'torch_cpu', 'torch_cuda_all', 'train_loader_generator',
                    'train_sampler_generator', 'validation_loader_generator'} <= checkpoint['rng_state'].keys()
        assert guard.digest_file(checkpoint_path) == metric['checkpoint_sha256']
        evaluated = checkpoint['ema']['module'] if phase == 'initial_ema' or phase.startswith('epoch_') else checkpoint['model']
        assert tensor_hash(evaluated) == metric['evaluated_tensor_sha256']
        counts = dict.fromkeys(ids, 0)
        predictions = read(stage / 'predictions.json')
        assert len(predictions) == 164400
        for prediction in predictions:
            assert prediction['image_id'] in ids and prediction['category_id'] in categories
            assert len(prediction['bbox']) == 4 and all(math.isfinite(v) for v in prediction['bbox'])
            assert math.isfinite(prediction['score'])
            counts[prediction['image_id']] += 1
        assert set(counts.values()) == {300}
        values = summaries(stage / 'evaluation_arrays.npz')
        maximum = max(abs(values[name] - metric['metrics'][name]) for name in NAMES)
        assert maximum <= 1e-12
        results.append({'phase': phase, 'epoch_index': manifest['epoch_index'],
                        'optimizer_updates': manifest['optimizer_updates'], 'validation_passes': manifest['validation_passes'],
                        'checkpoint_sha256': metric['checkpoint_sha256'],
                        'evaluated_tensor_sha256': metric['evaluated_tensor_sha256'],
                        'metrics': metric['metrics'], 'max_array_summary_difference': maximum,
                        'images': len(ids), 'predictions': len(predictions)})
    report = {'status': 'DIAGNOSTIC_SAVED_EVIDENCE_VERIFIED', 'mode': args.mode,
              'created_local_utc': datetime.now(timezone.utc).isoformat(), 'phases': results,
              'model_inference_passes': 0, 'evaluator_reruns': 0, 'optimizer_updates': 0,
              'interpretation_limit': 'Integrity of saved diagnostic evidence only; not a fair model comparison or proof of cause.'}
    with output.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({'status': report['status'], 'mode': args.mode, 'phases': len(results), 'report': str(output)}))

if __name__ == '__main__':
    main()
