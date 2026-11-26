"""Summarize verified saved diagnostic outputs; does not run models or evaluators."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import backup_guard as guard

WORK = guard.REMOTE_WORK
ROOT = WORK / 'diagnostic_backups_20261105' / guard.RUN_ID
OLD = WORK / 'remote_evidence/interrupted_BASE_20261105/runs/BASE/epochs.jsonl'
OLD_PROTOCOL = '8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'
LABELS = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle',
          'awning-tricycle', 'bus', 'motor']

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def describe(phase):
    guard.read_ack_for_previous(ROOT, phase)
    stage = ROOT / phase
    manifest = guard.parse_manifest((stage / 'manifest.json').read_bytes())
    metric = read(stage / 'metrics.json')
    with np.load(stage / 'evaluation_arrays.npz', allow_pickle=False) as arrays:
        precision = arrays['precision']
        assert precision.shape == (10, 101, 10, 4, 3)
        category_ap = []
        for k in range(10):
            values = precision[:, :, k, 0, 2]
            values = values[values > -1]
            category_ap.append(float(values.mean()) if values.size else -1.0)
    predictions = read(stage / 'predictions.json')
    assert len(predictions) == 548 * 300
    tops = np.full(548, -np.inf)
    counts = np.zeros(548, dtype=np.int32)
    for item in predictions:
        image = item['image_id']
        score = item['score']
        assert type(image) is int and 0 <= image < 548 and math.isfinite(score)
        tops[image] = max(tops[image], score)
        counts[image] += 1
    assert bool(np.all(counts == 300)) and bool(np.isfinite(tops).all())
    return {'phase': phase, 'epoch_index': manifest['epoch_index'],
            'optimizer_updates': manifest['optimizer_updates'], 'validation_passes': manifest['validation_passes'],
            'AP': metric['metrics']['AP'], 'category_AP': dict(zip(LABELS, category_ap)),
            'top_score_by_image_quantiles': dict(zip(['min', 'q25', 'median', 'q75', 'max'],
                                                       map(float, np.quantile(tops, [0, .25, .5, .75, 1])))),
            'checkpoint_sha256': metric['checkpoint_sha256'],
            'evaluated_tensor_sha256': metric['evaluated_tensor_sha256']}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['partial', 'final'], required=True)
    args = parser.parse_args()
    OUT = WORK / ('diagnostic_analysis_' + args.mode + '_20261105')
    guard.check_plans()
    assert not OUT.exists() and not OUT.is_symlink()
    names = ['initial_ema', 'initial_raw'] + [
        'epoch_%04d' % e for e in range(214, 225) if (ROOT / ('epoch_%04d' % e) / 'ack.json').is_file()]
    if (ROOT / 'terminal_raw/ack.json').is_file():
        names.append('terminal_raw')
    if args.mode == 'final':
        assert names[-1] == 'terminal_raw'
    assert names[:2] == ['initial_ema', 'initial_raw']
    observations = [describe(name) for name in names]
    epochs = [o['epoch_index'] for o in observations if o['phase'].startswith('epoch_')]
    assert epochs == list(range(214, 214 + len(epochs)))
    historic = [json.loads(line) for line in OLD.read_text(encoding='utf-8').splitlines()]
    assert len(historic) == 225 and all(x['epoch'] == i for i, x in enumerate(historic))
    assert all(x['protocol_sha256'] == OLD_PROTOCOL for x in historic[213:225])
    old_points = [{'epoch_index': x['epoch'], 'AP': x['metrics']['AP']} for x in historic[213:225]]
    OUT.mkdir(exist_ok=False)
    result = {'status': 'SAVED_DIAGNOSTIC_TRAJECTORY_SUMMARIZED', 'mode': args.mode,
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'new_diagnostic': observations, 'interrupted_original_ema': old_points,
              'stop_threshold_AP': observations[0]['AP'] - 0.05,
              'model_inference_passes': 0, 'evaluator_reruns': 0, 'optimizer_updates': 0,
              'scope': 'Descriptive comparison only. Initial RNG state was unavailable, so the new trajectory is not an exact replay; neither curve proves a cause or a fair BASE/FULL comparison.',
              'top_score_note': 'Top score is the highest prediction score per image; top predictions across phases need not correspond to the same object or class.'}
    with (OUT / 'trajectory.json').open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    ema = [x for x in observations if x['phase'] == 'initial_ema' or x['phase'].startswith('epoch_')]
    raw = [x for x in observations if x['phase'] in ['initial_raw', 'terminal_raw']]
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.plot([x['epoch_index'] for x in old_points], [x['AP'] for x in old_points],
            color='#818a93', linestyle='--', marker='o', markersize=3.5, label='Interrupted original: EMA')
    ax.plot([x['epoch_index'] for x in ema], [x['AP'] for x in ema],
            color='#1769a5', marker='o', markersize=4, linewidth=2, label='New diagnostic: EMA')
    ax.scatter([x['epoch_index'] for x in raw], [x['AP'] for x in raw],
               color='#bd4b36', marker='D', s=40, zorder=4, label='New diagnostic: raw model')
    ax.axhline(result['stop_threshold_AP'], color='#b17b18', linestyle=':', linewidth=1.5,
               label='Diagnostic stop threshold (AP drop 0.05)')
    ax.set(xlabel='Training epoch index (zero-based)', ylabel='AP on 548 validation images',
           title='BASE validation trajectory and diagnostic checkpoints')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=.22)
    ax.legend(frameon=False, fontsize=8.5, loc='best')
    fig.text(.5, -.015, 'New trajectory uses reset RNG; it is not an exact replay or a fair BASE/FULL comparison.',
             ha='center', fontsize=8, color='#454b50')
    fig.savefig(OUT / 'trajectory.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT / 'trajectory.pdf', bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'status': result['status'], 'phase_count': len(observations), 'output': str(OUT)}))

if __name__ == '__main__':
    main()
