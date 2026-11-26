"""Verify saved training observations and export actual supervision rows.

Descriptive engineering fixture only; gradient magnitudes do not prove causes.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aligned_iou(a, b):
    aa = np.concatenate([a[:, :2] - a[:, 2:] / 2, a[:, :2] + a[:, 2:] / 2], axis=1)
    bb = np.concatenate([b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], axis=1)
    wh = np.maximum(np.minimum(aa[:, 2:], bb[:, 2:]) - np.maximum(aa[:, :2], bb[:, :2]), 0)
    inter = wh.prod(axis=1)
    union = np.maximum(a[:, 2:], 0).prod(axis=1) + np.maximum(b[:, 2:], 0).prod(axis=1) - inter
    return inter / np.maximum(union, 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    capture = Path(args.capture)
    manifest = json.loads((capture / 'manifest.json').read_text(encoding='utf8'))['files']
    for name, expected in manifest.items():
        assert digest(capture / name) == expected, name
    evidence = json.loads((capture / 'evidence.json').read_text(encoding='utf8'))
    assert evidence['status'] == 'CAPTURED' and not evidence['errors']
    arrays = np.load(capture / 'arrays.npz', allow_pickle=False)
    assert len(arrays.files) == evidence['arrays']
    assert sum(arrays[k].nbytes for k in arrays.files) == evidence['array_bytes']
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    pair_rows = []
    pair_sets = {}
    loss_summaries = []
    weighted_count = 0
    for event in evidence['events']:
        if event['kind'] == 'criterion':
            for name, value in event['weighted_loss_scalars'].items():
                assert float(arrays['weighted_loss.' + name]) == value
                weighted_count += 1
        if event['kind'] not in ('matcher', 'loss_call'):
            continue
        assert event['stage'] != 'UNRESOLVED'
        for batch in event['batches']:
            b = batch['batch']
            prefix = event['prefix'] + f'.batch{b}'
            q = arrays[prefix + '.query_index']
            g = arrays[prefix + '.GT_row_index']
            assert len(q) == len(g) == batch['pairs']
            assert len(np.unique(q)) == batch['unique_queries']
            assert len(np.unique(g)) == batch['unique_GT_rows']
            assert np.all((g >= 0) & (g < len(arrays[f'target.{b}.labels'])))
            pairs = set(zip(q.tolist(), g.tolist()))
            pair_sets[(event['kind'], event['stage'], event.get('loss'), b)] = pairs
            if event['kind'] == 'matcher':
                continue
            predicted = arrays[prefix + '.pred_boxes']
            target = arrays[prefix + '.target_boxes']
            labels = arrays[prefix + '.effective_labels']
            logits = arrays[prefix + '.pred_logits']
            assert np.array_equal(target, arrays[f'target.{b}.boxes'][g])
            # Effective labels can differ for class-agnostic encoder supervision.
            assert np.all((labels >= 0) & (labels < logits.shape[-1]))
            iou = aligned_iou(predicted, target)
            probability = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
            pred_class = logits.argmax(axis=-1)
            source_key = f'target.{b}.verified_source_annotation_ids'
            source_ids = arrays[source_key][g] if source_key in arrays else [None] * len(g)
            for j in range(len(g)):
                pair_rows.append(dict(stage=event['stage'], loss=event['loss'], DN=event['is_DN'], batch=b,
                    query=int(q[j]), GT_row=int(g[j]), source_annotation_id=source_ids[j],
                    original_class=int(arrays[f'target.{b}.labels'][g[j]]), effective_class=int(labels[j]),
                    predicted_class=int(pred_class[j]), effective_class_probability=float(probability[j, labels[j]]),
                    IoU=float(iou[j]), input_short_edge_pixels=float(min(target[j, 2:]) * arrays['input.normalized_tensor'].shape[-1])))
            loss_summaries.append(dict(stage=event['stage'], loss=event['loss'], DN=event['is_DN'], batch=b,
                pairs=len(g), unique_GT_rows=len(np.unique(g)), unique_queries=len(np.unique(q)),
                IoU_mean=float(iou.mean()) if len(g) else None, geometric_pairs_IoU50=int((iou >= .5).sum()),
                geometric_pairs_IoU75=int((iou >= .75).sum()), effective_class_correct=int((pred_class == labels).sum())))
    comparisons = []
    for (kind, stage, loss, batch), pairs in pair_sets.items():
        if kind != 'loss_call':
            continue
        own = pair_sets.get(('matcher', stage, None, batch))
        if own is not None:
            comparisons.append(dict(stage=stage, loss=loss, batch=batch, actual_supervision_pairs=len(pairs),
                own_layer_matcher_pairs=len(own), shared_pairs=len(pairs & own),
                supervision_pairs_outside_own_matcher=len(pairs - own), own_matcher_pairs_unused=len(own - pairs)))
    with (output / 'actual_supervision_pairs.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader(); writer.writerows(pair_rows)
    with (output / 'activation_gradient_stats.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=['name', 'shape', 'numel', 'nonfinite', 'finite_l2', 'finite_max_abs'])
        writer.writeheader(); writer.writerows(evidence['gradients'])
    report = dict(status='VERIFIED_SAVED_TRAINING_EVIDENCE', immutable_files_verified=len(manifest),
        arrays_verified=len(arrays.files), actual_supervision_rows=len(pair_rows), weighted_losses_verified=weighted_count,
        matcher_calls=sum(e['kind']=='matcher' for e in evidence['events']), loss_calls=len([e for e in evidence['events'] if e['kind']=='loss_call']),
        activation_gradient_observations=len(evidence['gradients']), activation_gradient_nonfinite=sum(e['nonfinite'] for e in evidence['gradients']),
        parameter_gradient_phases=list(evidence['parameter_gradient_snapshots']), actual_optimizer_steps=evidence['actual_optimizer_steps_in_sample'],
        own_layer_matcher_vs_actual_supervision=comparisons, descriptive_loss_pair_summaries=loss_summaries,
        limits='Actual indices and saved arithmetic verified for one engineering fixture. Pair geometry is not AP or COCO recall; repeated union and denoising GT rows are not independent objects. No historical learning evidence or root-cause inference. Gradient magnitudes alone cannot identify causal failure.')
    (output / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('own_layer_matcher_vs_actual_supervision','descriptive_loss_pair_summaries','limits')}))


if __name__ == '__main__':
    main()
