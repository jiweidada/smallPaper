"""Post-proposal descriptive audit of saved development predictions/matches.

No matcher replay, model creation/execution, gradients, optimizer, remote access,
or heldout data. Quality selection below is an ADAPTED pair-selection control,
not a reproduction of Align-DETR training.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[2]
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
sys.path.insert(0, str(WORK/'英文小论文/code/dn_pre_frozen_diagnostic_20261020_v1/runtime'))
import torch


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(output):
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    base = WORK/'英文小论文/data/08_匹配聚合覆盖探针_20261019_v1/isolated_v3'
    rows, sources = [], []
    for side, seed in [('A', 1), ('B', 2)]:
        for batch in range(8):
            ap = base/side/f'development/batch{batch:02d}_analysis.json'
            pp = base/side/f'frozen_predictions/development/batch{batch:02d}_predictions.pt'
            analysis = json.loads(ap.read_text(encoding='utf-8'))
            prediction_sha = sha(pp)
            assert prediction_sha == analysis['prediction_sha256']
            sources.append({'analysis': str(ap), 'analysis_sha256': sha(ap), 'predictions': str(pp), 'prediction_sha256': prediction_sha})
            payload = torch.load(pp, map_location='cpu', weights_only=False)
            for image, item in enumerate(analysis['inputs']):
                maps = {stage: dict(zip(qg[0], qg[1])) for stage, qg in analysis['branch_matches'][image].items()}
                pre = item['branches']['pre']['native_union']
                final = item['branches']['final']['native_union']
                union = dict(zip(pre['pair_query'], pre['pair_GT']))
                assert union == dict(zip(final['pair_query'], final['pair_GT']))
                pre_iou = dict(zip(pre['pair_query'], pre['pair_iou']))
                pre_l1 = dict(zip(pre['pair_query'], pre['pair_l1']))
                pre_giou = dict(zip(pre['pair_query'], pre['pair_giou']))
                logits = payload['outputs']['pre_outputs']['pred_logits'][image].sigmoid()
                labels = payload['targets'][image]['labels']
                selected, unchanged_pre, encoder_only, destination = [], [], [], Counter()
                source_set_sizes, source_sets = Counter(), Counter()
                equal_share_receiver_coefficients = Counter()
                bg, other, other_same_class, other_different_class = 0, 0, 0, 0
                for q, g in union.items():
                    if maps['pre'].get(q) == g:
                        unchanged_pre.append(q)
                        continue
                    origin = next((stage for stage in ['aux_0', 'aux_1', 'final'] if maps[stage].get(q) == g), None)
                    if origin is None:
                        assert maps['enc_0'].get(q) == g
                        encoder_only.append(q)
                        continue
                    selected.append(q)
                    destination[origin] += 1
                    source_set = tuple(stage for stage in ['aux_0', 'aux_1', 'final'] if maps[stage].get(q) == g)
                    source_set_sizes[len(source_set)] += 1
                    source_sets['+'.join(source_set)] += 1
                    for stage in source_set:
                        equal_share_receiver_coefficients[stage] += 1 / len(source_set)
                    if q not in maps['pre']:
                        bg += 1
                    else:
                        other += 1
                        if int(labels[maps['pre'][q]]) == int(labels[g]):
                            other_same_class += 1
                        else:
                            other_different_class += 1
                # Same per-GT number of transferred pairs, chosen only with the
                # original pre joint score, no future/final outputs for choice.
                selected_gt_counts = Counter(union[q] for q in selected)
                quality_selected, cutoff_ties, non_tied_symdiff = [], 0, 0
                for g, count in selected_gt_counts.items():
                    candidates = [q for q, gg in union.items() if gg == g]
                    quality = {q: float(logits[q, int(labels[g])])**.25 * max(0., pre_iou[q])**.75 for q in candidates}
                    ordered = sorted(candidates, key=lambda q: (quality[q], q))
                    quality_selected.extend(ordered[:count])
                    is_cutoff_tie = count < len(ordered) and quality[ordered[count-1]] == quality[ordered[count]]
                    if is_cutoff_tie:
                        cutoff_ties += 1
                    else:
                        native_g = {q for q in selected if union[q] == g}
                        non_tied_symdiff += len(native_g ^ set(ordered[:count]))
                selected_set, quality_set = set(selected), set(quality_selected)
                assert len(selected_set) == len(quality_set)
                losses = {q: 5*pre_l1[q] + 2*(1-pre_giou[q]) for q in union}
                total_loss = sum(losses.values())
                own_not_union = sum(union.get(q) != g for q, g in maps['pre'].items())
                row = {
                    'seed': seed, 'batch': batch, 'image': image, 'GT': item['GT'],
                    'union_pairs': len(union), 'pre_own_pairs': len(maps['pre']),
                    'kept_pre_own_union_pairs': len(unchanged_pre),
                    'encoder_only_union_pairs_not_transferred': len(encoder_only),
                    'transfer_pairs': len(selected), 'destination_counts': dict(destination),
                    'decoder_source_set_sizes': dict(source_set_sizes),
                    'decoder_source_sets': dict(source_sets),
                    'equal_share_receiver_coefficients_before_original_denominator': dict(equal_share_receiver_coefficients),
                    'pre_background_query_transfer_pairs': bg,
                    'pre_other_GT_query_transfer_pairs': other,
                    'pre_other_GT_same_class_transfer_pairs': other_same_class,
                    'pre_other_GT_different_class_transfer_pairs': other_different_class,
                    'pre_own_pairs_not_in_original_union': own_not_union,
                    'transfer_affected_GT': len(selected_gt_counts),
                    'quality_selection_overlap': len(selected_set & quality_set),
                    'quality_selection_symmetric_difference': len(selected_set ^ quality_set),
                    'quality_cutoff_tied_GT': cutoff_ties,
                    'quality_selection_symmetric_difference_on_non_tied_GT': non_tied_symdiff,
                    'original_pre_geometry_sum': total_loss,
                    'transferred_pre_geometry_sum': sum(losses[q] for q in selected),
                    'batch_original_num_boxes_go': sum(x['K'] for x in analysis['inputs']),
                    'transfer_pre_IoU_mean': sum(pre_iou[q] for q in selected)/len(selected) if selected else None,
                    'kept_pre_own_IoU_mean': sum(pre_iou[q] for q in unchanged_pre)/len(unchanged_pre) if unchanged_pre else None,
                    'transfer_share_of_original_pre_weighted_geometry_sum': sum(losses[q] for q in selected)/total_loss if total_loss else None,
                }
                assert len(union) == len(selected) + len(unchanged_pre) + len(encoder_only)
                assert len(selected) == bg + other == sum(destination.values())
                rows.append(row)
    aggregates = []
    for seed in [1, 2]:
        subset = [r for r in rows if r['seed'] == seed]
        fields = ['GT', 'union_pairs', 'transfer_pairs', 'pre_background_query_transfer_pairs', 'pre_other_GT_query_transfer_pairs', 'pre_other_GT_same_class_transfer_pairs', 'pre_other_GT_different_class_transfer_pairs', 'encoder_only_union_pairs_not_transferred', 'quality_selection_overlap', 'quality_selection_symmetric_difference', 'quality_cutoff_tied_GT', 'quality_selection_symmetric_difference_on_non_tied_GT', 'pre_own_pairs_not_in_original_union']
        agg = {'seed': seed, 'images': len(subset), **{key: sum(r[key] for r in subset) for key in fields}}
        agg['destination_counts'] = {stage: sum(r['destination_counts'].get(stage, 0) for r in subset) for stage in ['aux_0', 'aux_1', 'final']}
        agg['decoder_source_set_sizes'] = {str(k): sum(r['decoder_source_set_sizes'].get(k, 0) for r in subset) for k in [1, 2, 3]}
        agg['decoder_source_sets'] = dict(sum((Counter(r['decoder_source_sets']) for r in subset), Counter()))
        agg['equal_share_receiver_coefficients_before_original_denominator'] = {stage: sum(r['equal_share_receiver_coefficients_before_original_denominator'].get(stage, 0) for r in subset) for stage in ['aux_0', 'aux_1', 'final']}
        agg['sum_of_original_pre_loss_over_batches'] = sum(r['original_pre_geometry_sum']/r['batch_original_num_boxes_go'] for r in subset)
        agg['sum_of_transferred_pre_loss_over_batches'] = sum(r['transferred_pre_geometry_sum']/r['batch_original_num_boxes_go'] for r in subset)
        agg['transferred_fraction_of_original_pre_loss'] = agg['sum_of_transferred_pre_loss_over_batches']/agg['sum_of_original_pre_loss_over_batches']
        agg['images_with_multiple_receiver_layers'] = sum(len(r['destination_counts']) > 1 for r in subset)
        aggregates.append(agg)
    result = {'scope': 'Exploratory descriptive audit, 8 saved development batches/seed, same 32 augmented inputs across seeds. No causal effect or independent validation claim. Original saved matches, no matcher replay. Selection control uses pre sigmoid class^.25 times IoU^.75 with deterministic query-ID tie break; same per-GT selected counts, NOT full Align-DETR training nor matched destination-budget control. No heldout or model runs.', 'sources': sources, 'aggregate': aggregates, 'rows': rows}
    with Path(output).open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(aggregates, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
