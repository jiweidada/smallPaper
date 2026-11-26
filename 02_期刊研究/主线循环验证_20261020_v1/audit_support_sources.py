"""Cache-only source/support audit, original same-forward GPU matches/boxes.

No model construction, network differentiation, optimizer, or heldout reads.
"""
import hashlib, json, os
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = WORK/'英文小论文/data/08_匹配聚合覆盖探针_20261019_v1/isolated_v3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
sys.path.insert(0, str(WORK/'英文小论文/code/dn_pre_frozen_diagnostic_20261020_v1/runtime'))
import torch
import numpy as np
import psutil


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path, obj):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def xyxy(x):
    return np.concatenate([x[..., :2]-x[..., 2:]/2, x[..., :2]+x[..., 2:]/2], axis=-1)


def aligned_iou(a, b):
    inter = np.maximum(0, np.minimum(a[:, 2:], b[:, 2:])-np.maximum(a[:, :2], b[:, :2])).prod(-1)
    union = (a[:, 2:]-a[:, :2]).prod(-1)+(b[:, 2:]-b[:, :2]).prod(-1)-inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def main():
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    p = psutil.Process(); p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    p.cpu_affinity(p.cpu_affinity()[-1::-2][:2])
    # Independent synthetic checks before loading any outcome tensors.
    rr = np.array([[.5, .5, .1, .1]])
    lo = np.concatenate([rr[:, :2]-1.5*rr[:, 2:], rr[:, :2]-.5*rr[:, 2:]], -1)
    hi = np.concatenate([rr[:, :2]+.5*rr[:, 2:], rr[:, :2]+1.5*rr[:, 2:]], -1)
    target = xyxy(rr)
    assert aligned_iou(np.clip(target, lo, hi), target)[0] == 1
    far = np.array([[.8, .8, .9, .9]])
    assert aligned_iou(np.clip(far, lo, hi), far)[0] == 0
    spec = {'scope': 'BASE24 saved development00-07, same-forward boxes and native branch matches; per-pair exposures, not independent GT',
        'primary_cohort': 'final native union, IoU<0.1 and exact GT outside closed fixed FDR support',
        'separate_definitions': ['exact GT outside support', 'maximum achievable IoU<.5'],
        'capacity_size_strata': 'all / image GT<=Q / image GT>Q, crossed with all / normalized input shortside<4 at640',
        'questions': ['fraction of bad outside-support pairs absent from final own match',
                      'are the same query fixed supports actually different across decoder layers',
                      'does outside support imply predicted offsets near requested endpoints'],
        'no_acceptance_gate': 'Claude proposed30percent but it is not adopted as an evidence-derived or preregistered stopping threshold. Descriptive table only; no training on nonzero counts.',
        'sources': 'old GPU same-forward analysis/prediction pair, never data10 CPU indices',
        'tests_before_data': 'support algebra for up=.5/reg_scale4 recovers prior bounds; all rows checked against saved IoU',
        'script_sha256': sha(__file__), 'new_model_forwards': 0, 'new_backwards': 0, 'heldout_reads': False}
    save(OUT/'05_来源与支持核查预先口径.json', spec)
    rows, sources = [], []
    layer_reference_error = 0.
    pre_reference_error = 0.
    max_iou_error = 0.
    for side, seed in [('A', 1), ('B', 2)]:
        for batch in range(8):
            ap = DATA/side/f'development/batch{batch:02d}_analysis.json'
            pp = DATA/side/f'frozen_predictions/development/batch{batch:02d}_predictions.pt'
            analysis = json.loads(ap.read_text(encoding='utf8'))
            digest = sha(pp); assert digest == analysis['prediction_sha256']
            sources.append({'analysis': str(ap), 'analysis_sha256': sha(ap), 'prediction': str(pp), 'prediction_sha256': digest})
            v = torch.load(pp, map_location='cpu', weights_only=False)
            outputs = v['outputs']
            reg_scale, up = abs(float(outputs['reg_scale'])), abs(float(outputs['up'].reshape(-1)[0]))
            limit = 2*reg_scale*up
            Q = outputs['pred_boxes'].shape[1]
            for image, item in enumerate(analysis['inputs']):
                native = item['branches']['final']['native_union']
                qs, gs = np.array(native['pair_query'], dtype=int), np.array(native['pair_GT'], dtype=int)
                maps = {stage: dict(zip(pair[0], pair[1])) for stage, pair in analysis['branch_matches'][image].items()}
                ref = outputs['ref_points'][image, qs].numpy().astype(float)
                prediction = xyxy(outputs['pred_boxes'][image, qs].numpy().astype(float))
                gt_box = v['targets'][image]['boxes'][gs].numpy().astype(float)
                gt = xyxy(gt_box)
                assert np.all(ref[:, 2:] > 0)
                required = np.concatenate([(ref[:, :2]-gt[:, :2])*reg_scale/ref[:, 2:]-reg_scale/2,
                                           (gt[:, 2:]-ref[:, :2])*reg_scale/ref[:, 2:]-reg_scale/2], axis=-1)
                actual = np.concatenate([(ref[:, :2]-prediction[:, :2])*reg_scale/ref[:, 2:]-reg_scale/2,
                                         (prediction[:, 2:]-ref[:, :2])*reg_scale/ref[:, 2:]-reg_scale/2], axis=-1)
                wanted_outside = np.abs(required) > limit
                lower = np.concatenate([ref[:, :2]-(.5+limit/reg_scale)*ref[:, 2:],
                                        ref[:, :2]+(.5-limit/reg_scale)*ref[:, 2:]], axis=-1)
                upper = np.concatenate([ref[:, :2]-(.5-limit/reg_scale)*ref[:, 2:],
                                        ref[:, :2]+(.5+limit/reg_scale)*ref[:, 2:]], axis=-1)
                clipped = np.clip(gt, lower, upper)
                ceiling = aligned_iou(clipped, gt)
                iou = aligned_iou(prediction, gt)
                max_iou_error = max(max_iou_error, float(np.abs(iou-np.array(native['pair_iou'])).max()))
                assert np.allclose(iou, native['pair_iou'], atol=3e-5, rtol=0)
                for aux in outputs['aux_outputs']:
                    layer_reference_error = max(layer_reference_error,
                        float((aux['ref_points'][image, qs].double()-outputs['ref_points'][image, qs].double()).abs().max()))
                pre_reference_error = max(pre_reference_error,
                    float((outputs['pre_outputs']['pred_boxes'][image, qs].double()-outputs['ref_points'][image, qs].double()).abs().max()))
                for j, (q, g) in enumerate(zip(qs.tolist(), gs.tolist())):
                    origins = [stage for stage in ['final', 'aux_0', 'aux_1', 'pre', 'enc_0'] if maps[stage].get(q) == g]
                    assert origins
                    own = maps['final'].get(q) == g
                    rows.append({'seed': seed, 'batch': batch, 'image': image, 'q': q, 'g': g,
                        'image_GT': item['GT'], 'Q': Q, 'shortside640': float(gt_box[j, 2:].min()*640),
                        'IoU': float(iou[j]), 'outside_exact_support': bool(wanted_outside[j].any()),
                        'support_max_IoU': float(ceiling[j]), 'final_own_pair': own, 'origins': origins,
                        'nonfinal_only': not own,
                        'only_pre_or_encoder_sources': all(stage in ['pre', 'enc_0'] for stage in origins),
                        'requested_outside_edge_near_endpoint90': bool(np.any(wanted_outside[j] & (np.sign(required[j])*actual[j] >= .9*limit)))})
    groups = []
    for seed in [1, 2]:
        for capacity in ['all', 'N<=Q', 'N>Q']:
            for size in ['all', 'input_shortside<4']:
                for group in ['all', 'IoU<.1', 'IoU<.1_outside_exact_support', 'IoU<.1_ceiling<.5', 'IoU==0_outside_exact_support']:
                    subset = [r for r in rows if r['seed'] == seed and
                              (capacity == 'all' or (capacity == 'N<=Q' and r['image_GT'] <= r['Q']) or (capacity == 'N>Q' and r['image_GT'] > r['Q'])) and
                              (size == 'all' or r['shortside640'] < 4) and
                              (group == 'all' or (group == 'IoU<.1' and r['IoU'] < .1) or
                               (group == 'IoU<.1_outside_exact_support' and r['IoU'] < .1 and r['outside_exact_support']) or
                               (group == 'IoU<.1_ceiling<.5' and r['IoU'] < .1 and r['support_max_IoU'] < .5) or
                               (group == 'IoU==0_outside_exact_support' and r['IoU'] == 0 and r['outside_exact_support']))]
                    n = len(subset)
                    nonfinal = sum(r['nonfinal_only'] for r in subset)
                    groups.append({'seed': seed, 'capacity': capacity, 'size': size, 'group': group,
                        'pairs': n, 'nonfinal_only_pairs': nonfinal,
                        'nonfinal_only_fraction': nonfinal/n if n else None,
                        'final_own_pairs': n-nonfinal,
                        'only_pre_or_encoder_sources': sum(r['only_pre_or_encoder_sources'] for r in subset),
                        'outside_exact_support_pairs': sum(r['outside_exact_support'] for r in subset),
                        'requested_outside_edge_near_endpoint90': sum(r['requested_outside_edge_near_endpoint90'] for r in subset)})
    result = {'spec': spec, 'sources': sources, 'groups': groups,
              'aux_to_final_reference_max_abs': layer_reference_error,
              'pre_box_to_FDR_reference_max_abs': pre_reference_error,
              'recomputed_IoU_max_abs_error': max_iou_error,
              'interpretation_limits': 'Sources and static support do not establish harmful gradients or detection benefit. '
                                       'If aux refs equal final refs, no claim that support only becomes infeasible at final. '
                                       'Exact GT outside support is not failure to attain IoU.5. Lower training loss after deletion would be tautological.'}
    save(OUT/'06_同源缓存_配对来源与固定支持.json', result)
    save(OUT/'06b_同源缓存逐配对明细.json', {'rows': rows})
    print(json.dumps({'checks': {k: result[k] for k in ['aux_to_final_reference_max_abs', 'pre_box_to_FDR_reference_max_abs', 'recomputed_IoU_max_abs_error']},
                      'primary_all_size_and_capacity': [r for r in groups if r['capacity'] == 'all' and r['size'] == 'all']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
