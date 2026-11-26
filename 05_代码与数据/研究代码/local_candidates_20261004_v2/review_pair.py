"""Review two completed candidate runs; incomplete/technical failures never become performance negatives."""
import argparse
import json
from pathlib import Path
import yaml


def read_arm(path):
    root = Path(path)
    complete = json.loads((root / 'complete.json').read_text(encoding='utf8'))
    evaluation = json.loads((root / 'independent_val_selected_ema/evaluation.json').read_text(encoding='utf8'))
    audit = [json.loads(line) for line in (root / 'training_audit.jsonl').read_text(encoding='utf8').splitlines()]
    if [r['epoch'] for r in audit] != list(range(96)) or evaluation['images'] != 548:
        raise ValueError('Require full96 and full548val before a performance decision')
    if any(r['optimizer_steps'] + r['amp_skips_proxy'] != r['batches'] for r in audit):
        raise ValueError('Update/AMP accounting does not close')
    if not evaluation['FP32_no_deploy'] or evaluation['maxDets'] != [1, 10, 100]:
        raise ValueError('Evaluation protocol differs')
    if complete['best_AP'] != max(r['metrics']['AP'] for r in audit):
        raise ValueError('Highest overallAP selection does not match all96 native evaluations')
    return complete, evaluation, audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--control', required=True); p.add_argument('--candidate', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); control = read_arm(a.control); candidate = read_arm(a.candidate)
    cb, nb = control[0]['bindings'], candidate[0]['bindings']
    for key in ['implementation_sha256', 'annotation_sha256', 'init', 'torch', 'GPU_uuid',
                'threads', 'interop_threads', 'TF32_matmul', 'TF32_cudnn', 'cudnn_benchmark', 'cudnn_deterministic']:
        if cb[key] != nb[key]: raise ValueError(f'Paired runtime binding differs: {key}')
    if control[1]['annotation_sha256'] != candidate[1]['annotation_sha256']:
        raise ValueError('Val annotation differs')
    c_cfg, n_cfg = dict(cb['config']), dict(nb['config'])
    if c_cfg['kind'] == n_cfg['kind'] == 'dual_view' and not (c_cfg['arm'] in ['independent', 'local_residual'] and n_cfg['arm'] == 'aligned'):
        raise ValueError('Dual candidate is aligned; control is independent or explicit local-residual ablation')
    for cfg in (c_cfg, n_cfg): cfg.pop('priority', None); cfg.pop('arm', None)
    if c_cfg['kind'] == n_cfg['kind'] == 'queries':
        c_cov, n_cov = dict(c_cfg.pop('coverage')), dict(n_cfg.pop('coverage'))
        if c_cov.pop('reserve') != 0 or n_cov.pop('reserve') != 75 or c_cov != n_cov:
            raise ValueError('Query pair must differ only by reserve0/75')
    if c_cfg != n_cfg: raise ValueError('Other candidate settings differ')
    for x, y in zip(control[2], candidate[2]):
        if x['batches'] != y['batches'] or x['steps'][0]['first_inputs_sha256'] != y['steps'][0]['first_inputs_sha256']:
            raise ValueError('Epoch batch count/first actual inputs differ')
        if len(x['steps'])!=len(y['steps']):raise ValueError('Different per-epoch batch count')
        if any(s['image_ids'] != t['image_ids'] or s.get('windows') != t.get('windows') or
               s.get('input_and_targets_sha256') != t.get('input_and_targets_sha256') or
               s.get('annotation_ids') != t.get('annotation_ids')
               for s, t in zip(x['steps'], y['steps'])):
            raise ValueError('Actual per-batch image IDs/crop windows differ')
    delta = {k: (candidate[1]['metrics'][k] - control[1]['metrics'][k]) * 100 for k in control[1]['metrics']}
    direction = delta['APs'] > 0 and delta['AP'] >= 0 and delta['AP75'] >= 0 and sum(delta[k] > 0 for k in ['AP', 'APs', 'AP75']) >= 2
    result = {'status': 'candidate_pair_preliminary_review_not_root_scientific_acceptance',
              'candidate_minus_control_pp': delta, 'direction_gate': direction,
              'per_class_APs_delta_pp': {k: None if control[1]['per_class_APs'][k] is None or candidate[1]['per_class_APs'][k] is None
                  else (candidate[1]['per_class_APs'][k] - control[1]['per_class_APs'][k]) * 100 for k in control[1]['per_class_APs']},
              'remaining': ['per-batch sampler/target-exposure audit', 'independent evaluator agreement',
                            'runtime/checkpoint/source and full raw artifact archival review', 'real cost and repeat-seed evidence'],
              'not_statistical_significance': True}
    with Path(a.output).open('x', encoding='utf8') as handle: json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == '__main__': main()
