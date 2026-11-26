"""Cached vector arithmetic only: no network forward/backward or training optimizer.

Recreate exactly the two virtual Adam endpoints already defined and evaluated in
data09, and dot their difference with the already saved ordinary gradient.
No new grid, loss evaluation, or model execution.
"""
import argparse
import json
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK/'英文小论文/code/dn_pre_frozen_diagnostic_20261020_v1'))
from common import torch, resource_limits, sha, adam_virtual, clip_factor, dot, norm


def main(output):
    limits = resource_limits()
    data = WORK/'英文小论文/data/09_DN初始化冻结诊断_20261020_v1'
    original = json.loads((data/'03_全部局部反事实原始结果.json').read_text(encoding='utf-8'))['rows']
    rows, source_hashes = [], []
    for seed in [1, 2]:
        caches = {}
        for b in [0, 1]:
            p = data/f'seed{seed}/batch{b:02d}_frozen_head.pt'
            source_hashes.append({'file': str(p), 'sha256': sha(p)})
            caches[b] = torch.load(p, map_location='cpu', weights_only=False)
        for generator, evaluation in [(0, 0), (0, 1), (1, 1), (1, 0)]:
            source, target = caches[generator], caches[evaluation]
            included = adam_virtual(source['weights'], source['gradient_total_last'], source['optimizer_bindings'], clip_factor(source['global_grad_norm_include_DN']))
            excluded = adam_virtual(source['weights'], source['gradient_total_without_DN_last'], source['optimizer_bindings'], clip_factor(source['global_grad_norm_exclude_DN_last']))
            delta = [a - b for a, b in zip(included, excluded)]
            old = next(r for r in original if r['seed'] == seed and r['generator_batch'] == generator and r['evaluation_batch'] == evaluation)
            assert abs(norm(delta) - old['update_difference_norm']) < 1e-14
            exact = old['groups']['all_ordinary']['DN_effect']
            linear = dot(target['gradient_ordinary'], delta)
            rows.append({'seed': seed, 'generator': generator, 'evaluation': evaluation, 'cross_batch': generator != evaluation, 'first_order_cached_gradient_dot_endpoint_delta': linear, 'original_exact_finite_loss_extra': exact, 'same_sign': (linear > 0) == (exact > 0), 'finite_minus_first_order': exact - linear, 'endpoint_norm_replay_matches_original': True})
    for item in source_hashes:
        assert sha(item['file']) == item['sha256']
    with Path(output).open('x', encoding='utf-8') as f:
        json.dump({'scope': 'Post-proposal cache-only vector arithmetic. Replays previously evaluated cloned-parameter endpoints, not a training step or a new model run. Uses cached native FP32 ordinary gradient, while original exact finite loss was evaluated with FP64 cached-head arithmetic. Repeated development batches, no validation accuracy claim.', 'resources': limits, 'source_hashes': source_hashes, 'rows': rows}, f, ensure_ascii=False, indent=2)
    for row in rows:
        print(row['seed'], f"{row['generator']}->{row['evaluation']}", f"linear={row['first_order_cached_gradient_dot_endpoint_delta']:.10g}", f"exact={row['original_exact_finite_loss_extra']:.10g}", row['same_sign'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
