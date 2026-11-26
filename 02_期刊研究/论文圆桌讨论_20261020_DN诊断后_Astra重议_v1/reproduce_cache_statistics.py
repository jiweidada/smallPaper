"""Reproduce descriptive statistics from existing development caches only.

No model construction, forward/backward, optimizer, remote operations or heldout reads.
Output must be a new file. Run with D:/Anaconda/python.exe -s -X utf8 -B.
"""
import argparse
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


def xyxy(a):
    return torch.cat([a[:, :2] - a[:, 2:] / 2, a[:, :2] + a[:, 2:] / 2], 1)


def aligned_iou(a, b):
    a, b = xyxy(a.double()), xyxy(b.double())
    inter = (torch.minimum(a[:, 2:], b[:, 2:]) - torch.maximum(a[:, :2], b[:, :2])).clamp(min=0).prod(1)
    union = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(1) + (b[:, 2:] - b[:, :2]).clamp(min=0).prod(1) - inter
    return inter / union.clamp(min=1e-15)


def main(output):
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    rows, inputs = [], []
    for seed in [1, 2]:
        for batch in [0, 1]:
            p = WORK/f'英文小论文/data/09_DN初始化冻结诊断_20261020_v1/seed{seed}/batch{batch:02d}_frozen_head.pt'
            inputs.append({'file': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
            c = torch.load(p, map_location='cpu', weights_only=False)
            for image, ((q, g), t) in enumerate(zip(c['ordinary']['indices'], c['targets'])):
                n = len(t['labels'])
                m = torch.bincount(g, minlength=n)
                vals = aligned_iou(c['native_pre_boxes'][image][q], t['boxes'][g])
                sums = torch.bincount(g, weights=vals, minlength=n)
                means = sums/m.clamp(min=1)
                best = torch.zeros(n, dtype=torch.float64)
                best.scatter_reduce_(0, g, vals, reduce='amax', include_self=True)
                size = t['boxes'].double()[:, 2:].prod(1).sqrt()
                assert int(m.sum()) == len(q)
                for label, mask in [('single', m == 1), ('multiple', m >= 2)]:
                    k = int(mask.sum())
                    rows.append({
                        'seed': seed, 'batch': batch, 'image': image, 'GT_total': n,
                        'capacity_group': 'N_le_Q' if n <= 300 else 'N_gt_Q',
                        'group': label, 'unique_GT': k, 'ordinary_pairs': int(m[mask].sum()),
                        'omitted_in_image': int((m == 0).sum()),
                        'per_GT_mean_pre_IoU': float(means[mask].mean()) if k else None,
                        'per_GT_mean_best_assigned_pre_IoU': float(best[mask].mean()) if k else None,
                        'median_sqrt_normalized_area': float(size[mask].median()) if k else None,
                    })
    with Path(output).open('x', encoding='utf-8') as f:
        json.dump({'scope': 'Post-proposal descriptive cache-only analysis; repeated development inputs, not causal or independent validation; m=0 has no assigned IoU and is not zero-imputed.', 'inputs': inputs, 'rows': rows}, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
