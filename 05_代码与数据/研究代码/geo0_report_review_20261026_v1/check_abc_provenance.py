"""Reproduce existing A/B/C counts and audit what B actually measures; no forward."""
import json
import hashlib
from pathlib import Path
import time
import torch

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parent
EN = ROOT.parents[1]
TMP = EN.parent/'tmp'
CACHE = EN/'code/unit_instance_completion_20261021_v1/342_codex_stage3_20261022_v1'
expected = json.loads((TMP/'360_geo0_abc_v1.json').read_text(encoding='utf-8'))
cross = json.loads((TMP/'360_geo0_abc_v2.json').read_text(encoding='utf-8'))
start = time.monotonic()
result = {}
for seed in (1, 2):
    counts = {r: {'A': 0, 'B': 0, 'C': 0, 'B_score_le_0_001': 0,
                  'B_score_above_0_001_not_in_global_top300': 0} for r in (8, 16, 32)}
    images, unique_pairs = set(), set()
    total, n2 = 0, 0
    for path in sorted((CACHE/f'342_cache_val_v2_seed{seed}').glob('342_*.pt')):
        d = torch.load(path, map_location='cpu', weights_only=True)
        pb, lg = d['final300_boxes'].float(), d['final300_logits'].float()
        g, labels = d['gt_boxes_640'].float(), d['gt_labels'].long()
        b = pb*640
        edges = torch.cat((b[:, :2]-b[:, 2:]/2, b[:, :2]+b[:, 2:]/2), dim=1)
        wh = (torch.minimum(g[:, None, 2:], edges[None, :, 2:])
              -torch.maximum(g[:, None, :2], edges[None, :, :2])).clamp(min=0)
        inter = wh[..., 0]*wh[..., 1]
        ga = (g[:, 2]-g[:, 0])*(g[:, 3]-g[:, 1])
        pa = (edges[:, 2]-edges[:, 0])*(edges[:, 3]-edges[:, 1])
        I = inter/(ga[:, None]+pa[None, :]-inter+1e-12)
        selected = ((d['gt_short_side_640_all'] < 4) & (I.max(dim=1).values < .5)).nonzero().flatten()
        total += len(selected)
        if not len(selected):
            continue
        score, classes = lg.sigmoid().max(dim=1)
        survived = torch.zeros(300, dtype=torch.bool)
        survived[d['fullp_query_index'].long()] = True
        centers = (g[:, :2]+g[:, 2:])/2
        dist = torch.linalg.vector_norm(b[None, :, :2]-centers[:, None, :], dim=-1)
        same = classes[None, :] == labels[:, None]
        best_iou, best_q = torch.where(same, I, torch.zeros_like(I)).max(dim=1)
        for gidx in selected.tolist():
            if best_iou[gidx] >= .1:
                n2 += 1
                image_id = int(d['image_id'])
                images.add(image_id)
                unique_pairs.add((image_id, int(best_q[gidx])))
            for r in (8, 16, 32):
                in_window = (dist[gidx] <= r).nonzero().flatten()
                if not len(in_window):
                    counts[r]['A'] += 1
                    continue
                q = in_window[score[in_window].argmax()]
                if not survived[q]:
                    counts[r]['B'] += 1
                    key = ('B_score_le_0_001' if score[q] <= .001
                           else 'B_score_above_0_001_not_in_global_top300')
                    counts[r][key] += 1
                else:
                    assert I[gidx, q] < .5
                    counts[r]['C'] += 1
    assert total == expected[str(seed)]['geo0']
    for r in (8, 16, 32):
        for key in ('A', 'B', 'C'):
            assert counts[r][key] == expected[str(seed)]['win'][str(r)][key], (seed, r, key)
        assert counts[r]['B'] == counts[r]['B_score_le_0_001']+counts[r]['B_score_above_0_001_not_in_global_top300']
    result[str(seed)] = {
        'Geo0': total, 'windows': counts, 'criterion2_GT': n2,
        'criterion2_distinct_images': len(images), 'criterion2_distinct_image_query_pairs': len(unique_pairs),
        'criterion2_outside_C_by_radius': {r: cross[str(seed)]['cross'][str(r)]['has_ori2']['A']
                                         +cross[str(seed)]['cross'][str(r)]['has_ori2']['B'] for r in (8, 16, 32)}}
sources = [TMP/name for name in ('360_geo0_abc_v1.py', '360_geo0_abc_v1.json',
                                '360_geo0_abc_v2.py', '360_geo0_abc_v2.json',
                                '360_axisB_fdr_top_v2_ori2.py', '360_axisB_fdr_top_v2_ori2.json')]
sources += [EN/'持续寻找研究主线_20261020_v2/360_Geo0三分解与候选_20261026_v1.md']
output = {'status': 'existing A/B/C counts reproduced from caches; no model forward',
          'seeds': result, 'elapsed_seconds': time.monotonic()-start,
          'source_hashes': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
          'new_model_forwards': 0, 'training': False}
with (ROOT/'abc_provenance_audit.json').open('x', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(json.dumps({k:v for k,v in output.items() if k != 'source_hashes'}, ensure_ascii=True, indent=2))
