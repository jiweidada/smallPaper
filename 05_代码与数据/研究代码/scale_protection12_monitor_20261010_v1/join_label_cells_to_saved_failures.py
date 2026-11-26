"""GT-identity join of label-cell collisions to already verified saved endpoints."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path.cwd()
P = ROOT / 'outputs/scale_protection12_20261010_v1'
OUT = ROOT / 'outputs/sod_encoder_supervision_20261010_v2'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

with np.load(OUT / 'val_label_geometry.npz', allow_pickle=False) as archive:
    label = {k: archive[k] for k in archive.files}
assert len(label['annotation_id']) == len(set(label['annotation_id'].tolist())) == 38759
indices = {int(gid): i for i, gid in enumerate(label['annotation_id'])}
cell_xy = np.floor(label['centre640'] / 8).astype(np.int64)
keys = [(int(iid), int(xy[0]), int(xy[1])) for iid, xy in zip(label['image_id'], cell_xy)]
counts = Counter(keys)
collision = np.asarray([counts[key] > 1 for key in keys], dtype=np.bool_)
tiny = label['shortside640'] < 4
assert int(tiny.sum()) == 5096 and int((tiny & collision).sum()) == 2239
result = {}
provenance = {}
for seed in (1, 2):
    folder = P / f'stage_analysis/seed{seed}_target_protect_v1'
    manifest_path = folder / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    result[str(seed)] = {}
    provenance[str(seed)] = dict(stage_analysis_manifest_sha256=sha(manifest_path), array_sha256={})
    for arm in ('baseline', 'target_protect'):
        file = folder / ('baseline_per_GT.npz' if arm == 'baseline' else 'target_protect_per_GT.npz')
        assert sha(file) == manifest['files'][file.name]
        provenance[str(seed)]['array_sha256'][arm] = sha(file)
        with np.load(file, allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        order = np.asarray([indices[int(gid)] for gid in data['annotation_id']], dtype=np.int64)
        assert len(order) == len(set(order.tolist())) == 38759
        assert np.array_equal(data['image_id'], label['image_id'][order])
        assert np.array_equal(data['category_id'], label['category_id'][order])
        result[str(seed)][arm] = {}
        for name, mask in (('under4_single_GT_cell', tiny[order] & ~collision[order]),
                           ('under4_multi_GT_cell', tiny[order] & collision[order])):
            n = int(mask.sum())
            row = dict(GT=n)
            for threshold, column in ((.5, 0), (.75, 5)):
                tag = str(int(threshold * 100))
                supports = (data['max_iou'][mask] >= threshold).sum(axis=0)
                native_tp = int(data['native_TP'][mask, column].sum())
                row[tag] = dict(initial_geometric_support=int(supports[0]), pre_geometric_support=int(supports[1]),
                    final_geometric_support=int(supports[2]), native_TP=native_tp,
                    final_no_geometric_support=n-int(supports[2]), final_support_percent=100*int(supports[2])/n,
                    native_TP_percent=100*native_tp/n)
            result[str(seed)][arm][name] = row
        assert sum(v['GT'] for v in result[str(seed)][arm].values()) == 5096

report = dict(status='EXACT_GT_JOIN_TO_LABEL_CELL_GROUPS', groups=result, provenance=provenance,
    label_geometry_sha256=sha(OUT / 'val_label_geometry.npz'), code_sha256=sha(__file__),
    scope='Descriptive association on the two original annotation-cell groups. No augmented-training representation, TopK token identities, assignment gradients or causal effect is observed. P3 label collisions are a constraint for naive local auxiliary labels, not proof of a failure mechanism or a new acceptance gate.',
    new_model_forwards=0, new_evaluator_calls=0, new_optimizer_updates=0, new_training_epochs=0,
    current_random_controls_unchanged=True)
assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
with (OUT / 'saved_failure_cell_groups_v1.json').open('x', encoding='utf8') as stream:
    json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
print(json.dumps(dict(status=report['status'], groups=result)), flush=True)
