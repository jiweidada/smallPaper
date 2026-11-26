"""Parameter-free denoising-reference geometry; not a detector or training estimate."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import io
import json
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'A10_跨实例负去噪参考框几何审计_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def tape_for(boxes, seed):
    n = max(len(x) for x in boxes)
    g = max(100 // n, 1) if n else 0
    shape = (len(boxes), n * 2 * g)
    rng = np.random.Generator(np.random.PCG64(seed))
    return {
        'label_u': rng.random(shape, dtype=np.float32),
        'new_labels': rng.integers(0, 12, shape, dtype=np.int32),
        'sign': rng.integers(0, 2, shape + (4,), dtype=np.int32).astype(np.float32),
        'box_u': rng.random(shape + (4,), dtype=np.float32),
    }


def numpy_generator(boxes, labels, tape):
    """Dome BGT-padding/grouping/noise/abs/inverse-sigmoid operator,with shared tapes."""
    bs = len(boxes)
    n = max(len(x) for x in boxes)
    if not n:
        return np.zeros((bs, 0), np.int32), np.zeros((bs, 0, 4), np.float32), [], 0, np.zeros((bs, 0, 4), np.float32)
    g = max(100 // n, 1)
    ids = np.full((bs, n), 12, np.int32)
    b = np.zeros((bs, n, 4), np.float32)
    valid = np.zeros((bs, n), bool)
    for i, (bb, ll) in enumerate(zip(boxes, labels)):
        ids[i, :len(ll)] = ll
        b[i, :len(ll)] = bb
        valid[i, :len(ll)] = True
    ids = np.tile(ids, (1, 2 * g))
    b = np.tile(b, (1, 2 * g, 1))
    valid = np.tile(valid, (1, 2 * g))
    neg = np.tile(np.r_[np.zeros(n), np.ones(n)], g).astype(np.float32)[None, :, None]
    positive_indices = [np.flatnonzero((1 - neg[0, :, 0]).astype(bool) & valid[i]).astype(np.int64) for i in range(bs)]
    ids = np.where((tape['label_u'] < np.float32(.25)) & valid, tape['new_labels'], ids)
    known = np.concatenate((b[..., :2] - b[..., 2:] * np.float32(.5), b[..., :2] + b[..., 2:] * np.float32(.5)), axis=-1)
    diff = np.tile(b[..., 2:] * np.float32(.5), (1, 1, 2))
    sign = tape['sign'] * np.float32(2) - np.float32(1)
    part = (tape['box_u'] + np.float32(1)) * neg + tape['box_u'] * (1 - neg)
    known += sign * part * diff
    np.clip(known, 0, 1, out=known)
    cxy = (known[..., :2] + known[..., 2:]) * np.float32(.5)
    extent = known[..., 2:] - known[..., :2]
    signed = np.concatenate((cxy, extent), axis=-1)
    b = np.abs(signed)
    clipped = np.clip(b, 0, 1)
    logits = np.log(np.maximum(clipped, np.float32(1e-5)) / np.maximum(1 - clipped, np.float32(1e-5)))
    return ids, logits, positive_indices, g, signed


def initial_refs(logits):
    return np.float32(1) / (np.float32(1) + np.exp(-logits))


def corners(cx):
    return np.concatenate((cx[..., :2] - cx[..., 2:] * np.float32(.5), cx[..., :2] + cx[..., 2:] * np.float32(.5)), axis=-1)


def pair_iou(a, b):
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.maximum(hi - lo, 0).prod(-1)
    area_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(-1)
    area_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(-1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def cluster_ci(rows, key, seed=2026103001):
    rows = [r for r in rows if r[key] is not None]
    if not rows:
        return {'mean': None, 'low': None, 'high': None, 'images': 0, 'groups': 0}
    groups = sorted({r['group'] for r in rows})
    values = [[r[key] for r in rows if r['group'] == g] for g in groups]
    rng = np.random.Generator(np.random.PCG64(seed))
    boot = np.asarray([np.mean([x for i in rng.integers(len(groups), size=len(groups)) for x in values[i]]) for _ in range(2000)])
    return {'mean': float(np.mean([r[key] for r in rows])), 'low': float(np.quantile(boot, .025)), 'high': float(np.quantile(boot, .975)), 'images': len(rows), 'groups': len(groups)}


def main():
    p = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    manifest = json.loads((OUT / 'freeze_manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        assert sha(ROOT / entry['path']) == entry['sha256']
    eq = json.loads((OUT / 'author_operator_equivalence.json').read_text(encoding='utf-8'))
    assert eq['status'] == 'PASS_PARAMETER_FREE_AUTHOR_OPERATOR_EQUIVALENCE'
    assert p['status'] == 'FROZEN_BEFORE_ANY_A10_DATASET_NOISE_DRAWS'
    (OUT / 'started.json').open('x', encoding='utf-8').write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'pid': __import__('os').getpid()}))
    archive = Path(p['source_archive'])
    assert sha(archive) == p['source_archive_sha256']
    outputs = OUT / 'references'
    outputs.mkdir()
    per_gt = []
    per_image = []
    traces = []
    total_negative_queries = 0
    with tarfile.open(archive) as tar:
        for spec in p['trace_manifest']:
            raw = tar.extractfile(spec['path']).read()
            assert hashlib.sha256(raw).hexdigest() == spec['sha256']
            with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                gt = z['gt_boxes'].astype(np.float32) / np.float32(640)
                labels = z['gt_labels'].astype(np.int32) + 1
                uids = z['gt_uids'].astype(str)
                sizes = z['gt_sqrt_area'].astype(np.float32) * np.float32(1.25)
            assert np.all((labels >= 1) & (labels <= 10))
            n = len(gt)
            assert n > 0, 'The fixed source cohort must contain at least one valid GT per image.'
            cxy = (gt[:, :2] + gt[:, 2:]) * np.float32(.5)
            boxes = np.concatenate((cxy, gt[:, 2:] - gt[:, :2]), axis=-1)
            name = Path(spec['path']).stem
            metrics = {}
            saved = []
            saved_labels = []
            own_counts = []
            for draw in range(16):
                seed = int.from_bytes(hashlib.sha256((str(p['seed']) + '|' + name + '|' + str(draw)).encode()).digest()[:8], 'little')
                ids, logits, _, groups, signed = numpy_generator([boxes], [labels], tape_for([boxes], seed))
                refs = initial_refs(logits)[0]
                neg = (np.arange(len(refs)) // n) % 2 == 1
                neg_refs = refs[neg]
                neg_labels = ids[0, neg]
                origins = np.tile(np.arange(n), groups)
                first = np.arange(len(origins)) < n
                iou = pair_iou(corners(neg_refs), gt)
                other = origins[:, None] != np.arange(n)[None, :]
                class_agree = neg_labels[:, None] == labels[None, :]
                for threshold in p['iou_thresholds']:
                    overlaps = iou >= threshold
                    for mode, mask in [('one_group_same_input_class', first[:, None] & other & class_agree), ('one_group_any_input_class', first[:, None] & other), ('native_B1_all_groups_same_input_class', other & class_agree)]:
                        key = mode + '_iou' + str(threshold)
                        metrics.setdefault(key, []).append((overlaps & mask).any(axis=0))
                own = (iou[np.arange(len(origins)), origins] >= .5) & (neg_labels == labels[origins])
                own_counts.append(int(own[first].sum()))
                saved.append(neg_refs)
                saved_labels.append(neg_labels)
                total_negative_queries += len(neg_refs)
            data = {k: np.stack(v).astype(np.uint8) for k, v in metrics.items()}
            destination = outputs / (name + '.npz')
            np.savez_compressed(destination, negative_refs=np.stack(saved), negative_input_labels=np.stack(saved_labels), origins=origins, gt_boxes=gt, gt_labels=labels, gt_sizes800=sizes, gt_uids=uids, own_one_group_counts=np.asarray(own_counts), **data)
            traces.append({'path': str(destination.relative_to(OUT)), 'sha256': sha(destination)})
            row = {'file': name + '.jpg', 'group': name.split('_')[0], 'gt': n, 'groups_B1': groups, 'native_B1_negative_queries_per_draw': len(origins), 'own_one_group_iou50_rate': sum(own_counts) / (16 * n)}
            cohorts = {'all': np.ones(n, bool), 'lt8': sizes < 8, '8to16': (sizes >= 8) & (sizes < 16), 'ge16': sizes >= 16}
            for cohort, mask in cohorts.items():
                row[cohort + '_n'] = int(mask.sum())
                for key, values in data.items():
                    row[cohort + '_' + key] = float(values[:, mask].mean()) if mask.any() else None
            per_image.append(row)
            primary = data[p['primary_key']]
            for j in range(n):
                per_gt.append({'uid': str(uids[j]), 'file': row['file'], 'group': row['group'], 'size800': float(sizes[j]), 'class_id': int(labels[j]), 'exposed_draws': int(primary[:, j].sum()), 'draws': 16})
    assert len(per_image) == 100 and len(per_gt) == 4491
    primary = cluster_ci(per_image, 'lt8_' + p['primary_key'])
    eligible = sum(r['lt8_n'] for r in per_image)
    enough = eligible >= 200 and primary['groups'] >= 15
    if enough and primary['mean'] >= .05 and primary['low'] > .02:
        decision = 'GEOMETRIC_EVENT_PASSES_FREQUENCY_GATE_ONLY_NEEDS_PRIOR_CLEARANCE_AND_CAUSAL_TEST'
    elif enough and primary['high'] < .02:
        decision = 'DEPRIORITIZE_CROSS_INSTANCE_NEGATIVE_DN_FROM_LOW_GEOMETRIC_EXPOSURE'
    else:
        decision = 'INCONCLUSIVE_FREQUENCY_GATE_NO_TRAINING_OR_NEW_MAINLINE'
    estimates = {}
    for cohort in ('all', 'lt8', '8to16', 'ge16'):
        for mode in ('one_group_same_input_class', 'one_group_any_input_class', 'native_B1_all_groups_same_input_class'):
            estimates[cohort + '_' + mode + '_iou0.5'] = cluster_ci(per_image, cohort + '_' + mode + '_iou0.5')
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'COMPLETED_PARAMETER_FREE_GEOMETRY_ONLY', 'decision': decision, 'images': len(per_image), 'gt': len(per_gt), 'tiny_gt': eligible, 'negative_references': total_negative_queries, 'draws_per_image': 16, 'primary': primary, 'estimates': estimates, 'own_one_group_mean': float(np.mean([r['own_one_group_iou50_rate'] for r in per_image])), 'new_full_model_forwards': 0, 'cumulative_full_model_forwards': 200, 'new_fits': 0, 'main_model_updates': 0, 'mainline_validated': False, 'scope': p['interpretation_limits']}
    write(OUT / 'per_image.json', per_image)
    (OUT / 'per_gt.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in per_gt), encoding='utf-8')
    write(OUT / 'reference_manifest.json', {'files': traces})
    write(OUT / 'summary.json', result)
    write(OUT / 'completed.json', {'status': 'COMPLETED', 'summary_sha256': sha(OUT / 'summary.json'), 'protocol_sha256': sha(OUT / 'protocol.json'), 'new_full_model_forwards': 0, 'new_fits': 0})
    print(json.dumps({k: result[k] for k in ('status', 'decision', 'images', 'gt', 'tiny_gt', 'negative_references', 'primary', 'own_one_group_mean')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
