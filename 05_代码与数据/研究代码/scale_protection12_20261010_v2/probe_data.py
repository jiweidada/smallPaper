"""Bound native CPU data-only comparison; no detector/criterion construction."""
import argparse
from copy import deepcopy
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def main():
    p = argparse.ArgumentParser(); p.add_argument('--prefix', required=True)
    p.add_argument('--output', required=True); args = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    os.nice(10); sys.dont_write_bytecode = True
    old = Path(args.prefix) / 'DEIM/experiments/fine_residual_screen24_20261009_v2'
    new = Path(args.prefix) / 'DEIM/experiments/scale_protection12_20261010_v1'
    output = Path(args.output).resolve()
    assert output.parent == new/'diagnostics' and not output.exists()
    protocol = json.loads((new/'protocol.json').read_text())
    samples = protocol['data_only_panel_images']; assert samples == 128
    sys.path.insert(0, str(old/'code'))
    from experiment import verify_bundle
    from runtime import activate_training_snapshot
    receipt, plan, paths = verify_bundle(old/'bundles/baseline_seed1')
    activate_training_snapshot(paths['native_root'])
    from evaluator_binding import activate
    activate(paths['evaluator_root'], receipt['identity']['evaluator_binding'])
    import numpy as np
    import torch
    from engine.core import YAMLConfig
    sys.path.insert(0, str(new/'code'))
    from scale_guard import CONTRACT, install, training_profile
    assert not torch.cuda.is_available()
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.set_default_device('cpu')
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    tensor = lambda v: v.as_subclass(torch.Tensor)
    digest = lambda v: hashlib.sha256(tensor(v).contiguous().numpy().tobytes()).hexdigest()
    def reset(seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    def rng():
        return dict(python=hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
                    numpy=hashlib.sha256(np.random.get_state()[1].tobytes()+repr(np.random.get_state()[2:]).encode()).hexdigest(),
                    torch=digest(torch.get_rng_state()))
    cfg = YAMLConfig(str(old/'bundles/baseline_seed1/resolved.yml'))
    loader = cfg.train_dataloader; dataset = loader.dataset; collate = loader.collate_fn
    assert not dataset.cache_flag and collate.copyblend_prob == 0
    assert sha(paths['train_ann']) == protocol['train_annotation_sha256']
    profile = training_profile(dataset)
    positions = [round(i*(len(dataset.ids)-1)/(samples-1)) for i in range(samples)]
    assert len(set(positions)) == samples
    cat = {int(k)+1: int(v['category_id']) for k, v in dataset.coco.anns.items()}
    original_load = dataset.load_item; raw = {}; tagging = False
    def traced_load(idx):
        image, target = original_load(idx); iid = int(dataset.ids[idx]); w, h = image.size
        anns = [v for v in dataset.coco.loadAnns(dataset.coco.getAnnIds(imgIds=[iid])) if not v.get('iscrowd', 0)]
        boxes = torch.tensor([v['bbox'] for v in anns], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]; boxes[:, 0::2].clamp_(0, w); boxes[:, 1::2].clamp_(0, h)
        keep = (boxes[:, 2:] > boxes[:, :2]).all(1)
        valid = [v for v, k in zip(anns, keep.tolist()) if k]
        assert torch.equal(boxes[keep], tensor(target['boxes']))
        assert target['labels'].tolist() == [v['category_id'] for v in valid]
        ids = [int(v['id'])+1 for v in valid]
        side = ((boxes[keep, 2:] - boxes[keep, :2])*torch.tensor([640/w, 640/h])).min(-1).values.tolist()
        raw[iid] = dict(zip(ids, side))
        if tagging:
            target['labels'] = torch.tensor(ids, dtype=torch.int64)
        return image, target
    dataset.load_item = traced_load
    started = time.monotonic(); modes = {}; prefix_checks = []
    try:
        loader.set_epoch(0); reset(1)
        original_images, original_targets = collate([dataset[i] for i in positions[:4]])
        state = rng(); guard = install(dataset, 1, 'target_protect', profile=profile)
        reset(1); trial_images, trial_targets = collate([dataset[i] for i in positions[:4]])
        assert digest(original_images) == digest(trial_images) and rng() == state
        assert all(digest(a['boxes']) == digest(b['boxes']) and torch.equal(a['labels'], b['labels'])
                   for a, b in zip(original_targets, trial_targets))
        assert guard.occurrence == 0; guard.close(); prefix_checks.append('epoch0_identity_RNG_and_no_Mosaic_calls')
        loader.set_epoch(4)
        for seed in (1, 2):
            for arm in ('native', 'identity', 'target_protect', 'random_protect'):
                records = []; guard = None
                if arm != 'native':
                    guard = install(dataset, seed, arm, records.append, profile)
                normal = []; tagged_samples = []; batches = []
                try:
                    for encoded in (False, True):
                        tagging = encoded; reset(seed); records.clear()
                        if guard:
                            guard.reset()
                        for b in range(samples//4):
                            items = []; sample_rows = []
                            for pos in positions[b*4:b*4+4]:
                                im, target = dataset[pos]
                                row = dict(position=pos, image_id=int(dataset.ids[pos]), image=digest(im), boxes=digest(target['boxes']),
                                    ids=target['labels'].tolist(), normalized_wh=tensor(target['boxes'])[:, 2:].tolist(),
                                    source_image_ids=target.get('mosaic_source_image_ids', torch.empty(0, dtype=torch.int64)).tolist())
                                sample_rows.append(row); items.append((im, target))
                            images, targets = collate(items)
                            batch = dict(image=digest(images), boxes=[digest(t['boxes']) for t in targets],
                                labels=[t['labels'].tolist() for t in targets], shape=list(images.shape), rng_after=rng())
                            assert torch.isfinite(images).all() and all(torch.isfinite(t['boxes']).all() for t in targets)
                            assert all((tensor(t['boxes'])[:, 2:] > 0).all() for t in targets)
                            if not encoded:
                                normal.append(dict(samples=sample_rows, batch=batch))
                            else:
                                prior = normal[b]
                                assert all(batch[k] == prior['batch'][k] for k in ('image', 'boxes', 'shape', 'rng_after'))
                                assert [[cat[x] for x in ids] for ids in batch['labels']] == prior['batch']['labels']
                                for row, before in zip(sample_rows, prior['samples']):
                                    assert all(row[k] == before[k] for k in ('position', 'image_id', 'image', 'boxes', 'normalized_wh', 'source_image_ids'))
                                    assert [cat[x] for x in row['ids']] == before['ids']
                                    exposures = {}
                                    for gid, wh in zip(row['ids'], row['normalized_wh']):
                                        exposures.setdefault(gid, []).append(min(wh)*images.shape[-1])
                                    primary = [dict(annotation_id=gid-1, raw_shortside640=s,
                                        occurrences=len(exposures.get(gid, [])), survived=gid in exposures,
                                        actual_input_shortside=exposures[gid][0] if len(exposures.get(gid, [])) == 1 else None)
                                        for gid, s in raw[row['image_id']].items()]
                                    tagged_samples.append(dict(image_id=row['image_id'], position=row['position'], primary=primary,
                                        source_image_ids=row['source_image_ids'], exposures_before_mixup=len(row['ids'])))
                                batches.append(dict(batch=b, shape=batch['shape'], rng_after=batch['rng_after'],
                                    target_exposures_after_mixup=sum(len(t['labels']) for t in targets)))
                    primary = [r for s in tagged_samples for r in s['primary']]
                    retained = [r for r in primary if r['occurrences'] == 1]
                    summary = dict(raw_primary_GT=len(primary), retained_unique_primary_GT=len(retained),
                        missing_primary_GT=sum(not r['survived'] for r in primary), duplicate_excluded=sum(r['occurrences'] > 1 for r in primary),
                        primary_input_lt4=sum(r['actual_input_shortside'] < 4 for r in retained),
                        primary_input_ge4=sum(r['actual_input_shortside'] >= 4 for r in retained),
                        exposures_before_mixup=sum(s['exposures_before_mixup'] for s in tagged_samples),
                        exposures_after_mixup=sum(b['target_exposures_after_mixup'] for b in batches),
                        Mosaic_events=len(records) if guard else None, substituted_events=sum(r['applied'] for r in records))
                    modes[f'{seed}:{arm}'] = dict(summary=summary, samples=tagged_samples, batches=batches,
                        final_rng=rng(), guard_records=deepcopy(records), ordinary_hashes=normal)
                    print(json.dumps(dict(progress='MODE_COMPLETE', seed=seed, arm=arm, summary=summary)), flush=True)
                finally:
                    if guard:
                        guard.close()
        comparisons = {}
        for seed in (1, 2):
            native = modes[f'{seed}:native']; identity = modes[f'{seed}:identity']
            assert all(native[k] == identity[k] for k in ('samples', 'batches', 'final_rng', 'ordinary_hashes'))
            for arm in ('identity', 'target_protect', 'random_protect'):
                trial = modes[f'{seed}:{arm}']
                assert trial['final_rng'] == native['final_rng']
                assert [b['rng_after'] for b in trial['batches']] == [b['rng_after'] for b in native['batches']]
                old_gt = {r['annotation_id']: r for s in native['samples'] for r in s['primary']}
                new_gt = {r['annotation_id']: r for s in trial['samples'] for r in s['primary']}
                assert set(old_gt) == set(new_gt)
                common = [g for g in old_gt if old_gt[g]['occurrences'] == new_gt[g]['occurrences'] == 1]
                comparisons[f'{seed}:{arm}'] = dict(
                    same_retained_GT=len(common), native_retained_trial_missing=sum(v['survived'] and not new_gt[g]['survived'] for g, v in old_gt.items()),
                    native_missing_trial_retained=sum(not v['survived'] and new_gt[g]['survived'] for g, v in old_gt.items()),
                    common_native_lt4_trial_ge4=sum(old_gt[g]['actual_input_shortside'] < 4 <= new_gt[g]['actual_input_shortside'] for g in common),
                    common_native_ge4_trial_lt4=sum(new_gt[g]['actual_input_shortside'] < 4 <= old_gt[g]['actual_input_shortside'] for g in common))
        ready = all(modes[f'{s}:target_protect']['summary']['retained_unique_primary_GT'] >= modes[f'{s}:native']['summary']['retained_unique_primary_GT']
                    and modes[f'{s}:target_protect']['summary']['primary_input_ge4'] > modes[f'{s}:native']['summary']['primary_input_ge4'] for s in (1, 2))
        result = dict(status='COMPLETE_DATA_ONLY_SCALE_PROTECTION', data_only_physical_gate_passed=ready,
            identity=receipt['identity'], contract=CONTRACT, profile=profile, positions=positions, epoch_zero_based=4,
            seeds=[1, 2], prefix_checks=prefix_checks, native_identity_exact=True, native_global_rng_exact_all_arms=True,
            normal_tagged_classes_and_pixels_exact=True, comparisons=comparisons, modes=modes,
            model_constructed=False, model_forward_calls=0, optimizer_steps=0, new_training_epochs=0,
            completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
            protocol_sha256=sha(new/'protocol.json'), implementation={n:sha(new/'code'/n) for n in ('scale_guard.py','probe_data.py')},
            scope='Fixed training-data panel, not detector performance or actual epoch4 learning history. Full-source substitution changes donor/context/density distribution; matching only image-membership rate does not force realized Mosaic substitutions equal. Four pixels is a descriptive scale bucket, not a detection limit.')
        with output.open('x', encoding='utf8') as f:
            json.dump(result, f, indent=2, allow_nan=False)
        print(json.dumps(dict(status=result['status'], data_only_physical_gate_passed=ready, output_sha256=sha(output))), flush=True)
    finally:
        dataset.load_item = original_load


if __name__ == '__main__':
    main()
