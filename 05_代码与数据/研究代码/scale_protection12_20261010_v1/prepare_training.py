"""Prepare immutable native-recipe bundles; no Torch/model/optimizer import."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from runtime import HERE, COMMON_INIT_SHA, read, write, sha, implementation_hashes, resolved_config, verify_source


def prepare(prefix, seed, arm):
    from evaluator_binding import fingerprint
    protocol = read(HERE/'training_protocol.json')
    assert seed in protocol['execution_allowlist'][arm]
    root = Path(prefix)/'DEIM/experiments/scale_protection12_20261010_v1'
    assert sha(root/'protocol.json') == protocol['experiment_protocol_sha256']
    data = root/'diagnostics/data_probe.json'; observed = read(data)
    assert sha(data) == protocol['data_only_report_sha256']
    assert observed['status'] == 'COMPLETE_DATA_ONLY_SCALE_PROTECTION'
    assert observed['data_only_physical_gate_passed'] is True
    assert observed['protocol_sha256'] == protocol['experiment_protocol_sha256']
    assert observed['new_training_epochs'] == observed['optimizer_steps'] == observed['model_forward_calls'] == 0
    assert observed['native_identity_exact'] and observed['native_global_rng_exact_all_arms']
    if arm == 'random_protect':
        promotion = read(root/'dual_seed_native_promotion.json')
        assert promotion['both_target_protect_fixed12_native_gates_passed'] is True
        assert promotion['experiment_protocol_sha256'] == protocol['experiment_protocol_sha256']
    old = Path(prefix)/'DEIM/experiments/fine_residual_screen24_20261009_v2'
    legacy = old/f'bundles/baseline_seed{seed}'
    original = read(legacy/'receipt.json'); old_plan = read(legacy/'plan.json')
    assert original['identity']['arm'] == 'baseline' and original['identity']['seed'] == seed
    assert sha(legacy/'plan.json') == original['identity']['plan_sha256']
    assert sha(legacy/'resolved.yml') == original['resolved_sha256']
    assert all(sha(old/'code'/n) == h for n,h in original['identity']['implementation'].items())
    assert all(sha(HERE/n) == original['identity']['implementation'][n] for n in ('geometry.py','residual.py'))
    paths = {k: Path(v) for k,v in original['paths'].items()}
    paths.update(experiment_root=root, screen_root=root/f'seed{seed}_{arm}',
        output=root/f'seed{seed}_{arm}/run', legacy_bundle=legacy,
        baseline_checkpoint=Path(original['paths']['output'])/'checkpoint0011.pth',
        baseline_first100=Path(original['paths']['output'])/'first100_training_inputs.jsonl',
        baseline_cached=Path(protocol['baseline_reuse'][str(seed)]['source']),
        data_only_report=data, profile=root/'training_profile.json',
        image_manifest=root/f'image_manifest_seed{seed}.json',
        image_verification=root/f'image_verification_seed{seed}.json')
    reference = protocol['references'][str(seed)]
    for k in ('train_ann','val_ann','common_init','baseline_checkpoint','baseline_first100'):
        assert sha(paths[k]) == reference[k+'_sha256'], k
    assert sha(paths['common_init']) == COMMON_INIT_SHA
    assert observed['identity']['train_annotation_sha256'] == sha(paths['train_ann'])
    for k, h in protocol['baseline_reuse'][str(seed)]['pinned_files'].items():
        assert sha(paths['baseline_cached']/k) == h, k
    verify_source(paths['native_root'])
    evaluator = fingerprint(paths['evaluator_root']); assert evaluator == reference['evaluator_binding']
    if not paths['profile'].exists():
        write(paths['profile'], observed['profile'])
    assert read(paths['profile']) == observed['profile']
    assert sha(paths['profile']) == protocol['training_profile_sha256']
    assert sum(v['protect'] for v in observed['profile'].values()) == protocol['expected_protected_training_image_count']
    # Reuse the already frozen whole-dataset byte manifest, verify present data once.
    if not paths['image_manifest'].exists():
        copied = old/'bundles'/f'baseline_seed{seed}'/'image_manifest.json'
        if not copied.exists():
            copied = Path(reference['image_manifest_source']) if 'image_manifest_source' in reference else root/'frozen_image_manifest.json'
        assert sha(copied) == reference['image_manifest_sha256']
        paths['image_manifest'].write_bytes(copied.read_bytes())
    assert sha(paths['image_manifest']) == reference['image_manifest_sha256']
    manifest = read(paths['image_manifest'])
    if not paths['image_verification'].exists():
        checked = count_bytes = 0
        for split, n in (('train',6471),('val',548)):
            entries = manifest[split]; assert len(entries) == n
            images = paths[split+'_images']
            assert {p.name for p in images.glob('*.jpg')} == set(entries)
            for name, item in entries.items():
                p = images/name
                assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], str(p)
                checked += 1; count_bytes += item['bytes']
        assert checked == 7019
        write(paths['image_verification'], dict(status='ALL_CURRENT_IMAGES_MATCH_FROZEN_DATASET',
            files_verified=checked, bytes_verified=count_bytes, image_manifest_sha256=sha(paths['image_manifest']),
            new_model_forwards=0, new_training_updates=0))
    assert read(paths['image_verification'])['image_manifest_sha256'] == sha(paths['image_manifest'])
    summary = read(paths['baseline_cached']/'endpoint_epoch12/summary.json')
    assert summary['identity']['seed'] == seed and summary['identity']['arm'] == 'baseline'
    assert summary['epoch_zero_based'] == 11 and summary['evaluated_weights'] == 'EMA'
    for key in ('common_init_sha256','baseline_checkpoint_sha256','baseline_first100_sha256','evaluator_binding'):
        assert summary['identity'][key] == reference[key], key
    plan = dict(schema=protocol['schema'], arm=arm, seed=seed,
        historical_resolved_config=read(legacy/'resolved.yml'), model_recipe=old_plan['model_recipe'],
        legacy_receipt_sha256=sha(legacy/'receipt.json'), legacy_plan_sha256=sha(legacy/'plan.json'),
        historical_baseline_stats=summary['native_stats'], max_epochs=12, recipe_epochs=132,
        fixed_endpoint='native_epoch12_EMA', extra_parameters=0,
        scope='Native model, training-only source protection; no baseline training or forward replay.')
    cfg = resolved_config(plan, paths)
    screen = paths['screen_root']; bundle = screen/'bundle'
    assert not screen.exists() and not paths['output'].exists()
    screen.mkdir(); bundle.mkdir()
    for file in HERE.glob('*.py'):
        ast.parse(file.read_text(encoding='utf8'))
    write(bundle/'plan.json', plan); write(bundle/'resolved.yml', cfg)
    identity = dict(schema=plan['schema'], arm=arm, seed=seed, plan_sha256=sha(bundle/'plan.json'),
        implementation=implementation_hashes(), protocol_sha256=sha(HERE/'training_protocol.json'),
        experiment_protocol_sha256=protocol['experiment_protocol_sha256'],
        common_init_sha256=COMMON_INIT_SHA, evaluator_binding=evaluator,
        val_annotation_sha256=sha(paths['val_ann']), train_annotation_sha256=sha(paths['train_ann']),
        training_profile_sha256=sha(paths['profile']), data_only_report_sha256=sha(data),
        image_manifest_sha256=sha(paths['image_manifest']), image_verification_sha256=sha(paths['image_verification']),
        native_parameter_count=10231997, added_parameter_count=0,
        baseline_training_epochs_added=0, baseline_model_forwards_added=0)
    for k in ('train_ann','val_ann','baseline_checkpoint','baseline_first100'):
        identity[k+'_sha256'] = sha(paths[k])
    receipt = dict(status='PREPARED_NOT_TRAINED', identity=identity,
        paths={k:str(v) for k,v in paths.items()}, resolved_sha256=sha(bundle/'resolved.yml'))
    write(bundle/'receipt.json', receipt)
    assert 'torch' not in sys.modules
    print(json.dumps(dict(status=receipt['status'], seed=seed, arm=arm, bundle=str(bundle),
                         current_images_verified=7019, new_baseline_forwards=0, optimizer_updates=0)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prefix', required=True); p.add_argument('--seed', type=int, choices=(1,2), required=True)
    p.add_argument('--arm', choices=('target_protect','random_protect'), required=True)
    a = p.parse_args(); prepare(a.prefix, a.seed, a.arm)
