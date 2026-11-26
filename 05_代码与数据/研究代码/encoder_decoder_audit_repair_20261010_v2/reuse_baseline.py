"""Verify and reference an existing BASE12 evaluation without model execution."""
from pathlib import Path
import datetime
from runtime import HERE, read, write, sha, resolved_config


def verify_cached_baseline(receipt, plan, paths):
    spec = read(HERE / 'protocol.json')['baseline_reuse']
    source = Path(spec['source'])
    for name, expected in spec['pinned_files'].items():
        assert sha(source / name) == expected, 'Cached baseline changed: ' + name
    completion = read(source / 'completion.json')
    comparison = read(source / 'historical_comparison.json')
    assert completion['historical_log_reproduction_pass'] is True
    assert completion['all_parameters_frozen_like_native_EMA'] is True
    assert completion['checkpoint_sha256'] == receipt['identity']['baseline_checkpoint_sha256']
    assert completion['images'] == 548 and completion['optimizer_steps'] == 0
    assert comparison['historical_stats'] == plan['historical_baseline_stats']
    assert comparison['current_stats'] == plan['historical_baseline_stats']
    assert comparison['max_absolute_fraction_difference'] == 0
    checked = 0
    for folder, code_key, code_file in (
        ('endpoint_epoch12', 'capture_code_sha256', 'results.py'),
        ('stages_epoch12', 'trace_code_sha256', 'stage_trace.py'),
    ):
        manifest = read(source / folder / 'manifest.json')
        assert sha(source / folder / 'manifest.json') == completion[
            'endpoint_manifest_sha256' if folder == 'endpoint_epoch12' else 'stage_manifest_sha256']
        assert manifest[code_key] == sha(HERE / code_file)
        for name, expected in manifest['files'].items():
            assert Path(name).name == name
            assert sha(source / folder / name) == expected, 'Cached evidence changed: ' + name
            checked += 1
    summary = read(source / 'endpoint_epoch12/summary.json')
    assert summary['epoch_zero_based'] == 11 and summary['evaluated_weights'] == 'EMA'
    assert summary['identity']['arm'] == 'baseline' and summary['identity']['seed'] == plan['seed']
    assert summary['native_stats'] == plan['historical_baseline_stats']
    assert summary['identity']['evaluated_checkpoint_sha256'] == receipt['identity']['baseline_checkpoint_sha256']
    for key in ('common_init_sha256', 'baseline_checkpoint_sha256', 'baseline_first100_sha256',
                'train_ann_sha256', 'val_ann_sha256', 'image_manifest_sha256', 'evaluator_binding'):
        assert summary['identity'][key] == receipt['identity'][key], 'Different baseline binding: ' + key
    from evaluator_binding import validate_result_source
    validate_result_source(summary['evaluator_source'], summary['identity'])
    expected_config = resolved_config(plan, dict(paths, output=source), baseline=True)
    assert read(source / 'baseline.yml') == expected_config, 'Cached baseline config differs'
    return source, {
        'status': 'REUSE_PREVIOUSLY_VERIFIED_BASE12', 'source': str(source),
        'seed': plan['seed'], 'pinned_files': spec['pinned_files'],
        'evidence_files_verified': checked, 'historical_stats_max_abs_difference': 0,
        'new_model_forwards': 0, 'new_optimizer_steps': 0, 'new_epochs': 0,
        'original_evaluation_completion': completion['completed_remote_utc'],
        'original_identity_preserved': True,
        'verified_remote_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def attach_cached_baseline(receipt, plan, paths):
    source, proof = verify_cached_baseline(receipt, plan, paths)
    target = paths['screen_root'] / 'baseline_eval'
    assert not target.exists() and not target.is_symlink()
    # Keep the original evaluation's files, identity and timestamps untouched.
    target.symlink_to(source, target_is_directory=True)
    write(paths['screen_root'] / 'baseline_reuse.json', proof)
    return proof
