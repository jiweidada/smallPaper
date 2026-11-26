"""Read and attach the original frozen native BASE12 packet, no forwards."""
from pathlib import Path
from runtime import HERE, read, sha, write


def attach_cached_baseline(receipt, plan, paths):
    source = paths['baseline_cached']; spec = read(HERE/'training_protocol.json')['baseline_reuse'][str(plan['seed'])]
    assert str(source) == spec['source']
    for n,h in spec['pinned_files'].items():
        assert sha(source/n) == h, n
    completion = read(source/'completion.json'); history = read(source/'historical_comparison.json')
    assert completion['historical_log_reproduction_pass'] and completion['all_parameters_frozen_like_native_EMA']
    assert completion['images'] == 548 and completion['optimizer_steps'] == 0
    assert completion['checkpoint_sha256'] == receipt['identity']['baseline_checkpoint_sha256']
    assert history['current_stats'] == history['historical_stats'] == plan['historical_baseline_stats']
    assert history['max_absolute_fraction_difference'] == 0
    count = 0
    for name in ('endpoint_epoch12','stages_epoch12'):
        manifest = read(source/name/'manifest.json')
        for file, digest in manifest['files'].items():
            assert Path(file).name == file and sha(source/name/file) == digest
            count += 1
    summary = read(source/'endpoint_epoch12/summary.json')
    assert summary['identity']['arm'] == 'baseline' and summary['identity']['seed'] == plan['seed']
    assert summary['native_stats'] == plan['historical_baseline_stats']
    from evaluator_binding import validate_result_source
    validate_result_source(summary['evaluator_source'], summary['identity'])
    target = paths['screen_root']/'baseline_eval'
    assert not target.exists() and not target.is_symlink()
    target.symlink_to(source, target_is_directory=True)
    proof = dict(status='REUSE_VERIFIED_NATIVE_BASE12', original_source=str(source), evidence_files_verified=count,
                 new_model_forwards=0, new_optimizer_steps=0, new_epochs=0, original_identity_preserved=True)
    write(paths['screen_root']/'baseline_reuse.json', proof)
    return proof
