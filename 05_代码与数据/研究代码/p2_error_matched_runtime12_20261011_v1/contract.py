"""Pure metadata contract; original replay status is preserved for each seed."""
def validate(seed,proposal,checked,completed,replay,original,protocol):
    assert type(seed) is int and seed in (1,2)
    binding=protocol['bindings'][str(seed)]
    assert checked['seed']==proposal['seed']==replay['seed']==original['seed']==seed
    assert checked['status']=='PASS_ACTUAL_CPU_RANK16_NORM_DC_ERROR_AND64_BASIS_OPERATOR_CHECK'
    assert completed['exit_code']==0 and completed['automatic_retry'] is False
    assert checked['rank']==16 and checked['basis64_exact'] is True and checked['CUDA_initialized'] is False
    assert checked['GPU_evaluation_completed'] is False and checked['root_or_efficacy_proven'] is False
    assert checked['norm_error']<1e-6 and abs(checked['error_difference'])<1e-6
    assert all(checked[k]==0 for k in ('full_detector_forwards','BASE_forwards','parent_forwards','evaluator_calls','optimizer_updates','training_epochs'))
    assert proposal['mode']=='svd16_dc_error_norm' and proposal['rank']==16
    assert proposal['model_replay_tolerance_changed'] is False
    assert replay['status']==binding['reference_replay_status']
    assert replay['strict_loaded_tensors']==801 and replay['parameters']==10249437
    assert replay['identity']==original['identity'] and replay['arm']=='p2_skip'
    assert replay['checkpoint_sha256']==original['checkpoint_sha256']==proposal['parent_checkpoint_sha256']==binding['checkpoint_sha256']
    assert replay['candidate_model_forwards']==137 and replay['images']==548
    assert replay['baseline_model_forwards']==replay['optimizer_updates']==replay['epochs_added']==0
    if seed==1:
        assert replay['status']=='CANDIDATE_RELOAD_MISMATCH_NO_RETRY'
        assert not replay['native_GT_hits_exact'] and not replay['all548_stage_arrays_exact'] and replay['stats_max_abs_fraction_difference']>0
    else:
        assert replay['status']=='PASS_SAVED_CANDIDATE_RELOAD'
        assert replay['native_GT_hits_exact'] and replay['all548_stage_arrays_exact'] and replay['stats_max_abs_fraction_difference']==0
    assert protocol['mode']=='svd16_dc_error_norm' and protocol['model_forwards_per_seed']==137
    assert protocol['parent_forwards_added']==protocol['BASE_forwards_added']==protocol['training_epochs_added']==0
