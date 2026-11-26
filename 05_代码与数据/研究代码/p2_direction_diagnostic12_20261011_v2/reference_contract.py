"""Pure metadata contract: this does not upgrade a failed replay to a pass."""
def validate(preparation, completed, replay, protocol):
    assert preparation['seed']==replay['seed']==1 and replay['arm']=='p2_skip'
    assert preparation['status']=='PASS_ACTUAL_FIXED12_CPU_RANK16_NORM_MATCHED_WEIGHT_INTERVENTIONS'
    assert completed['exit_code']==0 and not completed['automatic_retry']
    assert replay['status']=='CANDIDATE_RELOAD_MISMATCH_NO_RETRY'
    assert replay['strict_loaded_tensors']==801 and replay['parameters']==10249437
    assert not replay['all548_stage_arrays_exact'] and not replay['native_GT_hits_exact']
    assert replay['candidate_model_forwards']==137 and replay['baseline_model_forwards']==0
    assert replay['optimizer_updates']==replay['epochs_added']==0 and not replay['automatic_retry']
    assert replay['images']==548 and replay['stats_max_abs_fraction_difference']>0
    assert preparation['identity']==replay['identity']
    assert preparation['checkpoint_sha256']==replay['checkpoint_sha256']==protocol['parent_checkpoint_sha256']
    assert preparation['source_sha256']==protocol['cpu_prepare_source_sha256']
    assert preparation['protocol_sha256']==protocol['cpu_protocol_sha256']
    assert completed['report_sha256']==protocol['CPU_report_sha256']
    assert completed['weights_npz_sha256']==protocol['CPU_weights_sha256']
    assert [x['mode'] for x in preparation['cases']]==protocol['interventions']==['dc16_norm','svd16_norm']
    assert all(x['numerical_rank']==16 and x['basis64_equal'] and x['relative_Frobenius_norm_error']<1e-6 for x in preparation['cases'])
    assert protocol['primary_contrast']=='svd16_norm_minus_dc16_norm'
    assert protocol['reference_replay_status']=='CANDIDATE_RELOAD_MISMATCH_NO_RETRY'
    assert protocol['parent_forwards_added']==protocol['BASE_forwards_added']==protocol['training_epochs_added']==0
    assert protocol['formal_training_efficacy_or_unique_root_proven'] is False
