"""Read-only fine-gate binding. Does not import Torch or evaluate a model."""
from pathlib import Path
from runtime import HERE,read,sha

def verify_fine_evidence(seed):
    protocol=read(HERE/'protocol.json');folder=HERE/'promotion';index=read(folder/'index.json')
    assert sha(folder/'index.json')==protocol['fine_promotion_sha256']
    assert index['status']=='BOTH_FINE_FIXED12_RELOAD_AND_LOCAL_VERIFICATION_PASS'
    assert index['scientific_plan_sha256']==protocol['scientific_plan_sha256']
    for n in (1,2):
        entry=index['seeds'][str(n)];report=read(folder/f'seed{n}_verification_report.json')
        assert sha(folder/f'seed{n}_verification_report.json')==entry['local_verification_sha256']
        assert report['status']=='PASS_FIXED12_DOWNLOADED_EVIDENCE_AND_SAVED_WEIGHT_RELOAD'
        assert report['seed']==n and report['files_verified']==2202
        assert report['archive_sha256']==entry['terminal_archive']['sha256']
        assert report['all548_saved_weight_stage_arrays_exact'] and report['stage_geometry_TP_conflicts']==0
        base=report['baseline_overall_percent'];fine=report['candidate_overall_percent']
        assert fine['APs']>base['APs'] and fine['AP']>=base['AP'] and report['native_gate_passed_this_seed']
        assert report['cost']['candidate_parameters']==13806625 and report['cost']['added_parameters']==3574628
        count=report['final_training_counts'];assert count['attempts_total']==count['EMA_updates']==19404
        assert 0<count['actual_optimizer_steps_total']<=19404
        done=read(folder/f'seed{n}_training_completion.json');replay=read(folder/f'seed{n}_reload_verification.json')
        assert sha(folder/f'seed{n}_training_completion.json')==entry['training_completion_sha256']
        assert sha(folder/f'seed{n}_reload_verification.json')==entry['reload_verification_sha256']
        assert done['completed_epochs']==12 and done['entered_epochs']==list(range(12)) and done['native_gate_passed']
        assert done['identity']['arm']=='fine' and done['identity']['seed']==n
        assert done['identity']['implementation']['small_object_model.py']==protocol['prototype_model_sha256']['small_object_model.py']
        assert done['identity']['implementation']['geometry_memory.py']==protocol['prototype_model_sha256']['geometry_memory.py']
        assert replay['status']=='PASS_SAVED_CANDIDATE_RELOAD' and replay['all548_stage_arrays_exact']
        assert replay['checkpoint_sha256']==done['last_checkpoint_sha256']==entry['checkpoint_sha256']
        assert replay['optimizer_updates']==replay['epochs_added']==replay['baseline_model_forwards']==0
        assert replay['candidate_model_forwards']==137 and replay['images']==548
    own=index['seeds'][str(seed)];root=Path(own['remote_root']);screen=root/f'seed{seed}_fine'
    assert read(screen/'controller_completion.json')['status']=='COMPLETE_FIXED12'
    assert read(screen/'candidate_completion.json')['exit_code']==0
    assert sha(screen/'run/training_completion.json')==own['training_completion_sha256']
    assert sha(screen/'run/last.pth')==own['checkpoint_sha256']
    assert sha(screen/'candidate_reload_v1/verification.json')==own['reload_verification_sha256']
    assert read(screen/'candidate_reload_v1/completion.json')['exit_code']==0
    metadata=read(root/f'terminal_evidence_seed{seed}_metadata.json')
    assert metadata==own['terminal_archive'] and sha(root/f'terminal_evidence_seed{seed}.zip')==metadata['sha256']
    for name,pk,tk in [('controller_start.json','pid','start_ticks'),('candidate_start.json','child_pid','child_start_ticks')]:
        record=read(screen/name);proc=Path(f"/proc/{record[pk]}/stat")
        assert not proc.exists() or proc.read_text().split()[21]!=str(record[tk])
    return index

def verify_promotion(receipt,plan,paths):
    assert plan['variant']=='degraded' and plan['seed'] in (1,2)
    protocol=read(HERE/'protocol.json');assert protocol['execution_allowlist']=={'fine':[],'degraded':[1,2]}
    index=verify_fine_evidence(plan['seed']);proof=read(paths['screen_root']/'dual_seed_fine_promotion.json')
    assert proof['both_same_seed_native_gates_passed'] and proof['protocol_sha256']==sha(HERE/'protocol.json')
    assert proof['fine_promotion_sha256']==protocol['fine_promotion_sha256']==receipt['identity']['fine_promotion_sha256']
    identity=receipt['identity'];fine=read(HERE/'promotion'/f"seed{plan['seed']}_training_completion.json")['identity']
    for key in ('common_init_sha256','train_ann_sha256','val_ann_sha256','baseline_checkpoint_sha256','baseline_first100_sha256','image_manifest_sha256','evaluator_binding'):
        assert identity[key]==fine[key],key
    return index
