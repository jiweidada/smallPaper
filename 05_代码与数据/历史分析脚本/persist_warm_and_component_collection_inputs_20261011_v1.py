from pathlib import Path
import argparse,base64,hashlib,json
p=argparse.ArgumentParser();p.add_argument('--input-base64',required=True);a=p.parse_args()
data=json.loads(base64.b64decode(a.input_base64))
def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def checked(item):
    raw=item['payload'].encode();m=item['transport']
    assert len(raw)==m['bytes'] and hashlib.sha256(raw).hexdigest()==m['sha256']
    return raw,json.loads(raw)
warm=Path('outputs/warm_stem12_20261011_v1');deploy={};observed={};transport={}
for seed in (1,2):
    key=str(seed)
    raw,o=checked(data['deployment'][key]);assert o['preparation_exit_code']==0 and o['source_SHA_verified'] and o['source_files']==26
    assert o['images_verified']==7019 and o['parameters']==10243812 and o['extra_parameters']==11815
    assert o['training_epochs']==o['BASE_forwards']==o['existing_process_signals']==0
    (warm/f'remote_deployment_seed{seed}_v1.json').write_bytes(raw);deploy[key]=o
    raw,o=checked(data['warm_observation'][key]);assert o['source_SHA_verified'] and o['dispatcher']['same_live'] and not o['failure']
    assert o['dispatcher']['pid']==deploy[key]['handle']['pid'] and o['dispatcher']['start_ticks']==deploy[key]['handle']['start_ticks']
    (warm/f'remote_observation_seed{seed}_v1.json').write_bytes(raw);observed[key]=o
    transport[key]={n:data[n][key]['transport'] for n in ('deployment','warm_observation')}
write(warm/'transport_verification_v1.json',dict(status='PASS_BOTH_DEPLOYMENT_AND_LIVE_DISPATCHER_EXACT_BYTES_SHA',seed=transport))
write(warm/'execution_state_v1.json',dict(status='PREPARED_BOTH_SEEDS_RESOURCE_GATED_DISPATCHERS_VERIFIED_LIVE',
    previous_goal_turn='progress',current_goal_turn='progress',no_progress_streak=0,goal_complete=False,
    CPU_engineering_pass='../warm_stem_prototype_20261011_v2/CPU_validation_summary_v1.json',
    CUDA_Stem_engineering_pass='../warm_stem_gpu_preflight_20261011_v1/GPU_validation_summary_v1.json',
    full_detector_preflight_passed=False,actual_new_training_epochs={s:len(o['training_metrics']) for s,o in observed.items()},
    native_original_tensors_retained=794,extra_parameters=11815,per_configuration_epoch_cap=12,
    existing_BASE_epochs_added=0,existing_BASE_forwards_added=0,existing_process_signals=0,
    dispatchers={s:o['dispatcher'] for s,o in observed.items()},phases={s:o['live']['phase'] for s,o in observed.items()},
    resource_rule='Current3 older GPU jobs reserve27648MiB including the candidate, above24576MiB total. Wait for a native trainer to finish. No old job is interrupted; new candidate runs once after the full detector preflight.',
    AP_improvement_proven=False,root_or_novelty_proven=False,
    pending='Full CUDA detector/native criterion preflight, one12-epoch training per seed, exact12 EMA, saved-weight replay and paired error-stage analysis.'))
root=Path('outputs/stem_component_primary_collection_20261011_v2');root.mkdir(exist_ok=False)
for seed in (1,2):
    key=str(seed);raw,o=checked(data['HF_collection'][key]);assert o['exit_code']==0
    m=o['metadata'];assert m['seed']==seed and m['arm']=='highfreq_rgb' and m['status']=='COLLECTED_STOPPED_EXACT12_PRIMARY_ARTIFACTS'
    assert m['collector_sha256']==hashlib.sha256(Path('英文小论文/code/stem_component_primary_collection_20261011_v2/collect.py').read_bytes()).hexdigest()
    assert m['collector_protocol_sha256']==hashlib.sha256(Path('英文小论文/code/stem_component_primary_collection_20261011_v2/protocol.json').read_bytes()).hexdigest()
    assert m['new_model_forwards']==m['new_evaluator_calls']==m['new_training_epochs']==0 and not m['saved_EMA_independent_replay_included']
    dest=root/f'seed{seed}_highfreq_rgb_v1';dest.mkdir()
    (dest/'remote_collection_execution_v1.json').write_bytes(raw)
    write(dest/'remote_collection_transport_v1.json',data['HF_collection'][key]['transport'])
    write(dest/'remote_collection_metadata_v1.json',m)
print(json.dumps(dict(status='PERSISTED_ACTUAL_TWO_WARM_DISPATCHERS_AND_TWO_COMPONENT_COLLECTIONS',new_training_epochs=0,HF_archives_pending_download=True)))
