from pathlib import Path
import hashlib,json
read=lambda p:json.loads(Path(p).read_bytes())
source=Path('英文小论文/code/stem_random_primary_collection12_20261011_v1/collect_stopped.py')
assert hashlib.sha256(source.read_bytes()).hexdigest()=='3670dd90d0919c80d97c6c98f2dcd6c354e3917d4797318bbb5a38070f9b96d0'
inputs=read('outputs/stem_random_primary_collection12_20261011_v1_inputs.json')
root=Path('outputs/stem_random_primary_collection12_20261011_v1');root.mkdir(exist_ok=False)
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
for seed,obs in inputs['observations'].items():
    decoded={}
    for label,item in obs.items():
        b=item['payload'].encode('utf8');t=item['transport']
        assert len(b)==t['bytes'] and hashlib.sha256(b).hexdigest()==t['sha256']
        decoded[label]=json.loads(b);(root/f'seed{seed}_{label}.json').write_bytes(b)
    p=decoded['precheck'];v=decoded['collection'];m=v['metadata']
    assert not p['trainer']['same_live'] and not p['controller']['same_live'] and not p['output_exists']
    assert p['training_completion']['completed_epochs']==12 and p['candidate_completion']['exit_code']==0
    assert v['status']=='STOPPED_RANDOM_NATIVE_STEM_PRIMARY12_PACKAGE_COLLECTED' and v['exit_code']==0
    assert m['seed']==int(seed) and m['arm']=='native_stem_scratch' and not m['saved_EMA_independent_replay_included']
    assert v['receipt']['wrapper_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    assert m['new_model_forwards']==m['new_evaluator_calls']==m['new_training_epochs']==0
    assert m['last_checkpoint_sha256']==p['training_completion']['last_checkpoint_sha256']
    folder=Path(f'outputs/stem_primary_collection_20261011_v1/seed{seed}_native_stem_scratch_v1');folder.mkdir(exist_ok=False)
    (folder/'remote_collection_execution_v1.json').write_bytes(obs['collection']['payload'].encode('utf8'))
    write(folder/'remote_collection_transport_v1.json',obs['collection']['transport'])
    write(folder/'remote_collection_metadata_v1.json',m)
write(root/'transport.json',{s:{l:i['transport'] for l,i in o.items()} for s,o in inputs['observations'].items()})
write(root/'execution_state_v1.json',dict(status='BOTH_STOPPED_RANDOM_NATIVE_STEM_PRIMARY12_REMOTE_PACKAGES_COLLECTED_DOWNLOAD_PENDING',
    native_saved_EMA_independent_replay_pending=True,new_BASE_forwards=0,new_training_epochs=0,goal_complete=False))
print(json.dumps(dict(status='BOTH_REMOTE_PACKAGES_COLLECTED_LOCAL_DESTINATIONS_PREPARED',seeds=[1,2],goal_complete=False)))
