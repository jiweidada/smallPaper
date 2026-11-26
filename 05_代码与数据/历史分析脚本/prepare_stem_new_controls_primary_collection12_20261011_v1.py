from pathlib import Path
import hashlib
import json

read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs=read('outputs/stem_new_controls_primary_collection12_20261011_v1_inputs.json')
values={}
for key in ('deployment','observation'):
    raw=inputs[key]['payload'].encode('utf8');t=inputs[key]['transport']
    assert len(raw)==t['bytes'] and hashlib.sha256(raw).hexdigest()==t['sha256']
    values[key]=json.loads(raw)
o=values['observation'];d=values['deployment']
assert o['collector']['same_live'] is False and o['failure'] is None
assert o['completion']['status']=='BOTH_STOPPED_SEED2_NATIVE12_CONTROL_PRIMARY_PACKAGES_COLLECTED'
assert o['collector']['pid']==d['handle']['pid'] and o['collector']['start_ticks']==d['handle']['start_ticks']
assert o['source_sha256']==sha('英文小论文/code/stem_new_controls_primary_collect12_20261011_v1/collect_completed.py')
root=Path('outputs/stem_new_controls_primary_collection12_20261011_v1');root.mkdir(exist_ok=False)
for key in ('deployment','observation'):
    (root/f'{key}.json').write_bytes(inputs[key]['payload'].encode('utf8'))
(root/'transport.json').write_text(json.dumps({k:inputs[k]['transport'] for k in ('deployment','observation')},indent=2),encoding='utf8')
(root/'setup_path_correction.txt').write_text('Initial preparation read the wrong collector path and exited before creating a worker or running any model. Existing deployed code was found at code/collect.py; only the new read-only packaging wrapper was corrected. No training or model replay was retried.\n'+inputs['setup_failure'],encoding='utf8')
for arm,item in o['values'].items():
    assert arm in ('attention_spatial_mean','early_final_downsample')
    meta=item['metadata'];raw=item['execution_raw'].encode('utf8');t=item['execution_transport']
    assert len(raw)==t['bytes'] and hashlib.sha256(raw).hexdigest()==t['sha256']
    assert json.loads(raw)==dict(exit_code=0,metadata=meta)
    assert meta in o['completion']['results'] and meta['seed']==2 and meta['arm']==arm
    assert meta['archive']==f'/24085403014/DEIM/experiments/stem_component_primary_collection_20261011_v2/seed2_{arm}_v1/primary_artifacts_v1.zip'
    assert meta['new_model_forwards']==meta['new_evaluator_calls']==meta['new_training_epochs']==0
    assert meta['saved_EMA_independent_replay_included'] is False
    target=Path('outputs/stem_component_primary_collection_20261011_v2')/f'seed2_{arm}_v1'
    target.mkdir(exist_ok=False)
    (target/'remote_collection_metadata_v1.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    (target/'remote_collection_execution_v1.json').write_bytes(raw)
    (target/'remote_collection_transport_v1.json').write_text(json.dumps(t,indent=2),encoding='utf8')
    print(json.dumps(dict(arm=arm,bytes=meta['bytes'],sha256=meta['sha256'],download_pending=True)))
