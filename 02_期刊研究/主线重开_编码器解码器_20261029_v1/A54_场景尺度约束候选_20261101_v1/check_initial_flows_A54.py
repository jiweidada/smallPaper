from pathlib import Path
import hashlib,json,time
P=Path(__file__).resolve().parent;B=Path('/root/uav_a49_20261101_v1/arm_B')
keys=['order_sha256','first_eight_actual_input_sha256','first_eight_pre_forward_rng_sha256','all_batches_targets_including_ignore_tiles_sha256']
rows=[]
for epoch in [1,2]:
    files={'B':B/('epoch%02d_input_flow.json'%epoch),'M':P/'arm_M'/('epoch%02d_input_flow.json'%epoch)}
    data={a:json.loads(f.read_text()) for a,f in files.items()}
    checks={k:data['B'][k]==data['M'][k] for k in keys};assert all(checks.values())
    rows.append({'epoch':epoch,'checks':checks,'source_SHA256':{a:hashlib.sha256(f.read_bytes()).hexdigest() for a,f in files.items()}})
init=json.loads((P/'arm_M/initialization.json').read_text())
assert init['scratch_no_capacity_weights'] and init['all_native_initial_tensors_identical']
result={'status':'PASS','observation_only':True,'server_unix':time.time(),'completed_epochs_independently_compared':[1,2],
    'rows':rows,'native_init_sha256':init['common_init_sha256'],'new_module_init_sha256':init['new_module_initialization_sha256'],
    'input_scope':'Order and all-batch target/ignore metadata hashes; actual pixels and pre-forward RNG hashes cover the first8 batches of each epoch. This is not an assertion that all809 RNG states were individually stored.',
    'new_whole_model_forwards':0,'new_optimizer_updates':0}
(P/'M_first2_flow_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
