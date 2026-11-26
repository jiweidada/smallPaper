"""Validate the new reader against real runtime identities and cached BASE arrays."""
from pathlib import Path
from copy import deepcopy
import hashlib,importlib.util,json,sys
R=Path(__file__).resolve().parent.parent
HERE=R/'英文小论文/code/p2_lowrank12_local_reader_20261011_v1'
OUT=R/'outputs/p2_lowrank12_local_reader_20261011_v1'
read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('actual_rank16_reader',HERE/'verify_terminal_local.py');reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
protocol=reader.local_protocol();review=read(OUT/'source_review.json')
assert {p.name:sha(p) for p in HERE.iterdir() if p.is_file()}==review['files']
captured=read(R/'outputs/turn7_lowrank_local_reader_actual_inputs.json');data={}
for k,v in captured.items():
    raw=v['payload'].encode();assert len(raw)==v['transport']['bytes'] and hashlib.sha256(raw).hexdigest()==v['transport']['sha256'];data[k]=json.loads(raw)
actual=[];negative=[]
for seed in (1,2):
    j=data[f'turn7_lowrank_local_reader_actual_runtime_inputs_seed{seed}']
    assert not j['terminal_exists'] and not j['stage_trace_manifest_exists']
    assert j['GPU_source_files']==protocol['implementation_sha256']
    f=j['files'];receipt=f['receipt']['report'];plan=f['plan']['report'];identity=receipt['identity'];cpu=f['cpu']['report'];runtime=f['runtime']['report']
    reader.validate_identity(identity,plan,protocol)
    assert identity['plan_sha256']==f['plan']['sha256'] and identity['cpu_initialization_sha256']==f['cpu']['sha256']
    assert identity==runtime['identity']==f['AMP']['report']['identity']
    init=cpu['initialization'];assert runtime['initialization']==init==f['AMP']['report']['initialization']
    expected=dict(parameters=10238173,extra_parameters=6176,exact_shared_tensors=794,new_state_tensors=8,removed_native_tensors=0,state_tensors=802)
    assert all(init[k]==v for k,v in expected.items()) and init['strict_load'] and init['all_shared_tensors_exact'] and init['new_state_unchanged']
    assert runtime['optimizer_coverage']==f['AMP']['report']['optimizer_coverage']==dict(trainable_tensors=493,all_native_parameter_groups_unchanged=True,all_parameters_exactly_once=True,added_parameters=6176)
    assert f['AMP']['report']['status']=='PASS_LOW_RANK_CONFIG_FULL_AMP_LOSS_BACKWARD_NO_UPDATES'
    assert [x['side'] for x in f['AMP']['report']['cases']]==[640,832]
    assert all(x['finite_gradients'] and all(x['branch'].values()) for x in f['AMP']['report']['cases'])
    assert f['AMP_completion']['report']['exit_code']==0
    module=reader.pure_module('actual_rank16_config_'+str(seed),'runtime.py')
    from pathlib import PurePosixPath
    cfg=module.resolved_config(plan,{k:PurePosixPath(v) for k,v in receipt['paths'].items()})
    serialized=json.dumps(cfg,ensure_ascii=False,indent=2).encode()
    assert hashlib.sha256(serialized).hexdigest()==receipt['resolved_sha256'],'Actual frozen config hash differs'
    fixture=OUT/f'actual_configuration_seed{seed}';(fixture/'bundle').mkdir(parents=True,exist_ok=False);(fixture/'run').mkdir()
    (fixture/'bundle/resolved.yml').write_bytes(serialized)
    (fixture/'run/args.json').write_text(json.dumps(dict(yaml_cfg=cfg)),encoding='utf8')
    reader.config_check(receipt,plan,fixture)
    cached=reader.reuse_sources(identity,protocol);annotation=read(cached/'reference/val_annotations.json')
    base,flags,checks=reader.native_endpoint(cached/'baseline_eval/endpoint_epoch12',annotation)
    reader.stages(cached/'baseline_eval',base['identity'],annotation)
    assert base['native_stats']==plan['historical_baseline_stats']
    actual.append(dict(seed=seed,real_runtime_identity_validated=True,actual_resolved_configuration_hash_exact=True,
        initialization_counts=expected,actual_AMP640_832_receipts_verified=True,cached_BASE_native_PR12_and_matching_flags=checks,
        cached_BASE_all548_stage_arrays_validated=True,cached_BASE_APs=base['overall_percent']['APs'],
        new_lowrank_terminal_exists=False,new_lowrank_formal_terminal_validated=False,metadata_capture_UTC=j['UTC']))
    for name,where,key,value in [('wrong_schema','identity','schema','original_stem12_training_v1'),('wrong_arm','identity','arm','p2_skip'),
        ('wrong_seed','identity','seed',3),('wrong_parameter_count','identity','added_parameter_count',17440),
        ('new_BASE_forward','identity','baseline_model_forwards_added',1),('overtraining','plan','max_epochs',13),
        ('wrong_endpoint','plan','fixed_endpoint','best_epoch'),('wrong_core','identity','implementation',{})]:
        ident,p=deepcopy(identity),deepcopy(plan);{'identity':ident,'plan':p}[where][key]=value
        try:reader.validate_identity(ident,p,protocol)
        except (AssertionError,KeyError):negative.append(dict(seed=seed,name=name))
        else:raise AssertionError(name)
    changed=deepcopy(cfg);changed['epoches']=12
    # Replay only the configuration checker with a deliberately changed saved
    # file; this is a labeled local checker fixture, not a remote model run.
    bad=OUT/f'bad_configuration_seed{seed}';(bad/'bundle').mkdir(parents=True);(bad/'run').mkdir()
    (bad/'bundle/resolved.yml').write_text(json.dumps(changed),encoding='utf8');(bad/'run/args.json').write_text(json.dumps(dict(yaml_cfg=changed)),encoding='utf8')
    try:reader.config_check(receipt,plan,bad)
    except AssertionError:negative.append(dict(seed=seed,name='changed_native_recipe'))
    else:raise AssertionError('changed_native_recipe')
for name,names in [('parent_directory',['../escape']),('absolute',['/escape']),('Windows_drive',['C:/escape']),('duplicates',['x','x']),('case_duplicate',['x','X'])]:
    try:reader.validate_names(names)
    except AssertionError:negative.append(dict(name=name))
    else:raise AssertionError(name)
assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
report=dict(status='PASS_REAL_TWO_SEED_RUNTIME_AND_CACHED_BASE_LOCAL_READER_PREPARATION',actual=actual,negative_cases=negative,
    new_terminal_positive_records=0,source_review_sha256=sha(OUT/'source_review.json'),local_reader_protocol_sha256=sha(HERE/'local_reader_protocol.json'),
    new_model_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,new_training_epochs=0,
    scope='Actual runtime identity/config/initialization and archived BASE data checked. Local bad-input fixtures reject bad identities, recipe changes and paths. No new candidate fixed12 or saved-EMA replay exists yet; full terminal verification remains pending.',
    validation_script_sha256=sha(__file__))
with (OUT/'verification.json').open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(status=report['status'],real_seed_checks=[x['seed'] for x in actual],negative_cases=len(negative),new_lowrank_formal_terminals=0,new_model_forwards=0)))
