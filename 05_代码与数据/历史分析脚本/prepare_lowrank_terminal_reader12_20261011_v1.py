"""Freeze a pure local terminal verifier before either new fixed12 endpoint exists."""
from pathlib import Path
import ast,hashlib,json
R=Path(__file__).resolve().parent.parent
read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=R/'英文小论文/code/original_stem12_monitor_20261011_v1'
core=R/'英文小论文/code/p2_lowrank12_20261011_v3'
new=R/'英文小论文/code/p2_lowrank12_local_reader_20261011_v1'
out=R/'outputs/p2_lowrank12_local_reader_20261011_v1'
assert not new.exists() and not out.exists()
before={p.name:sha(p) for p in core.iterdir() if p.is_file()}
source=(old/'verify_terminal_local.py').read_text(encoding='utf8')
def replace(target,replacement):
    global source
    assert source.count(target)==1,target
    source=source.replace(target,replacement)
replace('Verify downloaded original-Stem training and exact saved EMA endpoint. No Torch.',
        'Verify new rank16 P2 fixed12 terminal packets; all saved EMA arrays exact. No Torch.')
replace("CORE = ROOT / '英文小论文/code/original_stem12_20261011_v1'", "CORE = ROOT / '英文小论文/code/p2_lowrank12_20261011_v3'")
replace("PASS = 'PASS_ORIGINAL_STEM_FIXED12_DOWNLOADED_EVIDENCE_AND_SAVED_EMA_RELOAD'", "PASS = 'PASS_LOW_RANK_P2_FIXED12_DOWNLOADED_EVIDENCE_AND_EXACT_SAVED_EMA_RELOAD'")
replace("p['schema'] == 'original_stem_fixed12_local_reader_v1'", "p['schema'] == 'p2_lowrank12_local_reader_v1'")
replace("manifest['schema'] == 'original_stem12_terminal_v1'", "manifest['schema'] == 'p2_lowrank12_terminal_v1'")
replace("identity['schema'] == plan['schema'] == 'original_stem12_training_v1'", "identity['schema'] == plan['schema'] == 'p2_lowrank12_training_v1'")
replace("identity['arm'] == plan['arm'] == 'original_stem'", "identity['arm'] == plan['arm'] == 'p2_lowrank16'")
replace("identity['native_parameter_count'] == 10231997 and identity['added_parameter_count'] == 5325", "identity['native_parameter_count'] == 10231997 and identity['added_parameter_count'] == 6176")
replace("plan['max_epochs'] == 12 and plan['recipe_epochs'] == 132 and plan['extra_parameters'] == 5325", "plan['max_epochs'] == 12 and plan['recipe_epochs'] == 132 and plan['extra_parameters'] == 6176 and plan['expected_parameters']==10238173")
start=source.index('def config_check(');end=source.index('\n\ndef stages(',start)
source=source[:start]+'''def config_check(receipt, plan, destination):
    from pathlib import PurePosixPath
    cfg=deepcopy(plan['historical_resolved_config'])
    assert cfg['model']=='ResidualGeometryDEIM' and cfg['ResidualGeometryDEIM']['arm']=='baseline'
    assert cfg['epoches']==132 and cfg['flat_epoch']==64 and cfg['no_aug_epoch']==12
    assert cfg['train_dataloader']['total_batch_size']==4
    assert cfg['train_dataloader']['dataset']['transforms']['policy']['epoch']==[4,64,120]
    assert cfg['seed']==plan['seed'] and cfg['use_amp'] and cfg['use_ema']
    assert not cfg.get('use_amp_fallback_fp32',False)
    monitor=cfg.get('baseline_log_monitor')
    assert monitor is None or (isinstance(monitor,dict) and monitor.get('enabled') is False)
    runtime=pure_module('lowrank12_pure_configuration','runtime.py')
    expected=runtime.resolved_config(plan,{k:PurePosixPath(v) for k,v in receipt['paths'].items()})
    assert read(destination/'bundle/resolved.yml')==expected
    assert read(destination/'run/args.json')['yaml_cfg']==expected
'''+source[end:]
start=source.index("    init=runtime['initialization']");end=source.index("    assert runtime['resume']",start)
source=source[:start]+'''    init=runtime['initialization']
    assert init['parameters']==10238173 and init['extra_parameters']==6176
    assert init['exact_shared_tensors']==794 and init['state_tensors']==802
    assert init['new_state_tensors']==8 and init['removed_native_tensors']==0
    assert init['removed_common_prefixes']==[] and init['new_state_prefixes']==['backbone.p2_skip.']
    assert init['user_arm']=='p2_lowrank16' and init['common_init_sha256']==identity['common_init_sha256']
    assert init['strict_load'] and init['all_shared_tensors_exact'] and init['new_state_unchanged']
    assert init==read(destination/'bundle/cpu_initialization.json')['initialization']
    assert runtime['optimizer_coverage']==dict(trainable_tensors=493,all_native_parameter_groups_unchanged=True,
        all_parameters_exactly_once=True,added_parameters=6176)
    assert runtime['user_source']==read(CORE/'user_sources.json')
    assert runtime['added_parameters']==6176 and 'matching_cost' not in runtime
    preflight=read(screen/'detector_preflight_v1/report.json')
    assert preflight['status']=='PASS_LOW_RANK_CONFIG_FULL_AMP_LOSS_BACKWARD_NO_UPDATES'
    assert preflight['identity']==identity and preflight['initialization']==init
    assert preflight['optimizer_coverage']==runtime['optimizer_coverage']
    assert preflight['candidate_detector_forwards']==2 and preflight['BASE_detector_forwards']==0
    assert preflight['optimizer_updates']==preflight['training_epochs']==preflight['evaluator_calls']==0
    assert preflight['states_discarded'] and [x['side'] for x in preflight['cases']]==[640,832]
    assert all(x['finite_gradients'] and all(x['branch'][k] for k in ('zero_restore_maintained','restore_gradient_active','upstream_bridge_gradients_initially_zero')) for x in preflight['cases'])
    assert preflight['preflight_source_sha256']==sha(CORE/'preflight.py')
    launch=read(screen/'detector_preflight_dispatch.json');completion=read(screen/'detector_preflight_completion.json')
    prefix={1:'/25085400120',2:'/24085403014'}[identity['seed']]
    python=prefix+('/envs/deim/bin/python' if identity['seed']==1 else '/deim_migration_runtime/envs/deim/bin/python')
    assert launch['command']==[python,'-B',prefix+'/DEIM/experiments/p2_lowrank12_20261011_v3/code/preflight.py','--bundle',receipt['paths']['screen_root']+'/bundle']
    assert completion['exit_code']==completion['training_epochs']==completion['optimizer_updates']==0
'''+source[end:]
replace("row['original_stem_only'] is True and row['candidate_active_from_epoch_zero'] is True", "row['user_backbone_arm']=='p2_lowrank16' and row['candidate_active_from_epoch_zero'] is True")
replace("row['added_parameters']==5325", "row['added_parameters']==6176")
replace("closeout['schema'] == 'original_stem12_closeout_v1'", "closeout['schema'] == 'p2_lowrank12_closeout_v1'")
replace("proof['strict_loaded_tensors'] == 831 and proof['parameters'] == 10237322", "proof['strict_loaded_tensors'] == 802 and proof['parameters'] == 10238173")
replace("saved['arm'] == 'original_stem'", "saved['arm'] == 'p2_lowrank16'")
new.mkdir();out.mkdir()
(new/'verify_terminal_local.py').write_text(source,encoding='utf8')
ast.parse(source)
base=read(old/'local_reader_protocol.json')
protocol=dict(schema='p2_lowrank12_local_reader_v1',endpoint=base['endpoint'],acceptance=base['acceptance'],
    core_files_sha256=before,implementation_sha256=before,
    reader_files_sha256={'verify_terminal_local.py':sha(new/'verify_terminal_local.py')},
    reused_local_evidence=base['reused_local_evidence'],common_init_relative_path=base['common_init_relative_path'],
    prepared_before_new_terminal_file_consumption=True,new_lowrank_terminal_packets_validated=0,
    new_model_forwards=0,new_training_epochs=0,source_reader_sha256=sha(old/'verify_terminal_local.py'),
    notes=['Adapted only the arm/schema/model/configuration/initialization and preflight checks to the real rank16 implementation.',
        'Native PR reconstruction, all548 original/reloaded stage arrays, zero replay tolerance, all12 epochs and both-seed efficacy conditions remain.',
        'The frozen remote metadata uses a legacy STEM status label. The manifest schema, exact arm and implementation bind this to rank16 P2; the label alone cannot prove identity.',
        'A replay failure prevents formal PASS, is preserved, and must not cause automatic retraining or tolerance increase.'])
(new/'local_reader_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
assert before=={p.name:sha(p) for p in core.iterdir() if p.is_file()}
review=dict(status='PREPARED_LOW_RANK_P2_FIXED12_LOCAL_VERIFIER_NOT_YET_CONSUMED_NEW_TERMINAL',
    files={p.name:sha(p) for p in new.iterdir()},remote_core_unchanged=True,
    new_terminal_positive_records=0,exact548_replay_requirement_unchanged=True,stats_fraction_tolerance=0,
    baseline_training_or_model_forwards_added=0,builder_sha256=sha(__file__))
(out/'source_review.json').write_text(json.dumps(review,indent=2),encoding='utf8')
print(json.dumps(review))
