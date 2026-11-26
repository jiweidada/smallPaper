from pathlib import Path
import ast,hashlib,json,difflib
root=Path('英文小论文/code/warm_stem_primary_collection12_20261011_v1')
old=Path('英文小论文/code/stem_component_primary_verify12_20261011_v2/verify_primary.py');s=old.read_text(encoding='utf8')
items=[
("assert arm in ('detail_lowres','highfreq_rgb','attention_spatial_mean','early_final_downsample')","assert arm=='warm_stem'"),
("outputs/stem_component_primary_collection_20261011_v2/seed","outputs/warm_stem_primary_collection12_20261011_v1/seed"),
("collector=ROOT/'英文小论文/code/stem_component_primary_collection_20261011_v2'","collector=ROOT/'英文小论文/code/warm_stem_primary_collection12_20261011_v1'"),
("['component_files_sha256']","['warm_files_sha256']"),
("'exact12_component_primary_artifact_collection_v2'","'exact12_warm_primary_artifact_collection_v1'"),
("'PASS_COMPONENT_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS'","'PASS_WARM_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS'"),
("choices=('detail_lowres','highfreq_rgb','attention_spatial_mean','early_final_downsample')","choices=('warm_stem',)")
]
for b,a in items:
 assert s.count(b)==1,b;s=s.replace(b,a)
b="    assert executed['exit_code']==0 and executed['metadata']==meta"
a=b+"""
    assert executed['status']=='STOPPED_WARM_PRIMARY12_PACKAGE_COLLECTED'
    assert executed['receipt']['seed']==seed and not executed['receipt']['automatic_retry']
    assert executed['receipt']['source_sha256']==sha(Path(__file__).parent/'collect.py')
    assert executed['receipt']['protocol_sha256']==sha(Path(__file__).parent/'protocol.json')"""
assert s.count(b)==1;s=s.replace(b,a)
b="    assert read(run/'first100_parity.json')['all100_rows_exact']"
a=b+"""
    receipt=read(destination/'bundle/receipt.json');plan=read(destination/'bundle/plan.json')
    assert receipt['identity']==identity and plan['schema']=='warm_stem12_training_v1' and plan['arm']=='warm_stem' and plan['seed']==seed
    assert identity['protocol_sha256']==sha(destination/'code/training_protocol.json')
    assert identity['plan_sha256']==sha(destination/'bundle/plan.json') and receipt['resolved_sha256']==sha(destination/'bundle/resolved.yml')
    assert identity['native_parameter_count']==10231997 and identity['added_parameter_count']==11815
    assert identity['common_init_sha256']=='85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91'
    runtime=module('warm_primary_native_config',ROOT/'英文小论文/code/warm_stem12_20261011_v1/runtime.py')
    cfg=runtime.resolved_config(plan,{k:Path(v) for k,v in receipt['paths'].items()})
    assert read(destination/'bundle/resolved.yml')==cfg and read(run/'args.json')['yaml_cfg']==cfg
    start=read(run/'runtime_start.json')
    assert start['identity']==identity and start['optimizer_coverage']['all_parameters_exactly_once']
    full=read(destination/'screen/detector_preflight_v1/report.json')
    assert full['identity']==identity and full['status']=='PASS_CANDIDATE_FULL_DETECTOR_AMP_LOSS_BACKWARD_NO_UPDATES'
    assert full['optimizer_updates']==full['training_epochs']==0 and full['candidate_detector_forwards']==2
    for key in ('image_manifest','image_verification'):
        assert sha(destination/('provenance/'+key+'.json'))==identity[key+'_sha256']
    assert read(destination/'provenance/image_verification.json')['files_verified']==7019"""
assert s.count(b)==1;s=s.replace(b,a)
ast.parse(s);(root/'verify_primary.py').write_text(s,encoding='utf8',newline='\n')
(root/'verification_adaptation.diff').write_text(''.join(difflib.unified_diff(old.read_text(encoding='utf8').splitlines(True),s.splitlines(True),fromfile=str(old),tofile=str(root/'verify_primary.py'))),encoding='utf8')
print(json.dumps({'status':'WARM_PRIMARY_VERIFIER_PREPARED','source_sha256':hashlib.sha256((root/'verify_primary.py').read_bytes()).hexdigest(),
 'formal_terminal_gate_relaxed':False,'scope':'Own Warm identity/source bound; generic existing native-endpoint and stage-array helpers reused without edits.'}))
