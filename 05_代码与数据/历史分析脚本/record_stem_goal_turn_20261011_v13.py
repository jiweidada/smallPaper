"""Verify current Stem identity and read-only live observations; no model execution."""
from pathlib import Path
import ast, hashlib, json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'; REF=OUT/'native_stem_scratch12_20261011_v3'; MAIN=OUT/'stem_component12_20261011_v1'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
for version in (3,4):
    metadata=read(REF/f'compact_transport_v{version}.json'); observations=[]
    for seed in (1,2):
        p=REF/f'compact_payload_seed{seed}_v{version}.json'; b=p.read_bytes(); t=metadata[str(seed)]
        assert len(b)==t['expected_bytes'] and sha(p)==t['expected_sha256'], (version,seed,'transport fidelity failure')
        o=json.loads(b)
        assert o['seed']==seed and o['code_sha256_verified'] and not o['failures']
        assert all(x['same_live'] and x['state']!='T' for x in o['processes'].values())
        assert o['runtime']['first100_parity.json']['all100_rows_exact']
        assert o['completed_epochs']<=12
        observations.append(o)
    write(REF/f'compact_observation_v{version}.json',observations)
source_note=read(ROOT/'英文小论文/code/original_stem12_20261011_v1/original_stem_source.json')
source=ROOT/source_note['source'].replace('\\','/'); extracted=ROOT/'英文小论文/code/original_stem12_20261011_v1/original_stem.py'
assert sha(source)==source_note['source_sha256'] and sha(extracted)==source_note['extracted_file_sha256']
def definitions(p):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text(encoding='utf-8-sig')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
original_defs,extracted_defs=definitions(source),definitions(extracted)
assert all(original_defs[n]==extracted_defs[n] for n in source_note['extracted_definitions'])
snapshot=read(MAIN/'training_observation_v18.json'); verified=read(MAIN/'training_observation_v18_transport_verification.json')
assert sha(MAIN/'training_observation_v18.json')==verified['verified_snapshot_sha256']
progress=read(MAIN/'progress_v18.json'); assert progress['snapshot_sha256']==sha(MAIN/'training_observation_v18.json')
init=read(MAIN/'initialization_provenance_v1.json'); historical=read(MAIN/'historical_stem_recipe_audit_v1.json')
audit=dict(status='SOURCE_AND_TRANSPORT_VERIFIED_OBSERVATIONAL_REPORT',same_original_EnhancedStemBlock_AST=True,
    source=str(source),source_sha256=sha(source),extracted=str(extracted),extracted_sha256=sha(extracted),
    replaced_path='backbone.stem',current_context='S/B0, only Stem replaced; retain native encoder/decoder and outside-Stem initial tensors',
    original_native_pretrained_stem_tensors=init['original_native_stem_exact_pretrained_tensors'],enhanced_stem_initialization='random',
    initialization_causal_effect_established=False,fixed12_efficacy_established=False,
    historical_fixed12=[dict(label=k,**historical['records'][k]['first12'][11]) for k in ('old_native_baseline','old_enhanced_stem_relu')],
    current_full_latest=[max([r for r in progress['rows'] if r['seed']==s and r['arm']=='original_stem'],key=lambda x:x['epoch']) for s in (1,2)],
    random_native_latest=[dict(seed=o['seed'],completed_epochs=o['completed_epochs'],iteration=o['iteration'],AP=100*o['latest_metrics']['test_coco_eval_bbox'][0],APs=100*o['latest_metrics']['test_coco_eval_bbox'][3],observed_remote_UTC=o['observed_remote_UTC']) for o in observations],
    new_model_forwards=0,new_optimizer_updates=0,
    interpretation='Same module is not identical to the historical whole experiment. Current gap mixes structure and pretrained Stem removal. Random-native comparison is pending; no allocation of gap to initialization or structural failure is established.')
write(MAIN/'stem_identity_and_progress_v1.json',audit)
ref=read(REF/'execution_state_v2.json')
ref.update(supersedes='execution_state_v2.json',evidence='compact_observation_v4.json',
    completed_epochs={str(o['seed']):o['completed_epochs'] for o in observations},
    optimizer_updates={str(o['seed']):o['runtime']['training_audit_live.json']['actual_optimizer_steps'] for o in observations},
    latest_live_processes={str(o['seed']):o['processes'] for o in observations},transport_verification='compact_transport_v4.json exact SHA256 and byte length verified')
write(REF/'execution_state_v3.json',ref)
main=read(MAIN/'execution_state_v12.json')
main.update(supersedes='execution_state_v12.json',evidence='training_observation_v18.json',baseline_comparison='progress_v18.json',
    reference_state='../native_stem_scratch12_20261011_v3/execution_state_v3.json',user_steering='核实是否用户原始增强Stem、未提升原因及随机原始Stem进度',
    identity_audit='stem_identity_and_progress_v1.json',current_goal_turn='progress',previous_goal_turn='progress',no_progress_streak=0,goal_complete=False,
    finding='Exact original EnhancedStem definitions verified; historical whole recipe differs. Both random-native arms completed1 epoch and run epoch2. Initialization does not yet explain the observed gap causally.')
for seed in (1,2):
    main['current_live_trainers']['seed'+str(seed)]={a:snapshot['seed'+str(seed)]['jobs'][a]['processes'] for a in ('original_stem','detail_lowres','highfreq_rgb')}
    assert all(p['same_live'] and p['state']!='T' for ps in main['current_live_trainers']['seed'+str(seed)].values() for p in ps)
write(MAIN/'execution_state_v13.json',main)
full=read(OUT/'original_stem12_20261011_v1/execution_state_v15.json')
full.update(supersedes='execution_state_v15.json',authoritative_state='../stem_component12_20261011_v1/execution_state_v13.json',reference_state='../native_stem_scratch12_20261011_v3/execution_state_v3.json')
write(OUT/'original_stem12_20261011_v1/execution_state_v16.json',full)
print(json.dumps(dict(identity=audit['same_original_EnhancedStemBlock_AST'],reference=audit['random_native_latest'],full=audit['current_full_latest']),ensure_ascii=False))
