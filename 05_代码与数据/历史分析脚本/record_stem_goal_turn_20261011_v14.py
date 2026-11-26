"""Preserve actual CPU recorder checks and same-epoch Stem comparisons."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; MAIN=OUT/'stem_component12_20261011_v1'; TRACE=OUT/'native_selection_trace_20261011_v1'
CODE=ROOT/'英文小论文/code/native_selection_trace_20261011_v1'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
code={p.name:sha(p) for p in CODE.glob('*.py')}
expected='76583a1c6cf7b35bf87fdb68e57cf33252a87d0bce62cc2da62d337f21acd75f'
for seed in (1,2):
    p=TRACE/f'validation_seed{seed}_v1.json'; assert sha(p)==expected
    d=read(p); assert d['code_sha256']==code and len(d['cases'])==16 and d['status']=='PASS_SOURCE_EXTRACTED_NATIVE_SELECTION_CPU_FIXTURES'
    assert d['source_sha256']=='b31408e49d4961e1e82692703b8f9fc3f07f1e280fd13ee12f896cb337f9bae5'
    assert not d['CUDA_initialized'] and d['detector_forwards']==d['bbox_head_calls']==d['optimizer_updates']==0
    assert all(all(c[k] for k in ('exact_native_return','actual_indices_checked','RNG_unchanged','inputs_unchanged','one_native_topk')) for c in d['cases'])
protocol=dict(schema='native_selection_diagnostic_preparation_v1',stage='CPU_RECORDER_CHECKS_COMPLETE_FULL_DETECTOR_RECORDING_PENDING',
    primary_API_source='https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/torch/overrides.py',
    decoder_source_sha256=d['source_sha256'],implementation_sha256=code,
    observed_native_boundary='Default max-class-score Top300 over8400 locations, before native selected-only bbox regression',
    copies=['dense_class_logits','dense_anchor_logits','actual_selected_indices','native_selected_anchor_logits'],
    dense_anchor_role='Unregressed positional prior. Dense predicted boxes do not exist at this native boundary.',
    permitted_next_use='Separate fixed12 EMA diagnostic processes after original Stem and random-native reference stop training and terminal packets pass; do not change frozen ongoing training or readers.',
    new_pretrained_BASE_forwards=0,new_pretrained_BASE_training_epochs=0,new_training_configs=0,
    actual_new_image_records=0,full_detector_equivalence_verified=False,
    decision_limit='Recorder readiness is engineering evidence only; no new root cause, AP improvement or novelty established.')
write(CODE/'protocol.json',protocol)
write(TRACE/'manifest.json',dict(status='EXACT_SCP_BYTES_AND_SOURCE_CHECKS_PASS',files={p.name:sha(p) for p in TRACE.glob('validation*.json')},code_sha256=code,protocol_sha256=sha(CODE/'protocol.json')))
progress=read(MAIN/'progress_v18.json');refs=read(OUT/'native_stem_scratch12_20261011_v3/compact_observation_v4.json'); pairs=[]
for o in refs:
    seed=o['seed'];assert o['completed_epochs']==1 and o['latest_metrics']['epoch']==0
    full=next(r for r in progress['rows'] if r['seed']==seed and r['arm']=='original_stem' and r['epoch']==1)
    v=o['latest_metrics']['test_coco_eval_bbox'];random_AP=100*v[0];random_APs=100*v[3]
    pairs.append(dict(seed=seed,epoch=1,enhanced_AP=full['AP'],random_native_AP=random_AP,delta_AP_pp=full['AP']-random_AP,
        enhanced_APs=full['APs'],random_native_APs=random_APs,delta_APs_pp=full['APs']-random_APs))
historical=read(MAIN/'historical_stem_recipe_audit_v1.json')
comparison=dict(status='ANALYZED_DESCRIPTIVE_SAME_EPOCH_EARLY_LOGS',current_S_pairs=pairs,historical_n_epoch12=historical['historical_comparison'][11],
    current_source_exact_original_stem=True,current_fixed12_results_available=False,
    historical_limit='Old n-scale whole recipes also differ in backbone aggregation/parameter count. The observed historical improvement is real in saved logs, but is not a Stem-only causal estimate.',
    current_limit='One common evaluated epoch only; do not compare enhanced epoch7/8 against random-native epoch1, or infer epoch12 outcome from epoch1.',
    evidence_sha256={'full_progress':sha(MAIN/'progress_v18.json'),'random_native':sha(OUT/'native_stem_scratch12_20261011_v3/compact_observation_v4.json'),'historical':sha(MAIN/'historical_stem_recipe_audit_v1.json')},
    model_forwards=0,optimizer_updates=0)
write(MAIN/'same_epoch_random_native_comparison_v1.json',comparison)
main=read(MAIN/'execution_state_v13.json')
main.update(supersedes='execution_state_v13.json',previous_goal_turn='progress',current_goal_turn='progress',no_progress_streak=0,goal_complete=False,
    candidate_selection_recorder='../native_selection_trace_20261011_v1/manifest.json',same_epoch_reference_comparison='same_epoch_random_native_comparison_v1.json',
    finding='New isolated recorder passed16 actual native-selector CPU cases on both servers, including ties/invalid anchors/teardown. Current common epoch1 does not show enhanced gain; old n-scale epoch12 saved logs do show a gain but whole recipes differ.',
    next='Keep bounded ongoing trials; obtain fixed12 EMA results. Use recorder in an independent terminal diagnostic replay to resolve selection-versus-selected-box-regression boundary without fabricating dense predicted boxes.')
write(MAIN/'execution_state_v14.json',main)
full=read(OUT/'original_stem12_20261011_v1/execution_state_v16.json')
full.update(supersedes='execution_state_v16.json',authoritative_state='../stem_component12_20261011_v1/execution_state_v14.json')
write(OUT/'original_stem12_20261011_v1/execution_state_v17.json',full)
print(json.dumps(dict(recorder_status=protocol['stage'],current_S_pairs=pairs,historical_n_epoch12=comparison['historical_n_epoch12']),ensure_ascii=False))
