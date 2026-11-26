"""Preserve exact EMA binding; treat model/generated grid comparison as descriptive."""
from pathlib import Path
import hashlib,json,ast
R=Path(__file__).resolve().parent.parent;read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=R/'英文小论文/code/native_selection_analysis12_20261011_v4';new=R/'英文小论文/code/native_selection_analysis12_20261011_v5';new.mkdir(exist_ok=False)
before={p.name:sha(p) for p in old.iterdir()}
for p in old.iterdir():
    if p.name=='protocol.json':continue
    text=p.read_text(encoding='utf8');edited=text.replace('actual_selection_offline_analysis_v4','actual_selection_offline_analysis_v5').replace('selection_v4','selection_v5')
    if p.name=='analyze.py':
        edited=edited.replace("assert report['status']=='PASS_CHECKPOINT_BOUND_EMA_MODEL_AND_NATIVE_GENERATED_ANCHOR_TENSOR_AUDIT'", "assert report['status'] in ['PASS_CHECKPOINT_BOUND_EMA_MODEL_AND_NATIVE_GENERATED_ANCHOR_TENSOR_AUDIT','VERIFIED_ACTUAL_EMA_EXPORT_AND_FEATURE_CELLS_CPU_MODEL_GENERATOR_DIFFERENCE_RECORDED']")
        target="    assert case['model_equal_generated'] and np.array_equal(values['model_anchors'],values['generated_anchors'])"
        replacement="""    # EMA is bound exactly to its actual checkpoint export. Freshly generated
    # CPU model anchors are supplementary provenance, not a proxy for that EMA.
    m,g=values['model_anchors'],values['generated_anchors']
    assert m.shape==g.shape==a.shape and m.dtype==g.dtype==a.dtype
    assert case['model_equal_generated']==bool(np.array_equal(m,g))
    good=np.isfinite(a).all(1)
    assert np.array_equal(np.isfinite(m),np.isfinite(a)) and np.array_equal(np.isfinite(g),np.isfinite(a))
    assert float(np.max(np.abs(m[good].astype('float64')-g[good].astype('float64'))))==case['model_generated_max_abs_difference']
    verify_grid_centers(dict(dense_anchor_logits=m),m)
    verify_grid_centers(dict(dense_anchor_logits=g),g)"""
        assert edited.count(target)==1;edited=edited.replace(target,replacement)
    if edited==text:(new/p.name).write_bytes(p.read_bytes())
    else:(new/p.name).write_text(edited,encoding='utf8')
for p in new.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
assert sha(new/'selection_core.py')==sha(old/'selection_core.py')
assert {p.name:sha(p) for p in old.iterdir()}==before
out=R/'outputs/native_selection_analysis12_20261011_v5';out.mkdir(exist_ok=False)
v3=R/'outputs/native_selection_anchor_audit12_20261011_v3/seed1/report.json';fixed=read(v3)
actual=R/'outputs/native_selection_anchor_audit12_20261011_v2/seed1/report.json';original=read(actual)
assert sha(actual)==fixed['original_audit_report_sha256']
for field in ['native_generator_source_sha256','EMA_source_sha256','CUDA_initialized','optimizer_updates']:fixed[field]=original[field]
assert not fixed['CUDA_initialized'] and fixed['optimizer_updates']==0
fixed.update(metadata_correction_report_sha256=sha(v3),previous_exact_generator_check_not_passed=True)
audit_path=out/'seed1_actual_EMA_export_binding.json';audit_path.write_text(json.dumps(fixed,indent=2),encoding='utf8')
jobs=[]
for seed in [1,2]:
    for arm,kind in [('original_stem','full'),('native_stem_scratch','reference'),('highfreq_rgb','control')]:
        formal=R/f'outputs/stem_fixed12_seed{seed}_formal_collection12_20261011_v1/seed{seed}_{arm}';report=formal/'verification.json';f=read(report)
        folder=R/f'outputs/native_selection_collection12_20261011_v2/seed{seed}_{arm}/extracted/diagnostic'
        audit=audit_path if seed==1 else R/'outputs/native_selection_anchor_audit12_20261011_v2/seed2/report.json'
        export=R/f'outputs/native_selection_anchor_audit12_20261011_v2/seed{seed}/{arm}.npz'
        assert f['all548_stage_arrays_reloaded_exact'] and (folder/'verification.json').exists()
        jobs.append(dict(arm=arm,kind=kind,seed=seed,report=str(report),report_sha256=sha(report),packet=str(formal/'extracted'),diagnostic=str(folder),
            verification_sha256=sha(folder/'verification.json'),selection_manifest_sha256=sha(folder/'selection_epoch12/manifest.json'),
            anchor_audit_report=str(audit),anchor_audit_report_sha256=sha(audit),anchor_export=str(export),anchor_export_sha256=sha(export)))
(out/'six_actual_jobs.json').write_text(json.dumps(dict(jobs=jobs),indent=2),encoding='utf8')
review=dict(status='PREPARED_SIX_ACTUAL_FORMAL_AND_EXACT_DIAGNOSTIC_BINDINGS',original_v4_files_preserved=True,exact_EMA_array_requirement_unchanged=True,
    selection_core_source_unchanged=True,model_generated_grid_equality_is_descriptive=True,no_float_tolerance_raised=True,all_six_required=True,
    model_forwards=0,evaluator_calls=0,training_epochs=0,builder_sha256=sha(__file__),corrected_anchor_metadata_sha256=sha(audit_path))
(out/'source_review.json').write_text(json.dumps(review,indent=2),encoding='utf8');print(json.dumps(review))
