"""Recompute raw gradient-array statistics, causal contrasts and mock audit evidence."""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import pathlib,hashlib,json,zipfile,math
import numpy as np
ROOT=pathlib.Path(r'C:\Users\jiwei\Desktop\小论文投稿')
OUT=ROOT/'outputs/encoder_decoder_gradient_diagnosis_20261010_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def extract_verify(archive,expected,dest,manifest):
    assert sha(archive)==expected
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for n in z.namelist():assert (dest/n).resolve().is_relative_to(dest.resolve())
        if not dest.exists():dest.mkdir();z.extractall(dest)
        assert {p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()}==set(z.namelist())
    files=read(dest/manifest)['files']
    for n,v in files.items():
        assert sha(dest/n)==(v['sha256'] if isinstance(v,dict) else v),n
        if isinstance(v,dict) and 'bytes' in v:assert (dest/n).stat().st_size==v['bytes']
    return len(files)+1
def main():
    archive=OUT/'gradient_evidence.zip';expected='9b7c0c8d64354d9aa6befa39af7c7f8876b705e6991e8e620fd333ef8a79b41f';dest=OUT/'collection';count=extract_verify(archive,expected,dest,'evidence_manifest.json')
    assert read(dest/'controller_completion.json')['exit_code']==0
    completion=read(dest/'results/completion.json');assert completion['forwards']==6 and completion['backwards']==6 and completion['optimizer_steps']==0 and completion['epochs_added']==0
    protocol=read(dest/'protocol.json');assert sha(dest/'probe.py')==protocol['probe_sha256']
    assert sha(dest/'controller.py')==protocol['controller_sha256']
    cases={};arrays_checked=0
    for case in protocol['cases']:
        name=case[0];row=read(dest/'results'/(name+'.json'));cases[name]=row
        assert row['parameters_unchanged_after_backward_clip'] and row['optimizer_steps']==0
        with np.load(dest/'results'/(name+'_gradients.npz')) as z:
            for phase in ['before_clip','after_clip']:
                for n,s in row[phase].items():
                    if not s['connected']:continue
                    a=z[phase+'::'+n];assert list(a.shape)==s['shape'];assert hashlib.sha256(a.tobytes()).hexdigest()==s['sha256']
                    assert bool(np.isfinite(a).all())==s['finite'];assert int(np.count_nonzero(a))==s['nonzero_elements']
                    if s['finite']:
                        norm=float(np.linalg.norm(a.astype(np.float64).reshape(-1)));maximum=float(np.max(np.abs(a)))
                        assert math.isclose(norm,s['norm64'],rel_tol=2e-10,abs_tol=1e-15),(name,n,norm,s['norm64'])
                        assert maximum==s['max_abs']
                    arrays_checked+=1
            for prefix,g in row['groups'].items():
                for phase in ['before_clip','after_clip']:
                    vals=[s for n,s in row[phase].items() if n.startswith(prefix)]
                    assert sum(v['nonzero_elements'] for v in vals)==g[phase]['nonzero_elements']
                    if g[phase]['finite']:
                        assert math.isclose(math.sqrt(sum(v['norm64']**2 for v in vals)),g[phase]['norm64'],rel_tol=1e-12,abs_tol=1e-15)
    prefixes=list(cases['common_fine_FP32']['groups']);assert len(prefixes)==4
    signatures={}
    for name in ['common_fine_FP32','activate_native_regression_FP32','activate_added_outputs_FP32','activate_both_FP32']:
        signatures[name]=[cases[name]['groups'][p]['before_clip']['nonzero_elements']>0 for p in prefixes]
    assert signatures['common_fine_FP32']==[False,True,False,False]
    assert signatures['activate_native_regression_FP32']==[False,True,True,True]
    assert signatures['activate_added_outputs_FP32']==[True,True,False,False]
    assert signatures['activate_both_FP32']==[True,True,True,True]
    with np.load(dest/'results/common_native_fine_predictions.npz') as z:
        for key in ['pred_boxes','pred_logits']:assert np.array_equal(z['common_native_FP32::'+key],z['common_fine_FP32::'+key])
    assert cases['common_native_FP32']['losses']==cases['common_fine_FP32']['losses']
    unit=read(dest/'results/norm_measurement_unit_case.json');assert unit['norm32']==0 and unit['norm64']>0 and unit['nonzero_elements']==2
    assert not cases['common_fine_AMP']['global_unscaled_all_parameters_finite']
    out2=ROOT/'outputs/encoder_decoder_audit_repair_20261010_v2';metadata=read(out2/'archive_metadata.json');d2=out2/'collection';n2=extract_verify(out2/'audit_repair_evidence.zip',metadata['sha256'],d2,'encoder_decoder_audit_repair_20261010_v2/combined_evidence_manifest.json')
    v2=d2/'encoder_decoder_audit_repair_20261010_v2';v1=d2/'encoder_decoder_audit_repair_20261010_v1'
    assert read(v1/'unit_terminal.json')['exit_code']==1
    assert read(v2/'unit_terminal.json')['exit_code']==0
    mock=read(v2/'mock_results/completion.json');assert mock['actual_optimizer_steps']==0 and mock['neural_forward_calls']==0 and mock['original_input_hash_RNG_parity_methods_AST_exact'] and mock['native_EMA_per_attempt_semantics_checked']
    assert sha(v2/'training_audit.py')==mock['repair_sha256']
    scalar=read(v2/'scalar_precision_probe.json');assert scalar['float16_cast']==0 and scalar['optimizer_steps']==0
    bound=read(OUT/'eight_update_source_bound.json');assert bound['coordinate_absolute_update_upper_bound']<bound['float16_round_to_zero_boundary']
    fallacies={'1_Simpson':'Four group signatures reported individually; no pooled AP.','2_Ecological':'One image engineering gradients cannot imply full VisDrone efficacy.','3_Berkson':'Fixed prior engineering fixture; not selected using current gradients.','4_Collider':'No query matching/collider-conditioned detection success claim.','5_Base_rates':'Exact nonzero tensor counts and norms retained; no new AP denominator.','6_Regression_to_mean':'No new accuracy measurement, best epoch or stopping endpoint.','7_Survivorship':'Failed real training and failed mock-v1 serialization preserved. AMP default-scale nonfinites explicitly reported.','8_Look_elsewhere':'Six intervention cases prospectively frozen; no scale/initializer search.','9_Forking_paths':'Manual activation only analytic; not silently replacing trained initialization; same configuration budget not reset.','10_Correlation_causation':'Controlled gate interventions establish conditional gradient blocking on the fixture, not original lost terminal tensors or detection improvement.','11_Reverse_causality':'No attribution of population small-target error to this startup behavior.'}
    result={'status':'PASS_LOCAL_GRADIENT_ARRAY_AND_PROSPECTIVE_AUDIT_VERIFICATION','gradient_archive_files_verified':count,'audit_archive_files_verified':n2,'raw_gradient_arrays_statistics_recomputed':arrays_checked,'causal_FP32_group_nonzero_signatures':signatures,'prefix_order':prefixes,'common_native_fine_predictions_and_losses_exact':True,'common_AMP_default_scale_nonfinite_gradients_observed':True,'FP32_norm_underflow_unit_case_verified':True,'native_EMA_per_attempt_audit_mock_verified':True,'original_input_RNG_methods_AST_exact':True,'failed_unit_v1_preserved':True,'full_forwards_added':6,'full_backwards_added':6,'actual_optimizer_updates_added':0,'training_epochs_added':0,'formal_configuration_restarted':False,'candidate_AP':None,'material_passport':{'skill':'academic-research-suite','workflow':'experiment-agent','mode':'run+validate','engineering_scope':'VERIFIED_RAW_GRADIENT_CONTRASTS_AND_MOCK_AUDIT_ONLY','scientific_scope':'ANALYZED_NO_EFFICACY'},'fallacy_scan':{'checked':11,'total':11,'items':fallacies},'limits':['One81GT validation image reused in TRAIN mode, not original10 training batches.','Eight-update source-derived FP16 bound is an analysis under explicit assumptions, not a recovered terminal weight trajectory.','The manual activation patterns are diagnostic interventions; no new initializer chosen or trained.','Actual whole-epoch training integration of the audit repair remains unverified.','No candidate AP improvement or population small-object root cause established.']}
    with (OUT/'validation_report.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ['fallacy_scan']},ensure_ascii=False))
if __name__=='__main__':main()
