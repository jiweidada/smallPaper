"""Verify complete seed1 nativeEMA replay and all six frozen-state contrasts."""
from pathlib import Path
import hashlib,json,zipfile,numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'outputs/baseline_ema_factorial_20261010_v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
archive=OUT/'collection/EMA_inference_correction_evidence_seed1.zip';expected='39a130ac8639fd61a41f47d0c4c5875166fceff6edb98346d5f3f0123698abd8';assert sha(archive)==expected
dest=OUT/'collection/seed1';dest.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name in z.namelist():assert (dest/name).resolve().is_relative_to(dest.resolve())
    z.extractall(dest)
files=json.loads((dest/'EMA_inference_correction_manifest_seed1.json').read_text())['files']
for name,v in files.items():assert sha(dest/name)==v['sha256'] and (dest/name).stat().st_size==v['bytes']
full=dest/'complete_replay_seed1_v1';result=json.loads((full/'completion.json').read_text())
assert result['current_stats']==result['historical_stats'] and result['historical_max_absolute_fraction_difference']==0 and result['historical_log_reproduction_pass']
assert not result['current_reference_reproduction_pass'] and not result['all_stage_arrays_exactly_equal_to_first_replay']
assert result['state_parameters_and_BN_unchanged'] and result['optimizer_steps']==0 and result['epochs_added']==0
with np.load(full/'endpoint_epoch12/precision_recall.npz') as z:
    p,r=z['precision'],z['recall']
    avg=lambda v:float(v[v>-1].mean())
    stats=[avg(p[:,:,:,0,2]),avg(p[0,:,:,0,2]),avg(p[5,:,:,0,2]),*[avg(p[:,:,:,a,2]) for a in (1,2,3)],*[avg(r[:,:,0,m]) for m in (0,1,2)],*[avg(r[:,:,a,2]) for a in (1,2,3)]]
assert max(abs(x-y) for x,y in zip(stats,result['current_stats']))<1e-12
old=ROOT/'outputs/small_object_encoder_decoder12_20261010_v1/collection/seed1/seed1_fine/baseline_eval'
with np.load(dest/'ema_factorial_seed1_v1/outputs.npz') as z:
    for field in ('boxes','logits'):
        native=np.stack([np.load(old/f'stages_epoch12/image_{i:06d}.npz')['final_'+field] for i in range(4)])
        frozen=np.stack([np.load(full/f'stages_epoch12/image_{i:06d}.npz')['final_'+field] for i in range(4)])
        for name in ('native_grad_flags','native_restored_flags'):assert np.array_equal(z[name+'_pred_'+field],native)
        for name in ('native_all_frozen','deepcopy_all_frozen','native_EMA','wrapper_all_frozen'):assert np.array_equal(z[name+'_pred_'+field],frozen)
        assert not np.array_equal(native,frozen)
for i in range(548):
    with np.load(full/f'stages_epoch12/image_{i:06d}.npz') as z:assert all(np.isfinite(z[k]).all() for k in z.files)
fallacies={'1_Simpson':'Seeds and ordinary/frozen EMA references remain separate. No pooled improvement.','2_Ecological':'Four-image causal flag toggle is engineering evidence; no population localization mechanism inferred.','3_Berkson':'Full548-image validation retained; diagnostic first4 selected before output inspection.','4_Collider':'Query order can change with flags; large slotwise deltas are not object-identity localization errors.','5_Base_rates':'Native12-stat arrays and original COCO denominators preserved; percentages and pp distinguished.','6_Regression_to_mean':'Fixed historical epoch12; no best epoch or candidate outcomes observed.','7_Survivorship':'Both failed v1 controllers, zero-forward config-suffix failures and rejected exact-current hypothesis preserved.','8_Look_elsewhere':'No architecture/hyperparameter/seed search or closed-route revival in correction.','9_Forking_paths':'Historical1e-8 gate remains; reference replacement proposed but never activated. New isolated v2 correction frozen before training.','10_Correlation_causation':'Same input/weights/buffers flag toggle establishes conditional engineering effect; not efficacy or particular-kernel attribution.','11_Reverse_causality':'Historical evaluation mismatch is not evidence for the proposed small-object error cause.'}
report={'status':'LOCAL_EMA_REPLAY_AND_CAUSAL_FLAG_ARRAY_VERIFICATION_PASS','archive_sha256':expected,'archive_files_verified':len(files)+1,'images':548,'native12_stats_exact_historical':True,'local_PR_recomputed_max_abs':max(abs(x-y) for x,y in zip(stats,result['current_stats'])),'six_forward_arrays_verified':True,'full_replay_process_exited1_current_reference_hypothesis_rejected':True,'historical_gate_separately_passed':True,'optimizer_steps':0,'epochs_added':0,'material_passport':{'skill':'academic-research-suite','workflow':'experiment-agent','mode':'run+validate','engineering_scope':'VERIFIED_ARRAYS_AND_FLAGS','scientific_scope':'ANALYZED_NO_EFFICACY'},'fallacy_scan':{'checked':11,'total':11,'items':fallacies},'limits':['Only seed1 full nativeEMA historical replay locally verified here; seed2 still pending.','No specific low-level kernel identified.','No candidate AP or accepted innovation.']}
(OUT/'validation_report.json').open('x',encoding='utf8').write(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='fallacy_scan'},ensure_ascii=False))
