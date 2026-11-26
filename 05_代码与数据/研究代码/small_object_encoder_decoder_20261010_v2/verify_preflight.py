"""Local collection integrity and saved engineering-record parity checks."""
from pathlib import Path
import base64,datetime,hashlib,json,math,zipfile
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=ROOT/'outputs/small_object_encoder_decoder_20261010_v2'
EXPECTED='efcf04541d8c3adcf522fd399e751001443bebbac89f6f60f1989fd8c7303fca'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    data=base64.b64decode((OUT/'evidence_bundle.zip.b64').read_text())
    assert len(data)==117487 and hashlib.sha256(data).hexdigest()==EXPECTED
    archive=OUT/'evidence_bundle.zip'
    if archive.exists():assert archive.read_bytes()==data
    else:archive.write_bytes(data)
    destination=OUT/'collection';destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.infolist())==52
        for info in z.infolist():
            target=(destination/info.filename).resolve()
            assert target.is_relative_to(destination.resolve()) and not info.is_dir()
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert target.read_bytes()==z.read(info)
            else:target.write_bytes(z.read(info))
        assert {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()}==set(z.namelist())
    failed=destination/'revision1';passed=destination/'revision2';capture=passed/'preflight_v2'
    assert read(failed/'controller_completion.json')['exit_code']==1
    failure=read(failed/'preflight_v1/failure.json');assert failure['forward_calls_completed']==5 and failure['backward_calls_completed']==0 and failure['optimizer_steps']==0
    assert read(passed/'controller_completion.json')['exit_code']==0 and not read(passed/'controller_completion.json')['timed_out']
    for directory in [failed,passed]:
        protocol=read(directory/'code/preflight_protocol.json')
        assert all(sha(directory/'code'/k)==v for k,v in protocol['implementation_sha256'].items())
        local_code=HERE if directory==passed else HERE.with_name('small_object_encoder_decoder_20261010_v1')
        assert all(sha(local_code/k)==v for k,v in protocol['implementation_sha256'].items())
        assert sha(directory/'code/preflight_protocol.json')==sha(local_code/'preflight_protocol.json')==read(directory/'controller_start.json')['protocol_sha256']
    manifest=read(capture/'manifest.json');assert all(sha(capture/k)==v for k,v in manifest['files'].items())
    completion=read(capture/'completion.json');assert completion['model_forward_calls']==17 and completion['model_backward_calls']==6 and completion['optimizer_steps']==completion['epochs_added']==0
    initializations=completion['initializations'];assert initializations[1]['parameters']==initializations[2]['parameters'] and initializations[1]['extra_parameters']==initializations[2]['extra_parameters']
    source=read(capture/'source_identity.json');prior=read(ROOT/'outputs/encoder_pool_diagnostic_20261010_v1/collection_seed1/capture_v1/source_identity.json')
    assert source['native_files_sha256']==prior['native_files_sha256'] and len(source['native_files_sha256'])==468
    checks=[]
    for amp in (0,1):
        for train in (0,1):
            key=f'common_amp{amp}_train{train}';native=read(capture/f'native_{key}.json')
            for variant in ('fine','degraded'):
                candidate=read(capture/f'{variant}_{key}.json')
                assert candidate['output_identity']==native['output_identity'] and candidate['losses']==native['losses'] and candidate['rng_after']==native['rng_after']
                assert all(x['exact'] and x['max_abs_error']==0 for x in candidate['comparison'].values())
                checks.append({'case':key,'variant':variant,'tensor_output_identity_loss_RNG_exact':True,'native_loss_count':len(candidate['losses']),'native_buffer_remote_assertion_verified_in_bound_code':True})
    with np.load(capture/'initial_predictions.npz',allow_pickle=False) as z:
        assert len(z.files)==12
        for amp in (0,1):
            key=f'common_amp{amp}_train0'
            for field in ('pred_boxes','pred_logits'):
                base=z['native.'+key+'.'+field]
                assert all(np.array_equal(base,z[v+'.'+key+'.'+field]) for v in ('fine','degraded'))
                for variant in ('native','fine','degraded'):
                    record=read(capture/f'{variant}_{key}.json')['output_identity']['.'+field]
                    dtype={'torch.float32':'float32','torch.float16':'float16','torch.float64':'float64'}[record['dtype']]
                    value=z[variant+'.'+key+'.'+field].astype(dtype)
                    assert list(value.shape)==record['shape'] and hashlib.sha256(value.tobytes()).hexdigest()==record['sha256']
    native=read(capture/'native_mature_eval.json')
    for v in ('fine','degraded'):assert read(capture/f'{v}_mature_eval.json')['output_identity']==native['output_identity']
    gradients=read(capture/'gradients.json');assert len(gradients)==6
    for item in gradients:
        assert all(x['connected'] and x['finite'] and math.isfinite(x['norm']) for x in item['gradients'].values())
        if item['heads_manually_activated']:
            assert len(item['activation'])==4
            for prefix,norm in item['group_gradient_norms'].items():
                # Python3.12+ sum(float) uses improved summation. Reproduce
                # the remote Python's left-to-right sum without changing any
                # numerical tolerance, norm, output or scientific check.
                accumulator=0.
                for name,g in item['gradients'].items():
                    if name.startswith(prefix):accumulator+=g['norm']**2
                recomputed=accumulator**.5
                assert recomputed==norm and norm>0
    fallacies={
      '1_Simpson':'FP32/AMP,EVAL/TRAIN and two variants retained separately; none pooled into scientific improvement.',
      '2_Ecological':'Repeated single-image engineering probes do not represent VisDrone or training trajectories.',
      '3_Berkson':'Existing image1/81GT fixture chosen before candidate outputs. It is not a representative tiny-object test cohort.',
      '4_Collider':'TopK/native quality losses affect query selection. This check does not identify a harmful selector or gradient.',
      '5_Base_rates':'No AP/recall estimated. Exact initial outputs and parameter counts have explicit engineering denominators.',
      '6_Regression_to_mean':'No learned before/after endpoint or best epoch; temporary manual activation disclosed/discarded.',
      '7_Survivorship':'Failed revision1 preserved with5 forwards0 backwards. Successful revision2 separately bound; both archived.',
      '8_Look_elsewhere':'Earlier negative routes remain closed. This is one architecture hypothesis with same-capacity degradation control, not an efficacy success.',
      '9_Forking_paths':'Architecture/check matrix before execution. One device-comparison correction logged after failure; no architecture/tolerance/fixture/gate changes.',
      '10_Correlation_causation':'Finite gradients/identity imply valid wiring only, not improved localization, manifold, root cause or novelty.',
      '11_Reverse_causality':'Hypothesized task conflict/fine spatial information loss remains unproven. Neither low AP75 nor small PCA dimension establishes it.'}
    report={'status':'PASS_LOCAL_ENCODER_DECODER_PREFLIGHT_COLLECTION_AND_PARITY','material_passport':{'skill':'academic-research-suite','workflow':'experiment-agent','mode':'run+validate','scientific_scope':'ANALYZED_DESIGN_ONLY','engineering_verification':'VERIFIED_BOUND_SOURCE_NATIVE_IDENTITY_AND_FINITE_GRADIENT_PREFLIGHT'},'observed_local_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive_sha256':EXPECTED,'archive_bytes':len(data),'archive_files':52,'capture_files_hash_verified':len(manifest['files']),'native_source_files':468,'failed_revision1':failure,'successful_completion':completion,'pair_checks':checks,'saved_NPZ_initial_prediction_arrays_exact':12,'native_parameters':initializations[0]['parameters'],'prototype_parameters':initializations[1]['parameters'],'added_parameters':initializations[1]['extra_parameters'],'parameter_growth_percent':100*(initializations[1]['parameters']/initializations[0]['parameters']-1),'same_capacity_variants':True,'total_forwards_this_design':22,'total_backwards_this_design':6,'optimizer_steps':0,'epochs_added':0,'AP_evidence':False,'new_mechanism_accepted':False,'formal_training_launched':False,'local_verifier_correction':{'first_verifier_preserved':'local_validation_first_version.py','first_failure':'Aggregated gradient norm differed by1ULP; all individual norm records exact.','resolved':'Python3.12+ sum changed float accumulation; explicit left-to-right addition reproduces remote result exactly for all8 groups. No tolerance loosened or model execution repeated.','norms_exact':8},'fallacy_scan':{'checked':11,'total':11,'items':fallacies},'limits':completion['limits']+['Source/array/record integrity is independently checked locally; nativeBN/constructor parameters assertions executed by bound remote code, not re-executed as a whole locally.','Remote and local clocks differ; freeze ordering established by exact protocol identity and tool sequence, not clock subtraction.']}
    save(OUT/'validation_report.json',report)
    print(json.dumps({k:report[k] for k in ['status','native_parameters','prototype_parameters','added_parameters','parameter_growth_percent','optimizer_steps','epochs_added']}))
if __name__=='__main__':main()
