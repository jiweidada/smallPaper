"""Verify immutable terminal evidence; never starts or resumes training."""
import hashlib,json,pathlib,zipfile
import numpy as np
ROOT=pathlib.Path(r'C:\Users\jiwei\Desktop\小论文投稿')
OUT=ROOT/'outputs/small_object_encoder_decoder12_20261010_v2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf8'))
archives=[('terminal_evidence_seed1.zip','051d4ef8e3f98f2ecfc4e1c2638afdc624f6064d0e36e41519e3ebe7f992ea0f','terminal_seed1','manifest.json'),('full_EMA_evidence_seed2.zip','c8cbbeb148fc50b5aa3e7d72ed6de05c404cae1e098111608e9ac0eddad927ea','full_EMA_seed2','manifest.json'),('bootstrap_evidence_seed1.zip','2035dc0c0a83bb2bb5be6da0d0b76adb7776d2206c34566551b5875791617e12','bootstrap_seed1','bootstrap_evidence_manifest_seed1.json')]
verified=[]
for name,expected,label,mname in archives:
    a=OUT/'collection'/name;assert sha(a)==expected,name
    dest=OUT/'collection'/label
    with zipfile.ZipFile(a) as z:
        assert z.testzip() is None
        for n in z.namelist():assert (dest/n).resolve().is_relative_to(dest.resolve())
        if not dest.exists():
            dest.mkdir();z.extractall(dest)
        assert {p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()}==set(z.namelist())
    m=read(dest/mname);fs=m['files']
    for n,v in fs.items():
        p=dest/n;assert p.is_file(),n
        h=v['sha256'] if isinstance(v,dict) else v
        assert sha(p)==h,n
        if isinstance(v,dict) and 'bytes' in v:assert p.stat().st_size==v['bytes'],n
    verified.append({'archive':name,'sha256':expected,'files_verified':len(fs)+1})
a=OUT/'collection/terminal_seed1';b=OUT/'collection/full_EMA_seed2'
replays=[]
for seed,d in [(1,a/'seed1_fine/baseline_eval'),(2,b/'complete_replay_seed2_v1')]:
    c=read(d/'completion.json');p=read(d/'endpoint_epoch12/summary.json')
    values=read(d/'historical_comparison.json') if (d/'historical_comparison.json').is_file() else c
    current=values['current_stats'];historical=values['historical_stats']
    assert current==historical and c['historical_log_reproduction_pass']
    with np.load(d/'endpoint_epoch12/precision_recall.npz') as z:
        precision,recall=z['precision'],z['recall'];avg=lambda v:float(v[v>-1].mean())
        stats=[avg(precision[:,:,:,0,2]),avg(precision[0,:,:,0,2]),avg(precision[5,:,:,0,2]),*[avg(precision[:,:,:,x,2]) for x in (1,2,3)],*[avg(recall[:,:,0,x]) for x in (0,1,2)],*[avg(recall[:,:,x,2]) for x in (1,2,3)]]
    error=max(abs(x-y) for x,y in zip(stats,current));assert error<1e-12
    imgs=list((d/'stages_epoch12').glob('image_*.npz'));assert len(imgs)==548
    for f in imgs:
        with np.load(f) as z:assert all(np.isfinite(z[n]).all() for n in z.files)
    replays.append({'seed':seed,'all12_native_stats_exact_historical':True,'local_PR_recomputed_max_abs':error,'images':548,'AP_percent':100*current[0],'APs_percent':100*current[3],'AP75_percent':100*current[2]})
source=read(a/'code/source_binding.json')['native_files_sha256']
for f in (a/'native_sources/engine').rglob('*.py'):assert sha(f)==source[f.relative_to(a/'native_sources').as_posix()]
run=a/'seed1_fine/run';rows=read(run/'first8_optimizer_steps.json');assert len(rows)==8
inputs=[json.loads(x) for x in (run/'first100_training_inputs.jsonl').read_text().splitlines()];assert len(inputs)==10
assert [x['attempt'] for x in inputs]==list(range(1,11))
assert [x['actual_optimizer_step'] for x in rows]==list(range(1,9))
assert rows[-1]['attempt']==10
prefixes=['encoder.geometry_field.','decoder.enc_bbox_head.geometry_proposal.','decoder.decoder.geometry_layers.','decoder.decoder.geometry_attention.']
groups={}
for prefix in prefixes:
    entries={n:v for n,v in rows[-1]['delta_from_initial'].items() if n.startswith(prefix)};assert entries,prefix
    grads=[v for n,v in rows[-1]['gradients_after_native_unscale_clip'].items() if n.startswith(prefix)]
    assert all(v is not None and v['finite'] for v in grads)
    groups[prefix]={'parameter_tensors':len(entries),'max_recorded_delta_norm':max(entries.values()),'max_recorded_clipped_gradient_norm':max(v['norm'] for v in grads)}
assert groups[prefixes[1]]['max_recorded_delta_norm']>0
assert all(groups[x]['max_recorded_delta_norm']==0 for x in [prefixes[0],prefixes[2],prefixes[3]])
assert read(a/'seed1_fine/candidate_completion.json')['exit_code']==1
assert read(a/'seed1_fine/controller_failure.json')['status']=='FAILED_NO_RETRY'
assert 'A new parameter group still has no actual update' in (a/'seed1_fine/candidate.log').read_text()
assert not list(run.rglob('*.pth')) and not (run/'training_completion.json').exists()
snapshot=read(a/'terminal_snapshot.json');assert snapshot['checkpoint_files']==[]
zero=OUT/'collection/bootstrap_seed1/zero_ema_preflight_v1'
with np.load(zero/'outputs.npz') as z:
    for k in ('pred_boxes','pred_logits'):assert np.array_equal(z['model_0_'+k],z['model_1_'+k])
assert read(zero/'summary.json')['pass']
fallacies={
 '1_Simpson':'Seeds kept separate; no pooled improvement claimed.',
 '2_Ecological':'Engineering gradients cannot establish dataset-level efficacy.',
 '3_Berkson':'All548 validation images retained; candidate early failure fully reported.',
 '4_Collider':'Slotwise query differences between inference flags are not object-identity errors.',
 '5_Base_rates':'Original COCO denominators and all12 statistics retained; units percent.',
 '6_Regression_to_mean':'Fixed historical12 endpoint; no best-epoch substitution.',
 '7_Survivorship':'Original failed controllers, partial8 actual updates and missing checkpoint retained.',
 '8_Look_elsewhere':'No new seed/configuration or automatic second-seed launch after failure.',
 '9_Forking_paths':'Historical exact gate retained; no tolerance/reference replacement activated.',
 '10_Correlation_causation':'Zero norm alone does not prove tensorwise zero or architecture ineffectiveness.',
 '11_Reverse_causality':'No candidate AP measured; no accuracy conclusion from audit failure.'}
result={'status':'PASS_LOCAL_TERMINAL_EVIDENCE_VERIFICATION','archives':verified,'baseline_replays':replays,'first_common_EMA_outputs_exact':True,'training_attempts':10,'actual_optimizer_updates':8,'completed_epochs':0,'candidate_AP':None,'candidate_checkpoint_saved':False,'last_added_group_statistics':groups,'seed2_candidate_started':False,'seed2_direct_model_equivalence_hypothesis':'FAILED; historical native EMA equivalence separately PASS','material_passport':{'skill':'academic-research-suite','workflow':'experiment-agent','mode':'run+validate','engineering_scope':'VERIFIED_TERMINAL_LOGS_HASHES_AND_RECOMPUTED_BASELINE_METRICS','scientific_scope':'ANALYZED_NO_CANDIDATE_EFFICACY'},'fallacy_scan':{'checked':11,'total':11,'items':fallacies},'limits':['Original first10 input/RNG equality enforced in remote audit, not independently recomputed here.','FP32 norm measurements may hide very small nonzero entries.','No raw terminal parameter tensors saved; exact tensorwise update status cannot be reconstructed.','No candidate endpoint12 AP, no accepted improvement, no retry or continuation.']}
with (OUT/'terminal_evidence_validation.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in result.items() if k not in ('fallacy_scan',)},ensure_ascii=False))
