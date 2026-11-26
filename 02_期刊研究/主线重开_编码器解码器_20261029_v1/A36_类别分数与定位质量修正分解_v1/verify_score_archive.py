from pathlib import Path,PurePosixPath
import hashlib,json,sys,zipfile
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=read(D/'score_protocol_FROZEN.json');done=read(D/'remote_complete.json');arc=D/'native_score_evidence.zip'
assert arc.stat().st_size==done['archive_bytes'] and sha(arc)==done['archive_sha256'] and done['head_linear_calls']==12
E=D/'downloaded_evidence';E.mkdir(exist_ok=False)
with zipfile.ZipFile(arc) as z:
    assert len(z.infolist())==done['archived_files']
    for info in z.infolist():
        p=PurePosixPath(info.filename);assert not p.is_absolute() and '..' not in p.parts
        dest=E.joinpath(*p.parts).resolve();assert E.resolve() in dest.parents;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(z.read(info))
files=read(E/'file_manifest.json')['files_sha256']
for n,h in files.items():assert sha(E/n)==h,n
assert len(files)+1==done['archived_files']
for n,h in read(D/'execution_inputs_manifest.json')['remote_files_sha256'].items():assert sha(E/n)==h and sha(D/n)==h,n
assert sha(E/'execution_inputs_manifest.json')==sha(D/'execution_inputs_manifest.json')
head=read(E/'head_result.json');assert head['head_linear_calls']==12 and all(head[k]==0 for k in ['full_model_calls','decoder_calls','optimizer_updates','GPU_calls'])
with np.load(E/'head_parameters.npz',allow_pickle=False) as z:W=z['weight'].astype(np.float64);bias=z['bias'].astype(np.float64)
linear_error=sigmoid_error=0.;arrays=0
for c in cfg['cases']:
    assert sha(c['local_capture'])==c['capture_sha256']
    with np.load(c['local_capture'],allow_pickle=False) as old,np.load(E/'captures'/('case%02d.npz'%c['index']),allow_pickle=False) as new:
        for a,b in [('boxes','depth3_boxes'),('native_logits','depth3_logits'),('native_probabilities','depth3_probabilities')]:
            assert np.array_equal(new[a],old[b]);arrays+=1
        expected=old['depth3_features'].astype(np.float64)@W.T+bias
        err=float(np.abs(expected-new['raw_logits']).max());linear_error=max(linear_error,err);assert err<=1e-4
        probability=1/(1+np.exp(-new['raw_logits'].astype(np.float64)))
        err=float(np.abs(probability-new['raw_probabilities']).max());sigmoid_error=max(sigmoid_error,err);assert err<=1.5e-7
        assert np.array_equal(new['offset_per_class'],new['native_logits'].astype(np.float64)-new['raw_logits'].astype(np.float64))
        proof=head['fidelity'][c['index']];assert proof['class_offset_spread']<=proof['rounding_bound'] and proof['native_predictions_exact']
expected=read(Path(cfg['cases'][0]['local_capture']).parent.parent/'raw_native_predictions_without_region_filter.json')
assert read(E/'predictions_without_region_filter.json')['NATIVE']==expected
out=dict(status='PASS_SCORE_ARCHIVE_AND_LINEAR_PROJECTION',archive_files=done['archived_files'],manifest_files_verified=len(files),
         native_arrays_compared=arrays,independent_float64_linear_max_error=linear_error,independent_sigmoid_max_error=sigmoid_error,
         head_linear_calls=12,new_verifier_model_calls=0,
         scope='Pinned final-headfunctional ablation;not independentsemantic/classqualitycalibration or causalmechanismproof.')
with (D/'archive_verification.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out))
